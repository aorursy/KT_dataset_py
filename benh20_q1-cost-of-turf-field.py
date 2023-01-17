import pandas as pd

import scipy.stats as stats

import numpy as np

from tqdm import tqdm

import statsmodels.api as sm

import statsmodels.formula.api as smf

import matplotlib.pyplot as plt



"""

Import datasets 

https://www.kaggle.com/c/nfl-playing-surface-analytics/data

"""



KAGGLE = True

if not KAGGLE:

    IR_data = pd.read_csv('InjuryRecord.csv')

    PL_data = pd.read_csv('PlayList.csv')

else:

    IR_data = pd.read_csv('../input/nfl-playing-surface-analytics/InjuryRecord.csv')

    PL_data = pd.read_csv('../input/nfl-playing-surface-analytics/PlayList.csv')



pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 500)



plt.rcParams['figure.dpi'] = 200

plt.rcParams.update({'font.size': 14})
def cleanIRdata(IR_data):



    # Mark whether we need to set a play for each row

    IR_data['PlayBool'] = [False if pd.isnull(i) else True for i in IR_data['PlayKey']] 



    # Get maximum play count for each play

    IR_data['HighPlay'] = [PL_data[PL_data['GameID'] == i]['PlayerGamePlay'].max() for i in IR_data['GameID']] 



    # Create a new playkey

    PlayKey_adj = []

    PlayKey = list(IR_data['PlayKey'])

    HighPlay = list(IR_data['HighPlay'])

    PlayBool = list(IR_data['PlayBool'])

    GameID = list(IR_data['GameID'])

    for i in range(len(IR_data)):

        if PlayBool[i]:

            PlayKey_adj.append(PlayKey[i])

        else:

            PlayKey_adj.append(GameID[i] + '-' + str(HighPlay[i]))



    IR_data['PlayKey_adj'] = PlayKey_adj

    return IR_data

    

IR_data = cleanIRdata(IR_data)

IR_data.head()
# Merge data

PL_data['PlayKey_adj'] = PL_data['PlayKey']  

df = pd.merge(PL_data, IR_data, on = 'PlayKey_adj', how='left').fillna(0)

df.head()
# This function calculates the number of games missed and appends it to our dataframe

def countGamesMissed(df):

    

    GM_array = []

    for i, row in df.iterrows():



        DM1  = row['DM_M1']

        DM7  = row['DM_M7']

        DM28 = row['DM_M28']

        DM42 = row['DM_M42']



        if DM42 == 1:

            GM_array.append(6.5)

        elif DM28 == 1:

            GM_array.append(4.5)

        elif DM7 == 1:

            GM_array.append(1.5)

        elif DM1 == 1:

            GM_array.append(.5)

        else:

            GM_array.append(0)



    df['GamesMissed'] = GM_array

    return df
df = countGamesMissed(df)

df[df['GamesMissed'] > 0][['PlayKey_adj', 'DM_M1','DM_M7','DM_M28','DM_M42','GamesMissed']].head()
PGs = ['LB', 'QB', 'DL', 'OL', 'SPEC', 'TE', 'WR', 'RB', 'DB']

injurySummary = pd.DataFrame(columns=['PG', 'Plays', 'Injuries'])

for i in range(len(PGs)):

    PG = PGs[i]

    injurySummary.loc[i] = [PG, len(df[df['PositionGroup'] == PG]), len(df[(df['PositionGroup'] == PG) & (df['DM_M1'] == 1)])]

    

injurySummary
syntheticPlays    = df[df['FieldType'] == 'Synthetic']

syntheticInjuries = df[(df['DM_M1'] == 1) & (df['FieldType'] == 'Synthetic')]

naturalPlays      = df[df['FieldType'] == 'Natural']

naturalInjuries   = df[(df['DM_M1'] == 1) & (df['FieldType'] == 'Natural')]

plays             = df

injuries          = df[df['DM_M1'] == 1]



print('There are', len(IR_data), 'plays in the full dataset')

print('There are', len(plays), 'plays in the merged dataset')

print(len(injuries), 'of', len(plays), "(%.4f" % (len(injuries) / len(plays) * 100) + '%)', 'of plays in the merged dataset are injuries')

print(len(syntheticInjuries), 'of', len(syntheticPlays), "(%.4f" % (len(syntheticInjuries) / len(syntheticPlays) * 100) + '%)', 'sythetic injuries')

print(len(naturalInjuries), 'of', len(naturalPlays), "(%.4f" % (len(naturalInjuries) / len(naturalPlays) * 100) + '%)', 'natural injuries')



# P-Test

natural_sample = [1] * len(naturalInjuries) + [0] * (len(naturalPlays) - len(naturalInjuries))

synthetic_sample = [1] * len(syntheticInjuries) + [0] * (len(syntheticPlays) - len(syntheticInjuries))

t_stat, p_val = stats.ttest_ind(natural_sample, synthetic_sample, equal_var=False)

print('The p-value that synthetic is worse than natural is', "%.5f" % p_val)
SYN_INJURIES = len(set(df[(df['DM_M1'] == 1) & (df['FieldType'] == 'Synthetic')]['GameID_x']))

SYN_GAMES = len(set(df[(df['FieldType'] == 'Synthetic')]['GameID_x']))

NAT_INJURIES = len(set(df[(df['DM_M1'] == 1) & (df['FieldType'] == 'Natural')]['GameID_x']))

NAT_GAMES = len(set(df[(df['FieldType'] == 'Natural')]['GameID_x']))



print(SYN_INJURIES, 'of', SYN_GAMES, "(%.4f" % (SYN_INJURIES / SYN_GAMES * 100) + '%)', 'of synthetic games had injuries')

print(NAT_INJURIES, 'of', NAT_GAMES, "(%.4f" % (NAT_INJURIES / NAT_GAMES * 100) + '%)', 'of natural games had injuries')

print("%.4f" %  ((SYN_INJURIES / SYN_GAMES) / (NAT_INJURIES / NAT_GAMES)), 'higher injury rate on turf')





# P-Test

natural_sample = [1] * NAT_INJURIES + [0] * (NAT_GAMES - NAT_INJURIES)

synthetic_sample = [1] * SYN_INJURIES + [0] * (SYN_GAMES - SYN_INJURIES)

t_stat, p_val = stats.ttest_ind(natural_sample, synthetic_sample, equal_var=False)

print('The p-value that synthetic is worse than natural (on a game level) is', "%.5f" % p_val)
syntheticGM       = sum(df[df['FieldType'] == 'Synthetic']['GamesMissed'])

naturalGM         = sum(df[df['FieldType'] == 'Natural']['GamesMissed'])

SYN_INJURY_AVG = round(syntheticGM / SYN_INJURIES,2)

NAT_INJURY_AVG = round(naturalGM / NAT_INJURIES,2)

print('Each synthetic injury averages', SYN_INJURY_AVG, 'games missed')

print('Each natural injury averages', NAT_INJURY_AVG, 'games missed')

print('Synthetic injuries miss', round(SYN_INJURY_AVG / NAT_INJURY_AVG,3), 'more games')



GM_NAT_LIST = list(df[(df['DM_M1'] == 1) & (df['FieldType'] == 'Natural')]['GamesMissed'])

GM_SYN_LIST = list(df[(df['DM_M1'] == 1) & (df['FieldType'] == 'Synthetic')]['GamesMissed'])



t_stat, p_val = stats.ttest_ind(GM_NAT_LIST, GM_SYN_LIST, equal_var=False)

print('The p-value that synthetic injuries are worse (in terms of games missed) than natural injuries is', "%.3f" % p_val)
def convertVariablesForRegression(tmp):



    def isIn(item, word):

        return item in str(word).lower()



    # Convert some fields to boolean

    tmp['Is_Synthetic'] = tmp['FieldType'] == 'Synthetic'

    tmp['Is_Rain'] = [True if isIn('rain', i) or isIn('shower', i) else False for i in tmp['Weather']]

    tmp['Is_Snow'] = [True if isIn('snow', i) else False for i in tmp['Weather']]



    # Clean some variables

    tmp['Temperature_Adj'] = [i if i != -999 else 72 for i in tmp['Temperature']]

    

    return tmp

    

df = convertVariablesForRegression(df)



display(df[['PlayKey_adj', 'PositionGroup', 'PlayerDay', 'PlayerGamePlay', 'Is_Synthetic', 'Temperature_Adj', 'Is_Rain', \

    'Is_Snow', 'GamesMissed']].head())
X = 'GamesMissed'

Y = ['C(PositionGroup) * Is_Synthetic', 'Is_Synthetic', 'PlayerDay', 'PlayerGamePlay', 'Temperature_Adj', 'Is_Rain', 'Is_Snow']

Y = ' + '.join(Y)



model = sm.GLM.from_formula(X + ' ~ ' + Y, data=df, family=sm.families.Poisson()).fit()

# model = smf.ols(X + ' ~ ' + Y, data = df).fit()

print(model.summary())
# Create testing dataframe to predict on

td = pd.DataFrame()

PG = list(set(df['PositionGroup']))



# Get average plays per game by position

PPG = []

for i in PG:

    POS_PLAYS = len(df[df['PositionGroup'] == i])

    POS_GAMES = len(set(df[df['PositionGroup'] == i]['GameID_x']))

    PPG.append(int(POS_PLAYS / POS_GAMES))



# Use mean for each group

def getMeanForPG(stat, PG=PG, df=df):

    return [df[df['PositionGroup'] == pos_group][stat].median() for pos_group in PG]



td['PositionGroup']     = PG

td['Is_Synthetic']      = [False] * len(PG)

td['PlayerDay']         = getMeanForPG('PlayerDay')

td['PlayerGamePlay']    = getMeanForPG('PlayerGamePlay')

td['Temperature_Adj']   = getMeanForPG('Temperature_Adj')

td['Is_Rain']           = [False] * len(PG)

td['Is_Snow']           = [False] * len(PG)

td['PPG']               = PPG

                    

# Predict

td['GM_Natural']        = model.predict(td)



# Predict on artificial

td['Is_Synthetic']      = [True]  * len(PG)

td['GM_Synthetic']      = model.predict(td)



# Set average values for QB and SPEC since insufficient data (0 injuries)

GM_Avg = {}

for fieldType in ['Natural', 'Synthetic']:

    GM_Avg[fieldType] = sum(df[df['FieldType'] == fieldType]['GamesMissed']) / len(df[df['FieldType'] == fieldType])

    for PG in ['QB', 'SPEC']:

        td.at[list(td[td['PositionGroup'] == PG].index)[0], 'GM_' + fieldType] = GM_Avg[fieldType]



# Differences

td['SyntheticDelta']    = td['GM_Synthetic'] - td['GM_Natural']

td['SyntheticRatio']    = td['GM_Synthetic'] / td['GM_Natural']      



# GM / Game

td['DeltaGMperGame'] = list(td['SyntheticDelta'] * td['PPG'])



# GM / other units

td['GM_Natural_Game'] = list(td['GM_Natural'] * td['PPG'])

td['GM_Synthetic_Game'] = list(td['GM_Synthetic'] * td['PPG'])

td['GM_Natural_HomeSeason'] = [i*8 for i in list(td['GM_Natural'] * td['PPG'])]

td['GM_Synthetic_HomeSeason'] = [i*8 for i in list(td['GM_Synthetic'] * td['PPG'])]  

        

# Remove "missing data" player group

td = td[td['PositionGroup'] != 'Missing Data'].reset_index(drop=True)



# Display

td.sort_values('SyntheticRatio', ascending=False)
def plotGamesMissedPG(PerGame = False, PerHomeSeason=False):



    # Example from here https://pythonspot.com/matplotlib-bar-chart/

    fig, ax = plt.subplots(figsize=(10,5))

    index = np.arange(len(td))

    bar_width = 0.35

    opacity = 0.75

    

    nat_bars = list(td['GM_Natural'])

    syn_bars = list(td['GM_Synthetic'])

    timeframe = 'play'

        

    if PerGame:

        nat_bars = td['GM_Natural_Game']

        syn_bars = td['GM_Synthetic_Game']

        timeframe = 'game'

        

    if PerHomeSeason:

        nat_bars = td['GM_Natural_HomeSeason']

        syn_bars = td['GM_Synthetic_HomeSeason']

        timeframe = 'home season'

    

    

    plt.bar(

                x      = index, 

                height = nat_bars, 

                width  = bar_width,

                alpha  = opacity,

                color  = '#48BB78', # color from tailwindcss

                label  = 'Natural'

           )



    plt.bar(

                x      = index + bar_width, 

                height = syn_bars, 

                width  = bar_width,

                alpha  = opacity,

                color  = '#2B6CB0', # color from tailwindcss

                label  = 'Synthetic'

            )



    ax.spines['top'].set_color('white')

    ax.spines['right'].set_color('white')

    plt.xlabel('Position')

    plt.ylabel('Expected Games Missed (per %s)' % timeframe)

    plt.title('Natural v. Synthetic Turf - Games Missed Due to Injury', fontsize=14, fontweight='bold', pad=20)

    plt.xticks(index + bar_width, list(td['PositionGroup']))

    plt.legend()

    plt.tight_layout()

    plt.show()

    

plotGamesMissedPG(PerGame=True)
plotGamesMissedPG(PerHomeSeason=True)
# Read in all snap data

teams = ['car', 'cle', 'crd', 'gnb', 'nwe', 'pit', 'tam', 'was']



if KAGGLE:

    PATH = '/kaggle/input/nflpfrdata/PFR_data/'

else:

    PATH = 'PFR_DATA/'



snaps = pd.read_csv(PATH + 'snap_count/pit.csv', header=1)

for team in teams[1:]:

    teamSnaps = pd.read_csv(PATH + '/snap_count/%s.csv' % team, header=1)

    snaps = pd.concat([snaps, teamSnaps])

    

snaps[['Player', 'Pos','Num']].head()
# We need to conert all positions to our position groups

# https://en.wikipedia.org/wiki/American_football_positions

positionGroupConvert = {

    'C'    : 'OL', # C

    'CB'   : 'DB', # C

    'CBDB' : 'DB', # C

    'DB'   : 'DB', # C

    'DE'   : 'DL', # Could also be LB

    'DT'   : 'DL', # C

    'FB'   : 'RB', # C

    'FS'   : 'DB', # C

    'FSS'  : 'DB', # C

    'FSSS' : 'DB', # C

    'FSSSS': 'DB', # C

    'G'    : 'OL', # C

    'K'    : 'SPEC', # c

    'LB'   : 'LB', # C

    'LS'   : 'SPEC', # Long snapper

    'NT'   : 'DL', # C

    'P'    : 'SPEC', # C

    'QB'   : 'QB', # C

    'RB'   : 'RB', # C

    'S'    : 'DB', # C

    'SS'   : 'DB', # C

    'SSS'  : 'DB', # C

    'T'    : 'OL', # C

    'TE'   : 'TE', # C

    'WR'   : 'WR', # C

    

    # Extras from salary dataset

    'DL'   : 'DL', # C

    'EDGE' : 'DL', # Same as defensive end

    'HB'   : 'RB', # Halfback

    'ILB'  : 'LB', # C

    'LB-DE': 'DL', # Suggs (DE)

    'LG'   : 'OL', # C

    'LT'   : 'OL', #C

    'NT'   : 'DL', # Nose tackle, this is center on defense

    'OG'   : 'OL', # C

    'OL'   : 'OL', # C

    'OLB'  : 'LB', # C 

    'OT'   : 'OL', # C 

    'QB/TE': 'TE', # Logan Thomas (TE)

    'RB-WR': 'RB', # Ty Montgomery (RB)

}
# Group and count

snaps['PositionGroup'] = [positionGroupConvert[i] for i in snaps['Pos']]

snaps.rename(columns={'Num': 'Off', 'Num.1': 'Def', 'Num.2':'Spec'}, inplace=True)

snaps['Snaps'] = snaps['Off'] + snaps['Def'] + snaps['Spec']

snaps[['Pos', 'Snaps', 'PositionGroup']]

snaps['TeamSnapsPerGame'] = round(snaps['Snaps'] / 16 / 8, 2)

groupedSnaps = snaps.groupby('PositionGroup').sum()

groupedSnaps
# Load data, preview

if KAGGLE:

    PATH = '/kaggle/input/nflpfrdata/PFR_data/'

else:

    PATH = 'PFR_DATA/'

salary = pd.read_csv(PATH + 'salary/salary.csv', header=0)

salary['PositionGroup'] = [i if pd.isnull(i) else positionGroupConvert[i] for i in salary['Pos']]

salary['Salary'] = [int(i.replace('$', '')) for i in salary['Salary']]

salary[['Player', 'Salary', 'PositionGroup']].head()
# Calculate Number of Men from each group on field

for i in ['Off', 'Def', 'Spec']:

    groupedSnaps[i + '_men'] = round(11 * groupedSnaps[i] / sum(groupedSnaps[i]), 2)

    

groupedSnaps['Men'] = groupedSnaps[['Off_men','Def_men','Spec_men']].max(axis=1)

groupedSnaps
# Calculate average position salary

positionSalary = []

for pos in list(groupedSnaps.index):

    numPosition = int(32 * groupedSnaps['Men'][pos])

    positionSalary.append(salary[salary['PositionGroup'] == pos][0:numPosition]['Salary'].mean())

    

groupedSnaps['Salary'] = positionSalary

groupedSnaps['SalaryFormatted'] = ['${:,.2f}m'.format(i/1000000) for i in positionSalary]



groupedSnaps.sort_values(by=['Salary'], ascending=False)
res = pd.merge(td, groupedSnaps, on = 'PositionGroup', how='left').fillna(0)

res[['PositionGroup', 'SyntheticDelta', 'TeamSnapsPerGame', 'Salary']]
# Calculations

HOME_GAMES = 8 / 16

res['InjuryHomeCostNatural'] = res['GM_Natural'] * res['TeamSnapsPerGame'] * res['Salary'] * HOME_GAMES

res['InjuryHomeCostSynthetic'] = res['GM_Synthetic'] * res['TeamSnapsPerGame'] * res['Salary'] * HOME_GAMES

res['InjuryHomeCostDelta'] = res['SyntheticDelta'] * res['TeamSnapsPerGame'] * res['Salary'] * HOME_GAMES



res['TeamSnapsPerGame_F'] = [int(i) for i in res['TeamSnapsPerGame']]

res['Salary_F'] = ['${:,.2f}m'.format(i/1000000) for i in res['Salary']]



res['InjuryHomeCostDelta_F'] = ['${:,.0f}k'.format(i/1000) for i in res['InjuryHomeCostDelta']]





res[['PositionGroup', 'TeamSnapsPerGame_F', 'Salary_F', \

    'SyntheticDelta', 'InjuryHomeCostDelta_F' ]]
# Example from here https://pythonspot.com/matplotlib-bar-chart/

fig, ax = plt.subplots(figsize=(10,5))

index = np.arange(len(td))

bar_width = 0.35

opacity = 0.75



totalNaturalCost = '${:,.2f}m'.format(float(sum(res['InjuryHomeCostNatural']) / 1000000))

totalSyntheticCost = '${:,.2f}m'.format(float(sum(res['InjuryHomeCostSynthetic']) / 1000000))

plt.bar(

            x      = index, 

            height = list(res['InjuryHomeCostNatural']), 

            width  = bar_width,

            alpha  = opacity,

            color  = '#48BB78', # color from tailwindcss

            label  = 'Natural (Total Cost of Injuries: ' + totalNaturalCost + ')'

       )



plt.bar(

            x      = index + bar_width, 

            height = list(res['InjuryHomeCostSynthetic']), 

            width  = bar_width,

            alpha  = opacity,

            color  = '#2B6CB0', # color from tailwindcss

            label  = 'Sythetic (Total Cost: ' + totalSyntheticCost + ')'

        )



ax.set_yticklabels(['${:,}'.format(int(x)) for x in ax.get_yticks().tolist()])

ax.spines['top'].set_color('white')

ax.spines['right'].set_color('white')

plt.xlabel('Position Group')

plt.ylabel('Cost to Team')

plt.title('Home Games on Natural v. Synthetic Turf - Injury Cost', fontsize=14, fontweight='bold', pad=20)

plt.xticks(index + bar_width, list(res['PositionGroup']))

plt.legend()



plt.tight_layout()

plt.show()