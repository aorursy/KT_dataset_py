import os

import re



import pandas as pd

import numpy as np

import seaborn as sns



import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker



from IPython.display import Markdown as md
pd.options.display.max_rows = 999
os.listdir('../input/fifa-18-demo-player-dataset')
complete_data = pd.read_csv('../input/fifa-18-demo-player-dataset/CompleteDataset.csv', index_col=0)

complete_data.head()
c_data = complete_data.copy()
c_data.shape
c_data.info()
c_data.describe()
c_data.duplicated().sum()
c_data = c_data.drop_duplicates()
def get_number_missing_values_orig(df):

    """Return Series object with number of missing values in columns of DataFrame.

    

    Series object contains data only about columns, where missing values occurs.

    """

    return df.isnull().sum()[df.isnull().any()]
def get_missing_values(df):

    """Return number and percent of missing values in columns of DataFrame.

    

    Contains data only about columns, where missing values occurs.

    """

    total = df.isnull().sum()[df.isnull().any()]

    percent = df.isnull().sum()[df.isnull().any()]/df.shape[0]

    missing = pd.concat([total, percent], axis=1)

    missing.columns = ['Total', 'Percent']

    return missing
nomv = get_missing_values(c_data)

nomv
club_missing_values = c_data[c_data['Club'].isnull()]

club_missing_values.shape[0]
club_missing_values['Wage'].value_counts()
print(f'{club_missing_values.shape[0]} players do not play for any club and those players do not have wage (\u20ac0)')
playing_position_columns = nomv.index.tolist()[1:]
gks = c_data[(c_data['Preferred Positions'] == 'GK ') & (c_data[playing_position_columns].isnull().all(1))]

gks.shape[0]
md(f'Goalkeepers has no values for parameters connected with position on the field (there are {gks.shape[0]} players in this position).')
c_data['Acceleration'].unique()
def str_int(s):

    """Return integer value from string object.

    

    Solves a simple equation contained in a string and returns a solution as integer value.

    """

    pattern = re.compile(r'(\d+)([+-]*)(\d*)')

    if isinstance(s, int):

        return s

    else:

        m = pattern.match(s)

        sign = m.group(2)

        if sign == '-':

            return int(m.group(1)) - int(m.group(3))

        elif sign == '+':

            return int(m.group(1)) + int(m.group(3))

        else:

            return int(m.group(1))
for column in c_data.loc[:,'Acceleration':'Volleys']:

    c_data[column] = c_data[column].apply(str_int)
assert c_data.loc[:, 'Acceleration':'Volleys'].dtypes.all() == np.dtype('int64')
c_data[['Value', 'Wage']].head()
def get_amount_value(s):

    """Return amount value from string object.

    

    Converts the value of an amount with additional symbols to a floating point value.

    """

    pattern = re.compile(r'(\D*)(\d+\.\d*|\d+)(\D*)')

    m = pattern.match(s)

    if m.group(3) == 'M':

        return float(m.group(2)) * 1000000

    elif m.group(3) == 'K':

        return float(m.group(2)) * 1000

    else:

        return float(m.group(2))
c_data['Value'] = c_data['Value'].apply(get_amount_value)

c_data['Wage'] = c_data['Wage'].apply(get_amount_value)
assert c_data['Value'].dtype == np.dtype('float64')

assert c_data['Wage'].dtype == np.dtype('float64')
c_data.head()
c_data.info()
def format_value_y(t_val, t_pos):

    """Formats numbers to be graphically friendly."""

    if t_val > 1000000:

        return f'{int(t_val/1000000)}M'

    if t_val >1000:

        return f'{int(t_val/1000)}K'

    else:

        return int(t_val)
fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey='row')

fig.suptitle('Distribution of values and wages of players', fontsize=20)

axes[0, 0].hist(c_data['Wage'], bins=80)

axes[0, 0].set_ylabel('count', fontsize=12)

axes[0, 0].xaxis.set_major_formatter(ticker.FuncFormatter(format_value_y))

axes[0, 1].hist(c_data['Value'], bins=80)

axes[0, 1].xaxis.set_major_locator(ticker.MaxNLocator(6))

axes[0, 1].ticklabel_format(style='plain', axis='x')

axes[0, 1].xaxis.set_major_formatter(ticker.FuncFormatter(format_value_y))

axes[1, 0].hist(c_data[c_data['Wage'] < 60000]['Wage'], bins=60)

axes[1, 0].set_xlabel('wage [\u20ac]', fontsize=12)

axes[1, 0].set_ylabel('count', fontsize=12)

axes[1, 0].xaxis.set_major_formatter(ticker.FuncFormatter(format_value_y))

axes[1, 1].hist(c_data[c_data['Value'] < 15000000]['Value'], bins=60)

axes[1, 1].set_xlabel('value [\u20ac]', fontsize=12)

axes[1, 1].xaxis.set_major_formatter(ticker.FuncFormatter(format_value_y))



plt.tight_layout()
(c_data['Potential'] - c_data['Overall']).value_counts().sort_index(ascending=True)
plt.rc('axes', labelsize=14)



fig, axes = plt.subplots(1, 2, figsize=(9, 5))

fig.suptitle('Players age distribution and comparison of overall and potential ratings', fontsize=20)



axes[0].hist(c_data['Age'], bins=30, density=True, alpha=.4)

axes[0].set_xlabel('age [years]')



axes[1].hist(c_data['Overall'], bins=40, density=True, alpha=.4, label='overall')

axes[1].hist(c_data['Potential'], bins=40, density=True, alpha=.4, label='potential')

axes[1].set_xlabel('overall/potential rating')

axes[1].legend(loc='best')



plt.tight_layout()
pair_attributes = ['Age', 'Wage', 'Value', 'Overall', 'Potential']
sns.pairplot(c_data[pair_attributes], diag_kind='kde');
columns = ['Name', 'Overall'] + c_data.columns[-28:].values.tolist()

field_players = c_data[c_data['Preferred Positions'] != 'GK '][columns]

field_players.head()
avg_points = field_players[field_players.columns.drop(['Overall', 'ID'])].mean(axis=1, numeric_only=True)

avg_points.index = field_players.Name

avg_points.sort_values(ascending=False).head()
md(f'The highest average of position attributes has: {avg_points.idxmax()}')
npn = 11 # minimum number of players from nationality
grouped_nat = c_data.groupby('Nationality').agg(['count', 'mean'])['Overall']

nat_ge_npn_players = grouped_nat[grouped_nat['count'] >= npn]



players_nat_ge_npn = c_data.loc[c_data['Nationality'].isin(nat_ge_npn_players.index)]

players_nat_npn = players_nat_ge_npn.groupby('Nationality').apply(lambda dfg: dfg.nlargest(npn, columns='Overall'))

players_nat_npn['Overall'].mean(level=0).sort_values(ascending=False)[:10]
assert (players_nat_npn.groupby(level=0).size() == npn).all()
positions = []

for pp in c_data['Preferred Positions'].unique():

    poss = [p for p in pp.split(' ') if p not in positions and p != '']

    positions = positions + poss

print(positions)
pref_posits = pd.Series(c_data['Preferred Positions'].values, index=c_data['ID'].values)

pref_posits = pref_posits.str.split(' ')

for e in pref_posits:

    e.pop(e.index(''))

pref_posits.head()
def get_preferred_positions_table(positions=positions, pref_posits=pref_posits):

    """Return the players' preferred positions encoded in dummy variables."""

    ppt = pd.DataFrame(0, index=pref_posits.index, columns=positions)

    for i in ppt.iterrows():

        pp = pref_posits.loc[i[0]]

        i[1].loc[pp] = 1

    return ppt
preferred_positions_table = get_preferred_positions_table()

preferred_positions_table.head()
formations = {'3-4-1-2': ['GK', 'CB', 'CB', 'CB', 'CM', 'CM', 'LM', 'RM', 'CF', 'ST', 'ST'],

              '3-4-2-1': ['GK', 'CB', 'CB', 'CB', 'CM', 'CM', 'LM', 'RM', 'LW', 'RW', 'ST'],

              '3-4-3': ['GK', 'CB', 'CB', 'CB', 'CM', 'CM', 'LM', 'RM', 'LW', 'ST', 'RW'],

              '3-5-2': ['GK', 'CB', 'CB', 'CB', 'CM', 'CM', 'LM', 'RM', 'CAM', 'ST', 'ST'],

              '4-1-2-1-2': ['GK', 'CB', 'CB', 'LB', 'RB', 'CDM', 'LM', 'RM', 'CAM', 'ST', 'ST'],

              '4-2-3-1': ['GK', 'CB', 'CB', 'LB', 'RB', 'CDM', 'CDM', 'CF', 'CAM', 'CAM', 'ST'],

              '4-2-2-2': ['GK', 'CB', 'CB', 'LB', 'RB', 'CDM', 'CDM', 'CAM', 'CAM', 'ST', 'ST'],

              '4-3-1-2': ['GK', 'CB', 'CB', 'LB', 'RB', 'CM', 'CM', 'CM', 'CF', 'ST', 'ST'],

              '4-3-2-1': ['GK', 'CB', 'CB', 'LB', 'RB', 'CM', 'CM', 'CM', 'LW', 'RW', 'ST'],

              '4-3-3': ['GK', 'CB', 'CB', 'LB', 'RB', 'CM', 'CM', 'CM', 'LW', 'ST', 'RW'],

              '4-4-1-1': ['GK', 'CB', 'CB', 'LB', 'RB', 'LM', 'CM', 'CM', 'RM', 'CF', 'ST'],

              '4-4-2': ['GK', 'CB', 'CB', 'LB', 'RB', 'CM', 'CM', 'LM', 'RM', 'ST', 'ST'],

              '4-5-1': ['GK', 'CB', 'CB', 'LB', 'RB', 'LM', 'CM', 'RM', 'CAM', 'CAM', 'ST'],

              '5-2-1-2': ['GK', 'LWB', 'CB', 'CB', 'CB', 'RWB', 'CM', 'CM', 'CF', 'ST', 'ST'],

              '5-2-2-1': ['GK', 'LWB', 'CB', 'CB', 'CB', 'RWB', 'CM', 'CM', 'LW', 'RW', 'ST'],

              '5-3-2': ['GK', 'LWB', 'CB', 'CB', 'CB', 'RWB', 'CM', 'CM', 'CM', 'ST', 'ST']

             }
def get_dream_team(formation, data=c_data, formations=formations, player_p=preferred_positions_table):

    """Return the best players from the given dataset for a specific formation."""

    dt = pd.DataFrame(columns=['ID', 'Name', 'Club', 'Overall', 'Preferred Positions', 'Position'])

    dt['ID'] = dt['ID'].astype('int64')

    dt['Overall'] = dt['Overall'].astype('int64')

    for pos in formations[formation]:

        ps = data[data['ID'].isin(player_p[player_p[pos] == 1].index)][['ID', 'Name', 'Club', 'Overall', 'Preferred Positions']].sort_values(by='Overall', ascending=False)

        if ps.empty:

            raise IndexError(f'No player for position {pos}')

        i = 0

        while (ps.iloc[i]['ID'] not in dt['ID'].values) == False:

            i += 1

        player = ps.iloc[i:i+1,:]

        player = player.assign(Position=pos)

        dt = dt.append(player, ignore_index=True)

    return dt
bc_data = c_data[['ID', 'Name', 'Club', 'Overall']]

bc_data.head()
bp_club = bc_data[['Club', 'Overall']].sort_values('Overall', ascending=False).groupby('Club', sort=False).mean().head()

bp_club.style.format('{:.1f}')
def get_formation_scores(data, formations=formations, player_p=preferred_positions_table):

    """Return the sum of players' overall score for the best team in specific formation from given dataset."""

    # TODO napisać, że wartość 0 oznacza brak zawodnika dla jakiejś pozycji

    formation_scores = {}

    for f in formations:

        try:

            formation_best_players = get_dream_team(f, data)

            formation_score = formation_best_players['Overall'].sum()

            formation_scores[f] = formation_score

        except IndexError as ie:

            formation_scores[f] = 0

    return formation_scores
def get_best_club_formation(data, formations=formations, player_p=preferred_positions_table):

    """Return the best formation data for a given club and the sum of the overall player ratings for that formation."""

    club_name = data['Club'].unique()

    assert len(club_name) == 1, 'Data contains info about players without a club or from more than one club'

    

    formation_scores = get_formation_scores(data)

    forms = []

    max_v = max(formation_scores.values())

    for k, v in formation_scores.items():

        if v == max_v:

            forms.append(k)

    return {'points': max_v, 'club': club_name[0], 'formations': forms}
def get_best_clubs_formation(data):

    """Return data about best club with they best formation(s) and sum of players' overall score in this formation(s)."""

    best_clubs = []

    data = data.dropna(subset=['Club'])

    clubs = data['Club'].unique()

    for club in clubs:

        club_data = data[data['Club'] == club]

        bcf = get_best_club_formation(club_data)

        if len(best_clubs) == 0 or bcf['points'] == best_clubs[0]['points']:

            best_clubs.append(bcf)

        elif bcf['points'] > best_clubs[0]['points']:

            best_clubs = [bcf]

    return best_clubs
get_best_clubs_formation(c_data)
formation = '4-4-2'

get_dream_team(formation)
pl_players = c_data[c_data['Nationality'] == 'Poland']

pl_players.head()
formation_pl = '4-4-2'

get_dream_team(formation_pl, data=pl_players)
mpl.rc('image', cmap='Pastel1')

cmap = mpl.cm.get_cmap(name='Pastel1')



fig, axes = plt.subplots(4, 2, figsize=(12, 16), sharey='row')

fig.suptitle('Polish players vs. world players', fontsize=20, y=1.01)



axes[0, 0].hist(pl_players['Age'], bins=20, density=True, alpha=.4, label='polish')

axes[0, 0].hist(c_data['Age'], bins=20, density=True, alpha=.4, label='world')

axes[0, 0].set_xlabel('age (years)')

axes[0, 0].set_ylabel('count')

axes[0, 0].legend(loc='best')



axes[0, 1].hist(pl_players['Overall'], bins=40, density=True, alpha=.4, label='polish')

axes[0, 1].hist(c_data['Overall'], bins=40, density=True, alpha=.4, label='world')

axes[0, 1].set_xlabel('overall score')

axes[0, 1].legend(loc='best')



axes[1, 0].scatter(pl_players['Age'], pl_players['Overall'], alpha=.4, s=6, label='polish')

axes[1, 0].set_xlabel('age (years)')

axes[1, 0].set_ylabel('overall score')

axes[1, 0].legend(loc='best')



axes[1, 1].scatter(c_data['Age'], c_data['Overall'], alpha=.4, color=cmap(.5), s=6, label='world')

axes[1, 1].set_xlabel('age (years)')

axes[1, 1].legend(loc='best')



axes[2, 0].scatter(pl_players['Overall'], pl_players['Value'], alpha=.2, s=6, label='polish')

axes[2, 0].set_xlabel('overal score')

axes[2, 0].set_ylabel('value [\u20ac]')

axes[2, 0].yaxis.set_major_formatter(ticker.FuncFormatter(format_value_y))

axes[2, 0].legend(loc='best')



axes[2, 1].scatter(c_data['Overall'], c_data['Value'], alpha=.2, color=cmap(.5), s=6, label='world')

axes[2, 1].set_xlabel('overal score')

axes[2, 1].legend(loc='best')



axes[3, 0].scatter(pl_players['Overall'], pl_players['Wage'], s=6, label='polish')

axes[3, 0].set_xlabel('overal score')

axes[3, 0].set_ylabel('wage [\u20ac]')

axes[3, 0].yaxis.set_major_formatter(ticker.FuncFormatter(format_value_y))

axes[3, 0].legend(loc='best')



axes[3, 1].scatter(c_data['Overall'], c_data['Wage'], s=6, label='world')

axes[3, 1].set_xlabel('overal score')

axes[3, 1].legend(loc='best')



plt.tight_layout()
club_player_values = c_data[['Club', 'Value']].groupby('Club').agg(['mean', 'min', 'max', 'sum']).sort_values(by=('Value', 'mean'), ascending=False)

club_player_values.head().style.format(' \u20ac{:.0f}')
player_weges = c_data[['Wage', 'Club']].groupby('Club').agg(['mean', 'min', 'max', 'sum']).sort_values(by=('Wage', 'mean'), ascending=False)

player_weges.head().style.format('\u20ac{:.0f}')
players_age = c_data[['Age', 'Club']].groupby('Club').mean().sort_values(by='Age')
print(f'The club with the lowest average age of players:\t{players_age["Age"].idxmin()} ({players_age["Age"].iloc[0]:.1f})')

print(f'The club with the highest average age of players:\t{players_age["Age"].idxmax()} ({players_age["Age"].iloc[-1]:.1f})')