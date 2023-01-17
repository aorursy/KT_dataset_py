# data structures

import numpy as np

import pandas as pd



# visualization

import plotly

import plotly.graph_objs as go

import plotly.express as px

plotly.offline.init_notebook_mode(connected=True)



from matplotlib import pyplot as plt

import seaborn as sns



from sklearn import linear_model

from sklearn.cluster import AgglomerativeClustering

from sklearn.decomposition import PCA

import scipy.cluster.hierarchy as sch



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print('Data files:', os.listdir("../input/ufcdataset"))



# Any results you write to the current directory are saved as output.
# read the dataset to a Pandas DataFrame

df = pd.read_csv('../input/ufcdataset/data.csv')
# "meta" stats

[ col for col in df.columns.values if 'Round' not in col ]
# round stats

[ col for col in df.columns.values if 'Round1' in col ]
df.describe()
df.head()
df[['B_Name', 'R_Name', 'Last_round']][df['Max_round'] == 4]
# replacing Max_round of 4 to 3 instead

df.loc[df['Max_round'] == 4, 'Max_round'] = 3



# print to double check

print('There are', df['Max_round'][df['Max_round'] == 3].size, 'fights with max 3 rounds')

print('There are', df['Max_round'][df['Max_round'] == 4].size, 'fights with max 4 rounds')

print('There are', df['Max_round'][df['Max_round'] == 5].size, 'fights with max 5 rounds')
df[['R_Name', 'R_Age', 'Date']][df['R_Name'] == 'Jose Aldo']
# find the date of the last fight

df['Date'].max()
# outliers should bubble to the top or bottom

df.sort_values('Date')['Date']
# setting outliers to standard slash format

df.at[12, 'Date'] = '02/16/2014'

df.at[197, 'Date'] = '06/08/2014'

df.at[78, 'Date'] = '06/29/2014'

df.at[384, 'Date'] = '10/04/2014'

df.at[449, 'Date'] = '11/17/2014'

df.at[334, 'Date'] = '12/20/2014'

df.at[686, 'Date'] = '05/23/2015'



# indicate added column with "_" prefix

df['_Date_year'] = df['Date'].transform(lambda date: int(date[-4:]))
# recalculate age

# formula: Age - (last_year - fight_year)

df['R_Age'] = df['R_Age'] - (df['_Date_year'].max() - df['_Date_year'])

df['B_Age'] = df['B_Age'] - (df['_Date_year'].max() - df['_Date_year'])
# checking our work

df[['R_Name', 'R_Age', 'Date']][df['R_Name'] == 'Jose Aldo']
# get names of fighters will missing weights

missing_weight_names = pd.concat([df['B_Name'][df['B_Weight'].isnull()], df['R_Name'][df['R_Weight'].isnull()]]).unique()

print(missing_weight_names)
# fill in with Googled values

weights = {

    'Lipeng Zhang': 70,

    'Antonio Carlos Junior': 84,

    'Aleksei Oleinik': 65,

    'Cat Zingano': 65,

    'Yao Zhikui': 56,

    'Jack Marshman': 84,

    'Allan Zuniga': 70 

}

for name in missing_weight_names:

    df['B_Weight'][df['B_Name'] == name] = weights[name]

    df['R_Weight'][df['R_Name'] == name] = weights[name]

sorted([ weight for weight in pd.concat([df['B_Weight'], df['R_Weight']]).unique() if weight <= 93 ])
df['B_Weight'][df['B_Weight'] == 76] = 77

df['R_Weight'][df['R_Weight'] == 76] = 77
df[['R_Name', 'R_Weight', 'Date']][df['R_Name'] == 'Diego Sanchez']
no_winby = df[df['winby'].isnull()]

print(len(no_winby), "rows where 'winby' is not set.")
print(df['winner'][df['winner'] == 'no contest'].size, 'no contests')

print(df['winner'][df['winner'] == 'draw'].size, 'draws')
print(no_winby['winner'][no_winby['winner'] == 'draw'].size, 'draws with an empty \'winby\' value')

print(no_winby['winner'][no_winby['winner'] == 'no contest'].size, 'no contest with an empty \'winby\' value')
# set on the original dataset, and recompute `no_winby`

df.loc[df['winner'] == 'draw', 'winby'] = 'DRAW'

no_winby = df[df['winby'].isnull()]
df[['B_Name', 'R_Name', 'Date', 'Last_round', 'winner', 'winby']][(df['winner'] == 'no contest') & (df['winby'].notnull())] # exclude the no-contests with no 'winby'
b = 'blue'

r = 'red'



df.at[40, 'winner'] = b

df.at[70, 'winner'] = r

df.at[234, 'winner'] = r

df.at[255, 'winner'] = r

df.at[301, 'winner'] = b

df.at[403, 'winner'] = r

df.at[428, 'winner'] = r

df.at[513, 'winner'] = r

df.at[628, 'winner'] = r

df.at[734, 'winner'] = r

df.at[894, 'winby'] = np.NaN

df.at[1179, 'winner'] = r

df.at[1389, 'winner'] = b

df.at[1473, 'winner'] = b

df.at[1475, 'winner'] = b

df.at[1522, 'winner'] = b

df.at[1664, 'winner'] = b

df.at[1764, 'winner'] = b



# recompute no_winby

no_winby = df[df['winby'].isnull()]
# extract any column that would help with Googling the match

no_winby[['B_Name', 'R_Name', 'Date', 'Last_round', 'winner']]
s = 'SUB'

d = 'DEC'

nc = 'NC'

k = 'KO/TKO'

dq = 'DQ'



df.at[36, 'winby'] = s # Omari Akhmedov	Gunnar Nelson: SUB

df.at[170, 'winby'] = s # Johnny Bedford	Rani Yahya: head bump NC

df.at[177, 'winby'] = d # Rashid Magomedov vs Rodrigo Damm: DEC

df.at[364, 'winby'] = d # Efrain Escudero	Leonardo Santos: DEC

df.at[493, 'winby'] = nc # Daron Cruickshank	KJ Noons: eye poke NC

df.at[622, 'winby'] = nc # Norifumi Yamamoto	Roman Salazar: eye poke NC

df.at[894, 'winby'] = nc # Jim Alers	Cole Miller: eye poke NC

df.at[973, 'winby'] = nc # Kevin Casey	Antonio Carlos Junior: eye poke NC

df.at[1450, 'winby'] = nc # Tim Means	Alex Oliveira: knees on downed opponent, NC

df.at[1576, 'winby'] = nc # Dustin Poirier	Eddie Alvarez: illegal knees, NC

df.at[1803, 'winby'] = k # Gilbert Burns	Jason Saggo: KO/TKO

df.at[1875, 'winby'] = dq # Mark Godbeer	Walt Harris: DQ (After a knee landed on Godbeer’s groin, the referee verbally signaled and seemingly touched Harris’ leg to call a time out. Just as he did that, though, Harris threw a head kick that also landed.)

df.at[2008, 'winby'] = dq # Hector Lombard	CB Dollaway: DQ (post-bell punches)

df.at[2234, 'winby'] = d # Liu Pingyuan	Damian Stasiak: DEC



# fights in "DW's Contender Series 2018: Week 4"

df.at[2286, 'winby'] = d # Joey Gomez	Kevin Aguilar: DEC

df.at[2287, 'winby'] = k # Alton Cunningham	Bevon Lewis: KO/TKO

df.at[2288, 'winby'] = d # Ricky Palacios	Toby Misech: DEC

df.at[2289, 'winby'] = k # Rilley Dutro	Jordan Espinosa: KO/TKO

df.at[2290, 'winby'] = k # Jamie Colleen	Maycee Barber: KO/TKO

df.at[2291, 'winby'] = s # Dom Pilarte	Vincent Morales: SUB

df.at[2292, 'winby'] = k # Josh Appelt	Jeff Hughes: KO/TKO



no_winby[['B_Name', 'R_Name', 'Date', 'Last_round', 'winner', 'winby']]
# fight in question

df[(df['Max_round'] == 3) & (df['winby'] == 'DEC') & (df['Last_round'] < 3 )]
# correct it to 3 (verified using Google)

df['Last_round'].iloc[334] = 3

print('"Last_round" set to', df['Last_round'].iloc[334], 'for the fight:', df['B_Name'].iloc[334], 'vs', df['R_Name'].iloc[334])
df.to_csv('cleaned_data.csv', index = False)
# store fight outcomes for reuse and to standardize ordering

fight_outcomes = df['winby'].unique().tolist()

print('Possible fight outcomes:', fight_outcomes)
# average number of rounds in max-3-round fight

print('Distribution of the # of rounds in a max-3-round fight')

print(df['Last_round'][df['Max_round'] == 3].describe(), '\n\n')



# average number of rounds in max-5-round fight

print('Distribution of the # of rounds in a max-5-round fight')

print(df['Last_round'][df['Max_round'] == 5].describe())
# 3- vs 5-round fights by win type

for outcome in ['KO/TKO', 'SUB']:

    max_3_round_stats = df['Last_round'][(df['Max_round'] == 3) & (df['winby'] == outcome)].describe().rename('Num Rounds with Max 3')

    max_5_round_stats = df['Last_round'][(df['Max_round'] == 5) & (df['winby'] == outcome)].describe().rename('Num Rounds with Max 5')

    print('Win by', outcome, ':', '\n', pd.concat([max_3_round_stats, max_5_round_stats], axis=1), '\n')
# collect data

all_fights = {3: {}, 5: {}}



# calculate outcomes by # max rounds

for num_max_rounds in [3, 5]:

    for outcome in fight_outcomes:

        fights_by_round_and_outcome = df['Last_round'][(df['Max_round'] == num_max_rounds) & (df['winby'] == outcome)].size

        if outcome in all_fights[num_max_rounds]:

            all_fights[num_max_rounds][outcome] += fights_by_round_and_outcome

        else:

            all_fights[num_max_rounds][outcome] = fights_by_round_and_outcome

        

all_3round = sum(all_fights[3].values())

all_5round = sum(all_fights[5].values())



fig = {

  "data": [

    {

      "values": list(all_fights[3].values()),

      "labels": list(all_fights[3].keys()),

      "text": list(all_fights[3].keys()),

      "domain": {"column": 0},

      "name": "Max 3 rounds " + str(all_3round),

      "hoverinfo":"label+percent+value",

      "hole": .4,

      "type": "pie"

    },

    {

      "values": list(all_fights[5].values()),

      "labels": list(all_fights[5].keys()),

      "text": list(all_fights[5].keys()),

      "textposition":"inside",

      "domain": {"column": 1},

      "name": "Max 5 rounds " + str(all_5round),

      "hoverinfo":"label+percent+value",

      "hole": .4,

      "type": "pie"

    }],

  "layout": {

        "title":"Fights by Win Type",

        "grid": {"rows": 1, "columns": 2},

        "annotations": [

            {

                "font": {

                    "size": 12

                },

                "showarrow": False,

                "text": "Max 3 rounds",

                "x": 0.16,

                "y": 0.55

            },

            {

                "font": {

                    "size": 12

                },

                "showarrow": False,

                "text": str(all_3round) + " examples",

                "x": 0.16,

                "y": 0.45

            },

            {

                "font": {

                    "size": 12

                },

                "showarrow": False,

                "text": "Max 5 rounds",

                "x": 0.84,

                "y": 0.55

            },

            {

                "font": {

                    "size": 12

                },

                "showarrow": False,

                "text": str(all_5round) + " examples",

                "x": 0.84,

                "y": 0.45

            },

        ]

    }

}

plotly.offline.iplot(fig, filename='donut')
# correlation won't work on nominal columns - we'll represent the 'winner' column numerically, where 1 is a blue win and 0 is a blue loss (red win)

bluewinner_df = pd.concat([df, (df['winner'] == 'blue').astype(int)], axis=1)

corr = bluewinner_df.corr().fillna(0)

corr
# sort features by correlation and print

# usually a correlation plot is helpful, but we have too many features here for it to be legible

corr_copy = corr['winner'].copy()

corr_copy.sort_values(inplace=True, ascending=False)

print("Sorted list of features that correlate with 'winner' - strongest correlations at the top and bottom (positive and negative correlation respectively)\n")

print(corr_copy)
# init values to 0

unique_names = pd.concat([df['B_Name'], df['R_Name']]).unique()

fighter_records = { name: {'wins': 0, 'losses': 0} for name in unique_names }



# calc wins and losses

blue_ratio = []

red_ratio = []

blue_wins = []

red_wins = []

blue_losses = []

red_losses = []

for index, row in df.iterrows():

    bluename = row['B_Name']

    redname = row['R_Name']

    

    # record stats

    blue_wins.append(fighter_records[bluename]['wins'])

    red_wins.append(fighter_records[redname]['wins'])

    blue_losses.append(fighter_records[bluename]['losses'])

    red_losses.append(fighter_records[redname]['losses'])

    

    # win ratio = wins / (wins + losses)

    blue_ratio.append(float(fighter_records[bluename]['wins']) / max(1, fighter_records[bluename]['wins']+fighter_records[bluename]['losses']))

    red_ratio.append(float(fighter_records[redname]['wins']) / max(1, fighter_records[redname]['wins']+fighter_records[redname]['losses']))



    # update stats

    if row['winby'] in ['KO/TKO', 'SUB', 'DEC', 'DQ']:

        winner = bluename if row['winner'] == 'blue' else redname

        loser = redname if row['winner'] == 'blue' else bluename

        fighter_records[winner]['wins'] += 1

        fighter_records[loser]['losses'] += 1



# save after aggregating

df['_B_WinRatio'] = pd.Series(blue_ratio)

df['_R_WinRatio'] = pd.Series(red_ratio)

df['_B_Prev_Wins'] = pd.Series(blue_wins)

df['_R_Prev_Wins'] = pd.Series(red_wins)

df['_B_Prev_Losses'] = pd.Series(blue_losses)

df['_R_Prev_Losses'] = pd.Series(red_losses)

df['BPrev'] = df['_B_Prev_Wins'] + df['_B_Prev_Losses']

df['RPrev'] = df['_R_Prev_Wins'] + df['_R_Prev_Losses']
# correlation won't work on nominal columns - we'll represent the 'winner' column numerically

record_df = pd.concat([(df['winner'] == 'blue').astype(int), df[['_B_WinRatio', '_R_WinRatio', '_B_Prev_Wins', '_R_Prev_Wins', '_B_Prev_Losses', '_R_Prev_Losses', 'BPrev', 'RPrev', 'winner']]], axis=1)

record_corr = record_df.corr().fillna(0)

ranked_metrics = record_corr['winner'].copy().iloc[1:]

ranked_metrics.sort_values(inplace=True, ascending=False)

print('blue winning correlates with:\n', ranked_metrics)
df['_B_Weight_Class'] = df['B_Weight'].apply(lambda x: int(x) if x <= 93 else 93)

df['_R_Weight_Class'] = df['R_Weight'].apply(lambda x: int(x) if x <= 93 else 93)



weight_classes = sorted(pd.concat([df['_B_Weight_Class'], df['_R_Weight_Class']]).unique())

print('The weight classes are:', weight_classes)
mismatched_fights = df[['_B_Weight_Class', '_R_Weight_Class']][df['_B_Weight_Class'] != df['_R_Weight_Class']]

print('There are', len(mismatched_fights), 'fights with mismatched weight classes')
df['winby'].unique()
bar_charts = []

for winby in ['DEC', 'KO/TKO', 'SUB', 'DRAW']:

    fights_by_class = []

    for cl in weight_classes:

        recent_df = df[df['_Date_year'] == df['_Date_year'].max()] # use only last year of data for reliable weight classes

        fights_for_class = len(recent_df[recent_df['winby'] == winby][(recent_df['B_Weight'] == cl) | (recent_df['R_Weight'] == cl)])

        fights_by_class.append(fights_for_class)

    bar = go.Bar(name=winby, x=[ str(cl)+'kg' for cl in weight_classes ], y=fights_by_class)

    bar_charts.append(bar)

    

layout = go.Layout(

    xaxis={'title': 'Weight Classes'},

    yaxis={'title': "Number fights by 'winby'"}

)

fig = go.Figure(data=bar_charts)

fig.update_layout(barmode='stack')

fig.show()
heavier_blue_wins = len(df[(df['_B_Weight_Class'] > df['_R_Weight_Class']) & (df['winner'] == 'blue')])

heavier_blue_total_fights = len(df[df['_B_Weight_Class'] > df['_R_Weight_Class']])



heavier_red_wins = len(df[(df['_B_Weight_Class'] < df['_R_Weight_Class']) & (df['winner'] == 'red')])

heavier_red_total_fights = len(df[df['_B_Weight_Class'] < df['_R_Weight_Class']])



heavier_win_probability = float(heavier_blue_wins + heavier_red_wins) / (heavier_blue_total_fights + heavier_red_total_fights)

print('Probability of heavier fighter winning: ', heavier_win_probability)
def map_win(winner):

    if winner == 'blue':

        return 1

    elif winner == 'red':

        return -1

    else:

        return 0



weight_diff = (df['_B_Weight_Class'] - df['_R_Weight_Class']).apply(lambda x: min(x, 27.2155)) # cap weight diffs >60lbs

wins = df['winner'].apply(map_win)



# Create linear regression object

regr = linear_model.LinearRegression()



# Train the model using the training sets

regr.fit(np.array(weight_diff).reshape(-1, 1), wins)

variance = '%.3f' % regr.score(np.array(weight_diff).reshape(-1, 1), wins)



layout = go.Layout(

    title='Correlation Coefficient = ' + variance,

    xaxis={'title': 'Weight Difference (kg)'},

    yaxis={'title': 'Did Win?'},

    shapes=[

        go.layout.Shape(

            type="line",

            x0=weight_diff.min(),

            y0=regr.predict([[weight_diff.min()]])[0],

            x1=weight_diff.max(),

            y1=regr.predict([[weight_diff.max()]])[0],

            line=dict(

                color="Red",

                width=3

            )

        )

    ]

)



fig = go.Figure(data=go.Scatter(x=weight_diff, y=wins, mode='markers'), layout=layout)

fig.show()
min_age = min(df['R_Age'].min(), df['B_Age'].min())

max_age = max(df['R_Age'].max(), df['B_Age'].max())



print('The youngest fighter in the dataset is', min_age)

print('The oldest fighter in the dataset is', max_age)
df_notnull = df[df['B_Age'].notnull() & df['R_Age'].notnull() & df['B_Weight'].notnull() & df['R_Weight'].notnull()]

ages = pd.concat([df_notnull['B_Age'], df_notnull['R_Age']])

weights = pd.concat([df_notnull['B_Weight'], df_notnull['R_Weight']])



# Create linear regression object

regr = linear_model.LinearRegression()



# Train the model using the training sets

regr.fit(np.array(ages).reshape(-1, 1), weights)

variance = '%.3f' % regr.score(np.array(ages).reshape(-1, 1), weights)



layout = go.Layout(

    title='Correlation Coefficient = ' + variance,

    xaxis={'title': 'Age'},

    yaxis={'title': 'Weight (kg)'},

    shapes=[

        go.layout.Shape(

            type="line",

            x0=19,

            y0=regr.predict([[19]])[0],

            x1=47,

            y1=regr.predict([[47]])[0],

            line=dict(

                color="Red",

                width=3

            )

        )

    ]

)



fig = go.Figure(data=go.Scatter(x=ages, y=weights, mode='markers'), layout=layout)

fig.show()
age_ranges = [(19,24), (25,29), (30,34), (35,39), (40,44), (45,49)]



# get total fights for each fight

mean_fights = []

for lower_age, upper_age in age_ranges:

    # get total number of fights with a fighter who is in the age range (double count fights with two such fighters)

    num_fights_for_age = len(df[(df['R_Age'] >= lower_age) & (df['R_Age'] <= upper_age)])

    num_fights_for_age += len(df[(df['B_Age'] >= lower_age) & (df['B_Age'] <= upper_age)])

    

    # calculate avg fights per fighter at that age

    fights_for_age = df[((df['R_Age'] >= lower_age) & (df['R_Age'] <= upper_age)) | ((df['B_Age'] >= lower_age) & (df['B_Age'] <= upper_age))]

    num_fighters = len(pd.concat([fights_for_age['B_Name'], fights_for_age['R_Name']]).unique())

    mean_fights_for_age = num_fights_for_age / num_fighters

    

    mean_fights.append(mean_fights_for_age)

        

# plot the results

layout = go.Layout(

    xaxis={'title': 'Age Ranges'},

    yaxis={'title': 'Fights per year'})

fig = go.Figure([go.Bar(x=[ str(age) for age in age_ranges ], y=mean_fights)], layout=layout)

fig.show()
# get relevant columns

clustering_columns = [ col for col in df.columns if ('Round' in col and 'TIP' not in col and 'Round4' not in col and 'Round5' not in col) ]

b_clustering_columns = [ col for col in clustering_columns if col.startswith('B_') ]

r_clustering_columns = [ col for col in clustering_columns if col.startswith('R_') ]



# filter out 5-round fights

df_3round = df[df['Max_round'] == 3]



# combine blue and red fighters into a single dataframe with the relevant stats, and add name column

clustering_df = pd.concat([df_3round[b_clustering_columns].rename(lambda x: x[3:], axis='columns'), df_3round[r_clustering_columns].rename(lambda x: x[3:], axis='columns')], ignore_index=True)

aggregated_columns = list(set([ col[7:] for col in clustering_df.columns.values if col ])) + ['Name'] # remove "RoundX_" from the start of each column

clustering_df['Name'] = pd.concat([df_3round['B_Name'], df_3round['R_Name']], ignore_index=True)



# aggregate round stats

new_rows = []

for index, row in clustering_df.iterrows():

    row_dict = {}

    for aggregated_column in aggregated_columns:

        if aggregated_column is 'Name':

            row_dict['Name'] = row['Name']

        else:

            col_sum = 0

            num_rounds = 0

            for round_num in range(1,4):

                round_column = 'Round' + str(round_num) + '_' + aggregated_column

                if not np.isnan(row[round_column]):

                    num_rounds += 1

                    col_sum += row[round_column]

            row_dict[aggregated_column] = col_sum / num_rounds if num_rounds > 0 else np.nan

    new_rows.append(row_dict)

    

# fill Nan values (missing stats for all 3 rounds) with the mean of each column

aggregated_clustering_df_raw = pd.DataFrame(new_rows)

aggregated_clustering_df_raw = aggregated_clustering_df_raw.fillna(aggregated_clustering_df_raw.mean())
# combine all rows for a single fighter into one row for that fighter

fighter_rows = []

names = aggregated_clustering_df_raw['Name'].unique()



for name in names:

    fights_for_fighter = aggregated_clustering_df_raw[aggregated_clustering_df_raw['Name'] == name]

    if len(fights_for_fighter) >= 5:

        fighter_row = fights_for_fighter.mean()

        fighter_row['Name'] = name # add name back in

        fighter_rows.append(fighter_row)



aggregated_clustering_df_raw_byname = pd.DataFrame(fighter_rows)
# let's look at the data

aggregated_clustering_df_raw_byname.describe()
# normalize the data - note the mean becomes 0 and the standard deviation becomes 1

pre_normalized = aggregated_clustering_df_raw_byname.drop('Name', axis=1)

aggregated_clustering_df = (pre_normalized - pre_normalized.mean()) / pre_normalized.std()

aggregated_clustering_df.describe()
clustering_data = aggregated_clustering_df.values

linked = sch.linkage(clustering_data, 'ward')

labelList = range(0, len(clustering_data))



plt.figure(figsize=(10, 7))

sch.dendrogram(linked,

            orientation='top',

            labels=labelList,

            distance_sort='descending',

            show_leaf_counts=True)

plt.show()
n_clusters = 3

cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')



cluster.fit_predict(clustering_data)



print('First 10 classifications:')

for name, label in zip(aggregated_clustering_df_raw_byname['Name'].as_matrix()[:10], cluster.labels_[:10]):

    print(name, label)
# 2D PCA

pca_model = PCA(n_components=2)

components = pca_model.fit_transform(clustering_data)



# plot the components with their cluster labels

cluster_data = pd.DataFrame(components, columns=['pca_x', 'pca_y'])

cluster_data['Name'] = aggregated_clustering_df_raw_byname['Name']

cluster_data['label'] = cluster.labels_

fig = px.scatter(cluster_data, x="pca_x", y="pca_y", color="label", hover_data=['Name'])

fig.show()
fighter_cluster_data = pd.concat([cluster_data, aggregated_clustering_df_raw_byname.drop('Name', axis=1)], sort=False, axis=1)



# we'll throw in age, height, weight, and win ratio for more context

def max_no_nan(series):

    return -1 if np.isnan(series.max()) else series.max()



# unfortunately, a join isn't possible with the original dataset (red and blue fighters), so we'll have to perform the lookup one-by-one

for idx, row in fighter_cluster_data.iterrows():

    fighter_as_blue = df[df['B_Name'] == row['Name']]

    fighter_as_red = df[df['R_Name'] == row['Name']]

    fighter_cluster_data.at[idx, 'Age'] = max(max_no_nan(fighter_as_blue['B_Age']), max_no_nan(fighter_as_red['R_Age']))

    fighter_cluster_data.at[idx, 'Height'] = max(max_no_nan(fighter_as_blue['B_Height']), max_no_nan(fighter_as_red['R_Height']))

    fighter_cluster_data.at[idx, 'Weight'] = max(max_no_nan(fighter_as_blue['B_Weight']), max_no_nan(fighter_as_red['R_Weight']))

    fighter_cluster_data.at[idx, 'Win_Ratio'] = (fighter_as_blue['_B_WinRatio'].mean() + fighter_as_red['_R_WinRatio'].mean()) / 2.0
fighter_cluster_data
fighter_cluster_data.to_csv('fighter_cluster_data.csv', index = False)
label_rows = []

for label in range(n_clusters):

    label_rows.append(fighter_cluster_data[fighter_cluster_data['label'] == label].mean())



label_cluster_data = pd.DataFrame(label_rows, columns=fighter_cluster_data.columns.values).drop(['Name', 'pca_x', 'pca_y'], axis=1)

label_cluster_data
label_cluster_data.to_csv('label_cluster_data.csv', index = False)
column_max_by_label = dict([(label, []) for label in range(n_clusters) ])

column_min_by_label = dict([(label, []) for label in range(n_clusters) ])



for column in label_cluster_data.drop('label', axis=1).columns.values:

    # compute the label that performs significantly more of this action

    column_top_two = label_cluster_data[column].nlargest(2)

    if max(column_top_two) >= min(column_top_two) * 1.25: # 25% more of a given action is our threshold

        max_label = label_cluster_data[column].argmax()

        column_max_by_label[max_label].append(column)

        

    # compute the label that performs significantly less of this action

    column_bottom_two = label_cluster_data[column].nsmallest(2)

    if min(column_bottom_two) <= max(column_bottom_two) * 0.75: # 25% less is our threshold

        min_label = label_cluster_data[column].argmin()

        column_min_by_label[min_label].append(column)



    

for label in range(n_clusters):

    print('label', label, 'performs more of the following actions than any other class:')

    print(column_max_by_label[label], '\n')

    print('label', label, 'performs less of the following actions than any other class:')

    print(column_min_by_label[label], '\n')
df.to_csv('final_data.csv', index = False)