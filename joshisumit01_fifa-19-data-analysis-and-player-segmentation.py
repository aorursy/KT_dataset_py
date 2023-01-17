# importing Required Liberary and Data Load

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import datetime

pd.set_option('display.max_columns', 100)

pd.set_option('display.max_rows', 100)

data = pd.read_csv('../input/fifa19/data.csv')
data.head()

data.info()
data.describe()
data.isnull().any()
# filling the missing value for the continous variables for proper data visualization



data['ShortPassing'].fillna(data['ShortPassing'].mean(), inplace = True)

data['ShortPassing'].value_counts(dropna = False)



data['Volleys'].fillna(data['Volleys'].mean(), inplace = True)

data['Volleys'].value_counts(dropna = False)



data['Dribbling'].fillna(data['Dribbling'].mean(), inplace = True)

data['Dribbling'].value_counts(dropna = False)



data['Curve'].fillna(data['Curve'].mean(), inplace = True)

data['Curve'].value_counts(dropna = False)



data['FKAccuracy'].fillna(data['FKAccuracy'], inplace = True)

data['FKAccuracy'].value_counts(dropna = False)



data['LongPassing'].fillna(data['LongPassing'].mean(), inplace = True)

data['LongPassing'].value_counts(dropna = False)



data['BallControl'].fillna(data['BallControl'].mean(), inplace = True)

data['BallControl'].value_counts(dropna = False)



data['HeadingAccuracy'].fillna(data['HeadingAccuracy'].mean(), inplace = True)

data['HeadingAccuracy'].value_counts(dropna = False)



data['Finishing'].fillna(data['Finishing'].mean(), inplace = True)

data['Finishing'].value_counts(dropna = False)



data['Crossing'].fillna(data['Crossing'].mean(), inplace = True)

data['Crossing'].value_counts(dropna = False)



data['Weight'].fillna('200lbs', inplace = True)

data['Weight'].value_counts(dropna = False)



data['Contract Valid Until'].fillna(2019, inplace = True)

data['Contract Valid Until'].value_counts(dropna = False)



data['Height'].fillna("5'11", inplace = True)

data['Height'].value_counts(dropna = False)



data['Loaned From'].fillna('None', inplace = True)

data['Loaned From'].value_counts(dropna = False)



data['Joined'].fillna('Jul 1, 2018', inplace = True)

data['Joined'].value_counts(dropna = False)



data['Jersey Number'].fillna(8, inplace = True)

data['Jersey Number'].value_counts(dropna = False)



data['Body Type'].fillna('Normal', inplace = True)



# replacing the single body types with the stocky body types



data['Body Type'].replace('Messi', 'Stocky', inplace = True)

data['Body Type'].replace('C. Ronaldo', 'Stocky', inplace = True)

data['Body Type'].replace('Courtois', 'Stocky', inplace = True)

data['Body Type'].replace('PLAYER_BODY_TYPE_25', 'Stocky', inplace = True)

data['Body Type'].replace('Shaqiri', 'Stocky', inplace = True)

data['Body Type'].replace('Neymar', 'Stocky', inplace = True)

data['Body Type'].replace('Akinfenwa', 'Stocky',inplace = True)

data['Body Type'].value_counts(dropna = False)



data['Position'].fillna('ST', inplace = True)

data['Position'].value_counts(dropna = False)



data['Club'].fillna('No Club', inplace = True)

data['Club'].value_counts(dropna = False)



data['Work Rate'].fillna('Medium/ Medium', inplace = True)

data['Work Rate'].value_counts(dropna = False)



data['Skill Moves'].fillna(data['Skill Moves'].median(), inplace = True)

data['Skill Moves'].value_counts(dropna = False)



data['Weak Foot'].fillna(3, inplace = True)

data['Weak Foot'].value_counts(dropna = False)



data['Preferred Foot'].fillna('Right', inplace = True)

data['Preferred Foot'].value_counts(dropna = False)



data['International Reputation'].fillna(1, inplace = True)

data['International Reputation'].value_counts(dropna = False)
data.isnull().any()
# preferences of foot over the players

plt.figure(figsize=(10,5))

data['Preferred Foot'].value_counts().plot.bar()
#  comparison of international reputation of the players

data['International Reputation'].value_counts()
#pie chart for the International reputation

labels = ['1', '2', '3', '4', '5']

sizes = [16532, 1261, 309, 51, 6]

colors = ['blue', 'yellow', 'green', 'red', 'orange']

explode = [0.1, 0.1, 0.2, 0.5, 0.9]

plt.figure(figsize = (7,7))

plt.pie(sizes, labels = labels, colors = colors, explode = explode, shadow = True)

plt.title('Repuatation of Football Players', fontsize = 25)

plt.legend()

plt.show()
# rate defines the higher power and control.

# which shows the power and control for the other foot preferred foot is higher

data['Weak Foot'].value_counts()
# pie chart for the weak foot

labels = ['5', '4', '3', '2', '1'] 

size = [229, 2662, 11349, 3761, 158]

colors = ['blue', 'yellow', 'green', 'red', 'orange']

explode = [0.1, 0.1, 0.1, 0.1, 0.1]

plt.figure(figsize=(7,7))

plt.pie(size, labels = labels, colors = colors, explode = explode, shadow = True)

plt.title('Representing Week Foot', fontsize = 25)

plt.legend()

plt.show()
# positions played by the players 

plt.figure(figsize = (12, 8))

sns.set(style = 'dark', palette = 'colorblind', color_codes = True)

ax = sns.countplot('Position', data = data, color = 'blue')

ax.set_xlabel(xlabel = 'Positions in Ground', fontsize = 16)

ax.set_ylabel(ylabel = 'Numbers Players', fontsize = 16)

ax.set_title(label = 'Player_s Positions', fontsize = 20)

plt.show()
data['Wage'].fillna('€200K', inplace = True)

data['Wage'].isnull().any()
# defining a function for cleaning the Weight data

def extract_value_from(value):

  out = value.replace('lbs', '')

  return float(out)



data['Weight'] = data['Weight'].apply(lambda x : extract_value_from(x))



data['Weight'].head()
# defining a function for cleaning the wage column

def extract_value_from(Value):

    out = Value.replace('€', '')

    if 'M' in out:

        out = float(out.replace('M', ''))*1000000

    elif 'K' in Value:

        out = float(out.replace('K', ''))*1000

    return float(out)
# applying the function to the wage column

data['Value'] = data['Value'].apply(lambda x: extract_value_from(x))

data['Wage'] = data['Wage'].apply(lambda x: extract_value_from(x))

data['Wage'].head()
# Comparing players' Wages

sns.set(style = 'dark', palette = 'bright', color_codes = True)

plt.figure(figsize = (100, 7))

x = data.Wage

sns.countplot(x, data = data, palette = 'rainbow')

plt.xlabel('Wage of Players', fontsize = 30)

plt.ylabel('Number_s Players', fontsize = 30)

plt.title('Comparing the wages', fontsize = 50)

plt.show()
data['Height'].head(10)
# Skill shown by the Players

plt.figure(figsize = (9, 5))

ax = sns.countplot(x = 'Skill Moves', data = data, palette = 'pastel')

ax.set_title(label = 'players\'s skill moves', fontsize = 20)

ax.set_xlabel(xlabel = 'Skill Moves', fontsize = 16)

ax.set_ylabel(ylabel = 'Count', fontsize = 16)

plt.show()
# Height of Players

plt.figure(figsize = (13, 8))

ax = sns.countplot(x = 'Height', data = data, palette = 'rainbow')

ax.set_title(label = 'player\'s Height', fontsize = 20)

ax.set_xlabel(xlabel = 'Height', fontsize = 16)

ax.set_ylabel(ylabel = 'Count', fontsize = 16)

plt.show()

# body weight of the players in the FIFA

plt.figure(figsize = (35, 8))

sns.countplot(x = 'Weight', data = data, palette = 'rainbow')

plt.title('Weight of the Players', fontsize = 20)

plt.xlabel('Weight ', fontsize = 16)

plt.ylabel('Players', fontsize = 16)

plt.show()
# Work rate of the players in the FIFA

plt.figure(figsize = (20, 8))

sns.countplot(x = 'Work Rate', data = data, palette = 'rainbow')

plt.title('work rates of the Players', fontsize = 20)

plt.xlabel('Work rates ', fontsize = 16)

plt.ylabel('Players', fontsize = 16)

plt.show()
# Special Score of the players in the FIFA

sns.set(style = 'dark', palette = 'colorblind', color_codes = True)

x = data.Special

plt.figure(figsize = (12, 8))

ax = sns.distplot(x, bins = 58, kde = False, color = 'blue')

ax.set_xlabel(xlabel = 'Special score range', fontsize = 16)

ax.set_ylabel(ylabel = 'Count of the Players',fontsize = 16)

ax.set_title(label = 'Scores of the Players', fontsize = 20)

plt.show()
# potential of the players in the FIFA 

sns.set(style = "dark", palette = "muted", color_codes = True)

x = data.Potential

plt.figure(figsize=(12,8))

ax = sns.distplot(x, bins = 58, kde = False, color = 'blue')

ax.set_xlabel(xlabel = "Player\'s Potential Scores", fontsize = 16)

ax.set_ylabel(ylabel = 'Number of players', fontsize = 16)

ax.set_title(label = 'Players Potential Scores', fontsize = 20)

plt.show()
# overall score of the players in the FIFA

sns.set(style = "dark", palette = "deep", color_codes = True)

x = data.Overall

plt.figure(figsize = (12,8))

ax = sns.distplot(x, bins = 52, kde = False, color = 'blue')

ax.set_xlabel(xlabel = "Player\'s Scores", fontsize = 16)

ax.set_ylabel(ylabel = 'Number of players', fontsize = 16)

ax.set_title(label = 'players Overall Scores', fontsize = 20)

plt.show()
# Countries participating in the FIFA

data['Nationality'].value_counts().plot.bar(color = 'blue')

plt.title('Nations Participating', fontsize = 20)

plt.xlabel('Name of The Country', fontsize = 20)

plt.ylabel('count', fontsize = 20)

plt.show()
# if there are people falling in same age group

sns.set(style = "dark", palette = "colorblind", color_codes = True)

x = data.Age

plt.figure(figsize = (12,8))

ax = sns.distplot(x, bins = 58, kde = False, color = 'blue')

ax.set_xlabel(xlabel = "Player\'s age", fontsize = 16)

ax.set_ylabel(ylabel = 'players', fontsize = 16)

ax.set_title(label = 'players age', fontsize = 20)

plt.show()
#comparing body types of the players

data['Body Type'].value_counts().plot.bar(color = 'blue', figsize = (7, 5))

plt.title('Different Body Types')

plt.xlabel('Body Types')

plt.ylabel('count')

plt.show()
data.columns
# extracting some specific fetures from the dataset

sel_col = ['Name', 'Age', 'Nationality', 'Overall', 'Potential', 'Club', 'Value',

                    'Wage', 'Special', 'Preferred Foot', 'International Reputation', 'Weak Foot',

                    'Skill Moves', 'Work Rate', 'Body Type', 'Position', 'Height', 'Weight',

                    'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',

                    'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',

                    'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',

                    'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',

                    'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',

                    'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',

                    'GKKicking', 'GKPositioning', 'GKReflexes', 'Release Clause']

data_sel = pd.DataFrame(data, columns = sel_col)

data_sel.columns
data_sel.sample(5)
# checking correlation

plt.rcParams['figure.figsize'] = (30, 30)

sns.heatmap(data_sel[['Age', 'Nationality', 'Overall', 'Potential', 'Club', 'Value',

                    'Wage', 'Special', 'Preferred Foot', 'International Reputation', 'Weak Foot',

                    'Skill Moves', 'Work Rate', 'Body Type', 'Position', 'Height', 'Weight',

                    'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',

                    'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',

                    'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',

                    'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',

                    'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',

                    'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',

                    'GKKicking', 'GKPositioning', 'GKReflexes', 'Release Clause']].corr(), annot = True)

plt.title('Histogram of the Dataset', fontsize = 30)

plt.show()
# top 15 players per each position based on their overall scores

data.iloc[data.groupby(data['Position'])['Overall'].idxmax()][['Position', 'Name', 'Age', 'Club', 'Nationality']].head(15)
# top 15 players from each positions based on their potential scores

data.iloc[data.groupby(data['Position'])['Potential'].idxmax()][['Position', 'Name', 'Age', 'Club', 'Nationality']].head(15)
# countries with highest number of players

data['Nationality'].value_counts().head(10)
# Nations' Player and their Weights

some_countries = ('England', 'Germany', 'Spain', 'Argentina', 'France', 'Brazil', 'Italy', 'Columbia', 'Japan', 'Netherlands')

data_countries = data.loc[data['Nationality'].isin(some_countries) & data['Weight']]

plt.rcParams['figure.figsize'] = (12, 7)

ax = sns.violinplot(x = data_countries['Nationality'], y = data_countries['Weight'], palette = 'rainbow')

ax.set_xlabel(xlabel = 'Countries', fontsize = 9)

ax.set_ylabel(ylabel = 'Weight', fontsize = 9)

ax.set_title(label = 'Weight of players from different countries', fontsize = 20)

plt.show()
# Nations' Player and their overall scores

some_countries = ('England', 'Germany', 'Spain', 'Argentina', 'France', 'Brazil', 'Italy', 'Columbia', 'Japan', 'Netherlands')

data_countries = data.loc[data['Nationality'].isin(some_countries) & data['Overall']]

plt.rcParams['figure.figsize'] = (12, 7)

ax = sns.barplot(x = data_countries['Nationality'], y = data_countries['Overall'], palette = 'rainbow')

ax.set_xlabel(xlabel = 'Countries', fontsize = 9)

ax.set_ylabel(ylabel = 'Overall Scores', fontsize = 9)

ax.set_title(label = 'scores of players from different countries', fontsize = 20)

plt.show()
# Every Nations' Player and their Wages

some_countries = ('England', 'Germany', 'Spain', 'Argentina', 'France', 'Brazil', 'Italy', 'Columbia', 'Japan', 'Netherlands')

data_countries = data.loc[data['Nationality'].isin(some_countries) & data['Wage']]

plt.rcParams['figure.figsize'] = (12, 7)

ax = sns.barplot(x = data_countries['Nationality'], y = data_countries['Wage'], palette = 'rainbow')

ax.set_xlabel(xlabel = 'Countries', fontsize = 9)

ax.set_ylabel(ylabel = 'Wage', fontsize = 9)

ax.set_title(label = 'Wages of players from different countries', fontsize = 15)

plt.show()
# Nations' Player and their International Reputation

some_countries = ('England', 'Germany', 'Spain', 'Argentina', 'France', 'Brazil', 'Italy', 'Columbia', 'Japan', 'Netherlands')

data_countries = data.loc[data['Nationality'].isin(some_countries) & data['International Reputation']]

plt.rcParams['figure.figsize'] = (12, 7)

ax = sns.violinplot(x = data_countries['Nationality'], y = data_countries['International Reputation'], palette = 'rainbow')

ax.set_xlabel(xlabel = 'Countries', fontsize = 9)

ax.set_ylabel(ylabel = 'Distribution of reputation', fontsize = 9)

ax.set_title(label = 'International Repuatation of players from different countries', fontsize = 15)

plt.show()
# top 10 clubs

data['Club'].value_counts().head(10)
top_clubs = ('CD Leganés', 'Southampton', 'RC Celta', 'Empoli', 'Fortuna Düsseldorf', 'Manchestar City',

             'Tottenham Hotspur', 'FC Barcelona', 'Valencia CF', 'Chelsea', 'Real Madrid')

data_clubs = data.loc[data['Club'].isin(top_clubs) & data['Overall']]

plt.rcParams['figure.figsize'] = (20, 8)

ax = sns.barplot(x = data_clubs['Club'], y = data_clubs['Overall'], palette = 'rainbow')

ax.set_xlabel(xlabel = 'Popular Clubs', fontsize = 9)

ax.set_ylabel(ylabel = 'Overall Score', fontsize = 9)

ax.set_title(label = 'Overall Score of popular Clubs', fontsize = 20)

plt.show()
# Age Distribution in top 10 clubs

top_clubs = ('CD Leganés', 'Southampton', 'RC Celta', 'Empoli', 'Fortuna Düsseldorf', 'Manchestar City',

             'Tottenham Hotspur', 'FC Barcelona', 'Valencia CF', 'Chelsea', 'Real Madrid')

data_club = data.loc[data['Club'].isin(top_clubs) & data['Wage']]

plt.rcParams['figure.figsize'] = (14, 8)

ax = sns.violinplot(x = 'Club', y = 'Age', data = data_club, palette = 'rainbow')

ax.set_xlabel(xlabel = 'popular Clubs', fontsize = 10)

ax.set_ylabel(ylabel = 'Distribution', fontsize = 10)

ax.set_title(label = 'Age in Popular Clubs', fontsize = 20)

plt.show()
# Wages Distribution in top 10 clubs

top_clubs = ('CD Leganés', 'Southampton', 'RC Celta', 'Empoli', 'Fortuna Düsseldorf', 'Manchestar City',

             'Tottenham Hotspur', 'FC Barcelona', 'Valencia CF', 'Chelsea', 'Real Madrid')

data_club = data.loc[data['Club'].isin(top_clubs) & data['Wage']]

plt.rcParams['figure.figsize'] = (16, 8)

ax = sns.violinplot(x = 'Club', y = 'Wage', data = data_club, palette = 'rainbow')

ax.set_xlabel(xlabel = 'popular Clubs', fontsize = 10)

ax.set_ylabel(ylabel = 'Distribution', fontsize = 10)

ax.set_title(label = 'Wages in Popular Clubs', fontsize = 20)

plt.show()
# international Reputation in top 10 clubs

top_clubs = ('CD Leganés', 'Southampton', 'RC Celta', 'Empoli', 'Fortuna Düsseldorf', 'Manchestar City',

             'Tottenham Hotspur', 'FC Barcelona', 'Valencia CF', 'Chelsea', 'Real Madrid')

data_club = data.loc[data['Club'].isin(top_clubs) & data['International Reputation']]

plt.rcParams['figure.figsize'] = (16, 8)

ax = sns.violinplot(x = 'Club', y = 'International Reputation', data = data_club, palette = 'rainbow')

ax.set_xlabel(xlabel = 'popular Clubs', fontsize = 10)

ax.set_ylabel(ylabel = 'Distribution', fontsize = 10)

ax.set_title(label = 'International Reputation in Popular Clubs', fontsize = 20)

plt.show()
# Weight distribution in top 10 clubs

top_clubs = ('CD Leganés', 'Southampton', 'RC Celta', 'Empoli', 'Fortuna Düsseldorf', 'Manchestar City',

             'Tottenham Hotspur', 'FC Barcelona', 'Valencia CF', 'Chelsea', 'Real Madrid')

data_clubs = data.loc[data['Club'].isin(top_clubs) & data['Weight']]

plt.rcParams['figure.figsize'] = (14, 8)

ax = sns.violinplot(x = 'Club', y = 'Weight', data = data_clubs, palette = 'rainbow')

ax.set_xlabel(xlabel = 'Popular Clubs', fontsize = 9)

ax.set_ylabel(ylabel = 'Weight', fontsize = 9)

ax.set_title(label = 'Weight in popular Clubs', fontsize = 20)

plt.show()
#10 youngest palyer in FIFA

young_player = data.sort_values('Age', ascending = True)[['Name', 'Age', 'Club', 'Nationality']].head(10)

print(young_player)
# 10 eldest players in FIFA

eldest_player = data.sort_values('Age', ascending = False)[['Name', 'Age', 'Club', 'Nationality']].head(10)

print(eldest_player)
data['Joined'].head()
# longest members of the club

import datetime

now = datetime.datetime.now()

data['Join_year'] = data.Joined.dropna().map(lambda x: x.split(',')[1].split(' ')[1])

data['member scene'] = (data.Join_year.dropna().map(lambda x: now.year - int(x))).astype('int')

membership = data[['Name', 'Club', 'member scene']].sort_values(by = 'member scene', ascending = False).head(10)

membership.set_index('Name', inplace=True)

membership
# most looked features in players

player_features = ('Acceleration', 'Aggression', 'Agility', 

                   'Balance', 'BallControl', 'Composure', 

                   'Crossing', 'Dribbling', 'FKAccuracy', 

                   'Finishing', 'GKDiving', 'GKHandling', 

                   'GKKicking', 'GKPositioning', 'GKReflexes', 

                   'HeadingAccuracy', 'Interceptions', 'Jumping', 

                   'LongPassing', 'LongShots', 'Marking', 'Penalties')

# Top four features for every position in football

for i, val in data.groupby(data['Position'])[player_features].mean().iterrows():

    print('Position {}: {}, {}, {}, {}'.format(i, *tuple(val.nlargest(4).index)))
# Top 10 lefty players

data[data['Preferred Foot'] == 'Left'][['Name', 'Age', 'Club', 'Nationality']].head(10)
# Top 10 Righty players

data[data['Preferred Foot'] == 'Right'][['Name', 'Age', 'Club', 'Nationality']].head(10)
# ballcontrol vs dribbing

sns.lmplot(x = 'BallControl', y = 'Dribbling', data = data, col = 'Preferred Foot')

sns.lmplot(x = 'Penalties', y = 'LongShots', data = data, col = 'Preferred Foot')

sns.lmplot(x = 'Stamina', y = 'Agility', data = data, col = 'Preferred Foot')
# top 10 clubs with highest number of different countries

data.groupby(data['Club'])['Nationality'].nunique().sort_values(ascending = False).head(10)
# top 10 clubs with Lowest number of different countries

data.groupby(data['Club'])['Nationality'].nunique().sort_values(ascending = True).head(10)
# for removing the null 

data[data['GKReflexes'].isnull()].index.tolist()

data.drop(data.index[[13236,13237,13238,13239,13240,13241,13242,13243,13244,13245,13246,13247,13248,13249,13250,13251,13252,

13253,13254,13255,13256,13257,13258,13259,13260,13261,13262,13263,13264,13265,13266,13267,13268,13269,13270,13271,13272,13273,

13274,13275,13276,13277,13278,13279,13280,13281,13282,13283]], inplace= True)
#data prepration for the clustring Analysis

dt1=data.iloc[:,54:88]

dt1.head()
dt1.describe()
# Normalizing  the data

def scale(x):

    return (x-np.mean(x))/np.std(x)

data_scaled=dt1.apply(scale,axis=0)

data_scaled.head()
## Create a cluster model

import sklearn.cluster as cluster

kmeans=cluster.KMeans(n_clusters=3,init="k-means++")

kmeans=kmeans.fit(data_scaled)

lab=kmeans.labels_

lab=list(lab)
# finding optimal value for "K"

from scipy.spatial.distance import cdist

np.min(cdist(data_scaled, kmeans.cluster_centers_, 'euclidean'),axis=1)

K=range(1,5)

wss = []

for k in K:

    kmeans = cluster.KMeans(n_clusters=k,init="k-means++")

    kmeans.fit(data_scaled)

    wss.append(sum(np.min(cdist(data_scaled, kmeans.cluster_centers_, 'euclidean'), 

                                      axis=1)) / data_scaled.shape[0])

import matplotlib.pyplot as plt

#plt.figure(figsize=(10,10))

plt.plot(K, wss, 'bx')

plt.xlabel('k')

plt.ylabel('Avg distor')

plt.title('Selecting k')

plt.show()
import sklearn.metrics as metrics

labels=cluster.KMeans(n_clusters=3,random_state=200).fit(data_scaled).labels_

metrics.silhouette_score(data_scaled,labels,metric="euclidean",sample_size=10000,random_state=200)
for i in range(2,4):

    labels=cluster.KMeans(n_clusters=i,random_state=200).fit(data_scaled).labels_

    print ("Silhoutte score for k= "+str(i)+" is "+str(metrics.silhouette_score(data_scaled,labels,metric="euclidean",

                                 sample_size=10000,random_state=200)))
# check for 2,3,4 clusters

kmeans=cluster.KMeans(n_clusters=3,random_state=200).fit(data_scaled)
colmeans=dt1.mean()
std=dt1.std(axis=0)
group_mean=dt1.groupby([kmeans.labels_]).mean()
x_mean=group_mean.sub(colmeans,axis=1)
x_mean.divide(std,axis=1)
dat_km = pd.concat([data_scaled.reset_index().drop('index', axis=1), pd.Series(kmeans.labels_)], axis =1)

dat_km.head()
dat_km.columns = ['index','Crossing', 'Finishing', 'HeadingAccu','Shotpass','volleys', 'Dribbling','curve', 'FKAccuracy',

                  'Longpassing', 'BallControl', 'Acceleration', 'Sprintspeed', 'Agility', 'Reactions', 'Balance', 

                  'Shotpower', 'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression', 'Interceptions', 'Positining',

                  'Vision', 'Penalties', 'Composure', 'Marking', 'Standing Tackle', 'SlidingTackle', 'GKHandling', 'GKKicking',

                  'GKPositining', 'GKReflexes', 'Clust_id']

dat_km.head()
# Check the count of observation per cluster

dat_km['Clust_id'].value_counts()
plt.rcParams['figure.figsize'] = (30, 30)

sns.heatmap(dat_km.corr(), annot = True)

plt.title('Histogram of the Dataset', fontsize = 15)

plt.show()
# Plot the Cluster with respect to the clusters obtained

plt.figure(figsize=(15,15))

plt.subplot(2,3,1)

sns.scatterplot(x='Balance', y ='Stamina', hue = 'Clust_id', legend='full', palette= ['red', 'blue', 'green'], data = dat_km)

plt.subplot(2,3,2)

sns.scatterplot(x='Aggression', y ='Penalties', hue = 'Clust_id', legend='full', palette= ['red', 'blue', 'green'], data = dat_km)

plt.subplot(2,3,3)

sns.scatterplot(x='Balance', y ='Strength', hue = 'Clust_id', legend='full', palette= ['red', 'blue', 'green'], data = dat_km)

plt.subplot(2,3,4)

sns.scatterplot(x='Longpassing', y ='Vision', hue = 'Clust_id', legend='full', palette= ['red', 'blue', 'green'], data = dat_km)

plt.subplot(2,3,5)

sns.scatterplot(x='Interceptions', y ='Vision', hue = 'Clust_id', legend='full', palette= ['red', 'blue', 'green'], data = dat_km)

plt.subplot(2,3,6)

sns.scatterplot(x='HeadingAccu', y ='Balance', hue = 'Clust_id', legend='full', palette= ['red', 'blue', 'green'], data = dat_km)
import pandas as pd

data = pd.read_csv("../input/fifa19/data.csv")