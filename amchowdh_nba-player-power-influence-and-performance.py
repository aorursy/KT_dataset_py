import pandas as pd

import statsmodels.api as sm

import numpy as np

import statsmodels.formula.api as smf

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.cluster import KMeans

color = sns.color_palette()

from IPython.core.display import display, HTML

display(HTML("<style>.container { width:100% !important; }</style>"))

%matplotlib inline
attendance_valuation_elo_df = pd.read_csv("../input/nba_2017_att_val_elo.csv");attendance_valuation_elo_df.head()
salary_df = pd.read_csv("../input/nba_2017_salary.csv");salary_df.head()

pie_df = pd.read_csv("../input/nba_2017_pie.csv");pie_df.head()
plus_minus_df = pd.read_csv("../input/nba_2017_real_plus_minus.csv");plus_minus_df.head()
br_stats_df = pd.read_csv("../input/nba_2017_br.csv");br_stats_df.head()


plus_minus_df.rename(columns={"NAME":"PLAYER", "WINS": "WINS_RPM"}, inplace=True)

players = []

for player in plus_minus_df["PLAYER"]:

    plyr, _ = player.split(",")

    players.append(plyr)

plus_minus_df.drop(["PLAYER"], inplace=True, axis=1)

plus_minus_df["PLAYER"] = players

plus_minus_df.head()


nba_players_df = br_stats_df.copy()

nba_players_df.rename(columns={'Player': 'PLAYER','Pos':'POSITION', 'Tm': "TEAM", 'Age': 'AGE', "PS/G": "POINTS"}, inplace=True)

nba_players_df.drop(["G", "GS", "TEAM"], inplace=True, axis=1)

nba_players_df = nba_players_df.merge(plus_minus_df, how="inner", on="PLAYER")

nba_players_df.head()


pie_df_subset = pie_df[["PLAYER", "PIE", "PACE", "W"]].copy()

nba_players_df = nba_players_df.merge(pie_df_subset, how="inner", on="PLAYER")

nba_players_df.head()
salary_df.rename(columns={'NAME': 'PLAYER'}, inplace=True)

salary_df["SALARY_MILLIONS"] = round(salary_df["SALARY"]/1000000, 2)

salary_df.drop(["POSITION","TEAM", "SALARY"], inplace=True, axis=1)

salary_df.head()
diff = list(set(nba_players_df["PLAYER"].values.tolist()) - set(salary_df["PLAYER"].values.tolist()))
len(diff)

nba_players_with_salary_df = nba_players_df.merge(salary_df); 
plt.subplots(figsize=(20,15))

ax = plt.axes()

ax.set_title("NBA Player Correlation Heatmap:  2016-2017 Season (STATS & SALARY)")

corr = nba_players_with_salary_df.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
sns.lmplot(x="SALARY_MILLIONS", y="WINS_RPM", data=nba_players_with_salary_df)

results = smf.ols('W ~POINTS', data=nba_players_with_salary_df).fit()

print(results.summary())

results = smf.ols('W ~WINS_RPM', data=nba_players_with_salary_df).fit()

print(results.summary())

results = smf.ols('SALARY_MILLIONS ~POINTS', data=nba_players_with_salary_df).fit()

print(results.summary())

results = smf.ols('SALARY_MILLIONS ~WINS_RPM', data=nba_players_with_salary_df).fit()

print(results.summary())

from ggplot import *



p = ggplot(nba_players_with_salary_df,aes(x="POINTS", y="WINS_RPM", color="SALARY_MILLIONS")) + geom_point(size=200)

p + xlab("POINTS/GAME") + ylab("WINS/RPM") + ggtitle("NBA Players 2016-2017:  POINTS/GAME, WINS REAL PLUS MINUS and SALARY")
wiki_df = pd.read_csv("../input/nba_2017_player_wikipedia.csv");wiki_df.head()

wiki_df.rename(columns={'names': 'PLAYER', "pageviews": "PAGEVIEWS"}, inplace=True)

median_wiki_df = wiki_df.groupby("PLAYER").median()



median_wiki_df_small = median_wiki_df[["PAGEVIEWS"]]
median_wiki_df_small = median_wiki_df_small.reset_index()

nba_players_with_salary_wiki_df = nba_players_with_salary_df.merge(median_wiki_df_small)

twitter_df = pd.read_csv("../input/nba_2017_twitter_players.csv");twitter_df.head()

nba_players_with_salary_wiki_twitter_df = nba_players_with_salary_wiki_df.merge(twitter_df)

nba_players_with_salary_wiki_twitter_df.head()


plt.subplots(figsize=(20,15))

ax = plt.axes()

ax.set_title("NBA Player Correlation Heatmap:  2016-2017 Season (STATS & SALARY & TWITTER & WIKIPEDIA)")

corr = nba_players_with_salary_wiki_twitter_df.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
#next, we want to see RPM in a scatterplot against some of the other "objective" positive outcomes

#in the dataset, such as PIE, wins, and salary.



import matplotlib.mlab as mlab

import matplotlib.pyplot as plt

import pylab



import matplotlib.pyplot

import pylab



matplotlib.pyplot.scatter(nba_players_with_salary_df['RPM'],nba_players_with_salary_df['PIE'])

matplotlib.pyplot.show()
matplotlib.pyplot.scatter(nba_players_with_salary_df['RPM'],nba_players_with_salary_df['SALARY_MILLIONS'])

matplotlib.pyplot.show()
matplotlib.pyplot.scatter(nba_players_with_salary_df['WINS_RPM'],nba_players_with_salary_df['W'])

matplotlib.pyplot.show()
matplotlib.pyplot.scatter(nba_players_with_salary_df['W'],nba_players_with_salary_df['SALARY_MILLIONS'])

matplotlib.pyplot.show()
matplotlib.pyplot.scatter(nba_players_with_salary_df['RPM'],nba_players_with_salary_df['W'])

matplotlib.pyplot.show()
sns.lmplot(x="RPM", y="W", data=nba_players_with_salary_wiki_twitter_df)

sns.lmplot(x="RPM", y="W", data=nba_players_with_salary_wiki_twitter_df)
sum(i > 2.5 for i in nba_players_with_salary_df['RPM'])/len(nba_players_with_salary_df['RPM'])
sns.lmplot(x="RPM", y="W", 

           data=nba_players_with_salary_wiki_twitter_df[(nba_players_with_salary_wiki_twitter_df.RPM >= 2.5)])
#first we want to see the distribution of RPM in the dataset

#we see that very few players have an RPM over 4



import matplotlib.mlab as mlab

import matplotlib.pyplot as plt

import pylab



plt.hist(nba_players_with_salary_df['RPM'])

plt.ylabel("Distribution")

plt.xlabel("RPM")

plt.show()
#first we want to see the distribution of RPM in the dataset

#we see that very few players have an RPM over 4



import matplotlib.mlab as mlab

import matplotlib.pyplot as plt

import pylab



plt.hist(nba_players_with_salary_df['W'])

plt.ylabel("Distribution")

plt.xlabel("W")

plt.show()
nba_players_with_salary_wiki_twitter_df.dtypes
results = smf.ols('WINS_RPM ~RPM + W', data=nba_players_with_salary_df).fit()

print(results.summary())
results = smf.ols('W~ RPM', data=nba_players_with_salary_wiki_twitter_df).fit()

print(results.summary())
x = nba_players_with_salary_df.drop(['RPM','PLAYER','POSITION','TEAM','WINS_RPM', 'ORPM', 'DRPM', 'W'], axis = 1)

y = nba_players_with_salary_df['RPM']



results = smf.ols('y ~ x', data = nba_players_with_salary_df).fit()

print(results.summary())
list(x)
#the above result is very interesting because it says that assists, steals, blocks, and turnovers have the

#largest effect on RPM, but points and shooting don't really matter

#Let's run that regression



results = smf.ols('RPM ~ AST + STL + BLK + TOV', 

                  data=nba_players_with_salary_wiki_twitter_df).fit()

print(results.summary())
rpm_factors = nba_players_with_salary_wiki_twitter_df[['PLAYER','AST','TOV','STL','BLK', 'RPM', 'W']].copy()

rpm_factors.head()
plt.subplots(figsize=(20,15))

ax = plt.axes()

ax.set_title("NBA Player Correlation Heatmap:  Major Factors Affecting RPM)")

corr = rpm_factors.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
#Since TOV is no longer significant, let's take it out of the model



results = smf.ols('RPM ~ AST + STL + BLK', 

                  data=nba_players_with_salary_wiki_twitter_df).fit()

print(results.summary())
p = ggplot(nba_players_with_salary_df,aes(x="AST", y="RPM", color="W")) + geom_point(size=200)

p + xlab("AST") + ylab("RPM") + ggtitle("NBA Players 2016-2017:  Assists, Real Plus Minus, and Wins")



#even more linear than I expected!
p = ggplot(nba_players_with_salary_df,aes(x="STL", y="RPM", color="W")) + geom_point(size=200)

p + xlab("STL") + ylab("RPM") + ggtitle("NBA Players 2016-2017:  Steals, Real Plus Minus, and Wins")
p = ggplot(nba_players_with_salary_df,aes(x="BLK", y="RPM", color="W")) + geom_point(size=200)

p + xlab("BLK") + ylab("RPM") + ggtitle("NBA Players 2016-2017:  Blocks, Real Plus Minus, and Wins")
nba_players_with_salary_df['ASTSTLBLK'] = nba_players_with_salary_df['AST'] + nba_players_with_salary_df['STL'] + nba_players_with_salary_df['BLK']



p = ggplot(nba_players_with_salary_df,aes(x="ASTSTLBLK", y="RPM", color="W")) + geom_point(size=200)

p + xlab("ASTSTLBLK") + ylab("RPM") + ggtitle("NBA Players 2016-2017:  Steals, Real Plus Minus, and Wins")
results = smf.ols('RPM ~ASTSTLBLK', 

                  data=nba_players_with_salary_df).fit()

print(results.summary())
#Now let's look at aststlblk vs wins



results = smf.ols('W ~ASTSTLBLK', 

                  data=nba_players_with_salary_df).fit()

print(results.summary())
p = ggplot(nba_players_with_salary_df,aes(x="AGE", y="ASTSTLBLK", color="RPM")) + geom_point(size=200)

p + xlab("AGE") + ylab("ASTSTLBLK") + ggtitle("NBA Players 2016-2017:  Steals, Age, and Real Plus Minus")
# Number of clusters

k_means = KMeans(n_clusters=3)



# Choose the columns that the clusters will be based upon

cluster_source = nba_players_with_salary_df.loc[:,["RPM", "W", "ASTSTLBLK"]]



# Create the clusters

kmeans = k_means.fit(cluster_source)



# Create a column, 'cluster,' denoting the cluster classification of each row

nba_players_with_salary_df['cluster'] = kmeans.labels_



# Create a scatter plot with colors based on the cluster

ax = sns.lmplot(x="ASTSTLBLK", y="RPM", data=nba_players_with_salary_df,hue="cluster", size=12, fit_reg=False)

ax.set(xlabel='ASTSTLBLK', ylabel='RPM', title="NBA player Wikipedia ASTSTLBLK vs RPM clustered on ASTSTLBLK, W, RPM:  2016-2017 Season")
bins = [-10, 2.5, np.inf]

labels = ['Low', 'High']

nba_players_with_salary_df['High_RPM'] = pd.cut(nba_players_with_salary_df['RPM'],bins,labels=labels)

nba_players_with_salary_df['High_RPM'].value_counts()
nba_players_with_salary_df["3P%"] = np.where(nba_players_with_salary_df["3P%"].isnull(), 0, nba_players_with_salary_df["3P%"])

nba_players_with_salary_df["FT%"] = np.where(nba_players_with_salary_df["FT%"].isnull(), 0, nba_players_with_salary_df["FT%"])
print(pd.isnull(nba_players_with_salary_df).sum())
#tutorial reference: https://www.datacamp.com/community/tutorials/exploratory-data-analysis-python



X = nba_players_with_salary_df.iloc[:,6:25]

Y = nba_players_with_salary_df.iloc[:,-1]



from sklearn.ensemble import RandomForestClassifier



# Isolate Data, class labels and column values

names = X.columns.values



# Build the model

rfc = RandomForestClassifier()



# Fit the model

rfc.fit(X, Y)



# Print the results

print("Features sorted by their score:")

print(sorted(zip(map(lambda x: round(x, 4), rfc.feature_importances_), names), reverse=True))