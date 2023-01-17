import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
color = sns.color_palette()
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
%matplotlib inline
from sklearn.feature_selection import RFE, f_regression
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, RandomizedLasso)
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
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
nba_players_with_salary_df = nba_players_df.merge(salary_df); 
# import wiki data
wiki_df = pd.read_csv("../input/nba_2017_player_wikipedia.csv");wiki_df.head()
wiki_df.rename(columns={'names': 'PLAYER', "pageviews": "PAGEVIEWS"}, inplace=True)
median_wiki_df = wiki_df.groupby("PLAYER").median()
median_wiki_df_small = median_wiki_df[["PAGEVIEWS"]]
median_wiki_df_small = median_wiki_df_small.reset_index()
# merge with wikipageviews
nba_players_with_salary_wiki_df = nba_players_with_salary_df.merge(median_wiki_df_small)
nba_players_with_salary_wiki_df.head()
# import twitter data
twitter_df = pd.read_csv("../input/nba_2017_twitter_players.csv");twitter_df.head()
# merge with twitter data
nba_players_with_salary_wiki_twitter_df = nba_players_with_salary_wiki_df.merge(twitter_df)
nba_players_with_salary_wiki_twitter_df.head()
# drop strings
nba_players_with_salary_wiki_twitter_df.drop(['Rk','PLAYER','POSITION','TEAM'],inplace=True, axis=1)
nba_players_with_salary_wiki_twitter_df.head()
# exam missing values
nba_players_with_salary_wiki_twitter_df.describe()
# drop columns with missing values
nba_players_with_salary_wiki_twitter_df.dropna(axis=0,inplace=True)
nba_players_with_salary_wiki_twitter_df.head()
# define training features and training labels
train_labels = nba_players_with_salary_wiki_twitter_df['SALARY_MILLIONS']
train_features = nba_players_with_salary_wiki_twitter_df.drop(['SALARY_MILLIONS'], axis=1)
# Store the column/feature names into a list "colnames"
colnames = train_features.columns
# Normalize features
train_features = pd.DataFrame(preprocessing.normalize(train_features,axis=0),columns= colnames)
train_features.head()
# exam shapes
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
#print('Testing Features Shape:', test_features.shape)
#print('Testing Labels Shape:', test_labels.shape)
# Define dictionary to store our rankings
ranks = {}
# Create our function which stores the feature rankings to the ranks dictionary
def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))
# Using Linear Regression
lr = LinearRegression(normalize=True)
lr.fit(train_features, train_labels)
ranks["LinReg"] = ranking(np.abs(lr.coef_), colnames)

# Using Ridge 
ridge = Ridge(alpha = 7)
ridge.fit(train_features, train_labels)
ranks['Ridge'] = ranking(np.abs(ridge.coef_), colnames)

# Using Lasso
lasso = Lasso(alpha=.05)
lasso.fit(train_features, train_labels)
ranks["Lasso"] = ranking(np.abs(lasso.coef_), colnames)
# random forest feature ranking
rf = RandomForestRegressor(n_jobs=-1, n_estimators=50, verbose=3)
rf.fit(train_features, train_labels)
ranks["RF"] = ranking(rf.feature_importances_, colnames);
# Create empty dictionary to store the mean value calculated from all the scores
r = {}
for name in colnames:
    r[name] = round(np.mean([ranks[method][name] 
                             for method in ranks.keys()]), 2)
 
methods = sorted(ranks.keys())
ranks["Mean"] = r
methods.append("Mean")
 
print("\t%s" % "\t".join(methods))
for name in colnames:
    print("%s\t%s" % (name, "\t".join(map(str, 
                         [ranks[method][name] for method in methods]))))
# Put the mean scores into a Pandas dataframe
meanplot = pd.DataFrame(list(r.items()), columns= ['Feature','Mean Ranking'])

# Sort the dataframe
meanplot = meanplot.sort_values('Mean Ranking', ascending=False)
# Let's plot the ranking of the features
sns.factorplot(x="Mean Ranking", y="Feature", data = meanplot, kind="bar", 
               size=14, aspect=1.9, palette='coolwarm')