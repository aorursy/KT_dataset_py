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
nba_players_with_salary_df.head()
#salary distribution
sns.kdeplot(nba_players_with_salary_df["SALARY_MILLIONS"], color="Blue", shade=True)
plt.subplots(figsize= (10, 5))
sns.countplot(x= "AGE", data =nba_players_with_salary_df)
plt.subplots(figsize= (10, 5))
sns.countplot(x= "POSITION", data =nba_players_with_salary_df)
ax=sns.pairplot(plus_minus_df)
fig = plt.subplots(figsize= (10, 5))
ax = sns.boxplot(y= "SALARY_MILLIONS" , x="AGE", data = nba_players_with_salary_df, orient="Vertical", width= 0.9)
plt.subplots(figsize=(20,15))
ax = plt.axes()
ax.set_title("NBA Player Correlation Heatmap:  2016-2017 Season (STATS & SALARY)")
corr = nba_players_with_salary_df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
           cmap="Accent",
           annot =True)
wiki_df = pd.read_csv("../input/nba_2017_player_wikipedia.csv");wiki_df.head()

wiki_df.rename(columns={'names': 'PLAYER', "pageviews": "PAGEVIEWS"}, inplace=True)

median_wiki_df = wiki_df.groupby("PLAYER").median()


median_wiki_df_small = median_wiki_df[["PAGEVIEWS"]]
median_wiki_df_small = median_wiki_df_small.reset_index()

nba_players_with_salary_wiki_df = nba_players_with_salary_df.merge(median_wiki_df_small)

twitter_df = pd.read_csv("../input/nba_2017_twitter_players.csv");twitter_df.head()

nba_players_with_salary_wiki_twitter_df = nba_players_with_salary_wiki_df.merge(twitter_df)

nba_players_with_salary_wiki_twitter_df.head()
#popularity distribution
sns.kdeplot(nba_players_with_salary_wiki_twitter_df["TWITTER_RETWEET_COUNT"], color="Blue", shade=True)

plt.subplots(figsize=(20,15))
ax = plt.axes()
ax.set_title("NBA Player Correlation Heatmap:  2016-2017 Season (STATS & SALARY & TWITTER & WIKIPEDIA)")
corr = nba_players_with_salary_wiki_twitter_df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(14,10))
OverallQual_scatter_plot = pd.concat([nba_players_with_salary_df['MPG'],nba_players_with_salary_df['SALARY_MILLIONS']],axis = 1)
sns.regplot(x='MPG',y = 'SALARY_MILLIONS',data = OverallQual_scatter_plot,scatter= True, fit_reg=True, ax=ax1)

TotalBsmtSF_scatter_plot = pd.concat([nba_players_with_salary_df['AGE'],nba_players_with_salary_df['SALARY_MILLIONS']],axis = 1)
sns.regplot(x='AGE',y = 'SALARY_MILLIONS',data = TotalBsmtSF_scatter_plot,scatter= True, fit_reg=True, ax=ax2)

GrLivArea_scatter_plot = pd.concat([nba_players_with_salary_wiki_twitter_df['TWITTER_RETWEET_COUNT'],nba_players_with_salary_wiki_twitter_df['SALARY_MILLIONS']],axis = 1)
sns.regplot(x='TWITTER_RETWEET_COUNT',y = 'SALARY_MILLIONS',data = GrLivArea_scatter_plot,scatter= True, fit_reg=True, ax=ax3)

GarageArea_scatter_plot = pd.concat([nba_players_with_salary_wiki_twitter_df['PAGEVIEWS'],nba_players_with_salary_wiki_twitter_df['SALARY_MILLIONS']],axis = 1)
sns.regplot(x='PAGEVIEWS',y = 'SALARY_MILLIONS',data = GarageArea_scatter_plot,scatter= True, fit_reg=True, ax=ax4)

results = smf.ols('SALARY_MILLIONS ~ TWITTER_RETWEET_COUNT', data=nba_players_with_salary_wiki_twitter_df).fit()
print(results.summary())
results = smf.ols('SALARY_MILLIONS ~ POINTS', data=nba_players_with_salary_df).fit()
print(results.summary())
results = smf.ols('SALARY_MILLIONS ~WINS_RPM', data=nba_players_with_salary_df).fit()
print(results.summary())
results = smf.ols('SALARY_MILLIONS ~ PAGEVIEWS', data=nba_players_with_salary_wiki_twitter_df).fit()
print(results.summary())