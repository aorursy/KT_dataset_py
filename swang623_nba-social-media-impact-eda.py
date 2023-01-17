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
### Start Import data and clean ###

attendance_valuation_elo_df = pd.read_csv("../input/nba_2017_att_val_elo.csv");

attendance_valuation_elo_df.head();
wiki_view_df = pd.read_csv("../input/nba_2017_player_wikipedia.csv");

wiki_view_df.head();
twit_df = pd.read_csv("../input/nba_2017_players_with_salary_wiki_twitter.csv");

twit_df.head()
salary_df = pd.read_csv("../input/nba_2017_salary.csv");

salary_df.head();
pie_df = pd.read_csv("../input/nba_2017_pie.csv")

pie_df.head();
plus_minus_df = pd.read_csv("../input/nba_2017_real_plus_minus.csv");

plus_minus_df.head();

# Quite Suprised that LBJ got higher +- than Curry!!!!!!!!!!!!!!!!!
br_stats_df = pd.read_csv("../input/nba_2017_br.csv");

br_stats_df.head();
plus_minus_df.rename(columns={"NAME":"PLAYER", "WINS": "WINS_RPM"}, inplace=True)

players = [] #create a new list to store only the name info

for player in plus_minus_df["PLAYER"]:

    plyr, _ = player.split(",") # split the NAME of player in to name and position

    players.append(plyr)

plus_minus_df.drop(["PLAYER"], inplace=True, axis=1) 

plus_minus_df["PLAYER"] = players # replace the NAME column with just the name info; i.e LeBron James, SF	LeBron James

plus_minus_df.head();
nba_players_df = br_stats_df.copy()

nba_players_df.rename(columns={'Player': 'PLAYER',

                               'Pos':'POSITION', 

                               'Tm': "TEAM", 

                               'Age': 'AGE', 

                               "PS/G": "POINTS"}, inplace=True)

nba_players_df.drop(["G", "GS", "TEAM"], inplace=True, axis=1)

nba_players_df = nba_players_df.merge(plus_minus_df, how="inner", on="PLAYER")

nba_players_df.head();
pie_df_subset = pie_df[["PLAYER", "PIE", "PACE", "W"]].copy()

nba_players_df = nba_players_df.merge(pie_df_subset, how="inner", on="PLAYER")

nba_players_df.head();
salary_df.rename(columns={'NAME': 'PLAYER'}, inplace=True)

salary_df["SALARY_MILLIONS"] = round(salary_df["SALARY"]/1000000, 2)

salary_df.drop(["POSITION","TEAM", "SALARY"], inplace=True, axis=1)

salary_df.head();
## Some format cleaning ##

diff = list(set(nba_players_df["PLAYER"].values.tolist()) - set(salary_df["PLAYER"].values.tolist()))

len(diff) # indecated the No. of players that missing salary info from the list. 111 from this dataset
nba_players_with_salary_df = nba_players_df.merge(salary_df); 

diff_check = list(set(nba_players_with_salary_df["PLAYER"].values.tolist()) - set(salary_df["PLAYER"].values.tolist()))

len(diff_check) # make sure no missing info
### Start Explain Data Here ###

## EDA Correlation Heatmap ##

plt.subplots(figsize=(20,15))

ax = plt.axes()

ax.set_title("NBA Player Correlation Heatmap:  2016-2017 Season (STATS & SALARY)")

corr = nba_players_with_salary_df.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values,

           cmap = "Purples")

# the problem with the correlation chart is redundant information. i.e. eFG% is calculated based on 2p%;

# 3p% and free throw %. Thus, just to see correlation between "how well a player shoot" and other factors

# don't necessarily required to put all the correlations in the list. It is hard to find the squares some time.
## Salary vs. real plus minus ##

sns.lmplot(x="SALARY_MILLIONS", y="WINS_RPM", data=nba_players_with_salary_df);

sns.lmplot(x="SALARY_MILLIONS", y="POINTS", data=nba_players_with_salary_df);



# higher salary usually means higher rpm or the other way around. So team managers knows more about 

# how players contribute to the team
sns.lmplot(x="SALARY_MILLIONS", y="TWITTER_FAVORITE_COUNT", data=twit_df);

sns.lmplot(x="SALARY_MILLIONS", y="PAGEVIEWS", data=twit_df);

# proof that fans are focusing only on a few players in the League. However, 

# these players don't necessary havevery high salaries

# Most of the players recived much less concentration. But in general, their 

# salary is positive correlation to their popularity.
### Additional Analysis ###

## The salary vs. positions ##

sns.boxplot(y= "SALARY_MILLIONS" , x="POSITION", data = twit_df, orient="Vertical")

# Top small forwards are still the most valuable players in the league. But Centers are

# paid the most overall.
## The salary vs. Age ##

sns.boxplot(x= "AGE" , y="SALARY_MILLIONS", data = twit_df, orient="v");

sns.lmplot(x="AGE", y="SALARY_MILLIONS", data=twit_df);

# The peak of players are 28-33, where they gain the most of the money.

# elder players did not lose much salary. Are their performances persists?
sns.boxplot(x= "AGE" , y="POINTS", data = twit_df, orient="v");

sns.lmplot(x="AGE", y="POINTS", data=twit_df);

# Elder players are actually contribute less to the team if look at the max value at different ages.

# The performance of an elder player is less predictable. Injuriesmay happen without a clue.

# Therefore, it is hard for elder players to keep their competitiveness on the court.
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