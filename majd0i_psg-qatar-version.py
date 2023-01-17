import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import sqlite3

from sklearn.metrics import accuracy_score

import warnings



%matplotlib inline

warnings.filterwarnings('ignore')

sns.set(style='whitegrid', context='notebook', palette='deep')

np.random.seed(2)



con = sqlite3.connect("../input/database.sqlite")

matches = pd.read_sql_query("SELECT * FROM Match;",con)

teams = pd.read_sql_query("SELECT * FROM Team;",con)

#leagues = pd.read_sql_query("SELECT * FROM League;",con)

countries = pd.read_sql_query("SELECT * FROM Country;",con)



matches.head()
matches = matches.drop(labels = ['home_player_X1','home_player_X2','home_player_X3',

                       'home_player_X4','home_player_X5','home_player_X6',

                       'home_player_X7','home_player_X8','home_player_X9',

                       'home_player_X10','home_player_X11',

                       'home_player_Y1','home_player_Y2','home_player_Y3',

                       'home_player_Y4','home_player_Y5','home_player_Y6',

                       'home_player_Y7','home_player_Y8','home_player_Y9',

                       'home_player_Y10','home_player_Y11','home_player_1',

                       'home_player_2','home_player_3','home_player_4',

                       'home_player_5','home_player_6','home_player_7',

                       'home_player_8','home_player_9','home_player_10',

                       'home_player_11','away_player_X1','away_player_X2',

                       'away_player_X3','away_player_X4',

                       'away_player_X5','away_player_X6','away_player_X7',

                       'away_player_X8','away_player_X9','away_player_X10',

                       'away_player_X11','away_player_Y2','away_player_Y3',

                       'away_player_Y4','away_player_Y5','away_player_Y6',

                       'away_player_Y7','away_player_Y1',

                       'away_player_Y8','away_player_Y9','away_player_Y10',

                       'away_player_Y11','away_player_1',

                       'away_player_2','away_player_3','away_player_4',

                       'away_player_5','away_player_6','away_player_7',

                       'away_player_8','away_player_9','away_player_10',

                       'away_player_11'],axis = 1)

matches = matches.drop(labels = ['goal','shoton','shotoff','foulcommit','card','cross','corner','possession'],axis = 1)

matches = matches.drop(labels = ['GBH','GBD','GBA','BSH','BSD','BSA','SJH','SJD','SJA','PSH','PSD','PSA'],axis = 1)

matches = matches.dropna(axis = 0).reset_index(drop=True)



matches.isnull().sum()
matches.shape
matches = matches.merge(countries,left_on="league_id",right_on="id")

matches = matches.drop(labels = ["id_y","league_id","country_id"],axis=1)

matches = matches.rename(columns={'name':'league_country'})

matches["result"] = (matches["home_team_goal"] - matches["away_team_goal"]).map(

         lambda s : 0 if s > 0 else 1 if s == 0 else 2 )



matches["B365"] = np.argmin(matches[["B365H","B365D","B365A"]].values,axis = 1)

matches["BW"] = np.argmin(matches[["BWH","BWD","BWA"]].values,axis = 1)

matches["IW"] = np.argmin(matches[["IWH","IWD","IWA"]].values,axis = 1)

matches["LB"] = np.argmin(matches[["LBH","LBD","LBA"]].values,axis = 1)

matches["WH"] = np.argmin(matches[["WHH","WHD","WHA"]].values,axis = 1)

matches["VC"] = np.argmin(matches[["VCH","VCD","VCA"]].values,axis = 1)
accuracy_score(matches["result"],matches["B365"])
accuracy_score(matches["result"],matches["BW"])
accuracy_score(matches["result"],matches["IW"])
accuracy_score(matches["result"],matches["LB"])
accuracy_score(matches["result"],matches["WH"])
accuracy_score(matches["result"],matches["VC"])
# Compute accuracy in each group in the groupby pandas objects

def acc_group(y_true_desc,y_pred_desc):

    def inner(group):

        return accuracy_score(group[y_true_desc],group[y_pred_desc])

    inner.__name__ = 'acc_group'

    return inner



matches.groupby("league_country").apply(acc_group("result","B365"))

league_seasons_accuracies = matches.groupby(("league_country","season")).apply(acc_group("result","B365"))

league_seasons_accuracies = league_seasons_accuracies.reset_index()

league_seasons_accuracies = league_seasons_accuracies.rename(columns={0:'accuracy'})



selected_countries = ["France","Spain","England","Germany","Italy"]



Five_leagues = league_seasons_accuracies[league_seasons_accuracies['league_country'].isin(selected_countries)]



g = sns.factorplot(x="season",y="accuracy",hue="league_country",data=Five_leagues,size=6,aspect=1.5)

g.set_xticklabels(rotation=45)

sns.plt.suptitle('Bet 365 accuracy for the 5 biggest soccer leagues')
PSG_id = teams.loc[teams["team_short_name"] == 'PSG','team_api_id'].ravel()[0]

PSG_games = matches.loc[((matches["home_team_api_id"] == PSG_id) | (matches["away_team_api_id"] == PSG_id))]

PSG_games["date"] = pd.to_datetime(PSG_games["date"])

seasons = list(PSG_games["season"].unique())

PSG_games["season"] = pd.to_numeric(PSG_games["season"].map(lambda s: s.split("/")[0]))



PSG_goals = []

Opp_goals = []

for team_home, hg, ag in PSG_games[["home_team_api_id","home_team_goal","away_team_goal"]].values:

    if team_home == PSG_id:

        PSG_goals.append(hg)

        Opp_goals.append(ag)

    else :

        PSG_goals.append(ag)

        Opp_goals.append(hg)



PSG_games["PSG_goals"] = PSG_goals

PSG_games["Opp_goals"] = Opp_goals



# 0 for win , 1 Draw and 2 for loss

PSG_games["PSG_result"] = (PSG_games["PSG_goals"] - PSG_games["Opp_goals"]).map(

         lambda s : 0 if s > 0 else 1 if s == 0 else 2)



PSG_games['Investor'] = PSG_games["season"].map(lambda s: 'Qatar' if s>=2011 else 'Other')

results_counts = (PSG_games.groupby(['Investor'])['PSG_result']

                     .value_counts(normalize=True)

                     .rename('percentage')

                     .mul(100)

                     .reset_index())



sns.set_style("darkgrid")

p = sns.barplot(x="PSG_result", y="percentage", hue="Investor", data=results_counts,alpha=0.7)

p.set_ylabel("Percentage")

p = p.set_xticklabels(["Win","Draw","Loss"])
fig,axs = plt.subplots(1,2,figsize=(8,5))

g = sns.barplot(y="PSG_goals",x="Investor", data=PSG_games,ax=axs[0],alpha=0.7)

g.set_ylabel("Goals/game")

g = sns.barplot(y="Opp_goals",x="Investor", data=PSG_games,ax=axs[1],alpha=0.7)

g.set_ylabel("Goals conceded /game")

plt.tight_layout()

plt.show()

plt.gcf().clear()
#Victory_series :

def get_best_streak(PSG_results,result_match):

    best_streak = 0

    max_streak = 0

    for i in PSG_results:

        if best_streak > max_streak:

            max_streak = best_streak

        

        if i in result_match:

            best_streak += 1

        else :

            best_streak = 0

    return max_streak

# 0 == Win , 1 == Draw, 2 == Loss

PSG_streaks = []

PSG_streaks.append(get_best_streak(PSG_games["PSG_result"],result_match=[0,1]))

PSG_streaks.append(get_best_streak(PSG_games["PSG_result"],result_match=[0]))

PSG_streaks.append(get_best_streak(PSG_games["PSG_result"],result_match=[2,1]))

PSG_streaks.append(get_best_streak(PSG_games["PSG_result"],result_match=[2]))



PSG_streaks.append(get_best_streak(PSG_games.loc[PSG_games["Investor"] == "Qatar","PSG_result"],result_match=[0,1]))

PSG_streaks.append(get_best_streak(PSG_games.loc[PSG_games["Investor"] == "Qatar","PSG_result"],result_match=[0]))

PSG_streaks.append(get_best_streak(PSG_games.loc[PSG_games["Investor"] == "Qatar","PSG_result"],result_match=[2,1]))

PSG_streaks.append(get_best_streak(PSG_games.loc[PSG_games["Investor"] == "Qatar","PSG_result"],result_match=[2]))



PSG_streaks.append(get_best_streak(PSG_games.loc[PSG_games["Investor"] != "Qatar","PSG_result"],result_match=[0,1]))

PSG_streaks.append(get_best_streak(PSG_games.loc[PSG_games["Investor"] != "Qatar","PSG_result"],result_match=[0]))

PSG_streaks.append(get_best_streak(PSG_games.loc[PSG_games["Investor"] != "Qatar","PSG_result"],result_match=[2,1]))

PSG_streaks.append(get_best_streak(PSG_games.loc[PSG_games["Investor"] != "Qatar","PSG_result"],result_match=[2]))



Investors = ["All"]*4

Investors.extend(["Qatar"]*4)

Investors.extend(["Other"]*4)

Streak_type = ["Undefeat_streak","Winning_streak","Nowin_streak","Loss_streak"]*3



PSG_streaks = pd.DataFrame({'Investor':Investors,'PSG_streak':PSG_streaks,'Streak_type':Streak_type})



g = sns.barplot(x="Streak_type",hue="Investor",y="PSG_streak",data=PSG_streaks,alpha=0.8,saturation=1)

g.set_ylabel("Games")

g = g.set_xticklabels(["Undefeat streak","Winning streak","No winning streak","Losing streak"])
colors = ['r','r','r','g','g','g','g','g']

g = sns.barplot(y="PSG_goals",x="season", data=PSG_games,estimator=sum,palette=sns.color_palette(colors),

                label='Before Quatar',saturation=1,alpha=0.7,ci=0)

g.set_ylabel("Goals / season")

g.set_xticklabels(seasons,rotation = 45)

g.legend()
g = sns.barplot(y="Opp_goals",x="season", data=PSG_games,estimator=sum,palette=sns.color_palette(colors),

                label='Before Quatar',saturation=1,alpha=0.7,ci=0)

g.set_ylabel("Goals conceded / season")

g.set_xticklabels(seasons,rotation = 45)

g.legend()
PSG_bets = PSG_games.groupby(("season","Investor")).apply(acc_group("result","B365"))



PSG_bets = PSG_bets.reset_index()

PSG_bets = PSG_bets.rename(columns={0:'accuracy'})



g = sns.factorplot(x='season',y="accuracy",hue="Investor",data=PSG_bets,size=5,aspect=1.5)

g.set_xticklabels(seasons,rotation = 45)

g = sns.plt.suptitle('Bet 365 accuracy of PSG games')