

import pandas as pd

import statsmodels.api as sm

import statsmodels.formula.api as smf

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()

%matplotlib inline
attendance_df = pd.read_csv("../input/nba_2017_attendance.csv");attendance_df.head()

# number of games the teamed in the season

# Total attendance for the whole season

# Percentage of average capacity of the stadium that is filled
endorsement_df = pd.read_csv("../input/nba_2017_endorsements.csv");endorsement_df.head()

# endorsement: money paid for advertisement
valuations_df = pd.read_csv("../input/nba_2017_team_valuations.csv");valuations_df.head()
salary_df = pd.read_csv("../input/nba_2017_salary.csv");salary_df.head()

# small forward

# Point Guard

# Center

# Power forward

# salary in dollar
pie_df = pd.read_csv("../input/nba_2017_pie.csv");pie_df.head()

# GP game played in season (no tie), 82 games in the whole season

# Win

# Loss

# Min: average minutes per game in this season

# OFFRTG: score gained everytime he get the ball

# DEFRTG: points allowed when he/she faced in players in 100 times

# NETRTG：offrtg - defrtg 

# AST ratio：

# OREB%：Offense Rebound percentage 

# DREB%：the higher the better

# TO Ratio: turnover ratio

plus_minus_df = pd.read_csv("../input/nba_2017_real_plus_minus.csv");plus_minus_df.head()

# RPM real plus minus 

# ORPM higher the better

# DRPM higher the better

# WINS 
br_stats_df = pd.read_csv("../input/nba_2017_br.csv");br_stats_df.head()

# rk ranking

# pos position

# tm team

# G games

# GS 

# MP minutes played per game 

# FG field goal 

# FGA field goal attempted

# ft% free through percentage

# ORB 

# DRB 

# TRB = orb + drb

# ast assist 

# stl steel 

# BLK block

# TOV turnover 

# PF personal false 

# PS/
elo_df = pd.read_csv("../input/nba_2017_elo.csv");elo_df.head()

attendance_valuation_df = attendance_df.merge(valuations_df, how="inner", on="TEAM")
attendance_valuation_df.head()

attendance_valuation_elo_df = pd.read_csv("../input/nba_2017_att_val_elo.csv")
attendance_valuation_elo_df.head()

endorsement_df
endorsement_df['SALARY'] = endorsement_df['SALARY'].str.replace(',', '')

endorsement_df['SALARY'] = endorsement_df['SALARY'].str.replace('$', '')

endorsement_df['SALARY'] = endorsement_df['SALARY'].astype(float)
endorsement_df['ENDORSEMENT'] = endorsement_df['ENDORSEMENT'].str.replace(',', '')

endorsement_df['ENDORSEMENT'] = endorsement_df['ENDORSEMENT'].str.replace('$', '')

endorsement_df['ENDORSEMENT'] = endorsement_df['ENDORSEMENT'].astype(float)
endorsement_df
# calculate total of salary and endorsement

endorsement_df["total"] = endorsement_df.ENDORSEMENT + endorsement_df.SALARY

# set general plot properties

plt.subplots(figsize = (20,15))

ax = plt.axes()

# Plot 1 - background - "total" (top) series

sns.set_color_codes("muted")

sns.barplot(x="total", y = "NAME", data = endorsement_df, label = "Endorsement", color = 'b')

#Plot 2 - overlay - "bottom" series

sns.set_color_codes("pastel")

sns.barplot(x="SALARY", y = "NAME", data = endorsement_df, label = "Salary", color = "b")

# Add a legend 

ax.legend(ncol=2, loc="lower right", frameon=True) 

# add label

ax.set(ylabel="Player Names",

       xlabel="Player Salary and Endorsement")

# remove rim of table

sns.despine(left=True, bottom=True)

# reference: https://github.com/noahgift/spot_price_machine_learning/blob/master/notebooks/spot_pricing_ml.ipynb

# reference: http://randyzwitch.com/creating-stacked-bar-chart-seaborn/
results = smf.ols('ENDORSEMENT ~ SALARY', data=endorsement_df).fit()
print(results.summary())
import numpy as np

A = endorsement_df['SALARY'].values

B = endorsement_df['ENDORSEMENT'].values

print (np.corrcoef(A,B))
player_stats_df
player_stats_df[['ENDORSEMENT','Age','NAME']].sort_values(by = 'ENDORSEMENT', ascending = False)
a = player_stats_df['ENDORSEMENT'].values
b = player_stats_df['Age'].values
print (np.corrcoef(a,b))
attendance_valuation_elo_df.info()
import plotly.plotly as py

import cufflinks as cf

print (cf.__version__)
elo_value = attendance_valuation_elo_df[['TEAM','VALUE_MILLIONS','ELO']].sort_values(by='VALUE_MILLIONS', ascending = False)
# need to set the index of table to be the teams

elo_value = elo_value.set_index('TEAM')
cf.go_offline()

elo_value.iplot(title="Team ELO and Value ",

                    xTitle="Teams",

                    yTitle="",

                   #bestfit=True, bestfit_colors=["pink"],

                   #subplots=True,

                   shape=(4,1),

                    #subplot_titles=True,

                    fill=True,)
#pie_df
br_stats_df.head(5)
endorsement_df
player_stats_df = pd.merge(endorsement_df,br_stats_df , left_on = 'NAME', right_on = 'Player')
player_stats_df
player_stats_df['ast_tov'] = player_stats_df['AST']/player_stats_df['TOV']
sns.lmplot(x="SALARY", y="ast_tov", data=player_stats_df)

plt.show()
sns.barplot(x = 'ast_tov', y = 'NAME', data = player_stats_df, )
wikipedia_df = pd.read_csv("../input/nba_2017_player_wikipedia.csv");wikipedia_df.head()
twitter_df = pd.read_csv("../input/nba_2017_twitter_players.csv");twitter_df.head()
player_stats_df.head(5)
player_stats_twitter_df = pd.merge(player_stats_df, twitter_df, left_on = 'NAME', right_on = 'PLAYER')
plt.subplots(figsize=(20,15))

ax = plt.axes()

ax.set_title("NBA Player Correlation Heatmap")

corr = player_stats_twitter_df.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values,

           cmap="Oranges")
from ggplot import *
p = ggplot(player_stats_twitter_df,aes(x="2P%", y="ENDORSEMENT")) + geom_point(size=100, color='orange') + stat_smooth(method='lm') 

p + xlab("2P%") + ylab("Endorsement") + ggtitle("NBA Players 2016-2017: Age vs Salary")
player_stats_twitter_df[['2P%','ENDORSEMENT']].corr()
sns.lmplot(x="FG%", y="ENDORSEMENT", data=player_stats_twitter_df)
player_stats_twitter_df[["FG%", "ENDORSEMENT"]].corr()
sns.lmplot(x="eFG%", y="ENDORSEMENT", data=player_stats_twitter_df)
player_stats_twitter_df[["eFG%", "ENDORSEMENT"]].corr()
sns.lmplot(x="TWITTER_FAVORITE_COUNT", y="ENDORSEMENT", data=player_stats_twitter_df)
player_stats_twitter_df[["TWITTER_FAVORITE_COUNT", "ENDORSEMENT"]].corr()