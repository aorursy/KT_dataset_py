import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
color = sns.color_palette()

from IPython.core.display import display, HTML
display(HTML("<style>.container {width:100% !important; }</style>"))
%matplotlib inline
import pandas as pd
from pandas import Series,DataFrame
import io

salary_wiki_twitter_df = pd.read_csv("../input/nba_2017_players_with_salary_wiki_twitter.csv")
attendance_df = pd.read_csv("../input/nba_2017_att_val_elo.csv")
team_valuation_df = pd.read_csv("../input/nba_2017_team_valuations.csv")

salary_wiki_twitter_df.head()
player = salary_wiki_twitter_df[['PLAYER','TEAM','POSITION','POINTS','WINS_RPM','PIE','RPM','SALARY_MILLIONS','PAGEVIEWS','TWITTER_FAVORITE_COUNT','TWITTER_RETWEET_COUNT']]
player.head()
attendance_df.head()
dictionary_map = {"TEAM":{"ATL":"Atlanta Hawks",
"BKN":"Brooklyn Nets",
"BOS":"Boston Celtics",
"CHA":"Charlotte Hornets",
"CHI":"Chicago Bulls",
"CLE":"Cleveland Cavaliers",
"DAL":"Dallas Mavericks",
"DEN":"Denver Nuggets",
"DET":"Detroit Pistons",
"GSW":"Golden State Warriors",
"HOU":"Houston Rockets",
"IND":"Indiana Pacers",
"LAC":"Los Angeles Clippers",
"LAL":"Los Angeles Lakers",
"MEM":"Memphis Grizzlies",
"MIA":"Miami Heat", 
"MIL":"Milwaukee Bucks",
"MIN":"Minnesota Timberwolves",
"NOP":"New Orleans Pelicans",
"NYK":"New York Knicks",
"OKC":"Oklahoma City Thunder",
"ORL":"Orlando Magic",
"PHI":"Philadelphia 76ers",
"PHX":"Phoenix Suns",
"POR":"Portland Trail Blazers",
"SAC":"Sacramento Kings",
"SAS":"San Antonio Spurs",
"TOR":"Toronto Raptors",
"UTA":"Utah Jazz",
"WAS":"Washington Wizards"}}
player.replace(dictionary_map,inplace = True)

player.head()
total_data_df = player.merge(attendance_df, how = 'inner', on = 'TEAM')


total_data_df.head()
attendance_df = attendance_df.rename(columns = {'TOTAL':'TOTALATTENDANCE'})
attendance_df.head()

visualization = attendance_df[['VALUE_MILLIONS','TOTALATTENDANCE', 'AVG','PCT','ELO','CONF']]
corr_elo = visualization.corr()
plt.subplots(figsize=(10,8))
ax = plt.axes()
ax.set_title("NBA Team Correlation Heatmap:  2016-2017 Season (ELO, AVG Attendance, VALUATION IN MILLIONS)")
sns.heatmap(corr_elo, 
            xticklabels=corr_elo.columns.values,
            yticklabels=corr_elo.columns.values)
corr_elo
!pip -q install ggplot
ax = sns.lmplot(x="TOTALATTENDANCE", y="VALUE_MILLIONS", data = visualization, hue="CONF", size=8)
ax.set(xlabel='Total Attendance', ylabel='VALUE_MILLIONS', title="NBA Team Valuation in Millions vs Total Attendance:  2016-2017 Season")
ax = sns.lmplot(x="ELO", y="VALUE_MILLIONS", data = visualization, hue="CONF", size=8)
ax.set(xlabel='ELO Score', ylabel='VALUE_MILLIONS', title="NBA Team Valuation in Millions vs ELO Ranking:  2016-2017 Season")
corr_elo1 = player.corr()
plt.subplots(figsize=(10,8))
ax = plt.axes()
ax.set_title("NBA Team Player Information Correlation Heatmap:  2016-2017 Season")
sns.heatmap(corr_elo1, 
            xticklabels=corr_elo1.columns.values,
            yticklabels=corr_elo1.columns.values)
corr_elo1
!pip -q install ggplot

from ggplot import *
p = ggplot(player,aes(x="WINS_RPM", y="SALARY_MILLIONS", color="POSITION")) + geom_point(size=100)
p + xlab("WINS_RPM") + ylab("SALARY_MILLION") + ggtitle("NBA Players 2016-2017: WINS_RPM, SALARY_MILLIONS and POSITION")
total_data_df.head()
team_level_data = total_data_df.groupby('TEAM').mean()
team_level_data.head()
team_level_data.drop(columns = ['Unnamed: 0','GMS'],inplace = True)
team_level_data.head()
corr_elo2 = team_level_data.corr()
plt.subplots(figsize=(10,8))
ax = plt.axes()
ax.set_title("NBA TEAM Correlation Heatmap:  2016-2017 Season")
sns.heatmap(corr_elo2, 
            xticklabels=corr_elo2.columns.values,
            yticklabels=corr_elo2.columns.values)
results = smf.ols('VALUE_MILLIONS ~ POINTS+WINS_RPM+PIE+RPM+SALARY_MILLIONS+PAGEVIEWS+TWITTER_RETWEET_COUNT+ TOTAL +AVG+PCT +ELO', data=team_level_data).fit()
print(results.summary())

from ggplot import *
p = ggplot(total_data_df,aes(x="SALARY_MILLIONS", y="VALUE_MILLIONS", color="TEAM")) + geom_point(size=100)
p + xlab("SALARY_MILLIONS") + ylab("TEAM VALUE MILLIONS") + ggtitle("NBA Players 2016-2017: SALARY MILLIONS, TEAM VALUATION, TEAM")