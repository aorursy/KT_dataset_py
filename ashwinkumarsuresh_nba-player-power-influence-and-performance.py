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
wiki_df = pd.read_csv("../input/nba_2017_player_wikipedia.csv");wiki_df.head()

twitter_df = pd.read_csv("../input/nba_2017_twitter_players.csv");twitter_df.head()
nba_overall = pd.read_csv("../input/nba_2016_2017_100.csv")
nba_overall.head(5)
plt.subplots(figsize=(30,25))
ax = plt.axes()

corr_overall = nba_overall.corr()
corr_overall

sns.heatmap(corr_overall, 
            xticklabels=corr_overall.columns.values,
            yticklabels=corr_overall.columns.values)
results1 = smf.ols('SALARY_MILLIONS ~ AGE +GP + W_PCT + MIN + TS_PCT + PIE + TWITTER_FOLLOWER_COUNT_MILLIONS ', data= nba_overall).fit()

print(results1.summary())
sns.barplot(nba_overall.AGE, nba_overall.SALARY_MILLIONS)
a = nba_overall.AGE.max()
nba_overall[(nba_overall['AGE'] == a )]
nba_overall[['TWITTER_FOLLOWER_COUNT_MILLIONS', 'SALARY_MILLIONS']].corr()
nba_overall.head()

nba_overall.describe()
value = pd.read_csv('../input/nba_2017_team_valuations.csv')
#plt.plots(figsize=(10,100))
#ax = plt.axes()
x = value.TEAM
y = value.VALUE_MILLIONS
test = sns.barplot(x , y  ) #,  'tickangle': 90)

for item in test.get_xticklabels():
    item.set_rotation(90)