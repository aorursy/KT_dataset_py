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
diff = list(set(nba_players_df["PLAYER"].values.tolist()) - set(salary_df["PLAYER"].values.tolist()))
len(diff)


nba_players_with_salary_df = nba_players_df.merge(salary_df); 

plt.subplots(figsize=(20,15))
ax = plt.axes()
ax.set_title("NBA Player Correlation Heatmap:  2016-2017 Season (STATS & SALARY)")
corr = nba_players_with_salary_df.corr()
sns.heatmap(corr, cmap='RdBu',
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
sns.lmplot(x="SALARY_MILLIONS", y="POINTS", data=nba_players_with_salary_df)

results = smf.ols('SALARY_MILLIONS ~POINTS', data=nba_players_with_salary_df).fit()

print(results.summary())

results = smf.ols('SALARY_MILLIONS ~FGA', data=nba_players_with_salary_df).fit()

print(results.summary())

results = smf.ols('SALARY_MILLIONS ~FTA', data=nba_players_with_salary_df).fit()

print(results.summary())

results = smf.ols('SALARY_MILLIONS ~MPG', data=nba_players_with_salary_df).fit()

print(results.summary())

results = smf.ols('SALARY_MILLIONS ~FT', data=nba_players_with_salary_df).fit()
print(results.summary())
results = smf.ols('SALARY_MILLIONS ~FG', data=nba_players_with_salary_df).fit()
print(results.summary())
from sklearn.tree import DecisionTreeRegressor  
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
#from sklearn.mode_selection import train_test_split

nba_point = pd.DataFrame({'SALARY_MILLIONS':nba_players_with_salary_df['SALARY_MILLIONS'],'POINTS':nba_players_with_salary_df['POINTS'],'MPG':nba_players_with_salary_df['MPG'],'FG':nba_players_with_salary_df['FG'],'FGA':nba_players_with_salary_df['FGA'],'FT':nba_players_with_salary_df['FT'],'FTA':nba_players_with_salary_df['FTA']})
nba_point.describe()
target='SALARY_MILLIONS' #EXPECTED OUTPUT
x_columns = [x for x in nba_point.columns if x not in [target]]

X = nba_point[x_columns]
y = nba_point['SALARY_MILLIONS']
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=33)

rf=RandomForestRegressor()
rf1 = rf.fit(X,y) 
y_predprob = rf1.predict(X)
rf1.score(X,y)
r2_score(y, y_predprob)
mean_squared_error(y,y_predprob)
list(zip(nba_point[x_columns],rf1.feature_importances_))