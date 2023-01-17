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
#the top most attended games: sort the dataframe by "total" attendence
attendance_valuation_elo_df  = pd.read_csv("../input/social-power-nba/nba_2017_att_val_elo.csv");attendance_valuation_elo_df.head()
attendance_valuation_elo_df_sorted = attendance_valuation_elo_df.sort_values(by=['TOTAL'], ascending=False)
attendance_valuation_elo_df_sorted.head(6)
sns.lmplot(x="TOTAL", y="VALUE_MILLIONS", data=attendance_valuation_elo_df)
arenas  = pd.read_csv("../input/nba-arenas-pop/NBA_Arenas_Pop.csv")
arenas.head(6)

val_atten = attendance_valuation_elo_df.copy()
val_arena = val_atten.merge(arenas, how="inner", on="TEAM")
df = val_arena.drop(["Unnamed: 0", "GMS"], axis=1)
df.head(6)
df.to_csv('df.csv', index=False)

corr = df.corr()
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, cmap = cmap,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

valuations2 = df.pivot("TEAM",  "POPULATION_2016", "VALUE_MILLIONS")
plt.subplots(figsize=(20,15))
ax = plt.axes()
ax.set_title("NBA Team AVG Attendance vs Valuation in Millions Vs Capacity of Arena")
sns.heatmap(valuations2,linewidths=.5, annot=True, fmt='g')
numerical_df = df.loc[:,["TOTAL", "ELO", "VALUE_MILLIONS", "POPULATION_2016","two teams in city", "CAPACITY", "OPENED"]]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
print(scaler.fit(numerical_df))
print(scaler.transform(numerical_df))
from sklearn.cluster import KMeans
k_means = KMeans(n_clusters=3)
kmeans = k_means.fit(scaler.transform(numerical_df))
df['cluster'] = kmeans.labels_
df.sort_values(by = ["cluster"], ascending = True)




#Top paid players (Salary)
salary_df = pd.read_csv("../input/social-power-nba/nba_2017_salary.csv");salary_df.head()
salary_df_sorted = salary_df.sort_values(by = "SALARY",ascending=False )
salary_df_sorted.head(12) 
#Player Impact Estimate, Top PIE
#a player’s impact on each individual game they play
pie_df = pd.read_csv("../input/social-power-nba/nba_2017_pie.csv");pie_df.head()
pie_df_sorted = pie_df.sort_values(["PIE"],ascending = False)
pie_df_sorted.head(6)
# Real Plus_Minus (RPM), top RPM
# ESPN metrics that merely registers the net change in score (plus or minus) while each player is on the court.
plus_minus_df = pd.read_csv("../input/social-power-nba/nba_2017_real_plus_minus.csv");plus_minus_df.head()
plus_minus_df_sorted = plus_minus_df.sort_values (["RPM"],ascending = False) 
plus_minus_df_sorted.head(12)
# Basketball Reference Statistics
br_stats_df = pd.read_csv("../input/social-power-nba/nba_2017_br.csv");br_stats_df.head()
#rename columns in order to merge 
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


