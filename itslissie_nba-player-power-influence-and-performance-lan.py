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
attendance_valuation_elo_df = pd.read_csv("../input/nba_2017_att_val_elo.csv")
salary_df = pd.read_csv("../input/nba_2017_salary.csv")
pie_df = pd.read_csv("../input/nba_2017_pie.csv")
plus_minus_df = pd.read_csv("../input/nba_2017_real_plus_minus.csv")
br_stats_df = pd.read_csv("../input/nba_2017_br.csv")
plus_minus_df.rename(columns={"NAME":"PLAYER", "WINS": "WINS_RPM"}, inplace=True)
players = []
for player in plus_minus_df["PLAYER"]:
    plyr, _ = player.split(",")
    players.append(plyr)
plus_minus_df.drop(["PLAYER"], inplace=True, axis=1)
plus_minus_df["PLAYER"] = players
nba_players_df = br_stats_df.copy()
nba_players_df.rename(columns={'Player': 'PLAYER','Pos':'POSITION', 'Tm': "TEAM", 'Age': 'AGE', "PS/G": "POINTS"}, inplace=True)
nba_players_df.drop(["G", "GS", "TEAM"], inplace=True, axis=1)
nba_players_df = nba_players_df.merge(plus_minus_df, how="inner", on="PLAYER")
print(pd.isnull(nba_players_df).sum())
nba_players_detailed = nba_players_df.drop(['3P%', '2P%','FT%','POINTS','TEAM'], axis=1)
numerical_df = nba_players_detailed.ix[:,4:27]
numerical_df = numerical_df.apply(pd.to_numeric, errors='coerce')
numerical_df.head()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
print(scaler.fit(numerical_df))
print(scaler.transform(numerical_df))
distortions = []
for i in range(1, 11):
    km = KMeans(n_clusters=i,
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=0)
    km.fit(scaler.transform(numerical_df))
    distortions.append(km.inertia_)
    
plt.plot(range(1,11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.title("Detailed Indicators Elbow Method Cluster Analysis")
plt.show()
from sklearn.cluster import KMeans
k_means = KMeans(n_clusters=3)
kmeans = k_means.fit(scaler.transform(numerical_df))
nba_players_km = nba_players_df[["Rk","PLAYER","POSITION","AGE","POINTS","TEAM","RPM","WINS_RPM"]].copy()
nba_players_km['cluster'] = kmeans.labels_
nba_players_km.head()
pie_df_subset = pie_df[["PLAYER", "PIE", "PACE", "W"]].copy()
nba_players_km = nba_players_km.merge(pie_df_subset, how="inner", on="PLAYER")
nba_players_km.head()
salary_df.rename(columns={'NAME': 'PLAYER'}, inplace=True)
salary_df["SALARY_MILLIONS"] = round(salary_df["SALARY"]/1000000, 2)
salary_df.drop(["POSITION","TEAM", "SALARY"], inplace=True, axis=1)
salary_df.head()
nba_players_km_salary = nba_players_km.merge(salary_df,how="inner", on="PLAYER")
nba_players_km_salary.head()
plt.subplots(figsize=(10,8))
ax = plt.axes()
ax.set_title("NBA Player Correlation Heatmap:  2016-2017 Season (STATS & SALARY)")
corr = nba_players_km_salary.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values, cmap="Blues")
#EDA for relationship between Wins_RPM and other variables
sns.lmplot(x="POINTS", y="WINS_RPM", data=nba_players_km_salary); 
sns.lmplot(x="RPM", y="WINS_RPM", data=nba_players_km_salary);
sns.lmplot(x="cluster", y="WINS_RPM", data=nba_players_km_salary);
sns.lmplot(x="SALARY_MILLIONS", y="WINS_RPM", data=nba_players_km_salary);
import numpy as np
from matplotlib import cm
from sklearn.metrics import silhouette_samples
km = KMeans(n_clusters=3,
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=0)
y_km = km.fit_predict(scaler.transform(numerical_df))
cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(scaler.transform(numerical_df),
                                     y_km,
                                     metric='euclidean')
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(i)/n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, edgecolor='none',color=color)
    yticks.append((y_ax_lower + y_ax_upper)/2)
    y_ax_lower += len(c_silhouette_vals)
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg,
            color="red",
            linestyle="--")
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.title('Silhouette Plot Team Valuation')
plt.figure(figsize=(20,10))
plt.show()
results = smf.ols(' WINS_RPM~cluster', data=nba_players_km_salary).fit()
print(results.summary())
results = smf.ols('WINS_RPM ~RPM', data=nba_players_km_salary).fit()
print(results.summary())
results = smf.ols(formula='WINS_RPM ~ POINTS+RPM+cluster+SALARY_MILLIONS', data=nba_players_km_salary).fit()
print(results.summary())