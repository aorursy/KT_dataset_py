import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
%matplotlib inline
salary_df = pd.read_csv("../input/nba_2017_salary.csv");salary_df.head()

pie_df = pd.read_csv("../input/nba_2017_pie.csv");pie_df.head()
plus_minus_df = pd.read_csv("../input/nba_2017_real_plus_minus.csv")
plus_minus_df["NAME"] = plus_minus_df["NAME"].apply(lambda x: x.split(",")[0])
plus_minus_df.rename(columns = {"NAME" : "PLAYER"}, inplace = True)
plus_minus_df.head()
br_df = pd.read_csv("../input/nba_2017_br.csv");br_df.head()
players_df = br_df.copy()
players_df.rename(columns={'Player': 'PLAYER','Pos':'POSITION', 'Tm': "TEAM", 'Age': 'AGE', "PS/G": "POINTS"}, inplace=True)
players_df.drop(["G", "GS", "TEAM"], inplace=True, axis=1)
players_df = players_df.merge(plus_minus_df, how="inner", on="PLAYER")
players_df = players_df.merge(pie_df[["PLAYER", "PIE", "PACE", "W"]], how="inner", on="PLAYER")

salary_df.rename(columns={'NAME': 'PLAYER'}, inplace=True)
salary_df["SALARY_MILLIONS"] = round(salary_df["SALARY"]/1000000, 2)
salary_df.drop(["POSITION","TEAM", "SALARY"], inplace=True, axis=1)
salary_df.head()
players_df = players_df.merge(salary_df) 
players_df.head(5)
players_df.columns
#check NULL values
players_df.apply(axis=0, func=lambda x : any(pd.isnull(x)))
#fill NULL with mean
players_df["3P%"] = players_df["3P%"].fillna(players_df["3P%"].mean())
players_df["FT%"] = players_df["FT%"].fillna(players_df["FT%"].mean())
sns.set(style="white")

players_corr = players_df.corr()
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(30, 30))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(players_corr, cmap=cmap,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True, fmt=".2f")
##Euclidean Distance between two players based on their abilities
#Import Euclidean Distance Packages
from sklearn.preprocessing import normalize

from scipy.spatial.distance import pdist, squareform
#normalize vector for each player 
norm_df = pd.DataFrame(normalize(players_df[['Rk', 'AGE', 'MP', 'FG', 'FGA', 'FG%', '3P',
       '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB',
       'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'POINTS', 'GP',
       'MPG', 'ORPM', 'DRPM', 'RPM', 'WINS', 'PIE', 'PACE', 'W',
       'SALARY_MILLIONS']] , axis=1, copy=True, return_norm=False))
#Calculate Euclidean Distance between each player

dist = pdist(norm_df, 'euclidean')
dis_df = pd.DataFrame(squareform(dist))
dis_df.set_index(players_df["PLAYER"],inplace = True)
dis_df.columns = list(dis_df.index)
dis_df
sns.clustermap(dis_df,cmap=cmap)
plt.figure(figsize=(8,8))
plt.imshow(dis_df, cmap='YlGnBu_r')
cbar = plt.colorbar()
cbar.set_label('Eculidean Distance')
closest = np.where(dis_df.eq(dis_df[dis_df != 0].min(),0),dis_df.columns,False)
# Remove false from the array and get the column names as list
close_player_df = pd.DataFrame()
close_player_df["PLAYER"] = players_df["PLAYER"]
close_player_df['CLOSE_PLAYER_ABILITY'] = [i[i.astype(bool)].tolist() for i in closest]
close_player_df.head(10)
import scipy.cluster.hierarchy as hac
cluster_hac = hac.linkage(norm_df,method="ward")
plt.figure(figsize=(50, 200))
dendogram = hac.dendrogram(cluster_hac, leaf_font_size=50,orientation='right',show_leaf_counts = True,show_contracted=True,labels=dis_df.index)
plt.title('Hierarchical Clustering Dendrogram', fontsize=50)
plt.xlabel('Distance', fontsize=50)
plt.ylabel('Player', fontsize=50)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist


from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation #For clustering

from sklearn.mixture import GaussianMixture #For GMM clustering
#Determine the number of Clusters
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(norm_df)
    kmeanModel.fit(norm_df)
    distortions.append(sum(np.min(cdist(norm_df, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / dis_df.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
#k = 3
kmeans = KMeans(n_clusters=3, random_state=0).fit(norm_df)
kmeans_df = pd.DataFrame(kmeans.labels_)
players_df.insert((players_df.shape[1]),'kmeans',kmeans_df)
players_df
plt.figure(figsize=(20, 20))
scatter = plt.scatter(players_df['W'],players_df["SALARY_MILLIONS"],
                     c=kmeans_df[0],s=50)
plt.title('K-Means Clustering')
plt.xlabel('Win')
plt.ylabel('Salary')
plt.colorbar(scatter,ticks=np.linspace(0,3,4))
Agg = AgglomerativeClustering(n_clusters=3, affinity = 'euclidean', linkage = 'ward').fit_predict(norm_df)
Agg_df = pd.DataFrame(Agg)
players_df.insert((players_df.shape[1]),'AgglomerativeClustering',Agg_df)
plt.figure(figsize=(20, 20))
scatter = plt.scatter(players_df['W'],players_df["SALARY_MILLIONS"],
                     c=Agg_df[0],s=50)
plt.title('Agglomerative Clustering')
plt.xlabel('Win')
plt.ylabel('Salary')
plt.colorbar(scatter,ticks=np.linspace(0,3,4))

gau_model = GaussianMixture(n_components=3,init_params='kmeans')
gau_model.fit(norm_df)
gau_label = gau_model.predict(norm_df)
gau_df = pd.DataFrame(gau_label)
players_df.insert((players_df.shape[1]),'gmm',gau_df)
plt.figure(figsize=(20, 20))
scatter = plt.scatter(players_df['W'],players_df["SALARY_MILLIONS"],
                     c=gau_df[0],s=50)
plt.title('Gaussian Mixture Clustering')
plt.xlabel('Win')
plt.ylabel('Salary')
plt.colorbar(scatter,ticks=np.linspace(0,3,4))
