# importation des différentes librairies

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt # plotting



# Parcours des jeux de données

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# lectures des données

nRowsRead = None # nombre de lignes lues, 'None' pour lire le fichier entierement

df = pd.read_csv('/kaggle/input/tennis-20112019/atp.csv', delimiter=',', nrows = nRowsRead)

df.dataframeName = 'atp.csv'



# sélection des données

df1 = df[['ID', 'GRes_1', 'GRes_2', 'Age_CUR_1', 'Age_CUR_2', 'IsBirthDay_CUR_1', 'IsBirthDay_CUR_2', 'DaysFromLast_CUR_1', 'DaysFromLast_CUR_2', 'TotalPointsWon_1', 'TotalPointsWon_2', 'Serve1stWon_1', 'Serve1stWon_2', 'Serve2ndWon_1', 'Serve2ndWon_2', 'Surface']]

df2 = df[['ID', 'GRes_1', 'Age_CUR_1', 'IsBirthDay_CUR_1', 'DaysFromLast_CUR_1', 'TotalPointsWon_1', 'Serve1stWon_1', 'Serve2ndWon_1', 'Surface']]

df3 = df[['ID', 'GRes_2', 'Age_CUR_2', 'IsBirthDay_CUR_2', 'DaysFromLast_CUR_2', 'TotalPointsWon_2', 'Serve1stWon_2', 'Serve2ndWon_2', 'Surface']]



def isCurOlder(ages):

    if ages[0] > ages[1]:

        return 1

    else:

        return 0



df2 = df2.assign(isOlder=pd.Series(map(isCurOlder, df1[['Age_CUR_1', 'Age_CUR_2']].to_numpy())))

df3 = df3.assign(isOlder=pd.Series(map(isCurOlder, df1[['Age_CUR_2', 'Age_CUR_1']].to_numpy())))



# création d'un jeu de données fusionné

df3 = df3.rename(columns={"GRes_2": "GRes_1", "Age_CUR_2": "Age_CUR_1", "IsBirthDay_CUR_2": "IsBirthDay_CUR_1", "DaysFromLast_CUR_2": "DaysFromLast_CUR_1", "TotalPointsWon_2": "TotalPointsWon_1", "Serve1stWon_2": "Serve1stWon_1", "Serve2ndWon_2": "Serve2ndWon_1"})

dfmerged = df2.append(df3)



# suppression des NaN, non supporté par sklearn

dfmerged = dfmerged[dfmerged['ID'].notna()]

dfmerged = dfmerged[dfmerged['GRes_1'].notna()]

dfmerged = dfmerged[dfmerged['Age_CUR_1'].notna()]

dfmerged = dfmerged[dfmerged['IsBirthDay_CUR_1'].notna()]

dfmerged = dfmerged[dfmerged['DaysFromLast_CUR_1'].notna()]

dfmerged = dfmerged[dfmerged['TotalPointsWon_1'].notna()]

dfmerged = dfmerged[dfmerged['Serve1stWon_1'].notna()]

dfmerged = dfmerged[dfmerged['Serve2ndWon_1'].notna()]

dfmerged = dfmerged[dfmerged['Surface'].notna()]



# informations sur nos données

nRow, nCol = dfmerged.shape

print(f'There are {nRow} rows and {nCol} columns')
# 1) Vérifications d'usage

#print(df1.dtypes)

#print(df1.shape)

#print(df1.count())

#print(df1.describe())



# 2) Histogramme & Boxplot & Secteur pour l'âge des joueurs

print(dfmerged.Age_CUR_1.describe())

plt.figure()

dfmerged.Age_CUR_1.plot.hist()

plt.show()



plt.figure()

dfmerged.Age_CUR_1.plot(kind="box")

plt.show()



# 2) Boxplot & Secteur pour les anniversaires

print(dfmerged.IsBirthDay_CUR_1.describe())



print("Nombre de joueurs jouant le jour de leur anniversaire : " + str(list(dfmerged.IsBirthDay_CUR_1).count(1)))



plt.figure()

dfmerged.IsBirthDay_CUR_1.value_counts().plot.pie(figsize=[5,5])

plt.show()



# 2) Boxplot & Secteur pour la fréquence de jeu

print(dfmerged.DaysFromLast_CUR_1.describe())

plt.figure()

dfmerged.DaysFromLast_CUR_1.plot.hist()

plt.show()



plt.figure()

dfmerged.DaysFromLast_CUR_1.plot(kind="box")

plt.show()
dfmergedDayReduced = dfmerged[dfmerged['DaysFromLast_CUR_1']<=31]

print(dfmergedDayReduced.DaysFromLast_CUR_1.describe())

plt.figure()

dfmergedDayReduced.DaysFromLast_CUR_1.plot.hist()

plt.show()



plt.figure()

dfmergedDayReduced.DaysFromLast_CUR_1.plot(kind="box")

plt.show()
bins = range(12, 65)

dfmerged['discretizedAges'] = pd.cut(dfmerged['Age_CUR_1'], bins=bins, labels=bins[:-1])

print(dfmerged)
plt.scatter(dfmerged.Age_CUR_1, dfmerged.DaysFromLast_CUR_1, s=2, edgecolor = 'none')
plt.scatter(dfmerged.discretizedAges, dfmerged.DaysFromLast_CUR_1, s=10, edgecolor = 'none', marker = '.')
plt.scatter(bins[:-1], dfmerged.groupby('discretizedAges').DaysFromLast_CUR_1.mean())
dfmerged.DaysFromLast_CUR_1 = dfmerged.DaysFromLast_CUR_1.fillna(0.0)

dataCor = dfmerged[['Age_CUR_1', 'DaysFromLast_CUR_1']]

print(dataCor.corr(method='pearson'))
import scipy.stats



# Premier service gagnant

dftmp = dfmerged.groupby('IsBirthDay_CUR_1').Serve1stWon_1.mean()

print(dftmp)

# calcul du chi²

serve1stWonAverage = dfmerged.Serve1stWon_1.mean()

print(scipy.stats.chisquare(f_obs=dftmp.to_numpy(), f_exp=serve1stWonAverage))



# Second service gagnant

dftmp = dfmerged.groupby('IsBirthDay_CUR_1').Serve2ndWon_1.mean()

print(dftmp)

# calcul du chi²

serve2ndWonAverage = dfmerged.Serve2ndWon_1.mean()

print(scipy.stats.chisquare(f_obs=dftmp.to_numpy(), f_exp=serve2ndWonAverage))



# Total des points gagnés

dftmp = dfmerged.groupby('IsBirthDay_CUR_1').TotalPointsWon_1.mean()

print(dftmp)

# calcul du chi²

TotalPointsWonAverage = dfmerged.TotalPointsWon_1.mean()

print(scipy.stats.chisquare(f_obs=dftmp.to_numpy(), f_exp=TotalPointsWonAverage))
print(dfmerged)



boxplot = dfmerged.boxplot(column='discretizedAges', by='GRes_1')
dfmergedOnlyVictory = dfmerged[dfmerged['GRes_1']==1.0]

dfmergedOnlyDefeat = dfmerged[dfmerged['GRes_1']==0.0]



print(dfmergedOnlyVictory.Age_CUR_1.mean())

print(dfmergedOnlyDefeat.Age_CUR_1.mean())



print(dfmergedOnlyVictory.Age_CUR_1.median())

print(dfmergedOnlyDefeat.Age_CUR_1.median())
from statsmodels.graphics.mosaicplot import mosaic



print(dfmerged.groupby('isOlder').GRes_1.value_counts())

mosaic(dfmerged, ['isOlder', 'GRes_1'])
dfCluster1 = df[['Aces_A_1', 'Aces_L5_1', 'BreakPointsConvertedPCT_A_1', 'BreakPointsConvertedPCT_L5_1', 'BreakPointsTotal_A_1', 

                'BreakPointsTotal_L5_1', 'ReceivingPointsWonPCT_A_1', 'ReceivingPointsWonPCT_L5_1', 'Serve1stPCT_A_1', 'Serve1stWonPCT_A_1', 

                'Serve2ndWonPCT_L5_1', 'Serve2ndWonPCT_A_1', 'Age_CUR_1', 'TotalPointsWon_A_1', 'TotalPointsWon_L5_1']]



dfCluster2 = df[['Aces_A_2', 'Aces_L5_2', 'BreakPointsConvertedPCT_A_2', 'BreakPointsConvertedPCT_L5_2', 'BreakPointsTotal_A_2', 

                'BreakPointsTotal_L5_2', 'ReceivingPointsWonPCT_A_2', 'ReceivingPointsWonPCT_L5_2', 'Serve1stPCT_A_2', 'Serve1stWonPCT_A_2',

                'Serve2ndWonPCT_L5_2', 'Serve2ndWonPCT_A_2', 'Age_CUR_2', 'TotalPointsWon_A_2', 'TotalPointsWon_L5_2']]



dfCluster2 = dfCluster2.rename(columns={"Aces_A_2": "Aces_A_1", "Aces_L5_2": "Aces_L5_1", "BreakPointsConvertedPCT_A_2": "BreakPointsConvertedPCT_A_1",

                                        "BreakPointsConvertedPCT_L5_2": "BreakPointsConvertedPCT_L5_1", "BreakPointsTotal_A_2": "BreakPointsTotal_A_1",

                                        "BreakPointsTotal_L5_2": "BreakPointsTotal_L5_1", "ReceivingPointsWonPCT_A_2": "ReceivingPointsWonPCT_A_1",

                                        "ReceivingPointsWonPCT_L5_2": "ReceivingPointsWonPCT_L5_1", "Serve1stPCT_A_2": "Serve1stPCT_A_1", "Serve1stWonPCT_A_2": "Serve1stWonPCT_A_1",

                                        "Serve2ndWonPCT_L5_2": "Serve2ndWonPCT_L5_1", "Serve2ndWonPCT_A_2": "Serve2ndWonPCT_A_1", "Age_CUR_2": "Age_CUR_1", 

                                        "TotalPointsWon_A_2": "TotalPointsWon_A_1", "TotalPointsWon_L5_2": "TotalPointsWon_L5_1"})



dfCluster = dfCluster1.append(dfCluster2)



dfCluster = dfCluster[dfCluster['Aces_A_1'].notna()]

dfCluster = dfCluster[dfCluster['Aces_L5_1'].notna()]

dfCluster = dfCluster[dfCluster['BreakPointsConvertedPCT_A_1'].notna()]

dfCluster = dfCluster[dfCluster['BreakPointsConvertedPCT_L5_1'].notna()]

dfCluster = dfCluster[dfCluster['BreakPointsTotal_A_1'].notna()]

dfCluster = dfCluster[dfCluster['BreakPointsTotal_L5_1'].notna()]

dfCluster = dfCluster[dfCluster['ReceivingPointsWonPCT_A_1'].notna()]

dfCluster = dfCluster[dfCluster['ReceivingPointsWonPCT_L5_1'].notna()]

dfCluster = dfCluster[dfCluster['Serve1stPCT_A_1'].notna()]

dfCluster = dfCluster[dfCluster['Serve1stWonPCT_A_1'].notna()]

dfCluster = dfCluster[dfCluster['Serve2ndWonPCT_L5_1'].notna()]

dfCluster = dfCluster[dfCluster['Serve2ndWonPCT_A_1'].notna()]

dfCluster = dfCluster[dfCluster['Age_CUR_1'].notna()]

dfCluster = dfCluster[dfCluster['TotalPointsWon_A_1'].notna()]

dfCluster = dfCluster[dfCluster['TotalPointsWon_L5_1'].notna()]



# informations sur nos données

nRow, nCol = dfCluster.shape

print(f'There are {nRow} rows and {nCol} columns')
from scipy.cluster.vq import kmeans

from scipy.spatial.distance import cdist,pdist

from sklearn.cluster import KMeans

from sklearn.preprocessing import normalize

from sklearn.decomposition import PCA

from matplotlib import cm



dfKMeans = dfCluster.sample(n=500, random_state=24, replace=False)



# normalisation des données avant K Means

dfKMeans_scaled = normalize(dfKMeans)

dfKMeans_scaled = pd.DataFrame(dfKMeans_scaled, columns=dfKMeans.columns)



# réduction à 2 dimensions (PCA)

pca = PCA(n_components=2).fit(dfKMeans_scaled)

X = pca.transform(dfKMeans_scaled)



# clustering avec K Means

nb_clusters = 20

nums_clusters = range(1,nb_clusters+1)



kmeans_out = [kmeans(X,k) for k in nums_clusters]

centroids = [cent for (cent,var) in kmeans_out]

dist_kmeans = [cdist(X, cent, 'euclidean') for cent in centroids]

cIdx = [np.argmin(D,axis=1) for D in dist_kmeans]

dist = [np.min(D,axis=1) for D in dist_kmeans]



# calcul des sommes de carrés intra et inter classes

within_sum_squares = [sum(d**2) for d in dist]

sum_squares = sum(pdist(X)**2)/X.shape[0]

between_sum_squares = sum_squares - within_sum_squares



# représentation graphique

kIdx = 8     

clr = cm.rainbow( np.linspace(0,1,10) ).tolist()

mrk = 'os^p<dvh8>+x.'



fig = plt.figure()

ax = fig.add_subplot(111)

ax.plot(nums_clusters, between_sum_squares/sum_squares*100, 'b*-')

ax.plot(nums_clusters[kIdx], between_sum_squares[kIdx]/sum_squares*100, marker='o', markersize=12, 

    markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')

ax.set_ylim((0,100))

plt.grid(True)

plt.xlabel('Nombre de clusters')

plt.ylabel('Pourcentage de variance expliquée (%)')

plt.title('Méthode du coude - KMeans clustering')
dfKMeans = dfCluster



data_sample = dfKMeans.sample(n=100, random_state=24, replace=False)



# normalisation des données avant K Means

data_scaled = normalize(data_sample)

data_scaled = pd.DataFrame(data_scaled, columns=data_sample.columns)



# initialisation de variables

NUM_CLUSTERS = 9

NUM_ITER = 3

NUM_ATTEMPTS = 5



# exécution du premier K Means

km = KMeans(n_clusters=NUM_CLUSTERS, init='random', max_iter=1, n_init=1)

km.fit(data_scaled)



print('Inertie sur l\'échantillon:', km.inertia_)



final_cents = []

final_inert = []



# on boucle sur 5 tentatives pour trouver les meilleurs centroïdes

for sample in range(NUM_ATTEMPTS):

    km = KMeans(n_clusters= NUM_CLUSTERS, init='random', max_iter=1, n_init=1) 

    km.fit(data_scaled)

    inertia_start = km.inertia_

    inertia_end = 0

    cents = km.cluster_centers_

        

    for iter in range(NUM_ITER):

        km = KMeans(n_clusters = NUM_CLUSTERS, init=cents, max_iter=1, n_init=1)

        km.fit(data_scaled)

        inertia_end = km.inertia_

        cents = km.cluster_centers_



    final_cents.append(cents)

    final_inert.append(inertia_end)

    print('Difference entre inertie finale et initiale: ', inertia_start-inertia_end)



# on récupère les meilleurs centroïdes

best_cents = final_cents[final_inert.index(min(final_inert))]

print("Meilleurs centroïdes trouvés:", best_cents)

    

# on exécute K Means sur l'ensemble des données avec les meilleurs centroïdes à l'initialisation

data_fullScaled = normalize(dfKMeans)

data_fullScaled = pd.DataFrame(data_fullScaled, columns=dfKMeans.columns)

fullKMeans = KMeans(n_clusters=NUM_CLUSTERS, init=best_cents, max_iter=100, n_init=1, verbose=1)

fullKMeans.fit(data_fullScaled)
pca2D = PCA(2)

pca2D.fit(data_fullScaled)

projected = pca2D.fit_transform(data_fullScaled)

plt.scatter(projected[:, 0], projected[:, 1],

            c=fullKMeans.labels_, edgecolor='none', alpha=0.5,

            cmap=plt.cm.get_cmap('rainbow', 10))

plt.xlabel('F1')

plt.ylabel('F2')

plt.title('Visualisation 2D des 9 clusters')

plt.colorbar();
pca = PCA(3)

pca.fit(data_scaled)

pca_data = pd.DataFrame(pca.transform(data_scaled))



from matplotlib import colors as mcolors 

import math 



colors = list(zip(*sorted(( 

                    tuple(mcolors.rgb_to_hsv( 

                          mcolors.to_rgba(color)[:3])), name) 

                     for name, color in dict( 

                            mcolors.BASE_COLORS, **mcolors.CSS4_COLORS 

                                                      ).items())))[1] 

    

skips = math.floor(len(colors[5 : -5])/NUM_CLUSTERS) 

cluster_colors = colors[5 : -5 : skips] 



from mpl_toolkits.mplot3d import Axes3D 

   

fig = plt.figure(figsize=(15, 13)) 



ax = fig.add_subplot(111, projection = '3d') 

ax.scatter(pca_data[0], pca_data[1], pca_data[2],  

           c = list(map(lambda label : cluster_colors[label], 

                                            km.labels_))) 

   

str_labels = list(map(lambda label:'% s' % label, km.labels_)) 

   

list(map(lambda data1, data2, data3, str_label: 

        ax.text(data1, data2, data3, s = str_label, size = 12.5, 

        zorder = 20, color = 'k'), pca_data[0], pca_data[1], 

        pca_data[2], str_labels)) 

   

plt.show()
data_sample_agg = dfCluster.sample(n=100, random_state=24, replace=False)



data_sample_agg = data_sample_agg[data_sample_agg['BreakPointsConvertedPCT_A_1'] != 0]

data_sample_agg = data_sample_agg[data_sample_agg['Serve2ndWonPCT_A_1'] != 0]



import scipy.cluster.hierarchy as shc

plt.figure(figsize=(10, 7))  

plt.title("Dendrograms")  

dend = shc.dendrogram(shc.linkage(data_sample_agg, method='ward'))



from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=9, affinity='euclidean', linkage='ward')  

cluster.fit_predict(data_sample_agg)



plt.figure(figsize=(9, 7))  

plt.scatter(data_sample_agg['BreakPointsConvertedPCT_A_1'], data_sample_agg['Serve2ndWonPCT_A_1'], c=cluster.labels_) 
globalVariance = data_sample.var()

globalMean = data_sample.mean()

nCluster = len(data_sample.index)

rows = ['Aces_A_1','Aces_L5_1','BreakPointsConvertedPCT_A_1','BreakPointsConvertedPCT_L5_1','BreakPointsTotal_A_1','BreakPointsTotal_L5_1',

        'ReceivingPointsWonPCT_A_1','ReceivingPointsWonPCT_L5_1','Serve1stPCT_A_1','Serve1stWonPCT_A_1','Serve2ndWonPCT_L5_1','Serve2ndWonPCT_A_1','Age_CUR_1',

        'TotalPointsWon_A_1','TotalPointsWon_L5_1']



# fonction permettant de calculer la valeur v-test

def vTest(data):

    mean = data[0]

    var = data[1]

    globalMean = data[2]

    n = float(data[3])

    nCluster = float(data[4])



    result = (mean-globalMean) / (np.sqrt( ((nCluster - n)/(nCluster - 1)) * (var/n) ))

    return result



# cluster 1

group1_data = data_sample[km.labels_==0]

group1_mean = group1_data.mean()

group1_mean = group1_mean.to_frame()

group1 = group1_mean

symbols = ['mean']

group1.columns = symbols



group1['var'] = globalVariance

group1['globalMean'] = globalMean

group1['n'] = len(group1_data.index)

group1['nCluster'] = len(data_sample.index)

tab = pd.Series(map(vTest, group1[['mean', 'var', 'globalMean', 'n', 'nCluster']].to_numpy()))

group1['vtest_values'] = pd.Series(map(vTest, group1[['mean', 'var', 'globalMean', 'n', 'nCluster']].to_numpy()))

idx=0

for val in tab:

    index = rows[idx]

    group1.loc[index,'vtest_values'] = val

    idx += 1

group1 = group1.drop(['mean', 'var', 'globalMean', 'n', 'nCluster'], axis=1)



# cluster 2

group2_data = data_sample[km.labels_==1]

group2_mean = group2_data.mean()

group2_mean = group2_mean.to_frame()

group2 = group2_mean

symbols = ['mean']

group2.columns = symbols



group2['var'] = globalVariance

group2['globalMean'] = globalMean

group2['n'] = len(group2_data.index)

group2['nCluster'] = len(data_sample.index)



tab = pd.Series(map(vTest, group2[['mean', 'var', 'globalMean', 'n', 'nCluster']].to_numpy()))

group2['vtest_values'] = pd.Series(map(vTest, group2[['mean', 'var', 'globalMean', 'n', 'nCluster']].to_numpy()))

idx=0

for val in tab:

    index = rows[idx]

    group2.loc[index,'vtest_values'] = val

    idx += 1

group2 = group2.drop(['mean', 'var', 'globalMean', 'n', 'nCluster'], axis=1)



# cluster 3

group3_data = data_sample[km.labels_==2]

group3_mean = group3_data.mean()

group3_mean = group3_mean.to_frame()

group3 = group3_mean

symbols = ['mean']

group3.columns = symbols



group3['var'] = globalVariance

group3['globalMean'] = globalMean

group3['n'] = len(group3_data.index)

group3['nCluster'] = len(data_sample.index)



tab = pd.Series(map(vTest, group3[['mean', 'var', 'globalMean', 'n', 'nCluster']].to_numpy()))

group3['vtest_values'] = pd.Series(map(vTest, group3[['mean', 'var', 'globalMean', 'n', 'nCluster']].to_numpy()))

idx=0

for val in tab:

    index = rows[idx]

    group3.loc[index,'vtest_values'] = val

    idx += 1

group3 = group3.drop(['mean', 'var', 'globalMean', 'n', 'nCluster'], axis=1)



# cluster 4

group4_data = data_sample[km.labels_==3]

group4_mean = group4_data.mean()

group4_mean = group4_mean.to_frame()

group4 = group4_mean

symbols = ['mean']

group4.columns = symbols



group4['var'] = globalVariance

group4['globalMean'] = globalMean

group4['n'] = len(group4_data.index)

group4['nCluster'] = len(data_sample.index)



tab = pd.Series(map(vTest, group4[['mean', 'var', 'globalMean', 'n', 'nCluster']].to_numpy()))

group4['vtest_values'] = pd.Series(map(vTest, group4[['mean', 'var', 'globalMean', 'n', 'nCluster']].to_numpy()))

idx=0

for val in tab:

    index = rows[idx]

    group4.loc[index,'vtest_values'] = val

    idx += 1

group4 = group4.drop(['mean', 'var', 'globalMean', 'n', 'nCluster'], axis=1)



# cluster 5

group5_data = data_sample[km.labels_==4]

group5_mean = group5_data.mean()

group5_mean = group5_mean.to_frame()

group5 = group5_mean

symbols = ['mean']

group5.columns = symbols



group5['var'] = globalVariance

group5['globalMean'] = globalMean

group5['n'] = len(group5_data.index)

group5['nCluster'] = len(data_sample.index)



tab = pd.Series(map(vTest, group5[['mean', 'var', 'globalMean', 'n', 'nCluster']].to_numpy()))

group5['vtest_values'] = pd.Series(map(vTest, group5[['mean', 'var', 'globalMean', 'n', 'nCluster']].to_numpy()))

idx=0

for val in tab:

    index = rows[idx]

    group5.loc[index,'vtest_values'] = val

    idx += 1

group5 = group5.drop(['mean', 'var', 'globalMean', 'n', 'nCluster'], axis=1)



# cluster 6

group6_data = data_sample[km.labels_==5]

group6_mean = group6_data.mean()

group6_mean = group6_mean.to_frame()

group6 = group6_mean

symbols = ['mean']

group6.columns = symbols



group6['var'] = globalVariance

group6['globalMean'] = globalMean

group6['n'] = len(group6_data.index)

group6['nCluster'] = len(data_sample.index)



tab = pd.Series(map(vTest, group6[['mean', 'var', 'globalMean', 'n', 'nCluster']].to_numpy()))

group6['vtest_values'] = pd.Series(map(vTest, group6[['mean', 'var', 'globalMean', 'n', 'nCluster']].to_numpy()))

idx=0

for val in tab:

    index = rows[idx]

    group6.loc[index,'vtest_values'] = val

    idx += 1

group6 = group6.drop(['mean', 'var', 'globalMean', 'n', 'nCluster'], axis=1)



# cluster 7

group7_data = data_sample[km.labels_==6]

group7_mean = group7_data.mean()

group7_mean = group7_mean.to_frame()

group7 = group7_mean

symbols = ['mean']

group7.columns = symbols



group7['var'] = globalVariance

group7['globalMean'] = globalMean

group7['n'] = len(group7_data.index)

group7['nCluster'] = len(data_sample.index)



tab = pd.Series(map(vTest, group7[['mean', 'var', 'globalMean', 'n', 'nCluster']].to_numpy()))

group7['vtest_values'] = pd.Series(map(vTest, group7[['mean', 'var', 'globalMean', 'n', 'nCluster']].to_numpy()))

idx=0

for val in tab:

    index = rows[idx]

    group7.loc[index,'vtest_values'] = val

    idx += 1

group7 = group7.drop(['mean', 'var', 'globalMean', 'n', 'nCluster'], axis=1)



# cluster 8

group8_data = data_sample[km.labels_==7]

group8_mean = group8_data.mean()

group8_mean = group8_mean.to_frame()

group8 = group8_mean

symbols = ['mean']

group8.columns = symbols



group8['var'] = globalVariance

group8['globalMean'] = globalMean

group8['n'] = len(group8_data.index)

group8['nCluster'] = len(data_sample.index)



tab = pd.Series(map(vTest, group8[['mean', 'var', 'globalMean', 'n', 'nCluster']].to_numpy()))

group8['vtest_values'] = pd.Series(map(vTest, group8[['mean', 'var', 'globalMean', 'n', 'nCluster']].to_numpy()))

idx=0

for val in tab:

    index = rows[idx]

    group8.loc[index,'vtest_values'] = val

    idx += 1

group8 = group8.drop(['mean', 'var', 'globalMean', 'n', 'nCluster'], axis=1)



# cluster 9

group9_data = data_sample[km.labels_==8]

group9_mean = group9_data.mean()

group9_mean = group9_mean.to_frame()

group9 = group9_mean

symbols = ['mean']

group9.columns = symbols



group9['var'] = globalVariance

group9['globalMean'] = globalMean

group9['n'] = len(group9_data.index)

group9['nCluster'] = len(data_sample.index)



tab = pd.Series(map(vTest, group9[['mean', 'var', 'globalMean', 'n', 'nCluster']].to_numpy()))

group9['vtest_values'] = pd.Series(map(vTest, group9[['mean', 'var', 'globalMean', 'n', 'nCluster']].to_numpy()))

idx=0

for val in tab:

    index = rows[idx]

    group9.loc[index,'vtest_values'] = val

    idx += 1

group9 = group9.drop(['mean', 'var', 'globalMean', 'n', 'nCluster'], axis=1)





categories = ['Aces_A_1','Aces_L5_1','BreakPointsConvertedPCT_A_1','BreakPointsConvertedPCT_L5_1','BreakPointsTotal_A_1','BreakPointsTotal_L5_1',

              'ReceivingPointsWonPCT_A_1','ReceivingPointsWonPCT_L5_1','Serve1stPCT_A_1','Serve1stWonPCT_A_1','Serve2ndWonPCT_L5_1','Serve2ndWonPCT_A_1','Age_CUR_1',

              'TotalPointsWon_A_1','TotalPointsWon_L5_1']



# construction du radar chart



import plotly.graph_objects as go



fig = go.Figure()



fig.add_trace(go.Scatterpolar(

      r=group1.vtest_values.to_numpy(),

      theta=categories,

      fill='toself',

      name='Cluster 1'

))

fig.add_trace(go.Scatterpolar(

      r=group2.vtest_values.to_numpy(),

      theta=categories,

      fill='toself',

      name='Cluster 2'

))

fig.add_trace(go.Scatterpolar(

      r=group3.vtest_values.to_numpy(),

      theta=categories,

      fill='toself',

      name='Cluster 3'

))

fig.add_trace(go.Scatterpolar(

      r=group4.vtest_values.to_numpy(),

      theta=categories,

      fill='toself',

      name='Cluster 4'

))

fig.add_trace(go.Scatterpolar(

      r=group5.vtest_values.to_numpy(),

      theta=categories,

      fill='toself',

      name='Cluster 5'

))

fig.add_trace(go.Scatterpolar(

      r=group6.vtest_values.to_numpy(),

      theta=categories,

      fill='toself',

      name='Cluster 6'

))

fig.add_trace(go.Scatterpolar(

      r=group7.vtest_values.to_numpy(),

      theta=categories,

      fill='toself',

      name='Cluster 7'

))

fig.add_trace(go.Scatterpolar(

      r=group8.vtest_values.to_numpy(),

      theta=categories,

      fill='toself',

      name='Cluster 8'

))

fig.add_trace(go.Scatterpolar(

      r=group9.vtest_values.to_numpy(),

      theta=categories,

      fill='toself',

      name='Cluster 9'

))



fig.update_layout(

  polar=dict(

    radialaxis=dict(

      visible=True,

      range=[-4, 7]

    )),

  showlegend=False

)



fig.show()
from sklearn.tree import DecisionTreeClassifier, export_graphviz # Import Decision Tree Classifier

from sklearn.model_selection import train_test_split # Import train_test_split function

from sklearn.preprocessing import OneHotEncoder #Import scikit-learn metrics module for accuracy calculation

from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation



# suppression des NaN, non supporté par sklearn

dfmerged = dfmerged[dfmerged['discretizedAges'].notna()]

dfmerged = dfmerged[dfmerged['DaysFromLast_CUR_1'].notna()]

dfmerged = dfmerged[dfmerged['GRes_1'].notna()]

dfmerged = dfmerged[dfmerged['Surface'].notna()]

dfmerged = dfmerged[dfmerged['TotalPointsWon_1'].notna()]



dfmerged.Surface = pd.Series(map(int, dfmerged[['Surface']].to_numpy()))



# normalisation de certaines valeurs

def func(arg):

    return (arg == 1.0)

dfTree = dfmerged.assign(IsBirthDay=pd.Series(map(func, dfmerged[['IsBirthDay_CUR_1']].to_numpy())))

dfTree = dfTree.assign(DaysFromLast=pd.Series(map(int, dfmerged[['DaysFromLast_CUR_1']].to_numpy())))

dfTree = dfTree.assign(GRes=pd.Series(map(func, dfmerged.GRes_1.to_numpy())))

dfTree = dfTree.assign(TotalPointsWon=pd.Series(map(int, dfmerged.TotalPointsWon_1.to_numpy())))



# categorisation de la variable surface

new_cols = pd.get_dummies(dfmerged.Surface.to_numpy(), prefix="Surface")

dfTree['Surface_1'] = new_cols[['Surface_1']]

dfTree['Surface_2'] = new_cols[['Surface_2']]

dfTree['Surface_3'] = new_cols[['Surface_3']]

dfTree['Surface_4'] = new_cols[['Surface_4']]

dfTree['Surface_5'] = new_cols[['Surface_5']]



# Création des jeux de données

feature_cols = ['discretizedAges', 'DaysFromLast', 'IsBirthDay', 'TotalPointsWon']

X = dfTree[feature_cols] # Features

y = dfTree.GRes # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)



# Génération de l'arbre

clf = DecisionTreeClassifier(max_depth = 3, criterion="gini", splitter="best")

clf = clf.fit(X_train,y_train) # entrainement

y_pred = clf.predict(X_test) # Test de prédiction



print("Accuracy : ", metrics.accuracy_score(y_test, y_pred))



# visualisation

export_graphviz(clf,

                 out_file='tree.dot',

                 max_depth = 3,

                 rounded = True,

                 impurity=False,

                 class_names=['loose', 'win'],

                 feature_names = X_test.columns.values,

                 filled=True)



# Convert to png

from subprocess import call

call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])



# Display in python

plt.figure(figsize = (14, 16))

plt.imshow(plt.imread('tree.png'))

plt.axis('off')

plt.show()
from sklearn.tree import DecisionTreeRegressor, export_graphviz # Import Decision Tree Regressor



# suppression des NaN, non supporté par sklearn

dfmerged = dfmerged[dfmerged['DaysFromLast_CUR_1'].notna()]

dfmerged = dfmerged[dfmerged['Serve1stWon_1'].notna()]

dfmerged = dfmerged[dfmerged['Serve2ndWon_1'].notna()]

dfmerged = dfmerged[dfmerged['TotalPointsWon_1'].notna()]



# normalisation des valeurs

dfTree = dfmerged.assign(DaysFromLast=pd.Series(map(int, dfmerged[['DaysFromLast_CUR_1']].to_numpy())))

dfTree = dfTree.assign(Serve1stWon=pd.Series(map(int, dfmerged[['Serve1stWon_1']].to_numpy())))

dfTree = dfTree.assign(Serve2ndWon=pd.Series(map(int, dfmerged[['Serve2ndWon_1']].to_numpy())))

dfTree = dfTree.assign(TotalPointsWon=pd.Series(map(int, dfmerged[['TotalPointsWon_1']].to_numpy())))

def func(arg):

    return (arg == 1.0)

dfTree = dfTree.assign(GRes=pd.Series(map(func, dfmerged.GRes_1.to_numpy())))



# categorisation de la variable surface

new_cols = pd.get_dummies(dfmerged.Surface.to_numpy(), prefix="Surface")

dfTree['Surface_1'] = new_cols[['Surface_1']]

dfTree['Surface_2'] = new_cols[['Surface_2']]

dfTree['Surface_3'] = new_cols[['Surface_3']]

dfTree['Surface_4'] = new_cols[['Surface_4']]

dfTree['Surface_5'] = new_cols[['Surface_5']]



# création des jeux de données

feature_cols = ['GRes_1', 'DaysFromLast_CUR_1', 'Serve1stWon_1', 'Serve2ndWon_1', 'TotalPointsWon_1', 'Surface_1', 'Surface_2', 'Surface_3', 'Surface_4', 'Surface_5']

X = dfTree[feature_cols] # Features

y = dfTree.Age_CUR_1 # Target variable



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

best_depth = 0

best_depth_score = 0

for i in range(1,10):

    clf = DecisionTreeRegressor(max_depth=i)

    clf = clf.fit(X_train, y_train)

    score = clf.score(X_test, y_test)

    print("score for max depth ", i, ' : ', score)

    if score > best_depth_score:

        best_depth = i

        best_depth_score = score



clf = DecisionTreeRegressor(max_depth=best_depth)

clf = clf.fit(X_train, y_train)



# visualisation

export_graphviz(clf,

                 out_file='tree.dot',

                 max_depth = 3,

                 rounded = True,

                 impurity=False,

                 feature_names = X_test.columns.values,

                 filled=True)



# conversion en png

from subprocess import call

call(['dot', '-Tpng', 'tree.dot', '-o', 'regressionTree.png', '-Gdpi=600'])



# affichage

plt.figure(figsize = (22, 20))

plt.imshow(plt.imread('regressionTree.png'))

plt.axis('off')

plt.show()
pca2D2 = PCA(2)

pca2D2.fit(data_scaled) 

X = pca2D2.fit_transform(data_scaled)



from sklearn.cluster import DBSCAN

outlier_detection = DBSCAN(

    eps = 0.03,

    metric="euclidean",

    min_samples = 3,

    n_jobs = -1)

clusters = outlier_detection.fit_predict(X)



# les outliers sont les données ne faisant pas partit du premier cluster

nb_outliers = clusters.size - clusters.tolist().count(0)

outliers_fraction = nb_outliers / clusters.size



print(clusters)

# visualisation des clusters

plt.scatter(X[:, 0], X[:, 1],

           c=clusters,

           cmap=cm.get_cmap('Accent'))

plt.xlabel('F1')

plt.ylabel('F2')

plt.title('Visualisation des outliers')

plt.colorbar();
from scipy import stats

import matplotlib.pyplot as plt

import matplotlib.font_manager



from sklearn import svm

from sklearn.covariance import EllipticEnvelope

from sklearn.ensemble import IsolationForest



# Parametres

n_samples = clusters.size

clusters_separation = [0, 1, 2]

rng = np.random.RandomState(42)



# define two outlier detection tools to be compared

classifiers = {

    "One-Class SVM": svm.OneClassSVM(nu=0.95 * outliers_fraction + 0.05,

                                     kernel="rbf", gamma=0.1),

    "Robust covariance": EllipticEnvelope(contamination=outliers_fraction),

    "Isolation Forest": IsolationForest(max_samples=n_samples,

                                        contamination=outliers_fraction,

                                        random_state=rng)}



# Compare given classifiers under given settings

xx, yy = np.meshgrid(np.linspace(-0.25, 0.25, 500), np.linspace(-0.25, 0.25, 500))

n_inliers = int((1. - outliers_fraction) * n_samples)

n_outliers = int(outliers_fraction * n_samples)

ground_truth = np.ones(n_samples, dtype=int)

ground_truth[-n_outliers:] = -1



# Fit the model

plt.figure(figsize=(10.8, 3.6))

for i, (clf_name, clf) in enumerate(classifiers.items()):

    # fit the data and tag outliers

    clf.fit(X)

    scores_pred = clf.decision_function(X)

    threshold = stats.scoreatpercentile(scores_pred,

                                        100 * outliers_fraction)

    y_pred = clf.predict(X)

    n_errors = (y_pred != ground_truth).sum()

    # plot the levels lines and the points

    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)

    subplot = plt.subplot(1, 3, i + 1)

    subplot.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),

                     cmap=plt.cm.Blues_r)

    a = subplot.contour(xx, yy, Z, levels=[threshold],

                        linewidths=2, colors='red')

    subplot.contourf(xx, yy, Z, levels=[threshold, Z.max()],

                     colors='orange')

    b = subplot.scatter(X[:-n_outliers, 0], X[:-n_outliers, 1], c='white')

    c = subplot.scatter(X[-n_outliers:, 0], X[-n_outliers:, 1], c='black')

    subplot.axis('tight')

    subplot.legend(

        [a.collections[0], b, c],

        ['learned decision function', 'true inliers', 'true outliers'],

        prop=matplotlib.font_manager.FontProperties(size=11),

        loc='lower right')

    subplot.set_title("%d. %s (errors: %d)" % (i + 1, clf_name, n_errors))

    subplot.set_xlim((-0.25, 0.25))

    subplot.set_ylim((-0.25, 0.25))



plt.show()