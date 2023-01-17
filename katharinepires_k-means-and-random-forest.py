import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.cm as cm #Color Map, make color map

from sklearn.preprocessing import MinMaxScaler

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA 

from sklearn.metrics import silhouette_samples, silhouette_score #Know the ideal number of clusters

from sklearn.ensemble import RandomForestClassifier 

from collections import Counter
data = pd.read_csv('../input/fifa19/data.csv')
data.shape
data.head()
#create indexes: will start at zero until the number of columns



for i, col in enumerate(data.columns):

    print(i, col)
cols = [21, 26, 27]

cols += range(54, 83)
data = data.iloc[:, cols]
data.head()
#missing values:

data.isna().sum(axis = 0)
len(data) - len(data.dropna())
(60 / len(data)) * 100
data = data.dropna()
data.isna().sum(axis = 0)
def hist_boxplot(feature):

  fig, ax = plt.subplots(1, 2)

  ax[0].hist(feature)

  ax[1].boxplot(feature)
data_stats = data.describe()

data_stats
hist_boxplot(data_stats.loc['min'])
hist_boxplot(data_stats.loc['mean'])
hist_boxplot(data_stats.loc['max'])
data['Height'].head()
data['Height'] = data['Height'].str.split('\'')

data['Height'] = [30.48 * int(elem[0]) + 2.54 * int(elem[1]) for elem in data['Height']]

hist_boxplot(data['Height'])
data['Weight'].head()
data['Weight'] = data['Weight'].str.split('l')

data['Weight'] = [int(elem[0]) * 0.453 for elem in data['Weight']]

hist_boxplot(data['Weight'])
position = np.array(data['Position'])

np.unique(position, return_counts = True)
data = data.drop(['Position'], axis = 1)

data.head()
scaler = MinMaxScaler()

train = scaler.fit_transform(data)
train
wcss = []

K = range(1, 12) #we have 11 players in a game

for k in K:

    KM = KMeans(n_clusters = k)

    KM = KM.fit(train)

    wcss.append(KM.inertia_)
plt.plot(K, wcss, 'bx-')

plt.xlabel('k')

plt.ylabel('WCSS')

plt.title('Elbow Method');
pca = PCA(n_components = 2)

data_pca = pca.fit_transform(train)

pca.explained_variance_ratio_

exp_var = [round(i, 1) for i in pca.explained_variance_ratio_ * 100]
range_n_clusters = range(2, 12)

for n_clusters in range_n_clusters:

  fig, (ax1, ax2) = plt.subplots(1, 2)

  fig.set_size_inches(18, 7)



  ax1.set_xlim([-0.1, 1])

  ax1.set_ylim([0, len(train) + (n_clusters + 1) * 10])



  clusterer = KMeans(n_clusters=n_clusters, random_state=10)

  cluster_labels = clusterer.fit_predict(train)

  #print(cluster_labels)

  #print(np.unique(cluster_labels))



  silhouette_avg = silhouette_score(train, cluster_labels)

  print("For n_clusters = ", n_clusters, " Average score: ", silhouette_avg)



  sample_silhouette_values = silhouette_samples(train, cluster_labels)

  #print(sample_silhouette_values)

  #print(len(sample_silhouette_values))



  y_lower = 10

  for i in range(n_clusters):

    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

    ith_cluster_silhouette_values.sort()

    #print(ith_cluster_silhouette_values.shape)



    size_cluster_i = ith_cluster_silhouette_values.shape[0]



    y_upper = y_lower + size_cluster_i

    #print(y_upper)

    

    ax1.fill_betweenx(np.arange(y_lower, y_upper), ith_cluster_silhouette_values)



    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))



    y_lower = y_upper + 10



  ax1.set_title("The silhouette plot for the various clusters")

  ax1.set_xlabel("The silhouette coefficient values")

  ax1.set_ylabel("Cluster label")   



  ax1.axvline(x = silhouette_avg, color = "red", linestyle = "--")



  ax1.set_yticks([])

  ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])



  colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)

  ax2.scatter(data_pca[:, 0], data_pca[:, 1], marker='.', s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')



  centers = clusterer.cluster_centers_

  centers = pca.transform(centers)

  ax2.scatter(centers[:, 0], centers[:,1], marker='o', c='white', alpha=1, s=200, edgecolor='k')



  for i, c in enumerate(centers):

    ax2.scatter(c[0], c[1], marker='$%d$' % i, s=50, edgecolor='k')



  ax2.set_title("The visualization of the clustered data")

  ax2.set_xlabel('PC1 (' + str(exp_var[0]) + '% variance explained')

  ax2.set_ylabel('PC2 (' + str(exp_var[1]) + '% variance explained')



  plt.suptitle(("Silhouette analysis for Kmeans clustering on sample data with n_clusters = %d" % n_clusters),

               fontsize=14, fontweight='bold')
km = KMeans(n_clusters = 4, n_init = 100, random_state = 0)

km.fit(train)
print(km.cluster_centers_) #it shows the position of the centers
position[2]
group = km.labels_ #will return in these records to which group it belongs (km.labels_)

comp = []

for i in range(0, len(position)):

    ele = tuple((position[i], group[i]))

    comp.append(ele)
comp[0:4]
#how many records per group

count = Counter(comp)

count
comp = pd.DataFrame({'Position': [i[0] for i in list(count.keys())],

                     'Group': [i[1] for i in list(count.keys())],

                     'Numbers': list(count.values())})
comp.head()
comp.shape
#ordering groups according to their position:

comp = comp.sort_values(['Position', 'Group'])

comp.head()
#analyzing in percentage

comp_per = pd.DataFrame()

pos = comp['Position'].unique()

pos
for p in pos:

    comp_p = comp[comp['Position'] == p] 

    sum_N = sum(comp_p['Numbers'])

    comp_p['Numbers'] = comp_p['Numbers'] / sum_N

    comp_per = comp_per.append(comp_p)

comp_per = comp_per.sort_values(['Group', 'Numbers', 'Position'])
comp_per.head()
comp_per.tail()
#let's create a bar graph with the percentage of frequencies for each group:



comp_barplot = pd.DataFrame({'Position': sum([[ele] * 4 for ele in np.unique(position)], []),

                             'Group': sum([['0', '1', '2', '3'] * len(np.unique(position))], []),

                             'Numbers': [0] * 4 * len(np.unique(position))})
comp_barplot.head(10)
#we will now find the value of 'Numbers':



for row in range(0, len(comp_barplot)):

    pos = comp_barplot.iloc[row, 0]

    gro = int(comp_barplot.iloc[row, 1])

    reg = comp_per.loc[(comp_per['Position'] == pos) & (comp_per['Group'] == gro), :]

if len(reg) > 0:

    comp_barplot.iloc[row, 2] = reg['Numbers'].values
comp_barplot.head()
comp_barplot.tail()
counter1 = Counter(comp_per[comp_per['Numbers'] >= 0.5]['Group'])

counter1
#for security and warranty, we will leave this ordering code

counter1 = dict(sorted(counter1.items(), key = lambda x: x[0]))

counter1
x = [str(ele) for ele in list(counter1.keys())]

x
p1 = plt.bar(x, counter1.values())

counter2 = Counter(comp_per[comp_per['Numbers'] < 0.5]['Group'])

counter2 = dict(sorted(counter2.items(), key = lambda x: x[0]))

x = [str(ele) for ele in list(counter2.keys())]



p2 = plt.bar(x, counter2.values(), bottom=list(counter1.values()))

plt.title('Number of positions designated to each group')

plt.xlabel('Group')

plt.ylabel('Number of positions')

plt.legend((p1[0], p2[0]), ('Proportion >= 0.5', 'Proportion < 0.5'))
#most frequent positions for each group:

for i in range(4):

    g = comp_per[(comp_per['Group'] == i) & (comp_per['Numbers'] >= 0.5)][['Position', 'Numbers']]

    g = g.sort_values(by = 'Numbers')

    plt.barh(g['Position'], g['Numbers'])

    plt.axvline(0.5, color = 'r', linestyle = '--')

    plt.title('Positions best associated with Group ' + str(i))

    plt.show()
rf = RandomForestClassifier()

rf.fit(train, group)
importances = rf.feature_importances_

importances
features = data.columns

imp = pd.DataFrame({'Features': features, 'Importance': importances})

imp.head()
#the 5 most important features



imp = imp.sort_values(by = 'Importance', ascending = False)

imp.head()
#the 5 least important features



imp.tail()
#accumulated sum



imp['Sum Importance'] = imp['Importance'].cumsum()

imp = imp.sort_values(by = 'Importance')

imp.head()
imp.tail()
plt.figure(figsize=(8,8))

plt.barh(imp['Features'], imp['Importance'])

l1 = plt.axhline(len(imp) - (len(imp['Features'][imp['Sum Importance'] < 0.50]) + 1.5), linestyle='-.', color = 'r')

l2 = plt.axhline(len(imp) - (len(imp['Features'][imp['Sum Importance'] < 0.90]) + 1.5), linestyle='--', color = 'r')

l3 = plt.axhline(len(imp) - (len(imp['Features'][imp['Sum Importance'] < 0.99]) + 1.5), linestyle='-', color = 'r')

plt.legend(title = 'Cut-offs of acumulated importance', handles=(l1, l2, l3), labels = ('50%', '90%', '99%'))

plt.title('Feature importance in group assignment')
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix



X_train, X_test, Y_train, Y_test = train_test_split(train, group, test_size = 0.2)
import seaborn as sns



sns.countplot(Y_train)
rf = RandomForestClassifier()

rf.fit(X_train, Y_train)



pred = rf.predict(X_test)

accuracy_score(pred, Y_test)
cm = confusion_matrix(pred, Y_test)

cm
from yellowbrick.classifier import ConfusionMatrix

confusion_matrix = ConfusionMatrix(rf)

confusion_matrix.fit(X_train, Y_train)

confusion_matrix.score(X_test, Y_test)

confusion_matrix.show();
new = X_test[0]

new
new = new.reshape(1, -1)

#will return which group it belongs to

rf.predict(new) 
#using probability of belonging to each group

rf.predict_proba(new)