import os

import numpy as np

import pandas as pd

import seaborn as sns

from tqdm import tqdm_notebook



%matplotlib inline

from matplotlib import pyplot as plt

plt.style.use(['seaborn-darkgrid'])

plt.rcParams['figure.figsize'] = (12, 9)

plt.rcParams['font.family'] = 'DejaVu Sans'



from sklearn import metrics

from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering

from sklearn.decomposition import PCA

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.svm import LinearSVC





#yo 

from sklearn import decomposition





RANDOM_STATE = 17
# change this if necessary

PATH_TO_SAMSUNG_DATA = "../input"
X_train = np.loadtxt(os.path.join(PATH_TO_SAMSUNG_DATA, "samsung_train.txt"))

y_train = np.loadtxt(os.path.join(PATH_TO_SAMSUNG_DATA,

                                  "samsung_train_labels.txt")).astype(int)



X_test = np.loadtxt(os.path.join(PATH_TO_SAMSUNG_DATA, "samsung_test.txt"))

y_test = np.loadtxt(os.path.join(PATH_TO_SAMSUNG_DATA,

                                  "samsung_test_labels.txt")).astype(int)
# Checking dimensions

assert(X_train.shape == (7352, 561) and y_train.shape == (7352,))

assert(X_test.shape == (2947, 561) and y_test.shape == (2947,))
# Your code here

# unsupervised no tiene test lol 



X = np.concatenate((X_train, X_test))

y = np.concatenate((y_train, y_test))
np.unique(y)
n_classes = np.unique(y).size
# Your code here



scaler = StandardScaler()

X = scaler.fit_transform(X)

# Your code here

pca = decomposition.PCA(random_state = RANDOM_STATE)

X_pca = pca.fit_transform(X)



pca.explained_variance_ratio_



# Your code here



n = 0

acum = 0



while acum <= 0.9:

    acum += pca.explained_variance_ratio_[n]

    n+=1



print(n, acum)

np.round(pca.explained_variance_ratio_[0] * 100) 
# Your code here

plt.scatter(X_pca[:,0], X_pca[:,1], c=y, s=20, cmap='viridis');
# Your code here



km = KMeans(n_clusters= n_classes, n_init=100, random_state=RANDOM_STATE)

km.fit(X_pca)





# Your code here



clusters = km.labels_



plt.scatter(X_pca[:,0] , X_pca[:,1], c=clusters, s=20, cmap='viridis')
tab = pd.crosstab(y, clusters, margins=True)

tab.index = ['walking', 'going up the stairs',

            'going down the stairs', 'sitting', 'standing', 'laying', 'all']

tab.columns = ['cluster' + str(i + 1) for i in range(6)] + ['all']

tab
ctab = np.transpose(tab)

cidx =  ['walking', 'going up the stairs',

            'going down the stairs', 'sitting', 'standing', 'laying']

re = []



bestN = 0

bestIdx = ""

for ii in cidx:

    maximun = np.max(ctab[ii][0: len(cidx)])

    allCount = ctab[ii]["all"]

    ratioBest = maximun/allCount 

    print(ii +"  "+ str(ratioBest))

    #print(maximun)

    #print(allCount)

    #print(ratioBest)

    

    

    if ratioBest > bestN:

        bestN = ratioBest

        bestIdx = ii



print()

print("best is " + bestIdx)



    

# en el tuto no lo llaman elbow method pero ok XD

# optimizamos el J(C) copypasteando lo de ellos





inertia = []

for k in tqdm_notebook(range(1, n_classes + 1)):

    kmeans = KMeans(n_clusters=k, n_init = 100, random_state=RANDOM_STATE).fit(X)

    inertia.append(np.sqrt(kmeans.inertia_))





plt.plot(range(1, n_classes + 1), inertia, marker='s');

plt.xlabel('$k$')

plt.ylabel('$J(C_k)$');
ag = AgglomerativeClustering(n_clusters=n_classes, 

                             linkage='ward').fit(X_pca)
# Your code here



km = KMeans(n_clusters=2, n_init = 100, random_state=RANDOM_STATE).fit(X)



ari_ag = metrics.adjusted_rand_score(y, ag.labels_)

ari_km = metrics.adjusted_rand_score(y, km.labels_)





print ('Adjusted Rand Index agglomerative ' + str(ari_ag))

print ('Adjusted Rand Index kmeans ' + str(ari_km))
# # Your code here

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test) 
svc = LinearSVC(random_state=RANDOM_STATE)

svc_params = {'C': [0.001, 0.01, 0.1, 1, 10]}
# %%time

# # Your code here

best_svc = GridSearchCV(svc, svc_params, cv=3)

best_svc.fit(X_train_scaled, y_train)
best_svc.best_params_, best_svc.best_score_
y_predicted = best_svc.predict(X_test_scaled)
tab = pd.crosstab(y_test, y_predicted, margins=True)

tab.index = ['walking', 'climbing up the stairs',

             'going down the stairs', 'sitting', 'standing', 'laying', 'all']

tab.columns = ['walking', 'climbing up the stairs',

             'going down the stairs', 'sitting', 'standing', 'laying', 'all']

tab
from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import accuracy_score

 

npc_y_predicted = y_predicted.copy()

precision = precision_score(y_test, y_predicted, average=None)

recall = recall_score(y_test, y_predicted, average=None)



# ['walking', 'climbing up the stairs','going down the stairs', 'sitting', 'standing', 'laying']

print(precision)

print(recall)
# Your code here

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test) 



pca = decomposition.PCA(n_components = 65, random_state = RANDOM_STATE)



X_train_pca = pca.fit_transform(X_train_scaled)

X_test_pca = pca.transform(X_test_scaled)
svc = LinearSVC(random_state = RANDOM_STATE)

svc_params = {'C': [0.001, 0.01, 0.1, 1, 10]}

clf = GridSearchCV(svc, svc_params, cv = 3)

clf.fit(X_train_pca, y_train)

print(clf.best_params_)
# agarramos el mejor C

best_svc = LinearSVC(C = 1, random_state = RANDOM_STATE)

best_svc.fit(X_train_pca, y_train)

y_predicted = best_svc.predict(X_test_pca)





# copy paste de arriba no shame

# tab2 = pd.crosstab(y_test, y_predicted, margins=True)

# tab2.index = ['walking', 'climbing up the stairs',

#               'going down the stairs', 'sitting', 'standing', 'laying', 'all']

# tab2.columns = ['walking', 'climbing up the stairs',

#              'going down the stairs', 'sitting', 'standing', 'laying', 'all']

# tab2


# What is the difference between the best quality (accuracy) for cross-validation in the case of all 561 initial 

# characteristics and in the second case, when the principal component method was applied? Round to the nearest percent.

print(npc_y_predicted - y_predicted)

print((accuracy_score(y_test, npc_y_predicted) *100)) # non pca

print((accuracy_score(y_test, y_predicted) *100)) # pca



print(np.round(accuracy_score(y_test, npc_y_predicted) *100)) # non pca

print(np.round(accuracy_score(y_test, y_predicted) *100)) # pca