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
X = np.vstack([X_train, X_test])

y = np.hstack([y_train, y_test])
np.unique(y)
n_classes = np.unique(y).size
ss = StandardScaler()

tr = ss.fit_transform(X, y)
pca = PCA(n_components=0.9, svd_solver='full', random_state=RANDOM_STATE).fit(tr)

X_pca = pca.transform(tr) 
X_pca.shape[1]
round(pca.explained_variance_ratio_[0] * 100)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, s=20, cmap='viridis');
k_means = KMeans(n_clusters=n_classes, n_init=100, random_state=RANDOM_STATE)

k_means.fit(X_pca)
len(k_means.labels_)
# Your code here

cluster_labels = k_means.labels_

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, s=20, cmap='viridis');
tab = pd.crosstab(y, cluster_labels, margins=True)

tab.index = ['walking', 'going up the stairs',

            'going down the stairs', 'sitting', 'standing', 'laying', 'all']

tab.columns = ['cluster' + str(i + 1) for i in range(6)] + ['all']

tab
inertia = []

for k in tqdm_notebook(range(1, n_classes + 1)):

    kmeans = KMeans(n_clusters=k, random_state=1).fit(X)

    inertia.append(np.sqrt(kmeans.inertia_))
plt.plot(range(1, n_classes + 1), inertia, marker='s');

plt.xlabel('$k$')

plt.ylabel('$J(C_k)$');
ag = AgglomerativeClustering(n_clusters=n_classes, linkage='ward').fit(X_pca)
from sklearn.metrics import adjusted_rand_score

print(adjusted_rand_score(y, ag.labels_))

print(adjusted_rand_score(y, k_means.labels_))
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
svc = LinearSVC(random_state=RANDOM_STATE)

svc_params = {'C': [0.001, 0.01, 0.1, 1, 10]}
%%time

best_svc = GridSearchCV(svc, svc_params, cv=3)

best_svc.fit(X_train_scaled, y_train)
best_score_svc = best_svc.best_score_

best_svc.best_params_, best_score_svc
y_predicted = best_svc.predict(X_test_scaled)
tab = pd.crosstab(y_test, y_predicted, margins=True)

tab.index = ['walking', 'climbing up the stairs',

             'going down the stairs', 'sitting', 'standing', 'laying', 'all']

tab.columns = ['walking', 'climbing up the stairs',

             'going down the stairs', 'sitting', 'standing', 'laying', 'all']

tab
print('precision:')

for t in tab.drop('all', 1):

    print(t + ":", tab[t][t] / tab[t][-1])
print('recall:')

for t in tab.drop('all', 1):

    print(t + ":", tab[t][t] / tab['all'][t])
# Your code here

pca = PCA(n_components=0.9, random_state=RANDOM_STATE)

X_train_pca = pca.fit_transform(X_train_scaled)



best_svc.fit(X_train_pca, y_train)



best_score_svc_pca = best_svc.best_score_

best_svc.best_params_, best_score_svc_pca
print("difference:", abs(best_score_svc_pca - best_score_svc))