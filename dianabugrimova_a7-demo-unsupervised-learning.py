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

from sklearn import decomposition

from sklearn import cluster



RANDOM_STATE = 17
# change this if necessary

PATH_TO_SAMSUNG_DATA = "../input"
X_train = np.loadtxt(os.path.join(PATH_TO_SAMSUNG_DATA, "samsung_train.txt"))

y_train = np.loadtxt(os.path.join(PATH_TO_SAMSUNG_DATA,

                                  "samsung_train_labels.txt")).astype(int)



X_test = np.loadtxt(os.path.join(PATH_TO_SAMSUNG_DATA, "samsung_test.txt"))

y_test = np.loadtxt(os.path.join(PATH_TO_SAMSUNG_DATA,

                                  "samsung_test_labels.txt")).astype(int)
assert(X_train.shape == (7352, 561) and y_train.shape == (7352,))

assert(X_test.shape == (2947, 561) and y_test.shape == (2947,))
X_train = pd.DataFrame(X_train)

X_test = pd.DataFrame(X_test)

y_train = pd.DataFrame(y_train)

y_test = pd.DataFrame(y_test)
X=pd.concat([X_train, X_test])

X.shape
y=pd.concat([y_train, y_test])

y.shape
np.unique(y)
n_classes = np.unique(y).size
scaler = StandardScaler()

X_scaled=scaler.fit_transform(X)
pca = decomposition.PCA(random_state=RANDOM_STATE).fit(X_scaled)



plt.figure(figsize=(10,7))

plt.plot(np.cumsum(pca.explained_variance_ratio_), color='k', lw=2)

plt.xlabel('Number of components')

plt.ylabel('Total explained variance')

plt.xlim(0, 75)

plt.yticks(np.arange(0, 1.1, 0.1))

plt.axvline(21, c='b')

plt.axhline(0.9, c='r')

plt.show();
pca = decomposition.PCA(n_components=66).fit(X_scaled)
x_pca = pca.transform(X_scaled)
X.head()
for i, component in enumerate(pca.components_):

    print("{} component: {}% of initial variance".format(i + 1, 

          round(100 * pca.explained_variance_ratio_[i], 2)))
plt.figure(figsize=(10,10))

plt.scatter(x_pca[:,0],x_pca[:,1], c=y[0], s=20, cmap='viridis');

plt.xlabel('First principal component')

plt.ylabel('Second Principal Component')

plt.legend(loc='best')
kmeans=KMeans(n_clusters=n_classes, n_init=100, random_state = RANDOM_STATE, n_jobs=1).fit(x_pca)
y_pred_km=kmeans.fit_predict(y)
cluster_labels=kmeans.labels_
plt.figure(figsize=(10,10))

plt.scatter(x_pca[:,0],x_pca[:,1], c=cluster_labels, s=20, cmap='viridis');
tab = pd.crosstab(y[0], cluster_labels, margins=True)

tab.index = ['walking', 'going up the stairs',

            'going down the stairs', 'sitting', 'standing', 'laying', 'all']

tab.columns = ['cluster' + str(i + 1) for i in range(6)] + ['all']

tab
inertia = []

for k in tqdm_notebook(range(1, n_classes + 1)):

    kmeans = KMeans(n_clusters=k, random_state=17).fit(x_pca)

    inertia.append(np.sqrt(kmeans.inertia_))



plt.plot(range(1, 7), inertia, marker='s');

plt.xlabel('$k$')

plt.ylabel('$J(C_k)$');
ag = AgglomerativeClustering(n_clusters=n_classes, 

                             linkage='ward').fit(x_pca)
from sklearn.metrics.cluster import adjusted_rand_score
print('KMeans: ARI =', metrics.adjusted_rand_score(y[0], cluster_labels))

print('Agglomerative CLustering: ARI =', 

      metrics.adjusted_rand_score(y[0], ag.labels_))
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.fit_transform(X_test)
svc = LinearSVC(random_state=RANDOM_STATE)

svc_params = {'C': [0.001, 0.01, 0.1, 1, 10]}
%%time

best_svc = GridSearchCV(svc, svc_params, cv=3)
best_svc.fit(X_train_scaled, y_train[0])
best_svc.best_params_, best_svc.best_score_
y_predicted = best_svc.predict(X_test_scaled)
tab = pd.crosstab(y_test[0], y_predicted, margins=True)

tab.index = ['walking', 'climbing up the stairs',

            'going down the stairs', 'sitting', 'standing', 'laying', 'all']

tab.columns = ['walking', 'climbing up the stairs',

            'going down the stairs', 'sitting', 'standing', 'laying', 'all']

tab
x_pca_test = pca.transform(X_test)
%%time

best_svc = GridSearchCV(svc, svc_params, cv=3)
best_svc.fit(x_pca_test, y_test[0])
best_svc.best_params_, best_svc.best_score_