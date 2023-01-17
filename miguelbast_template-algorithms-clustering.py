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
X = np.concatenate((X_train, X_test))

y = np.hstack((y_train, y_test))
classes = np.unique(y)
n_classes = classes.size
# Your code here

scaler = StandardScaler()

X = scaler.fit_transform(X)
# Your code here

pca = PCA(random_state = RANDOM_STATE)

X_pca = pca.fit_transform(X)

count = 0

for i, component in enumerate(pca.components_):

    count += round(100 * pca.explained_variance_ratio_[i], 2)

    if i == 0 or (i+1) % 10 == 0:

        print("With {} components: {}% of initial variance".format(i + 1, count))

    if count >= 90:

        print("With {} components: {}% of initial variance".format(i + 1, count))

        break
# Your code here

65
# Your code here

51
# Your code here

plt.scatter(X_pca[:,0], X_pca[:,1], c=y, s=20, cmap='viridis');
km = KMeans(n_clusters = n_classes, n_init = 100, random_state = RANDOM_STATE)

km.fit(X_pca)
plt.scatter(X_pca[:,0], X_pca[:,1], c=km.labels_, s=20, cmap='viridis');
tab = pd.crosstab(y, km.labels_, margins=True)

tab.index = ['walking', 'going up the stairs',

            'going down the stairs', 'sitting', 'standing', 'laying', 'all']

tab.columns = ['cluster' + str(i + 1) for i in range(6)] + ['all']

tab
inertia = []

for k in tqdm_notebook(range(1, n_classes + 1)):

    inertia.append(KMeans(n_clusters=k, n_init=100, random_state = RANDOM_STATE, n_jobs=1).fit(X).inertia_)
plt.plot(range(1, 7), inertia, marker='s');

plt.xlabel("Num Clusters")

plt.ylabel("Sum sqared distances")
# calculating slopes

d = {}

for k in range(2, 6):

    i = k - 1

    d[k] = (inertia[i] - inertia[i + 1]) / (inertia[i - 1] - inertia[i])

d
ag = AgglomerativeClustering(n_clusters=n_classes, 

                             linkage='ward').fit(X_pca)
# Your code here

print(f'Adjusted Rand Index for K-means: {metrics.adjusted_rand_score(y, km.labels_)}')

print(f'Adjusted Rand Index for Agglomerative Clustering: {metrics.adjusted_rand_score(y, ag.labels_)}')
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.fit_transform(X_test)
svc = LinearSVC(random_state=RANDOM_STATE)

svc_params = {'C': [0.001, 0.01, 0.1, 1, 10]}
%%time 

best_svc = GridSearchCV(svc, svc_params, n_jobs=1, cv=3, verbose=1)

best_svc.fit(X_train_scaled, y_train)
best_svc.best_params_, best_svc.best_score_
y_predicted = best_svc.predict(X_test_scaled)
tab = pd.crosstab(y_test, y_predicted, margins=True)

tab.index = ['walking', 'climbing up the stairs',

             'going down the stairs', 'sitting', 'standing', 'laying', 'all']

tab.columns = ['walking', 'climbing up the stairs',

             'going down the stairs', 'sitting', 'standing', 'laying', 'all']

tab
# Your code here

pca = PCA(random_state = RANDOM_STATE, n_components = 0.9)

X_pca = pca.fit_transform(X_train_scaled)

X_pca_test = pca.transform(X_test_scaled)

#count = 0

#for i, component in enumerate(pca.components_):

#    count += round(100 * pca.explained_variance_ratio_[i], 2)

#    if i == 0 or (i+1) % 10 == 0:

#        print("With {} components: {}% of initial variance".format(i + 1, count))

#    if count >= 90:

#        print("With {} components: {}% of initial variance".format(i + 1, count))

#        break





svc = LinearSVC(random_state=RANDOM_STATE)

svc_params = {'C': [0.001, 0.01, 0.1, 1, 10]}

best_svc_pca = GridSearchCV(svc, svc_params, n_jobs=1, cv=3, verbose=1)

best_svc_pca.fit(X_pca, y_train)

best_svc_pca.best_params_, best_svc_pca.best_score_
print(f'Mejor hiperparámetro: {best_svc_pca.best_params_}')
best_svc_pca.best_score_ - best_svc.best_score_