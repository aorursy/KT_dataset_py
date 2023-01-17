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
# Your code here

X = np.vstack((X_train, X_test))

y = np.concatenate([y_train, y_test])
# np.unique(y)

np.unique(y)
# n_classes = np.unique(y).size

n_classes = np.unique(y).size
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components = 0.9, random_state=RANDOM_STATE)

X_pca = pca.fit_transform(X_scaled)
pca.n_components_
print ("%0.0f" % (pca.explained_variance_ratio_[0]*100))
# Your code here

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, s=20, cmap='viridis');
X_kmeans = KMeans(n_clusters=n_classes, n_init=100, random_state=RANDOM_STATE).fit(X_pca)
# Your code here

cluster_labels = X_kmeans.labels_.astype(int)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, s=20, cmap='viridis');
tab = pd.crosstab(y, cluster_labels, margins=True)

tab.index = ['walking', 'going up the stairs',

            'going down the stairs', 'sitting', 'standing', 'laying', 'all']

tab.columns = ['cluster' + str(i + 1) for i in range(6)] + ['all']

tab
# # Your code here

# inertia = []

# for k in tqdm_notebook(range(1, n_classes + 1)):

#     pass

inertia = []

for k in tqdm_notebook(range(1, n_classes + 1)):

    test_kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE).fit(X_pca)

    inertia.append(np.sqrt(test_kmeans.inertia_))



plt.plot(range(1, n_classes+1), inertia, marker='s');

plt.xlabel('$k$')

plt.ylabel('$J(C_k)$');
ag = AgglomerativeClustering(n_clusters=n_classes, linkage='ward').fit(X_pca)
print("Agglomerative Clustering ARI: %0.3f" % metrics.adjusted_rand_score(y, ag.labels_))

print("KMeans ARI: %0.3f" % metrics.adjusted_rand_score(y, X_kmeans.labels_))

R = np.random.randint(1, high=n_classes, size=y.shape)

print("Random ARI: %0.3f" % metrics.adjusted_rand_score(y, R))
# Your code here

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
svc = LinearSVC(random_state=RANDOM_STATE, dual=False)

svc_params = {'C': [0.001, 0.01, 0.1, 1, 10]}
%%time



gs=GridSearchCV(svc, svc_params, cv=3, verbose=False)

gs.fit(X_train_scaled, y_train)
best_svc = gs

best_svc.best_params_, best_svc.best_score_
y_predicted = best_svc.predict(X_test_scaled)
tab = pd.crosstab(y_test, y_predicted, margins=True)

tab.index = ['walking', 'climbing up the stairs',

             'going down the stairs', 'sitting', 'standing', 'laying', 'all']

tab.columns = ['walking', 'climbing up the stairs',

             'going down the stairs', 'sitting', 'standing', 'laying', 'all']

tab
# precision = tp / (tp+fp)

# recall = tp / (tp+fn)

labels = ['walking', 'climbing up the stairs', 'going down the stairs', 'sitting', 'standing', 'laying']

precision = [494/508, 459/469, 413/413, 426/442, 517/589, 526/526]

recall = [494/496, 459/471, 413/420, 426/491, 517/532, 526/537]

testy = pd.DataFrame({'labels': labels, 'precision' : pd.Series(precision), 'recall' : pd.Series(recall)})

print (testy)
X_pca_train = pca.fit_transform(X_train_scaled)

X_pca_test = pca.transform(X_test_scaled)



svc = LinearSVC(random_state=RANDOM_STATE, dual=False)

svc_params = {'C': [0.001, 0.01, 0.1, 1, 10]}
%%time

gs2=GridSearchCV(svc, svc_params, cv=3, verbose=False)

gs2.fit(X_pca_train, y_train)

print ("")
best_svc2 = gs2

print(best_svc2.best_params_, best_svc2.best_score_)

print("There was a loss of %0.1f" % ((best_svc.best_score_ - best_svc2.best_score_)*100), "percent")

y_predicted2 = best_svc2.predict(X_pca_test)