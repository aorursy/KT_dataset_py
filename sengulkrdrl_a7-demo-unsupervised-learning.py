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
X_train.shape,X_test.shape
y_train.shape, y_test.shape
# Checking dimensions
assert(X_train.shape == (7352, 561) and y_train.shape == (7352,))
assert(X_test.shape == (2947, 561) and y_test.shape == (2947,))
X = np.vstack([X_train, X_test])
y = np.hstack([y_train, y_test])    # neden horizontal birleştirdik?
y.shape
X.shape
np.unique(y)
n_classes = np.unique(y).size
n_classes
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaledX = scaler.fit_transform(X)
# print(scaler.mean_)
pca = PCA(n_components=0.9, random_state=RANDOM_STATE).fit(scaledX)     # n_component ? 
X_pca = pca.transform(scaledX)
X_pca.shape   # 65 feature 
round(float(pca.explained_variance_ratio_[0] *100 ))   # varyansın %51'ini ilk pca açıklar/ kapsar. 
# Your code here
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, s=20, cmap='viridis');
kmeans = KMeans(n_clusters=n_classes, n_init=100, 
                random_state=RANDOM_STATE, n_jobs=1)  # random_state ile her zaman aynı örnekler mi alınıyor yoksa sadece aynı oranda mı bölmesini sağlıyor? 
kmeans.fit(X_pca)
cluster_labels = kmeans.labels_
kmeans.labels_
cluster_labels.shape
kmeans.cluster_centers_.shape
# Your code here
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, s=20, cmap='viridis');
tab = pd.crosstab(y, cluster_labels, margins=True)
tab.index = ['walking', 'going up the stairs',
             'going down the stairs', 'sitting', 'standing', 'laying', 'all']
tab.columns = ['cluster' + str(i + 1) for i in range(6)] + ['all']
tab
# # Your code here
inertia = []
for k in tqdm_notebook(range(1, n_classes + 1)):
    kmeans = KMeans(n_clusters=k, n_init=100,random_state=RANDOM_STATE, n_jobs=1).fit(X_pca)
    inertia.append(np.sqrt(kmeans.inertia_)) 
plt.plot(range(1, n_classes+1), inertia, marker='s')
ag = AgglomerativeClustering(n_clusters=n_classes, 
                              linkage='ward').fit(X_pca)
# K Means Clustering 
print(metrics.adjusted_rand_score(y, cluster_labels))
# Agglomerative Clustering
print(metrics.adjusted_rand_score(y, ag.labels_))
# # Your code here
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit transformu ne zaman birlikte ne zaman ayrı kullanıyoruz? 
X_test_scaled = scaler.fit_transform(X_test)
svc = LinearSVC(random_state=RANDOM_STATE)
svc_params = {'C': [0.001, 0.01, 0.1, 1, 10]}
%%time
# # Your code here
best_svc = GridSearchCV(svc, svc_params, n_jobs=1, cv=3, verbose=1)
best_svc.fit(X_train_scaled, y_train) # x_train'in scale edilmiş halini kullanırken neden y_train'i aldık?
best_svc.best_params_, best_svc.best_score_
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