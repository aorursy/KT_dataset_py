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
merx =np.vstack([X_train, X_test])
mery = np.hstack([y_train, y_test])
np.unique(mery)

n_classes = np.unique(mery).size
n_classes
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_x = scaler.fit_transform(merx)
scaled_x 
# Your code here
pca = PCA(n_components=0.9, random_state =RANDOM_STATE).fit(scaled_x)
X_pca = pca.fit_transform(scaled_x)

X_pca.shape
# Your code here
round(float(pca.explained_variance_ratio_[0] * 100))
# Your code here
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=mery, s=20, cmap='viridis');
# Your code here
k_means = KMeans(n_clusters=n_classes, n_init=100, random_state=RANDOM_STATE)
k_means.fit(X_pca)
# Your code here
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=k_means.labels_, s=20, cmap='viridis');
tab = pd.crosstab(mery, k_means.labels_, margins=True)
tab.index = ['walking', 'going up the stairs',
             'going down the stairs', 'sitting', 'standing', 'laying', 'all']
tab.columns = ['cluster' + str(i + 1) for i in range(6)] + ['all']
tab
 # Your code here
inertia = []
for k in tqdm_notebook(range(1, n_classes + 1)):
    kmeans = KMeans(n_clusters=k, random_state=1).fit(merx)
    inertia.append(np.sqrt(kmeans.inertia_))
plt.plot(range(1, 7), inertia, marker='s');
ag = AgglomerativeClustering(n_clusters=n_classes,
                              linkage='ward').fit(X_pca)
# Your code here
from sklearn.metrics import adjusted_rand_score
adjusted_rand_score(mery, ag.labels_)
adjusted_rand_score(mery, k_means.labels_)
# # Your code here
# scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

svc = LinearSVC(random_state=RANDOM_STATE)
svc_params = {'C': [0.001, 0.01, 0.1, 1, 10]}
%%time
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
# Your code here