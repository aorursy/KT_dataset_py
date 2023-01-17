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
y = np.concatenate((y_train, y_test))
np.unique(y)
n_classes = np.unique(y).size
n_classes
sc = StandardScaler()
X = sc.fit_transform(X)
from sklearn import decomposition
pca = decomposition.PCA(n_components=66, random_state = RANDOM_STATE)
X_pca = pca.fit_transform(X)
s = 0 
for i, component in enumerate(pca.components_):
    s += 100 * pca.explained_variance_ratio_[i]
    print("{} component: {}% of initial variance".format(i + 1, 
                                                         round(100 * pca.explained_variance_ratio_[i], 2)))
print(s)
plt.scatter(X_pca[:,0], X_pca[:,1], c=y, s=20, cmap='viridis');
plt.colorbar()
kmeans = KMeans(n_clusters = n_classes, n_init = 100, random_state=RANDOM_STATE)
kmeans.fit(X_pca)
cluster_labels= kmeans.labels_
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, s=20, cmap='viridis')
tab = pd.crosstab(y, cluster_labels, margins=True)
tab.index = ['walking', 'going up the stairs',
             'going down the stairs', 'sitting', 'standing', 'laying', 'all']
tab.columns = ['cluster' + str(i + 1) for i in range(6)] + ['all']
tab
max_activity_values = tab.drop('all', axis=1).max(1)
all_values  = tab.max(1)
percentages = round(max_activity_values/all_values,3)
percentages.drop('all')*100
inertia = []
for k in tqdm_notebook(range(1, n_classes + 1)):
    kmeans = KMeans(n_clusters=k, n_init = 100, random_state=RANDOM_STATE).fit(X)
    inertia.append(np.sqrt(kmeans.inertia_))

    
plt.plot(range(1, n_classes + 1), inertia, marker='s');
plt.xlabel('$k$')
plt.ylabel('$J(C_k)$');
ag = AgglomerativeClustering(n_clusters=n_classes, 
                             linkage='ward').fit(X_pca)
kmeans = KMeans(n_clusters=2, n_init = 100, random_state=RANDOM_STATE).fit(X)
ARI_agg = metrics.adjusted_rand_score(y, ag.labels_)
ARI_km = metrics.adjusted_rand_score(y, kmeans.labels_)
print ('ARI for agglomerative clustering is ' + str(round(ARI_agg,4)))
print ('ARI for kmeans is ' + str(round(ARI_km,4)))
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) 
svc = LinearSVC(random_state=RANDOM_STATE)
svc_params = {'C': [0.001, 0.01, 0.1, 1, 10]}
%%time
clf = GridSearchCV(svc, svc_params, cv=3)
clf.fit(X_train_scaled, y_train)
print(clf.best_params_)
best_svc = LinearSVC(C = 0.1, random_state=RANDOM_STATE)
best_svc.fit(X_train_scaled, y_train)
#best_svc.best_params_, best_svc.best_score_
y_predicted = best_svc.predict(X_test_scaled)
tab1 = pd.crosstab(y_test, y_predicted, margins=True)
tab1.index = ['walking', 'climbing up the stairs',
              'going down the stairs', 'sitting', 'standing', 'laying', 'all']
tab1.columns = ['walking', 'climbing up the stairs',
             'going down the stairs', 'sitting', 'standing', 'laying', 'all']
tab1
from sklearn.metrics import precision_recall_fscore_support as score

precision, recall, fscore, support = score(y_test, y_predicted)
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) 

pca = decomposition.PCA(n_components = 66, random_state = RANDOM_STATE)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
%%time
svc = LinearSVC(random_state = RANDOM_STATE)
svc_params = {'C': [0.001, 0.01, 0.1, 1, 10]}
clf = GridSearchCV(svc, svc_params, cv = 3)
clf.fit(X_train_pca, y_train)
print(clf.best_params_)
best_svc = LinearSVC(C = 0.1, random_state = RANDOM_STATE)
best_svc.fit(X_train_pca, y_train)
y_predicted = best_svc.predict(X_test_pca)
tab2 = pd.crosstab(y_test, y_predicted, margins=True)
tab2.index = ['walking', 'climbing up the stairs',
              'going down the stairs', 'sitting', 'standing', 'laying', 'all']
tab2.columns = ['walking', 'climbing up the stairs',
             'going down the stairs', 'sitting', 'standing', 'laying', 'all']
tab2
#accuracy with pca
#best accuracy is for walking
max_activity_values2 = tab2.drop('all', axis=1).max(1)
all_values2  = tab2.max(1)
percentages2 = round(max_activity_values2/all_values2,3)*100
percentages2.drop('all')
#accuracy without pca
max_activity_values1 = tab1.drop('all', axis=1).max(1)
all_values1  = tab1.max(1)
percentages1 = round(max_activity_values1/all_values1,3)*100
percentages1.drop('all')
import math
p = percentages1.drop('all').max() - percentages2.loc['walking']
p = math.ceil(p)
print('The difference between the best quality (accuracy) for cross-validation in the case of all initial characteristics and in the second case, when the principal component method was applied is ' + str(round(p,2)) + '%')