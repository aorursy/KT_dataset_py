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
X = np.concatenate((X_train,X_test),axis=0)
y = np.concatenate((y_train,y_test))
np.unique(y)
n_classes = np.unique(y).size
n_classes
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
X.shape
# Your code here
pca = PCA(n_components=0.9,random_state=RANDOM_STATE)
pca.fit(X)
X_pca = pca.transform(X)
X_pca.shape, sum(pca.explained_variance_ratio_), pca.explained_variance_ratio_
# Your code here
# Your code here
# Your code here
pca = PCA(n_components=2,random_state=RANDOM_STATE)
pca.fit(X)
X_pca = pca.transform(X)
X_pca.shape, sum(pca.explained_variance_ratio_), pca.explained_variance_ratio_
plt.scatter(X_pca[:,0],X_pca[:,1] , c=y, s=20, cmap='viridis');
plt.legend()
X_pca.shape
kmeans = KMeans(n_clusters=n_classes,n_init=100,random_state=RANDOM_STATE)
kmeans.fit(X_pca)
# Your code here
plt.scatter(X_pca[:,0],X_pca[:,1] , c=kmeans.labels_, s=20, cmap='viridis');
tab = pd.crosstab(y, kmeans.labels_, margins=True)
tab.index = ['walking', 'going up the stairs',
            'going down the stairs', 'sitting', 'standing', 'laying', 'all']
tab.columns = ['cluster' + str(i + 1) for i in range(6)] + ['all']
tab
mute_all_chat = tab['all']
tab = tab.drop(['all'],axis=1)
tb = np.array(tab)
tb.max(axis=1)/mute_all_chat
# # Your code here
inertia = []
for k in tqdm_notebook(range(1, n_classes + 1)):
    kmeans = KMeans(n_clusters=k,n_init=100,random_state=RANDOM_STATE).fit(X_pca)
    inertia.append(np.sqrt(kmeans.inertia_))

plt.plot(range(1, 7), inertia, marker='s');
plt.xlabel('$k$')
plt.ylabel('$J(C_k)$');
# ag = AgglomerativeClustering(n_clusters=n_classes, 
#                              linkage='ward').fit(X_pca)
# Your code here
# # Your code here
scaler = StandardScaler()
X_train_scaled =scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
svc = LinearSVC(random_state=RANDOM_STATE)
svc_params = {'C': [0.001, 0.01, 0.1, 1, 10]}
 %%time
# # Your code here

best_svc = GridSearchCV(svc,svc_params,cv=3)
best_svc.fit(X_train_scaled,y_train)
best_svc.best_params_, best_svc.best_score_
y_predicted = best_svc.predict(X_test_scaled)
tab = pd.crosstab(y_test, y_predicted, margins=True)
tab.index = ['walking', 'climbing up the stairs',
             'going down the stairs', 'sitting', 'standing', 'laying', 'all']
tab.columns = ['walking', 'climbing up the stairs',
             'going down the stairs', 'sitting', 'standing', 'laying', 'all']
mute_all = tab['all']
tab = tab.drop(['all'],axis=1)
tab = tab.drop(['all'],axis=0)
tab
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_predicted)

tb = np.array(tab)
tb
diag = tb.diagonal()
tb.sum(axis=1)
precisions = tp/tb.sum(axis=0)
precisions
#standing
recalls = tp/tb.sum(axis=1)
recalls
#sitting
pca.fit(X_train_scaled)
X_train_pca = pca.transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
svc = LinearSVC(random_state=RANDOM_STATE)
svc_params = {'C': [0.001, 0.01, 0.1, 1, 10]}

best_svc = GridSearchCV(svc,svc_params,cv=3)
best_svc.fit(X_train_pca,y_train)
best_svc.best_params_, best_svc.best_score_