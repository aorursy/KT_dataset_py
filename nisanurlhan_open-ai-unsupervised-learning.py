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

X = np.vstack([X_train, X_test])

y = np.hstack([y_train, y_test])
np.unique(y)
n_classes = np.unique(y).size

n_classes
# Your code here

from sklearn import preprocessing

X_scaled=preprocessing.StandardScaler().fit_transform(X) #with default parameter

X_scaled.shape #(10299, 561) array
# Your code here

from sklearn.decomposition import PCA

pca = PCA(n_components=0.9,random_state=17,svd_solver='auto')

pca = pca.fit(X_scaled)

X_pca=pca.transform(X_scaled)

X_pca.shape #(10299, 65)
import matplotlib.pyplot as plt



per_val=np.round(pca.explained_variance_ratio_*100,decimals=1)

print(f"Percentage of Explained Variance {(np.round(pca.explained_variance_ratio_*100))[0]}")

labels=['PC'+str(x) for x in range(0,len(per_val))]

plt.bar(x=range(0,len(per_val)),height=per_val,tick_label=labels)

plt.ylabel('Percentage of Explained Varinace')

plt.xlabel('Principal Component')

plt.xticks(rotation=60)

plt.title("Scree Plot")

plt.figure(figsize=(16,8))

plt.show()
# Your code here

plt.scatter(X_pca[:,0],X_pca[:,1] , c=y, s=20, cmap='viridis');
# Your code here

from sklearn.cluster import KMeans



kmeans=KMeans(n_clusters=n_classes,n_init = 100,random_state = RANDOM_STATE)

kmeans.fit(X_pca)

cluster_labels=kmeans.labels_

#kmeans.cluster_centers_

#kmeans.score(X_pca)
# Your code here

plt.scatter(X_pca[:,0],X_pca[:,1] , c=cluster_labels, s=20, cmap='viridis');
tab = pd.crosstab(y, cluster_labels, margins=True)

tab.index = ['walking', 'going up the stairs',

             'going down the stairs', 'sitting', 'standing', 'laying', 'all']

tab.columns = ['cluster' + str(i + 1) for i in range(6)] + ['all']

tab
# # Your code here

inertia = []

n_classes=10

for k in tqdm_notebook(range(1, n_classes + 1)):

    kmean_model=KMeans(n_clusters=k,n_init=100, 

                    random_state=RANDOM_STATE, n_jobs=1)

    kmean_model.fit(X_pca)

    inertia.append(kmean_model.inertia_)

plt.figure(figsize=(16,8))

plt.xlabel('k')

plt.ylabel('Inertia')

plt.title('Elbow Method')

plt.plot((range(1, n_classes + 1)),inertia,'bx-')

plt.show()
ag = AgglomerativeClustering(n_clusters=n_classes, 

                              linkage='ward').fit(X_pca)

ag_labels_=ag.labels_
# Your code here

import sklearn

#ARI = (RI - Expected_RI) / (max(RI) - Expected_RI)

print("KMeans",sklearn.metrics.cluster.adjusted_rand_score(y,cluster_labels))

print("Agglomerative Clustering",sklearn.metrics.cluster.adjusted_rand_score(y,ag_labels_))
# # Your code here

from sklearn import svm

from sklearn import preprocessing

from sklearn.model_selection import GridSearchCV

cv=3

scaler = preprocessing.StandardScaler()

X_train_scaled=scaler.fit_transform(X_train)

X_test_scaled=scaler.transform(X_test)
svc = LinearSVC(random_state=RANDOM_STATE)

svc_params = {'C': [0.001, 0.01, 0.1, 1, 10]}
%%time

# Your code here

#what is the n_jobs parameter? I got a error like "Using backend SequentialBackend with 1 concurrent workers." and this execute process takes a little long.

best_svc =GridSearchCV(svc,svc_params,n_jobs=1,cv=cv,verbose=1)

best_svc.fit(X_train_scaled,y_train)
best_svc.best_params_, best_svc.best_score_
y_predicted = best_svc.predict(X_test_scaled)
tab = pd.crosstab(y_test, y_predicted, margins=True)

tab.index = ['walking', 'climbing up the stairs',

              'going down the stairs', 'sitting', 'standing', 'laying', 'all']

tab.columns = ['walking', 'climbing up the stairs',

              'going down the stairs', 'sitting', 'standing', 'laying', 'all']

tab
from sklearn.metrics import classification_report,confusion_matrix

target_names = ['walking', 'going up the stairs',

             'going down the stairs', 'sitting', 'standing', 'laying', 'all']

print(confusion_matrix(y_test,y_predicted))


from sklearn.metrics import precision_score,recall_score

precision = precision_score(y_test, y_predicted,average=None)

print('Precision:', precision)

# recall: tp / (tp + fn)

recall = recall_score(y_test, y_predicted,average=None)

print('recall:', recall)
# Your code here

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=0.9, random_state=RANDOM_STATE)

X_train_pca = pca.fit_transform(X_train_scaled)

X_test_pca = pca.transform(X_test_scaled)

svc = LinearSVC(random_state=RANDOM_STATE)

svc_params = {'C': [0.001, 0.01, 0.1, 1, 10]}

best_svc_pca = GridSearchCV(svc, svc_params, n_jobs=1, cv=3, verbose=1)

best_svc_pca.fit(X_train_pca, y_train);

best_svc_pca.best_params_, best_svc_pca.best_score_

round(100 * (best_svc_pca.best_score_ - best_svc.best_score_))