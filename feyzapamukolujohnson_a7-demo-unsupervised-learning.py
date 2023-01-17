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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn import preprocessing
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
df_x1=pd.DataFrame(X_train)
df_x2=pd.DataFrame(X_test)
df_mx=pd.concat([df_x1, df_x2], axis=0, join='outer')
df_y1=pd.DataFrame(y_train)
df_y2=pd.DataFrame(y_test)
df_my=pd.concat([df_y1,df_y2], axis=0, join='outer')
df_mx.head()
plt.scatter(df_mx[1],df_mx[0])
np.unique(df_my)
n_classes = np.unique(df_my).size
n_classes
# Your code here
scaler = StandardScaler()
df_std = scaler.fit_transform(df_mx)

df_std.shape
np.mean(df_std),np.std(df_std)
# Your code here
pca = PCA(n_components=0.9 ,random_state=RANDOM_STATE).fit(df_std)
X_pca = pca.transform(df_std)
X_pca.shape

# Your code here

percent_var = np.round(pca.explained_variance_ratio_*100, decimals=1)
labels = ['PC' + str(x) for x in range(1, len(percent_var)+1)]
plt.figure(figsize=(20,20))
plt.bar(x=range(1, len(percent_var)+1), height = percent_var, tick_label = labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()
# Your code here

print("original shape:   ", df_std.shape)
print("transformed shape:", X_pca.shape)
round(float(pca.explained_variance_ratio_[0] * 100))
plt.scatter(X_pca[:, 0], X_pca[:, 1],
            c=df_my.loc[:,0], edgecolor='none', alpha=0.5, s=40,cmap=plt.cm.get_cmap('nipy_spectral', 10))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar();
# Your code here
kmeans = KMeans(n_clusters=n_classes, n_init=100, random_state=RANDOM_STATE ,n_jobs=1)
kmeans.fit(X_pca)
kmeans.cluster_centers_
# Your code here

plt.scatter(X_pca[:,0],X_pca[:,1], s = 20, c = kmeans.labels_,cmap=plt.cm.get_cmap('nipy_spectral', 10), label='centroids')
plt.legend()
plt.show()
tab = pd.crosstab(df_my.iloc[:,0], kmeans.labels_, margins=True)
tab.index = ['walking', 'walking upstairs',
             'walking downstairs', 'sitting', 'standing', 'laying', 'all']
tab.columns = ['cluster' + str(i + 1) for i in range(6)] + ['all']
tab
# # Your code here
inertia = []
for k in range(1, 7):
    kmeans = KMeans(n_clusters=k, random_state=17).fit(X_pca)
    inertia.append(np.sqrt(kmeans.inertia_))
plt.plot(range(1, 7), inertia, marker='s');
plt.xlabel('clusters')
plt.ylabel('inertia');
ag = AgglomerativeClustering(n_clusters=n_classes, 
                             linkage='ward')
ag.fit(X_pca)
# Your code here
from sklearn.metrics.cluster import adjusted_rand_score
print('KMeans: ARI =', metrics.adjusted_rand_score(df_my.iloc[:,0], kmeans.labels_))
print('Agglomerative CLustering: ARI =', 
      metrics.adjusted_rand_score(df_my.iloc[:,0], ag.labels_))

# # Your code here
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(df_x1)
X_test_scaled =  scaler.fit_transform(df_x2)
df_x1.head()
svc = LinearSVC(random_state=RANDOM_STATE)
svc_params = {'C': [0.001, 0.01, 0.1, 1, 10]}
C_svc= GridSearchCV(estimator=svc, param_grid=svc_params, cv= 3)
C_svc.fit(X_train_scaled, df_y1.loc[:,0])

%%time
best_svc = C_svc.fit(X_train_scaled, df_y1.loc[:,0])
best_svc.best_params_, best_svc.best_score_
y_predicted = best_svc.predict(X_test_scaled)
tab = pd.crosstab(y_test, y_predicted, margins=True)
tab.index = ['walking', 'climbing up the stairs',
              'going down the stairs', 'sitting', 'standing', 'laying', 'all']
tab.columns = ['walking', 'climbing up the stairs',
              'going down the stairs', 'sitting', 'standing', 'laying', 'all']
tab

from sklearn.metrics import confusion_matrix
 
confusion_matrix(y_test, y_predicted)
# Your code here
pca = PCA(n_components=0.9, random_state=RANDOM_STATE)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

svc = LinearSVC(random_state=RANDOM_STATE)
svc_params = {'C': [0.001, 0.01, 0.1, 1, 10]}
C_svc= GridSearchCV(estimator=svc, param_grid=svc_params, cv= 3)
C_svc.fit(X_train_pca, df_y1.loc[:,0])
C_svc.best_params_, C_svc.best_score_
round((C_svc.best_score_ -best_svc.best_score_)*100 )