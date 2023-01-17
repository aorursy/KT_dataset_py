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
import numpy as np
X=np.concatenate((X_train, X_test))
y=np.concatenate((y_train,y_test))
np.unique(y)
n_classes = np.unique(y).size
n_classes
xdata = pd.DataFrame(X)
ydata=pd.DataFrame(y)
ydata.replace(1, 'walking',inplace=True)
ydata.replace(2, 'walking upstairs',inplace=True)
ydata.replace(3, 'walking downstairs',inplace=True)
ydata.replace(4, 'sitting',inplace=True)
ydata.replace(5, 'standing',inplace=True)
ydata.replace(6, 'laying down',inplace=True)
ydata.rename(columns = {0:'target'}, inplace = True)
ydata.head()
ydata['target'].unique()  #determined labels as target
xscaled = StandardScaler().fit_transform(xdata) # scaling the features
xscaled.shape

#feat_cols = ['feature'+str(i) for i in range(xscaled.shape[1])]
#normalized_x = pd.DataFrame(xscaled,columns=feat_cols)
#normalized_x.tail()
from sklearn.decomposition import PCA
pca_x = PCA(n_components=0.9, random_state = RANDOM_STATE)
principalComponents_x = pca_x.fit_transform(xscaled)
#principal_x_Df = pd.DataFrame(data = principalComponents_x
 #            , columns = ['principal component 1', 'principal component 2'])
#principal_x_Df.head()
#print('Explained variation per principal component: {}'.format(pca_x.explained_variance_ratio_))
pca = PCA(n_components=100)
pca.fit(xscaled)
exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)
x_ticks = np.arange(0, 100, 5)
plt.xticks(x_ticks)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.axhline(0.9, c='r') #added red line for to see intersection.
plt.show;
pca.explained_variance_ratio_[0]*100 
# Your code here
# plt.scatter(, , c=y, s=20, cmap='viridis');
plt.figure(figsize=(8,6))
plt.scatter(principalComponents_x[:,0],principalComponents_x[:,1],c=y,cmap='viridis')

import seaborn as sns
plt.figure(figsize=(16,10))
sns.scatterplot(
    principalComponents_x[:,0],principalComponents_x[:,1],
    hue='target',
    palette=sns.color_palette("hls", 6),
    data=ydata,
    legend="full",
    alpha=0.8
)
# Your code here
kmeans_pca=KMeans(n_clusters= n_classes,n_init=100,random_state = RANDOM_STATE)
kmeans_pca.fit(principalComponents_x)

cluster_labels = kmeans_pca.predict(principalComponents_x) #y values predicted by using kmeans algorithm
cluster_labels
# Your code here
plt.scatter(principalComponents_x[:,0],principalComponents_x[:,1], c=cluster_labels, s=20, cmap='viridis');
tab = pd.crosstab(y, cluster_labels, margins=True)
tab.index = ['walking', 'going up the stairs',
             'going down the stairs', 'sitting', 'standing', 'laying', 'all']
tab.columns = ['cluster' + str(i + 1) for i in range(6)] + ['all']
tab
pd.Series(tab.iloc[:-1,:-1].max(axis=1).values / 
          tab.iloc[:-1,-1].values, index=tab.index[:-1])
# # Your code here
inertia = []
for k in tqdm_notebook(range(1, n_classes + 1)):
    kmeans_pca=KMeans(n_clusters= k,n_init=100,random_state = RANDOM_STATE)
    kmeans_pca.fit(principalComponents_x)
    inertia.append(kmeans_pca.inertia_)
pass
plt.figure(figsize=(16,8))
plt.plot(range(1,7), inertia, 'bx-')
plt.xlabel('k')
plt.ylabel('inertia')
plt.title('The Elbow Method showing the optimal k')
plt.show()
ag = AgglomerativeClustering(n_clusters=n_classes, 
                             linkage='ward').fit(principalComponents_x)
ag
# Your code here

from sklearn.metrics.cluster import adjusted_rand_score
K_Mean_ARI = metrics.adjusted_rand_score(y, cluster_labels) 
Ag_ARI = metrics.adjusted_rand_score(y, ag.labels_)

print('K Means',K_Mean_ARI )
print('Ag_ARI',Ag_ARI )
# # Your code here
# scaler = StandardScaler()
# X_train_scaled =
# X_test_scaled = 
svc = LinearSVC(random_state=RANDOM_STATE)
svc_params = {'C': [0.001, 0.01, 0.1, 1, 10]}
# %%time
# # Your code here
# best_svc = None
# best_svc.best_params_, best_svc.best_score_
# y_predicted = best_svc.predict(X_test_scaled)
# tab = pd.crosstab(y_test, y_predicted, margins=True)
# tab.index = ['walking', 'climbing up the stairs',
#              'going down the stairs', 'sitting', 'standing', 'laying', 'all']
# tab.columns = ['walking', 'climbing up the stairs',
#              'going down the stairs', 'sitting', 'standing', 'laying', 'all']
# tab
# Your code here