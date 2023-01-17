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
x=np.concatenate((X_train, X_test))
y=np.concatenate((y_train,y_test))
np.unique(y)
n_classes = np.unique(y).size
n_classes
xdata = pd.DataFrame(x)
ydata=pd.DataFrame(y)
ydata.replace(1, 'walking',inplace=True)
ydata.replace(2, 'walking upstairs',inplace=True)
ydata.replace(3, 'walking downstairs',inplace=True)
ydata.replace(4, 'sitting',inplace=True)
ydata.replace(5, 'standing',inplace=True)
ydata.replace(6, 'laying down',inplace=True)
xscaled = StandardScaler().fit_transform(xdata) # normalizing the features
xscaled.shape
np.mean(xscaled),np.std(xscaled)
xscaled
feat_cols = ['feature'+str(i) for i in range(xscaled.shape[1])]
normalized_x = pd.DataFrame(xscaled,columns=feat_cols)
normalized_x.tail()
from sklearn.decomposition import PCA
pca_x = PCA(n_components=2)
principalComponents_x = pca_x.fit_transform(normalized_x)
principal_x_Df = pd.DataFrame(data = principalComponents_x
             , columns = ['principal component 1', 'principal component 2'])
principal_x_Df.tail()
print('Explained variation per principal component: {}'.format(pca_x.explained_variance_ratio_))
plt.figure(figsize=(8,6))
plt.scatter(principalComponents_x[:,0],principalComponents_x[:,1],c=y,cmap='plasma')
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')

principal_x_Df['y'] = y
principal_x_Df.head()
new=pd.concat([principal_x_Df,ydata],axis=1)
new.columns=['pca1','pca2','y','target']
new.drop('y',axis=1,inplace=True)
new.head()
pca = PCA(n_components=100)
pca.fit(normalized_x)
exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)
x_ticks = np.arange(0, 100, 5)
plt.xticks(x_ticks)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
import seaborn as sns
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="pca1", y="pca2",
    hue="target",
    palette=sns.color_palette("hls", 6),
    data=new,
    legend="full",
    alpha=0.8
)
n_classes
kmeans_pca=KMeans(n_clusters= n_classes, init='random',n_init=100,random_state = RANDOM_STATE)
kmeans_pca.fit(principalComponents_x)



plt.scatter(principalComponents_x[:, 0], principalComponents_x[:, 1], c=kmeans_pca.labels_, s=20, cmap='viridis')
tab = pd.crosstab(y,  kmeans_pca.labels_, margins=True)
tab.index = ['walking', 'going up the stairs',
             'going down the stairs', 'sitting', 'standing', 'laying', 'all']
tab.columns = ['cluster' + str(i + 1) for i in range(6)] + ['all']
tab
inertia = []
for k in tqdm_notebook(range(1, n_classes + 1)):
    kmeans = KMeans(n_clusters=k, random_state=1).fit(x)
    inertia.append(np.sqrt(kmeans.inertia_))
plt.plot(range(1, 7), inertia, marker='s');
ag = AgglomerativeClustering(n_clusters=n_classes, 
                              linkage='ward').fit(principalComponents_x)
from sklearn.metrics import adjusted_rand_score
adjusted_rand_score(y, ag.labels_)
adjusted_rand_score(y, kmeans_pca.labels_)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
svc = LinearSVC(random_state=RANDOM_STATE)
svc_params = {'C': [0.001, 0.01, 0.1, 1, 10]}
%%time
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