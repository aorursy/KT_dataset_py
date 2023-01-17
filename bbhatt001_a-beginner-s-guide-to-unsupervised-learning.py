import numpy as np 
import pandas as pd 
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', None, 'display.max_columns', None)
data=pd.read_csv('../input/IRIS.csv')
data.head()
data.info()
data['species'].unique()
data.describe()
X=data.drop(['species'],axis=1)
X.shape
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X=scaler.fit_transform(X)
model=KMeans(n_clusters=3)
model.fit(X)
label=model.predict(X)
print(label)
x=X[:,0]
y=X[:,3]
plt.scatter(x,y,c=label)
species=data['species']
species.head()
df=pd.DataFrame({'label':label,'species':species})
ct= pd.crosstab(df['label'],df['species'])
print(ct)
Inertia = []
K = range(1,5)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(X)
    Inertia.append(km.inertia_)

plt.plot(K, Inertia, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.show()
from scipy.cluster.hierarchy import linkage,dendrogram, fcluster
fig, ax= plt.subplots(figsize=(18,8))
mergings=linkage(X,method='complete')
dendrogram(mergings,  leaf_rotation=90,leaf_font_size=7)
plt.axhline(y=5, c='grey', lw=1, linestyle='dashed')
plt.show()
fig, ax= plt.subplots(figsize=(8,8))
dendrogram(mergings, truncate_mode = 'level', p=3,leaf_font_size=7)
plt.show()
data=data.set_index('species')
fig, ax= plt.subplots(figsize=(18,8))
mergings=linkage(X,method='complete')
dendrogram(mergings, labels=data.index, leaf_rotation=90,leaf_font_size=7)
plt.show()
mergings=linkage(X,method='complete')
label_3 = fcluster(mergings, 3, criterion='distance')
pairs=pd.DataFrame({'labels':label_3, 'species':species})
pairs.sort_values('labels')
from sklearn.decomposition import PCA
pca=PCA().fit(X)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.xlim(0,5,1)
pca=PCA().fit(X)
features=range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xlabel('features')
plt.ylabel('variance')
pca=PCA(n_components=2)
X_pca=pca.fit_transform(X)
X_pca.shape
x_pca=X_pca[:,0]
y_pca=X_pca[:,1]
# to color code the clusters, I have number coded the species (this is the form accepted for color slection) 
d = {'Iris-versicolor':0, 'Iris-virginica':1, 'Iris-setosa':2}
labels = [d[spec] for spec in species]
plt.scatter(x_pca, y_pca, c=labels)
plt.xlabel('PCA_feature_1')
plt.ylabel('PCA_feature_2')
plt.title('Iris species after PCA' )
