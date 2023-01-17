# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("../input"))
data=pd.read_csv("../input/camera_dataset.csv")
data.head()
df_corr=data.corr(method='pearson')
fig, ax = plt.subplots(figsize=(8,8))
sns.heatmap(df_corr, annot=True, ax=ax)
plt.title('Correlation for camera attributes')
plt.show()
fig= plt.figure(figsize=(15,15))
ax2=fig.add_subplot(331)
plt.scatter(data['Low resolution'], data['Max resolution'])
plt.title('Low ressolution & High Resolution')
plt.xlabel('Low Resolution')
plt.ylabel('Max Resolution')
ax2=fig.add_subplot(332)
plt.scatter(data["Effective pixels"], data['Max resolution'])
plt.title('Effective pixels & Max resolution ')
plt.xlabel('Effective pixels')
plt.ylabel('Max resolution')
ax2=fig.add_subplot(333)
plt.scatter(data['Release date'],data['Effective pixels'])
plt.title('Release date & Effective pixels')
plt.xlabel('Release date')
plt.ylabel('Effective pixels')
plt.show()
data.fillna(0, inplace=True)
from sklearn.cluster import KMeans
import sklearn.metrics as sm
chosen=['Release date','Max resolution','Low resolution','Effective pixels','Zoom wide (W)','Zoom tele (T)','Normal focus range','Macro focus range','Storage included','Weight (inc. batteries)','Dimensions','Price']
X=data[chosen].values
model = KMeans(n_clusters=2)
model.fit(X)
colormap = np.array(['pink','purple'])
plt.scatter(data['Zoom wide (W)'], data['Low resolution'],c=colormap[model.labels_], s=40)
plt.title('K Mean Classification')
plt.show()
plt.scatter(data['Zoom wide (W)'], data['Weight (inc. batteries)'],c=colormap[model.labels_], s=40)
plt.title('K Mean Classification')
plt.show()
fig2 = plt.figure(figsize=(15, 15))
ax6 = fig2.add_subplot(331)
plt.scatter(data['Effective pixels'], data['Max resolution'],c=colormap[model.labels_], s=40)
plt.title('effective pixels and max resolution')
plt.xlabel('Effective pixels')
plt.ylabel('Max resolution')
ax6 = fig2.add_subplot(332)
plt.scatter(data['Effective pixels'], data['Low resolution'],c=colormap[model.labels_], s=40)
plt.title('effective pixels and low resolution')
plt.xlabel('Effective pixels')
plt.ylabel('Low resolution')
ax6=fig2.add_subplot(333)
plt.scatter(data['Max resolution'], data['Low resolution'],c=colormap[model.labels_], s=40)
plt.title('max resolution and low resolution')
plt.xlabel('Max resolution')
plt.ylabel('Low resolution')
ax6=fig2.add_subplot(334)
plt.scatter(data['Zoom wide (W)'], data['Weight (inc. batteries)'],c=colormap[model.labels_], s=40)
plt.title('Zoom and weight')
plt.xlabel('Zoom')
plt.ylabel('weight')
ax6=fig2.add_subplot(335)
plt.scatter(data['Effective pixels'], data['Release date'],c=colormap[model.labels_], s=40)
plt.title('Effective pixels and release date')
plt.xlabel('Effective pixels')
plt.ylabel('Release date')
ax6=fig2.add_subplot(336)
plt.scatter(data['Max resolution'], data['Release date'],c=colormap[model.labels_], s=40)
plt.xlabel('Max resolution')
plt.ylabel('Release date')
ax6=fig2.add_subplot(337)
plt.scatter(data['Low resolution'], data['Release date'],c=colormap[model.labels_], s=40)
plt.xlabel('Low resolution')
plt.ylabel('Release date')
ax6=fig2.add_subplot(337)
plt.scatter(data['Weight (inc. batteries)'], data['Dimensions'],c=colormap[model.labels_], s=40)
plt.xlabel('Weight')
plt.ylabel('Dimensions')
plt.show()

data1=data.drop(['Model','Release date'],axis=1)
sse = {}
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(data1)
    data1["clusters"] = kmeans.labels_
    #print(data["clusters"])
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("Distortion")
plt.show()
from sklearn.preprocessing import StandardScaler
x = StandardScaler().fit_transform(X)
print(x)
print('NumPy covariance matrix: \n%s' %np.cov(x.T))
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
principalDf
pca = PCA().fit(X)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()
print(pca.explained_variance_ratio_)

plt.scatter(principalDf['principal component 1'],principalDf['principal component 2'],s=30,c='goldenrod',alpha=0.5)
plt.title('plotting both variables')
plt.xlabel('Principal component 1')
plt.ylabel('Principal component 2')
plt.show()
model = KMeans(n_clusters=3)
model.fit(principalDf)
colormap = np.array(['blue','red','yellow','orange','purple'])
plt.scatter(principalDf['principal component 1'], principalDf['principal component 2'],c=colormap[model.labels_], s=40)
plt.title('K Mean Classification')
plt.show()
