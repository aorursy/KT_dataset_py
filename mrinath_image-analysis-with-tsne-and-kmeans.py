import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt #for plotting out the graphs

from sklearn import datasets  # used to load the dataset
from sklearn import manifold  # for creating folds in the data

from sklearn.cluster import KMeans # for applying Kmeans clustering

%matplotlib inline
data = datasets.fetch_openml(
 'mnist_784',
 version=1,
 return_X_y=True
)
pixel_values, targets = data
targets=targets.astype(int)
tsne=manifold.TSNE(n_components=2)
transformed_data = tsne.fit_transform(pixel_values[:3000, :])# only using the first 3000 images
tsne_df = pd.DataFrame(
np.column_stack((transformed_data, targets[:3000])),
columns=["x", "y", "targets"]
)
tsne_df.loc[:, "targets"] = tsne_df.targets.astype(int)
tsne_df.x=tsne_df.x/(max(tsne_df.x)) 
tsne_df.y=tsne_df.y/(max(tsne_df.y)) # scaling the tsne data to 0-1
tsne_df[:10]
grid = sns.FacetGrid(tsne_df, hue="targets", size=8)
grid.map(plt.scatter, "x", "y").add_legend()

X=tsne_df[['x','y']]
X.x=X.x/(max(X.x))
X.y=X.y/(max(X.y))
kmeans=KMeans(n_clusters=10, init='k-means++',random_state=4)
y_pred=kmeans.fit_predict(X)
X=X.values
plt.figure(figsize=(10,10))
plt.scatter(X[y_pred==0,0],X[y_pred==0,1],c='red',label='cluster 0')
plt.scatter(X[y_pred==1,0],X[y_pred==1,1],c='blue',label='cluster 1')
plt.scatter(X[y_pred==2,0],X[y_pred==2,1],c='green',label='cluster 2')
plt.scatter(X[y_pred==3,0],X[y_pred==3,1],c='yellow',label='cluster 3')
plt.scatter(X[y_pred==4,0],X[y_pred==4,1],c='orange',label='cluster 4')
plt.scatter(X[y_pred==5,0],X[y_pred==5,1],c='cyan',label='cluster5 ')
plt.scatter(X[y_pred==6,0],X[y_pred==6,1],c='black',label='cluster 6')
plt.scatter(X[y_pred==7,0],X[y_pred==7,1],c='magenta',label='cluster 7')
plt.scatter(X[y_pred==8,0],X[y_pred==8,1],c='brown',label='cluster 8')
plt.scatter(X[y_pred==9,0],X[y_pred==9,1],c='pink',label='cluster 9')
plt.legend()
plt.show()
plt.figure(figsize=(10,10))
plt.scatter(X[tsne_df.targets==0,0],X[tsne_df.targets==0,1],c='red')
plt.scatter(X[tsne_df.targets==1,0],X[tsne_df.targets==1,1],c='blue')
plt.scatter(X[tsne_df.targets==2,0],X[tsne_df.targets==2,1],c='green')
plt.scatter(X[tsne_df.targets==3,0],X[tsne_df.targets==3,1],c='yellow')
plt.scatter(X[tsne_df.targets==4,0],X[tsne_df.targets==4,1],c='orange')
plt.scatter(X[tsne_df.targets==5,0],X[tsne_df.targets==5,1],c='cyan')
plt.scatter(X[tsne_df.targets==6,0],X[tsne_df.targets==6,1],c='black')
plt.scatter(X[tsne_df.targets==7,0],X[tsne_df.targets==7,1],c='magenta')
plt.scatter(X[tsne_df.targets==8,0],X[tsne_df.targets==8,1],c='brown')
plt.scatter(X[tsne_df.targets==9,0],X[tsne_df.targets==9,1],c='pink')
plt.show()