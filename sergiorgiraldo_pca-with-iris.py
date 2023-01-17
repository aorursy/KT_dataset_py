import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

%matplotlib inline
# loading dataset into Pandas DataFrame

df = pd.read_csv("../input/iris.data"

                 , names=['sepal length','sepal width','petal length','petal width','target'])
df.head()
features = ['sepal length', 'sepal width', 'petal length', 'petal width']

x = df.loc[:, features].values

x
y = df.loc[:,['target']].values

y
x = StandardScaler().fit_transform(x)

x
pd.DataFrame(data = x, columns = features).head()
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents

             , columns = ['principal component 1', 'principal component 2'])
principalDf.head(5)
df[['target']].head()
finalDf = pd.concat([principalDf, df[['target']]], axis = 1)

finalDf.head(5)
fig = plt.figure(figsize = (8,8))

ax = fig.add_subplot(1,1,1) 

ax.set_xlabel('Principal Component 1', fontsize = 15)

ax.set_ylabel('Principal Component 2', fontsize = 15)

ax.set_title('2 Component PCA', fontsize = 20)





targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

colors = ['r', 'g', 'b']

for target, color in zip(targets,colors):

    indicesToKeep = finalDf['target'] == target

    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']

               , finalDf.loc[indicesToKeep, 'principal component 2']

               , c = color

               , s = 50)

ax.legend(targets)

ax.grid()
pca.explained_variance_ratio_