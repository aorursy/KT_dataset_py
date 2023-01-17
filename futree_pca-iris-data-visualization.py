# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/iris/Iris.csv')

df.rename(columns=

          {"SepalLengthCm": "sepal length", 

           "SepalWidthCm": "sepal width",

           "PetalLengthCm": "petal length",

           "PetalWidthCm": "petal width",

           "Species": "target"}, inplace = True)

df.drop(['Id'],axis=1)
df.info()
from sklearn.preprocessing import StandardScaler



features = ['sepal length', 'sepal width', 'petal length', 'petal width']



# Separating out the features

x = df.loc[:, features].values



# Separating out the target

y = df.loc[:,['target']].values



# Standardizing the features

x = StandardScaler().fit_transform(x)

x = pd.DataFrame(x)

x
from sklearn.decomposition import PCA



pca = PCA(n_components=2)

principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data = principalComponents

             , columns = ['PC1', 'PC2'])

principalDf
# concatenate PCs and target

finalDf = pd.concat([principalDf, df[['target']]], axis = 1)

finalDf
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8,8))

ax = fig.add_subplot(1,1,1) 

ax.set_xlabel("Principal Component 1") 

ax.set_ylabel("Principal Component 2") 

ax.set_title('2 component PCA') 



targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

colors = ['r', 'g', 'b']

# zip() maps similar indexes of diff containers

for target, color in zip(targets,colors):

 indicesToKeep = finalDf['target'] == target

 ax.scatter(finalDf.loc[indicesToKeep, 'PC1']

 , finalDf.loc[indicesToKeep, 'PC2']

 , c = color

 , s = 50)

ax.legend(targets)

ax.grid()
pca.explained_variance_ratio_