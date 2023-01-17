#in questo notebook proverò ad utilizzare alcune funzioni di sklearn per applicare la Pricipal

#Component analysis  al dataset IRIS



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn import metrics

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/iris-flower-dataset/IRIS.csv')

data.head()

data.info()
data['species'].value_counts()
X = data.drop(['species'], axis=1)

y = data['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=5)

print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
#provo il k-neighbours classifier

from sklearn.neighbors import KNeighborsClassifier



k = 15

knn = KNeighborsClassifier(n_neighbors=k)

knn.fit(X, y)

y_pred = knn.predict(X)

print(metrics.accuracy_score(y, y_pred))
#PCA è soggetta allo scaling 

from sklearn.preprocessing import StandardScaler

XScaled = StandardScaler().fit_transform(X)



#traformo i dati 4-dimensionali in 2-dimensionali

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

pca = PCA(n_components=2)

principal_comp = pca.fit_transform(XScaled)



principalDf = pd.DataFrame(data = principal_comp

             , columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, data[['species']]], axis = 1)







fig = plt.figure(figsize = (8,8))

ax = fig.add_subplot(1,1,1) 

ax.set_xlabel('Principal Component 1', fontsize = 15)

ax.set_ylabel('Principal Component 2', fontsize = 15)

ax.set_title('2 component PCA', fontsize = 20)



targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

colors = ['r', 'g', 'b']

for target, color in zip(targets,colors):

    indicesToKeep = finalDf['species'] == target

    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']

               , finalDf.loc[indicesToKeep, 'principal component 2']

               , c = color

               , s = 50)

ax.legend(targets)

ax.grid()
#controllo quanta varianza è contenuta nelle due componenti

pca.explained_variance_ratio_