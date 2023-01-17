# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd

import sklearn 

import matplotlib.pyplot as plt



from sklearn.decomposition import PCA

from sklearn.decomposition import IncrementalPCA



from sklearn.linear_model import LogisticRegression



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dt_heart = pd.read_csv('/kaggle/input/heart.csv')

dt_heart.head()
if __name__ == "__main__":

    



    print(dt_heart.head(5))



    dt_features  = dt_heart.drop(['target'], axis=1)

    dt_target = dt_heart['target']



    dt_features = StandardScaler().fit_transform(dt_features)



    X_train, X_test, y_train, y_test = train_test_split(dt_features, dt_target, test_size=0.3, random_state=42)

print(X_train.shape)

print(y_train.shape)
y_train.head()
pca = PCA(n_components=3)

pca.fit(X_train)



ipca = IncrementalPCA(n_components=3, batch_size=3)

ipca.fit(X_train)



plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_ratio_)

plt.show()



logistic = LogisticRegression(solver='lbfgs')



dt_train = pca.transform(X_train)

dt_test = pca.transform(X_test)

logistic.fit(dt_train,y_train)

print("SCORE PCA: ", logistic.score(dt_test, y_test))



dt_train = ipca.transform(X_train)

dt_test = ipca.transform(X_test)

logistic.fit(dt_train, y_train)

print("SCORE IPCA: ", logistic.score(dt_test, y_test))



#implementacion_algoritmo_pca
print(X_train.shape)

print(y_train.shape)