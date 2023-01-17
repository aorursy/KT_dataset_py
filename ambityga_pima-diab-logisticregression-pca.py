# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split



# Any results you write to the current directory are saved as output.
diab = pd.read_csv('../input/diabetes.csv')
X = diab.iloc[:,:7]

y = diab.iloc[:,8]
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.33, random_state=42)
model = LogisticRegression()
model.fit(X_train,y_train)
model.score(X_test,y_test)
from sklearn.decomposition import PCA
pca_diab = PCA(n_components=2)

#decreasing 7 features to 2 features
X1= pca_diab.fit_transform(X)

#fit so that it could find relations and transformed into 2 features

#Namely =PCA1 and PCA2
data_frame = pd.DataFrame(data = X1,columns = ['PCA1','PCA2'])
data_frame.head()
X_train,X_test,y_train,y_test = train_test_split(X1, y, test_size=0.33, random_state=42)
model_new = LogisticRegression()
model_new.fit(X_train,y_train)
model_new.score(X_test,y_test)