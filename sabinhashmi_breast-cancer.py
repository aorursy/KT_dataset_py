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
data=pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')
data.shape
data.head()
data.isnull().sum()
data.info()
del data['Unnamed: 32']

del data['id']
x=data.drop('diagnosis',axis=1)

y=data['diagnosis']
y=y.map({'M':0,'B':1})
from scipy.stats import zscore
x=x.apply(zscore)
from sklearn.decomposition import PCA
pca=PCA()

pca.fit(x)

np.cumsum(pca.explained_variance_ratio_)
pca=PCA(n_components=10)

x=pd.DataFrame(pca.fit_transform(x),columns=['PCA-1','PCA-2','PCA-3','PCA-4','PCA-5','PCA-6','PCA-7','PCA-8','PCA-9','PCA-10'])
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=43)
from sklearn.linear_model import LogisticRegression

log=LogisticRegression()
log.fit(x_train,y_train)
y_pred=log.predict(x_test)
from sklearn.metrics import accuracy_score,classification_report
accuracy_score(y_test,y_pred)