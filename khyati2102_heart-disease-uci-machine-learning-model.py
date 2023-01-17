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
data=pd.read_csv("../input/heart-disease-uci/heart.csv")
data.head()
data.isnull().values.any()
data.info()
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
cormat=data.corr()

cor=cormat.index

plt.Figure(figsize=(40,40))

g=sns.heatmap(data[cor].corr(),cmap='RdYlGn')
data.var()
X=data.iloc[:,0:13]

y=data.iloc[:,:-1]
from sklearn.preprocessing import StandardScaler

ss=StandardScaler()

X_scaled=ss.fit_transform(X)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,random_state=42)
from sklearn.ensemble import RandomForestRegressor

rc=RandomForestRegressor(n_estimators=12,random_state=0)

rc.fit(X_train,y_train)
print("training set score {:.2f}".format(rc.score(X_train,y_train)))

print("test set score {:.2f}".format(rc.score(X_test,y_test)))