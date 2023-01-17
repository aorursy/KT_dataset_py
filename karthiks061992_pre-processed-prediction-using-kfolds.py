# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataset=pd.read_csv("/kaggle/input/insurance/insurance.csv")

dataset.head()

#dataset.isnull().sum()

#No null values
X=dataset.iloc[:,:-1]

Y=dataset.iloc[:,6]

X.head()

X.region.value_counts()
import matplotlib.pyplot as plt

import seaborn as sn

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1,4,5])],remainder="passthrough")

X=np.array(ct.fit_transform(X))

#print(pd.DataFrame(X))

from sklearn.preprocessing import StandardScaler

scale=StandardScaler()

X=scale.fit_transform(X)

Y=np.array(Y)

Y=Y.reshape(-1,1)

Y=scale.fit_transform(Y)

print(pd.DataFrame(Y))

Y=pd.DataFrame(Y)

X=pd.DataFrame(X)

X.head()

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression

regressor=LinearRegression()

regressor.fit(xtrain,ytrain)

ypred=regressor.predict(xtest)

from sklearn.metrics import r2_score

print(r2_score(ytest,ypred)*100)
from sklearn.model_selection import cross_val_score

print(cross_val_score(estimator=regressor,X=xtrain,y=ytrain,cv=10))

accuracies=cross_val_score(estimator=regressor,X=xtrain,y=ytrain,cv=10)

#These are the accuracies for 10 different crossfolds 
print("accuracies:{:.2f}".format(accuracies.mean()*100))

#Mean accuracies using k-cross folds 