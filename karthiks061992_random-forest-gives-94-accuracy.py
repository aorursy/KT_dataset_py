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
dataset=pd.read_csv("/kaggle/input/social-network-ads/Social_Network_Ads.csv")

dataset.head()

features=['Gender','Age','EstimatedSalary','Purchased']

dataset=dataset[features]

dataset.head()
#check for null values

#No Null values present

dataset.isnull().sum()
X=dataset.iloc[:,:-1]

Y=dataset.iloc[:,3]

X.head()
from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')

X=ct.fit_transform(X)

X=pd.DataFrame(X)

X.head()

#Applying standard scaler

from sklearn.preprocessing import StandardScaler

scale=StandardScaler()

X=scale.fit_transform(X)

X=pd.DataFrame(X)

X.head()

X.shape
Y.shape
from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2)

from sklearn.linear_model import LogisticRegression

regressor=LogisticRegression()

regressor.fit(xtrain,ytrain)

ypred=regressor.predict(xtest)
from sklearn.metrics import accuracy_score

accuracy_score(ytest,ypred)*100
#using RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier

classifier=RandomForestClassifier()

classifier.fit(xtrain,ytrain)

ypred1=classifier.predict(xtest)
from sklearn.metrics import accuracy_score

accuracy_score(ytest,ypred1)*100