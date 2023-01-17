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
dataset=pd.read_csv("/kaggle/input/titanicdataset-traincsv/train.csv")

dataset.head()

features=['Survived','Pclass','Sex','Age','SibSp','Parch','Embarked']

dataset=dataset[features]

dataset.head()

dataset.dropna(inplace=True)

#split now into dependent and independent variables

X=dataset.iloc[:,[1,2,3,4,5,6]]

X.head()

Y=dataset.iloc[:,0]

Y.head()
#X['Pclass'].value_counts()

from sklearn.compose import ColumnTransformer# Pclass,Sex,Embarked

from sklearn.preprocessing import OneHotEncoder

ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0,1,5])],remainder="passthrough")

X=ct.fit_transform(X)

X=pd.DataFrame(X)

X.head()

#Data is completely converted and no need of standard scaler

#splitting into training and testing

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2)

from sklearn.linear_model import LogisticRegression

regressor=LogisticRegression()

regressor.fit(xtrain,ytrain)


ypred=regressor.predict(xtest)

ypred
from sklearn.metrics import accuracy_score

score=accuracy_score(ytest,ypred)

print(score*100)