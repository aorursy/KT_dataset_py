# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# import libraries:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# import dataset:
df=pd.read_csv("../input/Social_Network_Ads.csv")
df.head()
# visualization:
sns.pairplot(df)
sns.scatterplot(df['EstimatedSalary'],df['Age'],data=df)
sns.scatterplot(df['EstimatedSalary'],df['Gender'],data=df)
sns.countplot(df['EstimatedSalary'],data=df,hue='Gender')
sns.countplot(df['EstimatedSalary'],data=df,hue='Age')
sns.distplot(df['EstimatedSalary'],bins=50,kde=False)
sns.distplot(df['Age'],bins=20,kde=False)
# spliting the dataset into dependent and independent:
X=df.iloc[:,:-1].values
y=df.iloc[:,4].values
X
y
# encodeding:
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()
X[:,1]=labelencoder.fit_transform(X[:,1])
onehotencoder=OneHotEncoder(categorical_features=[1])
X=onehotencoder.fit_transform(X).toarray()
X=X[:,1:]
X
# spliting thwe datset into train and test:
from sklearn.model_selection import  train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

# featuer scalling:
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)
# fiting the datset on Naives_Bayes
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)
# prediction:
y_pred=classifier.predict(X_test)
y_pred
y_test
# matrix:
from sklearn.metrics import classification_report,confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm
cr=classification_report(y_test,y_pred)
cr
(64+27)/100
