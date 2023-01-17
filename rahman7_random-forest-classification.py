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
# import Librareis:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

# import dataset:
df=pd.read_csv("../input/Social_Network_Ads.csv")
df.head()
# visualization:
sns.pairplot(df)
sns.boxplot(df['EstimatedSalary'],hue='Age',data=df)
sns.scatterplot(df['EstimatedSalary'],df['Age'],data=df)
sns.distplot(df['EstimatedSalary'],bins=50,kde=False)
sns.distplot(df['Age'],bins=50,kde=False)
sns.boxplot(x=df['EstimatedSalary'],y=df['Age'],hue='Gender',data=df)
# spili the dataset into dependent and independent:
X=df.iloc[:,:-1].values
y=df.iloc[:,4].values
X
y
# ecoding:
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()
X[:,1]=labelencoder.fit_transform(X[:,1])
onehotencoder=OneHotEncoder(categorical_features=[1])
X=onehotencoder.fit_transform(X).toarray()
X=X[:,1:]
X
# spliting the dataset into train and test set:
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
# feature sacling:
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)
# fitting the dataset into the Random_forest:
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)
# predictions:
y_pred=classifier.predict(X_test)
y_pred
y_test
#making the matrix:
from sklearn.metrics import confusion_matrix,classification_report
cm=confusion_matrix(y_test,y_pred)

cm
cr=classification_report(y_test,y_pred)
cr
(64+30)/100
