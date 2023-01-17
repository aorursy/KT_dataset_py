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
#import libraries:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import dataset :
df=pd.read_csv("../input/Churn_Modelling.csv")
df.head()
# make the data independent and dependent:
X=df.iloc[:,3:13].values
y=df.iloc[:,13].values
X
y
# encoding the dataset:
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencode_X_1=LabelEncoder()
X[:,1]=labelencode_X_1.fit_transform(X[:,1])
labelencode_X_2=LabelEncoder()
X[:,2]=labelencode_X_2.fit_transform(X[:,2])
onehotencoder=OneHotEncoder(categorical_features=[1])
X=onehotencoder.fit_transform(X).toarray()
X=X[:,1:]
# spliting the dataset  into train and test:
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
# feature scaling:
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)
# Fitting the XGboost for training set:
from xgboost import XGBClassifier
classifier=XGBClassifier()
classifier.fit(X_train,y_train)
# predictions:
y_pred=classifier.predict(X_test)
y_pred
y_test
# making the confusion matrix,classification_result:
from sklearn.metrics import classification_report,confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm
cr=classification_report(y_test,y_pred)
cr
(1524+207)/2000
# Apply k-flod cross_vaidation:
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)
accuracies.mean()
accuracies.std()
# dataset visualization:
df.head()
sns.pairplot(df,hue='Gender')
sns.countplot(df['Age'],hue='Gender',data=df)
sns.scatterplot(df['CreditScore'],df['Age'],hue='Gender',data=df)
sns.distplot(df['Age'],bins=20,kde=False)
sns.lineplot(df['CreditScore'],df['EstimatedSalary'],hue='Gender',data=df)
