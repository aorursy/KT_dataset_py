import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix,accuracy_score
data=pd.read_csv('../input/social-network-ads/Social_Network_Ads.csv')

data.head()
data.describe()
data.isnull().sum()
x=data.iloc[:,[2,3]]

y=data.iloc[:,-1]
x.head()
y.head()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20)
sc=StandardScaler()

x_train=sc.fit_transform(x_train)

x_test=sc.transform(x_test)
classifier=DecisionTreeClassifier(criterion="entropy")

classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)

y_pred
accuracy_score=accuracy_score(y_test,y_pred)

accuracy_score
confusion_matrix=confusion_matrix(y_test,y_pred)

confusion_matrix