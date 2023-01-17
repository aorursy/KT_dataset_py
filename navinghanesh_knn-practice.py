import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
data=pd.read_csv('../input/social-network-ads/Social_Network_Ads.csv')
data.describe()
data.isnull().sum()
x=data.iloc[:,[2,3]].values
x
y=data.iloc[:,-1].values
y
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
classifier=KNeighborsClassifier()
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
y_pred
accuracy_score=accuracy_score(y_test,y_pred)
accuracy_score
