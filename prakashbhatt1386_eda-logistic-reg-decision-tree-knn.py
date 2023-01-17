import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
data = pd.read_csv("../input/datanew/heart_failure.csv")
data.head(5)
data.describe()
data.isnull().sum()
data['DEATH_EVENT'].value_counts()
data.loc[data['age'] <= 60, 'Age_category'] = 'Before_retire' 

data.loc[data['age'] > 60, 'Age_category'] = 'After_retire' 



print (data)
data.head()
data['Age_category'].value_counts()
pd.crosstab(data.DEATH_EVENT,data.Age_category)
pd.crosstab(data.DEATH_EVENT,data.Age_category).plot(kind='bar', stacked=True)

           

           
data_new= data.drop('Age_category',axis=1)


x =  data_new.drop('DEATH_EVENT',axis=1)

y= data_new['DEATH_EVENT']
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler_data= scaler.fit_transform(x)

scaler_data
x_s= pd.DataFrame(scaler_data,columns=x.columns[:])
x_s.head(5)
from sklearn.model_selection import train_test_split
x_sTrain,x_sTest,y_Train,y_Test = train_test_split(x_s,y,test_size=.30,random_state=0)
from sklearn.linear_model import LogisticRegression

logmodel= LogisticRegression()

logmodel.fit(x_sTrain,y_Train)
Predction = logmodel.predict(x_sTest)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_Test,Predction))
print(confusion_matrix(y_Test,Predction))
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(x_sTrain,y_Train)
predict_1 = dtree.predict(x_sTest)
print(classification_report(y_Test,predict_1))
print(confusion_matrix(y_Test,predict_1))
from sklearn.neighbors import KNeighborsClassifier
knn =KNeighborsClassifier(n_neighbors=1)
knn.fit(x_sTrain,y_Train)
predict_2 = knn.predict(x_sTest)
print(classification_report(y_Test,predict_2))

print(confusion_matrix(y_Test,predict_2))
knn =KNeighborsClassifier(n_neighbors=5)
knn.fit(x_sTrain,y_Train)
predict_3 = knn.predict(x_sTest)
print(classification_report(y_Test,predict_3))

print(confusion_matrix(y_Test,predict_3))