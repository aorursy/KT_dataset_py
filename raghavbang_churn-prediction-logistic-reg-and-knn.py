import pandas as pd

import numpy as np

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, accuracy_score
pd.set_option("display.max_columns",500)
data_final=pd.read_csv("../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
data=data_final.copy()

data["TotalCharges"]=pd.to_numeric(data["TotalCharges"], errors='coerce')

data.info()

data.describe()
data.describe(include="O")

data.isnull().sum()

data["TotalCharges"].describe()

data["TotalCharges"].fillna(1397.4750,inplace=True)

data.columns

data=data.drop(["customerID"],axis=1)
data["Churn"].value_counts()

data.corr()
data2=pd.get_dummies(data,drop_first=True)

data2.columns

colmn=list(data2.columns)

feature= list(set(colmn)-set(["Churn_Yes"])) 

print(feature)

y=data2["Churn_Yes"].values 

print(y)



x=data2[feature].values

print(x)
from  sklearn.preprocessing import StandardScaler

sn= StandardScaler();

x=sn.fit_transform(x)
train_x,test_x,train_y,text_y=train_test_split(x,y, test_size=0.3 ,random_state=None)

logistic=LogisticRegression() 

logistic.fit(train_x,train_y) 



p=logistic.predict(test_x) 



confusion_matrix=confusion_matrix(text_y,p)

print(confusion_matrix)

acc=accuracy_score(text_y,p)

print(acc)

data2.describe()



from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt

for i in range(1,30):

 Kn= KNeighborsClassifier(n_neighbors=i)

 Kn.fit(train_x,train_y)



 p=Kn.predict(test_x)

 acc=accuracy_score(text_y,p)

 print(acc)

 print(i)


