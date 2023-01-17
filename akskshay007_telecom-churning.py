import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import math

%matplotlib inline 
df=pd.read_csv("../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df.info()
df.describe()
#DEMOGRAPHIC DATA DROPPED 

list1=["customerID",'gender','Partner',"Dependents",'SeniorCitizen','DeviceProtection','PaperlessBilling','PaymentMethod']

df.drop(list1,axis=1,inplace=True)

df.head()
#DEMOGRAPHIC PRESENTATION
sns.countplot(x="Churn",data=df)
#sns.countplot(x="Churn",hue="gender",data=df)
#sns.countplot(x="Churn",hue="SeniorCitizen",data=df)
#sns.countplot(x="Churn",hue="Partner",data=df)
#sns.countplot(x="Churn",hue="Dependents",data=df)
#sns.countplot(x="Churn",hue="DeviceProtection",data=df)
#sns.countplot(x="Churn",hue="PaperlessBilling",data=df)
#sns.countplot(x="Churn",hue="PaymentMethod",data=df)
#Skewness of Data
sns.distplot(df["tenure"])
df["tenure"].skew()
df["MonthlyCharges"].skew()
df['TotalCharges'].replace(" ",np.NaN,inplace=True)

df['TotalCharges']=df['TotalCharges'].astype(float)

df['TotalCharges'].replace(np.NaN,np.mean(df["TotalCharges"]),inplace=True)

sns.distplot(df["TotalCharges"])
df["TotalCharges"].skew()
sns.distplot(np.log(df["TotalCharges"]))
np.log(df["TotalCharges"]).skew()
sns.distplot(np.sqrt(df["TotalCharges"]))
np.sqrt(df["TotalCharges"]).skew()
sns.distplot(np.cbrt(df["TotalCharges"]))
np.cbrt(df["TotalCharges"]).skew()
#AS WE CAN SEE,CUBE ROOT RETURNS THE MOST LESS SKEWED DATA HERE AND IS CLSER TO THE BELL CURVE THAN ANY OTHER METHOD LIKE SQRT AND LOG
df["TotalCharges"]=np.cbrt(df["TotalCharges"])

df["TotalCharges"].head()
df["TotalCharges"].skew()
#Null Value Treatment
df.isnull().sum()
#outlier treatment

sns.boxplot(x="tenure",data=df)

plt.show()
sns.boxplot(x="MonthlyCharges",data=df)

plt.show()
sns.boxplot(x="TotalCharges",data=df)

plt.show()
# No outlier found 
#Categorical Data TREATMENT

from sklearn.preprocessing import LabelEncoder

labelencoder=LabelEncoder()
df["PhoneService"]=labelencoder.fit_transform(df["PhoneService"])

df.head()
df["MultipleLines"]=labelencoder.fit_transform(df["MultipleLines"])
df["InternetService"]=labelencoder.fit_transform(df["InternetService"])
df["OnlineSecurity"]=labelencoder.fit_transform(df["OnlineSecurity"])
df["OnlineBackup"]=labelencoder.fit_transform(df["OnlineBackup"])
df["TechSupport"]=labelencoder.fit_transform(df["TechSupport"])
df["StreamingMovies"]=labelencoder.fit_transform(df["StreamingMovies"])
df["StreamingTV"]=labelencoder.fit_transform(df["StreamingTV"])
df["Contract"]=labelencoder.fit_transform(df["Contract"])
df.head()
Churn_d=pd.get_dummies(df['Churn'],drop_first=True)

Churn_d.head(5)
df.head()
#FEATURE SCALING
from sklearn.model_selection import train_test_split

x=df.drop("Churn",axis=1)

y=Churn_d

#from sklearn.cross_validation import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
from sklearn.preprocessing import StandardScaler

sc_X=StandardScaler()

X_train=sc_X.fit_transform(x_train)

X_test=sc_X.fit_transform(x_test)

print(X_train)
#TRAINING OF DATA
#logistics regression
from sklearn.linear_model import LogisticRegression

logmodel=LogisticRegression()

logmodel.fit(X_train,y_train)
#EVALUATION OF THE MODEL
predictions=logmodel.predict(X_test)
from sklearn.metrics import confusion_matrix

res=confusion_matrix(y_test,predictions)

res
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,predictions)*100)
from sklearn.metrics import classification_report

classification_report(y_test,predictions)
#KNN

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score

from sklearn.metrics import accuracy_score
#classofier

classifier=KNeighborsClassifier(n_neighbors=19,p=2,metric="euclidean")

classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix

res=confusion_matrix(y_test,y_pred)

res
from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred)
from sklearn.metrics import classification_report

classification_report(y_test,y_pred)
#decision Tree
from sklearn.tree import DecisionTreeClassifier

classifier_entropy=DecisionTreeClassifier(criterion="entropy",random_state=100,max_depth=3)

#create the model

classifier_entropy.fit(X_train,y_train)
#pediction

y_pred=classifier_entropy.predict(X_test)

print(y_pred)


print("accuracy is :",accuracy_score(y_test,y_pred)*100) 
print(confusion_matrix(y_test,y_pred))
from sklearn.metrics import classification_report

classification_report(y_test,y_pred)
#import pandas as pd

#WA_Fn_UseC__Telco-Customer-Churn = pd.read_csv("../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")