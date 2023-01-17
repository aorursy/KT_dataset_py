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
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import train_test_split
data = pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")
data.head()
print("MISSING VALUES: \n\n ",data.isnull().sum())
print("--"*15)
print("SHAPE : ",data.shape)
#target variable visualizing
data["Churn"].value_counts().plot(kind = "pie")
print(data["Churn"].value_counts())

plt.figure(figsize=(14,12))

#gender influence
plt.subplot(2,2,1)
sns.countplot(data["Churn"],hue = data["gender"])


#Senior citizen influence on target variable
plt.subplot(2,2,2)
sns.countplot(data["Churn"],hue = data["SeniorCitizen"])



#Partner influence on target variable
plt.subplot(2,2,3)
sns.countplot(data["Churn"],hue = data["Partner"])


#dependents influence on target value
plt.subplot(2,2,4)
sns.countplot(data["Churn"],hue = data["Dependents"])

# GENDER
# 1 - male
# 0 - female


data["gender"].replace(["Male","Female"],[1,0],inplace =True)


#PARTNER
data["Partner"].replace(["Yes","No"],[1,0],inplace =True)


#DEPENDENTS
data["Dependents"].replace(["Yes","No"],[1,0],inplace =True)


#MULTIPLELINES
data["MultipleLines"].replace(["Yes","No","No phone service"],[1,0,2],inplace = True)


#PHONESERVICE
data["PhoneService"].replace(["Yes","No"],[1,0],inplace =True)


#INTERNET SERVICES
data["InternetService"].replace(["Fiber optic","DSL","No"],[0,1,2],inplace= True)


#CONTARCT
data["Contract"].replace(["Month-to-month","Two year","One year"],[0,1,2],inplace = True)


#TECH SUPPORT
data["TechSupport"].replace(["No","Yes","No internet service"],[0,1,2],inplace = True)


#ONLINE SECURITY
data["OnlineSecurity"].replace(["No","Yes","No internet service"],[0,1,2],inplace = True)


#DEVICE PROTECTION
data["DeviceProtection"].replace(["No","Yes","No internet service"],[0,1,2],inplace =True)


#STREAMING MOVIES
data["StreamingMovies"].replace(["No","Yes","No internet service"],[0,1,2],inplace =True)


#STREAMING TV
data["StreamingTV"].replace(["No","Yes","No internet service"],[0,1,2],inplace =True)


#ONLINEBACKUP
data["OnlineBackup"].replace(["No","Yes","No internet service"],[0,1,2],inplace =True)


#PAPERLESS BILLING
data["PaperlessBilling"].replace(["No","Yes"],[0,1],inplace =True)


#PAYMENT METHOD
data["PaymentMethod"].replace(["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"],[0,1,2,3],inplace =True)


#total Payment
data["TotalCharges"].replace([" "],[0],inplace = True)
data["TotalCharges"] = data["TotalCharges"].astype('float')



#churn
data["Churn"].replace(["No","Yes"],[0,1],inplace =True)





plt.figure(figsize = (16,10))
sns.heatmap(data.corr())
#dropping customer id columns Since that don't have any influence on target variable
data.drop("customerID",axis = 1,inplace = True)
#splitting of dataset
x = data.iloc[:,:-1]
y = data.iloc[:,-1]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2 ,random_state = 2)
#cross validation
def val(a,b,classifier):
    val = cross_val_score(classifier,a,b,scoring = "accuracy",cv = 10)
    return val
    
    
#accuarcy
def accuracy(a,b):
    ac = accuracy_score(a,b)
    cr = classification_report(a,b)
    print("accuracy score : ",ac)
    print("--"*15)
    print("Classification report :",cr)
    return ac
#KNN algorithm
knc = KNeighborsClassifier(n_neighbors=23)
knc.fit(x_train,y_train)
knc_pred = knc.predict(x_test)
acc_knn = accuracy(y_test,knc_pred)
#Decision Tree
dtc = DecisionTreeClassifier(random_state=2)
dtc.fit(x_train,y_train)
dtc_pred = dtc.predict(x_test)

acc_dtc = accuracy(y_test,dtc_pred)
#svc
svc = SVC(random_state = 2) 
svc.fit(x_train,y_train)
svc_pred = svc.predict(x_test)

acc_svc = accuracy(y_test,svc_pred)
z = pd.DataFrame(data = [acc_dtc,acc_knn,acc_svc],index = ['Decision Tree','KNN','SVC'],columns=[ "accuracy"])
#comparing results
z.plot(kind = 'bar',title="accuracy vs algorithm")
z
