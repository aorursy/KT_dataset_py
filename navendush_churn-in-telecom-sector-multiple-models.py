# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
tel = pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
tel.info()
tel.describe()
tel.head()
tel.corr()
print(tel.isnull().sum())
tel.columns
#Replacing spaces with null values in total charges column

tel['TotalCharges'] = tel["TotalCharges"].replace(" ",np.nan)



#Dropping null values from total charges column which contain .15% missing data 

tel = tel[tel["TotalCharges"].notnull()]

tel = tel.reset_index()[tel.columns]



#convert to float type

tel["TotalCharges"] = tel["TotalCharges"].astype(float)
gender_num = pd.get_dummies(tel['gender'],drop_first=True,prefix='Gender')

Dependents_num = pd.get_dummies(tel['Dependents'],drop_first=True,prefix='Dependents')

Partner_num = pd.get_dummies(tel['Partner'],drop_first=True,prefix='Partner')

PhoneService_num = pd.get_dummies(tel['PhoneService'],drop_first=True,prefix='PhoneService')

OnlineBackup_num = pd.get_dummies(tel['OnlineBackup'],drop_first=True,prefix='OnlineBackup')

DeviceProtection_num = pd.get_dummies(tel['DeviceProtection'],drop_first=True,prefix='DeviceProtection')

TechSupport_num = pd.get_dummies(tel['TechSupport'],drop_first=True,prefix='TechSupport')

StreamingTV_num = pd.get_dummies(tel['StreamingTV'],drop_first=True,prefix='StreamingTV')

StreamingMovies_num = pd.get_dummies(tel['StreamingMovies'],drop_first=True,prefix='StreamingMovies')

Contract_num = pd.get_dummies(tel['Contract'],drop_first=True,prefix='Contract')

PaperlessBilling_num = pd.get_dummies(tel['PaperlessBilling'],drop_first=True,prefix='PaperlessBilling')

MultipleLines_num = pd.get_dummies(tel['MultipleLines'],drop_first=True,prefix='MultipleLines')

InternetService_num = pd.get_dummies(tel['InternetService'],drop_first=True,prefix='InternetService')

PaymentMethod_num = pd.get_dummies(tel['PaymentMethod'],drop_first=True,prefix='PaymentMethod')

Churn_num = pd.get_dummies(tel['Churn'],drop_first=True,prefix='Churn')

OnlineSecurity_num = pd.get_dummies(tel['OnlineSecurity'],drop_first=True,prefix='OnlineSecurity')
tel = pd.concat([tel,gender_num,Partner_num,Dependents_num,

       PhoneService_num,MultipleLines_num,InternetService_num,

       OnlineSecurity_num, OnlineBackup_num, DeviceProtection_num, TechSupport_num,

       StreamingTV_num, StreamingMovies_num, Contract_num, PaperlessBilling_num,

       PaymentMethod_num,Churn_num],axis=1)
tel.drop(['gender','Partner','Dependents',

       'PhoneService','MultipleLines','InternetService',

       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',

       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',

       'PaymentMethod','Churn'],axis=1,inplace=True)
tel.drop(['customerID'],axis=1,inplace=True)
te = tel[tel.columns[-1]]
te
tr = tel[tel.columns[:-1]]
tr.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(tr, te, test_size=0.30, random_state=101)
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,rfc_pred))

print('\n')

print(classification_report(y_test,rfc_pred))
from sklearn.svm import SVC

svm_model = SVC()

svm_model.fit(X_train,y_train)
svm_predictions = svm_model.predict(X_test)
print(confusion_matrix(y_test,svm_predictions))

print('\n')

print(classification_report(y_test,svm_predictions))
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 

grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
grid.fit(X_train,y_train)
grid.best_params_
grid.best_estimator_
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))

print('\n')

print(classification_report(y_test,grid_predictions))
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(tel.drop('Churn_Yes',axis=1))

scaled_features = scaler.transform(tel.drop('Churn_Yes',axis=1))
df_feat = pd.DataFrame(scaled_features,columns=tel.columns[:-1])

df_feat.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_feat,tel['Churn_Yes'],test_size=0.30, random_state =101)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)

pred = knn.predict(X_test)
print(confusion_matrix(y_test,pred))

print('\n')

print(classification_report(y_test,pred))
error_rate = []



# Will take some time

for i in range(1,40):

    

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))

plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=24)

knn.fit(X_train,y_train)

pred = knn.predict(X_test)
print(confusion_matrix(y_test,pred))

print('\n')

print(classification_report(y_test,pred))