import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Read the dataset into pandas dataframe
dataset = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')
dataset.head()
#Check for null values and data types of columns
dataset.info()
dataset['TotalCharges'] = pd.to_numeric(dataset['TotalCharges'],errors='coerce')
#No null values present. Now lets check some statistics for continuous variable columns
dataset.describe()
#Notice that TotalCharges column now has 11 missing values after its conversion to numeric
df = dataset[pd.isnull(dataset).any(axis=1)]
df
#For other records, total charges is somewhat close to the product of tenure and monthly charges. So i will be using 
# this formula to fill the missing values
dataset['TotalCharges'].fillna(dataset['tenure']*dataset['MonthlyCharges'],inplace=True)
df = dataset[pd.isnull(dataset).any(axis=1)]
df
#Check for outliers
plt.boxplot(dataset['tenure'])
plt.show()
plt.boxplot(dataset['MonthlyCharges'])
plt.show()
plt.boxplot(dataset['TotalCharges'])
plt.show()
#None of the three numeric columns have outliers
#Lets check the number of target values each of 'Yes' and 'No' in the churn flag
dataset['Churn'].value_counts()

#The ratio of Yes to No is almost 1:3. Not a bad spread of classification
dataset['Churn'].value_counts().plot(kind='bar')
plt.show()
set(dataset['Churn'])
#Lets begin by converting the categorical variables from string to integer
dataset.loc[dataset['Churn'] == 'Yes','Churn'] = 1
dataset.loc[dataset['Churn'] == 'No','Churn'] = 0
dataset['Churn'] = dataset['Churn'].astype(int)
set(dataset['gender'])
dataset.loc[dataset['gender'] == 'Male','gender'] = 1
dataset.loc[dataset['gender'] == 'Female','gender'] = 2
dataset['gender'] = dataset['gender'].astype(int)
dataset.loc[dataset['Partner'] == 'Yes','Partner'] = 1
dataset.loc[dataset['Partner'] == 'No','Partner'] = 0
dataset['Partner'] = dataset['Partner'].astype(int)
dataset.loc[dataset['Dependents'] == 'Yes','Dependents'] = 1
dataset.loc[dataset['Dependents'] == 'No','Dependents'] = 0
dataset['Dependents'] = dataset['Dependents'].astype(int)
set(dataset['PhoneService'])
dataset.loc[dataset['PhoneService'] == 'Yes','PhoneService'] = 1
dataset.loc[dataset['PhoneService'] == 'No','PhoneService'] = 0
dataset['PhoneService'] = dataset['PhoneService'].astype(int)
set(dataset['MultipleLines'])
dataset.loc[dataset['MultipleLines'] == 'No','MultipleLines'] = 0
dataset.loc[dataset['MultipleLines'] == 'Yes','MultipleLines'] = 1
dataset.loc[dataset['MultipleLines'] == 'No phone service','MultipleLines'] = 2
dataset['MultipleLines'] = dataset['MultipleLines'].astype(int)
set(dataset['InternetService'])
dataset.loc[dataset['InternetService'] == 'No','InternetService'] = 0
dataset.loc[dataset['InternetService'] == 'DSL','InternetService'] = 1
dataset.loc[dataset['InternetService'] == 'Fiber optic','InternetService'] = 2
dataset['InternetService'] = dataset['InternetService'].astype(int)
set(dataset['OnlineSecurity'])
dataset.loc[dataset['OnlineSecurity'] == 'No','OnlineSecurity'] = 0
dataset.loc[dataset['OnlineSecurity'] == 'Yes','OnlineSecurity'] = 1
dataset.loc[dataset['OnlineSecurity'] == 'No internet service','OnlineSecurity'] = 2
dataset['OnlineSecurity'] = dataset['OnlineSecurity'].astype(int)
set(dataset['OnlineBackup'])
dataset.loc[dataset['OnlineBackup'] == 'No','OnlineBackup'] = 0
dataset.loc[dataset['OnlineBackup'] == 'Yes','OnlineBackup'] = 1
dataset.loc[dataset['OnlineBackup'] == 'No internet service','OnlineBackup'] = 2
dataset['OnlineBackup'] = dataset['OnlineBackup'].astype(int)
set(dataset['DeviceProtection'])
dataset.loc[dataset['DeviceProtection'] == 'No','DeviceProtection'] = 0
dataset.loc[dataset['DeviceProtection'] == 'Yes','DeviceProtection'] = 1
dataset.loc[dataset['DeviceProtection'] == 'No internet service','DeviceProtection'] = 2
dataset['DeviceProtection'] = dataset['DeviceProtection'].astype(int)
set(dataset['TechSupport'])
dataset.loc[dataset['TechSupport'] == 'No','TechSupport'] = 0
dataset.loc[dataset['TechSupport'] == 'Yes','TechSupport'] = 1
dataset.loc[dataset['TechSupport'] == 'No internet service','TechSupport'] = 2
dataset['TechSupport'] = dataset['TechSupport'].astype(int)
set(dataset['StreamingTV'])
dataset.loc[dataset['StreamingTV'] == 'No','StreamingTV'] = 0
dataset.loc[dataset['StreamingTV'] == 'Yes','StreamingTV'] = 1
dataset.loc[dataset['StreamingTV'] == 'No internet service','StreamingTV'] = 2
dataset['StreamingTV'] = dataset['StreamingTV'].astype(int)
dataset.loc[dataset['StreamingMovies'] == 'No','StreamingMovies'] = 0
dataset.loc[dataset['StreamingMovies'] == 'Yes','StreamingMovies'] = 1
dataset.loc[dataset['StreamingMovies'] == 'No internet service','StreamingMovies'] = 2
dataset['StreamingMovies'] = dataset['StreamingMovies'].astype(int)
set(dataset['Contract'])
item_mapping = {"Month-to-month":1,"One year":2,"Two year":3}
dataset['Contract'] = dataset['Contract'].map(item_mapping)
dataset['Contract'] = dataset['Contract'].astype(int)
set(dataset['PaperlessBilling'])
dataset.loc[dataset['PaperlessBilling'] == 'No','PaperlessBilling'] = 0
dataset.loc[dataset['PaperlessBilling'] == 'Yes','PaperlessBilling'] = 1
dataset['PaperlessBilling'] = dataset['PaperlessBilling'].astype(int)
set(dataset['PaymentMethod'])
item_mapping = {"Bank transfer (automatic)":1,"Credit card (automatic)":2, "Electronic check":3,"Mailed check":4}
dataset['PaymentMethod'] = dataset['PaymentMethod'].map(item_mapping)
dataset['PaymentMethod'] = dataset['PaymentMethod'].astype(int)
dataset['Churn'] = dataset['Churn'].astype(int)
#How many senior citizens contained in the data
dataset['SeniorCitizen'].value_counts().plot(kind='bar')
plt.show()
#Are senior citizens prone to switching network providers as easily as the rest of the population
import seaborn as sns
plt.figure(figsize=(10,3))
g = sns.barplot(x='SeniorCitizen',y='Churn',data=dataset)
plt.show()
data1 = dataset[['SeniorCitizen','Churn']].groupby(['SeniorCitizen'],as_index=False).mean()
data1
#This implies there are a good number of senior citizens who are disappointed with a particular provider and have switched connections
#Lets check the impact of gender on churn flag
data1 = dataset[['gender','Churn']].groupby(['gender'],as_index=False).mean()
data1
#dataset['TotalCharges'] = dataset['TotalCharges'].astype(float)
dataset['MonthlyCharges'] = dataset['MonthlyCharges'].astype(float)
#Lets first see how the values Total Charges and Monthly CHarges have an impact on Churn
data1 = dataset[['MonthlyCharges','Churn']].groupby(['Churn'],as_index=False).mean()
data1
#The monthly charges are a little higher in the case of Churn = 1. THis could be one reason for the shift
#We should probably look at why the charges are high, i.e. what services have been acquired for the same
data1 = dataset[['tenure','Churn']].groupby(['Churn'],as_index=False).mean()
data1
#The average tenure for churn = 1 is lower than that of churn = 0. So the company is possibly retaining its old customers
query = dataset[dataset['Churn'] == 1]
query
g = sns.barplot(x='InternetService',y='Churn',data=dataset)
plt.show()
g = sns.barplot(x='PhoneService',y='Churn',data=dataset)
plt.show()
g = sns.barplot(x='MultipleLines',y='Churn',data=dataset)
plt.show()
g = sns.barplot(x='OnlineSecurity',y='Churn',data=dataset)
plt.show()
g = sns.barplot(x='OnlineBackup',y='Churn',data=dataset)
plt.show()
g = sns.barplot(x='Partner',y='Churn',data=dataset)
plt.show()
g = sns.barplot(x='Dependents',y='Churn',data=dataset)
plt.show()
g = sns.barplot(x='DeviceProtection',y='Churn',data=dataset)
plt.show()
g = sns.barplot(x='TechSupport',y='Churn',data=dataset)
plt.show()
g = sns.barplot(x='StreamingTV',y='Churn',data=dataset)
plt.show()
g = sns.barplot(x='StreamingMovies',y='Churn',data=dataset)
plt.show()
g = sns.barplot(x='Contract',y='Churn',data=dataset)
plt.show()
g = sns.barplot(x='PaperlessBilling',y='Churn',data=dataset)
plt.show()
g = sns.barplot(x='PaymentMethod',y='Churn',data=dataset)
plt.show()
g = sns.barplot(x='SeniorCitizen',y='Churn',data=dataset)
plt.show()
g = sns.barplot(x='gender',y='Churn',data=dataset)
plt.show()
#For now, we consider the following columns:
#Payment Method
#Paperless Billing
#Contract
#Techsupport, DeviceProtection
#Partnet
#Dependents
#Online Security, Online Backup
#InternetService
#Totalcharges
#SeniorCitizen
from sklearn.model_selection import train_test_split
cols = ['SeniorCitizen','Contract','PaymentMethod','TechSupport','DeviceProtection','Partner','Dependents','OnlineSecurity','OnlineBackup','InternetService','PaperlessBilling','TotalCharges']
X = dataset[cols]
y = dataset['Churn']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.27,random_state=0)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

clf = DecisionTreeClassifier(criterion = "gini", max_depth=5, min_samples_leaf=4,random_state=100)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

accuracy_score(y_test,y_pred,normalize=True)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=1000,random_state=32,max_depth=5)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

accuracy_score(y_test,y_pred,normalize=True)
from sklearn.metrics import roc_curve,auc

y_pred_sample_score = clf.predict_proba(X_test)

fpr,tpr,thresholds = roc_curve(y_test,y_pred_sample_score[:,1])
roc_auc = auc(fpr,tpr)

print(roc_auc)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(penalty='l2',C=1.0)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

accuracy_score(y_test,y_pred,normalize=True)
y_pred_sample_score = clf.predict_proba(X_test)

fpr,tpr,thresholds = roc_curve(y_test,y_pred_sample_score[:,1])
roc_auc = auc(fpr,tpr)

print(roc_auc)