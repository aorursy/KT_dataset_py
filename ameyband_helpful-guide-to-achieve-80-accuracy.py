import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('../input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv')
df.head()
#Look at the number and name of columns in dataset.
print(df.columns)
print(df.shape)
#Check the info() of the dataset whether all the columns in dataset have the same datatype or not.
df.info()
#Checking for missing values
df.isnull().sum()
#Check the distribution of data
df['default.payment.next.month'].value_counts().plot.bar()
df['SEX'].value_counts().plot.bar()
sns.distplot(df['AGE'],kde=True,bins=30)
df['EDUCATION'].value_counts().plot.bar()
df['MARRIAGE'].value_counts().plot.bar()
sns.countplot(x='SEX', data=df,hue="default.payment.next.month", palette="muted")
sns.countplot(x='EDUCATION',data=df,hue="default.payment.next.month",palette="muted")
sns.countplot(x='MARRIAGE',data=df,hue="default.payment.next.month", palette="muted")
df['PAY_0'].value_counts()
fill = (df.PAY_0 == 4) | (df.PAY_0==5) | (df.PAY_0==6) | (df.PAY_0==7) | (df.PAY_0==8)
df.loc[fill,'PAY_0']=4
df.PAY_0.value_counts()
fill = (df.PAY_2 == 4) | (df.PAY_2 == 1) | (df.PAY_2 == 5) | (df.PAY_2 == 7) | (df.PAY_2 == 6) | (df.PAY_2 == 8)
df.loc[fill,'PAY_2']=4
#df.PAY_2.value_counts()
fill = (df.PAY_3 == 4) | (df.PAY_3 == 1) | (df.PAY_3 == 5) | (df.PAY_3 == 7) | (df.PAY_3 == 6) | (df.PAY_3 == 8)
df.loc[fill,'PAY_3']=4
#df.PAY_3.value_counts()
fill = (df.PAY_4 == 4) | (df.PAY_4 == 1) | (df.PAY_4 == 5) | (df.PAY_4 == 7) | (df.PAY_4 == 6) | (df.PAY_4 == 8)
df.loc[fill,'PAY_4']=4
#df.PAY_4.value_counts()
fill = (df.PAY_5 == 4) | (df.PAY_5 == 7) | (df.PAY_5 == 5) | (df.PAY_5 == 6) | (df.PAY_5 == 8)
df.loc[fill,'PAY_5']=4
#df.PAY_5.value_counts()
fill = (df.PAY_6 == 4) | (df.PAY_6 == 7) | (df.PAY_6 == 5) | (df.PAY_6 == 6) | (df.PAY_6 == 8)
df.loc[fill,'PAY_6']=4
#df.PAY_6.value_counts()
df.columns = df.columns.map(str.lower)
col_to_norm = ['limit_bal', 'age', 'bill_amt1', 'bill_amt2', 'bill_amt3', 'bill_amt4', 'bill_amt5', 'bill_amt6', 'pay_amt1', 'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5', 'pay_amt6']
#you can inbuilt StandardScalar() or MinMaxScalar() also
df[col_to_norm] = df[col_to_norm].apply(lambda x :( x-np.mean(x))/np.std(x))
df.head()
correlation = df.corr()
plt.subplots(figsize=(30,10))
sns.heatmap(correlation, square=True, annot=True, fmt=".1f" )
df = df.drop(["id"],1)
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values
#We split the data into train(0.75) and test(0.25) size.
 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state = 1)
#Start with logistic regression model
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression(random_state=1)
logmodel.fit(X_train,y_train)
y_pred = logmodel.predict(X_test)
from sklearn.metrics import  accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
roc=roc_auc_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
results = pd.DataFrame([['Logistic Regression', acc,prec,rec, f1,roc]],
columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
results
#plotting the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, cmap="Blues", annot=True,annot_kws={"size": 16})
#Apply Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 100,criterion = 'entropy',random_state = 0)
rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)
roc=roc_auc_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
results = pd.DataFrame([['Random tree Classifier', acc,prec,rec, f1,roc]],
columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
results
#plotting the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, cmap="Blues", annot=True,annot_kws={"size": 16})
#Apply XGBoost classifier model
from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
y_pred =xgb.predict(X_test)
roc=roc_auc_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
results = pd.DataFrame([['XGBOOST Classifier', acc,prec,rec, f1,roc]],
columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
results
#plotting the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, cmap="Blues", annot=True,annot_kws={"size": 16})