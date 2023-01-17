# Importing necessary libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
# Reading the dataset

train_data = pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
train_data.head()
train_data = train_data.iloc[:,1:]

train_data.head()
print("Shape of data: ",train_data.shape)
train_data.info()
train_data.columns
# Looking ate the unique values of Categorical Features

print(train_data['ssc_b'].unique())

print(train_data['hsc_b'].unique())

print(train_data['hsc_s'].unique())

print(train_data['degree_t'].unique())

print(train_data['workex'].unique())

print(train_data['specialisation'].unique())

print(train_data['status'].unique())
train_data.isnull().sum()
null_salary_data = train_data[train_data['salary'].notna()]

null_salary_data = null_salary_data.reset_index(drop = True)
var = 'salary'

fig, ax = plt.subplots()

fig.set_size_inches(20, 8)

plt.xticks(rotation=90);

sns.countplot(x = var,palette="ch:.4", data = null_salary_data)

ax.set_xlabel('Salary', fontsize=15)

ax.set_ylabel('Count', fontsize=15)

ax.set_title('Salary Count Distribution', fontsize=15)

sns.despine()
print(train_data['status'].value_counts())
var = 'status'

fig, ax = plt.subplots()

fig.set_size_inches(5,5)

sns.countplot(x = var, data = train_data)

ax.set_xlabel('Status', fontsize=15)

ax.set_ylabel('Count', fontsize=15)

ax.set_title('Status Count Distribution', fontsize=15)

ax.tick_params(labelsize=15)

sns.despine()
print('Senior Secondary Board: ',train_data['ssc_b'].unique())
var = 'ssc_b'

fig, ax = plt.subplots()

fig.set_size_inches(5,5)

sns.countplot(x = var, data = train_data)

ax.set_xlabel('Senior Secondary', fontsize=15)

ax.set_ylabel('Count', fontsize=15)

ax.set_title('Senior Secondary Count Distribution', fontsize=15)

ax.tick_params(labelsize=15)

sns.despine()
print('Higher Secondary Boards: ',train_data['hsc_b'].unique())
var = 'hsc_b'

fig, ax = plt.subplots()

fig.set_size_inches(5,5)

sns.countplot(x = var, data = train_data)

ax.set_xlabel('Higher Secondary', fontsize=15)

ax.set_ylabel('Count', fontsize=15)

ax.set_title('Higher Secondary Count Distribution', fontsize=15)

ax.tick_params(labelsize=15)

sns.despine()
print('Higher Secondary Subjects: ',train_data['hsc_s'].unique())
var = 'hsc_s'

fig, ax = plt.subplots()

fig.set_size_inches(5,5)

sns.countplot(x = var, data = train_data)

ax.set_xlabel('Higher Secondary Subjects', fontsize=15)

ax.set_ylabel('Count', fontsize=15)

ax.set_title('Higher Secondary Count Distribution', fontsize=15)

ax.tick_params(labelsize=15)

sns.despine()
print('Work Experience: ',train_data['workex'].unique())
var = 'workex'

fig, ax = plt.subplots()

fig.set_size_inches(5,5)

sns.countplot(x = var, data = train_data)

ax.set_xlabel('Work Experience', fontsize=15)

ax.set_ylabel('Count', fontsize=15)

ax.set_title('Work Experience Count Distribution', fontsize=15)

ax.tick_params(labelsize=15)

sns.despine()
var = 'specialisation'

fig, ax = plt.subplots()

fig.set_size_inches(5,5)

sns.countplot(x = var, data = train_data)

ax.set_xlabel('Specialisation', fontsize=15)

ax.set_ylabel('Count', fontsize=15)

ax.set_title('Specialisation Count Distribution', fontsize=15)

ax.tick_params(labelsize=15)

sns.despine()

fig, ax = plt.subplots()

fig.set_size_inches(5,5)

sns.barplot(x='ssc_b', y='salary',hue='gender', data=train_data)

ax.set_xlabel('Senior Secondary Board', fontsize=15)

ax.set_ylabel('Salary', fontsize=15)

ax.set_title('Senior Secondary Board Salary On basis of Gender', fontsize=10)

ax.tick_params(labelsize=15)

plt.show() 
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows = 2, ncols = 2, figsize = (10,10))



sns.countplot(x = 'gender', hue = 'status', palette="ch:2.5,-.2,dark=.3",data = train_data, ax = ax1)

ax1.set_title('Status on basis of Gender', fontsize=15)

ax1.tick_params(labelsize=15)



sns.countplot(x = 'specialisation', hue = 'status', palette="ch:5,-.6,dark=.3", data = train_data, ax = ax2)

ax2.set_title('Status on basis of Specialisation', fontsize=15)

ax2.tick_params(labelsize=15)



sns.countplot(x = 'workex', hue = 'status', palette="ch:10.5,-8.2,dark=.3", data = train_data, ax = ax3)

ax3.set_title('Status on basis of Work Exp.', fontsize=15)

ax3.tick_params(labelsize=15)



sns.countplot(x = 'degree_t', hue = 'status', palette="ch:12.5,-11.2,dark=.3", data = train_data, ax = ax4)

ax4.set_title('Status on basis of Degree', fontsize=15)

ax4.tick_params(labelsize=15)



plt.subplots_adjust(wspace=0.5)

plt.tight_layout() 
Gender = train_data[['gender']]

Gender = pd.get_dummies(Gender,drop_first=True)

Gender.head()
ssc_b = train_data[['ssc_b']]

ssc_b = pd.get_dummies(ssc_b,drop_first=True)

hsc_b = train_data[['hsc_b']]

hsc_b = pd.get_dummies(hsc_b,drop_first=True)

hsc_s = train_data[['hsc_s']]

hsc_s = pd.get_dummies(hsc_s,drop_first=True)

degree_t = train_data[['degree_t']]

degree_t = pd.get_dummies(degree_t,drop_first=True)

workex = train_data[['workex']]

workex = pd.get_dummies(workex,drop_first=True)

specialisation = train_data[['specialisation']]

specialisation = pd.get_dummies(specialisation,drop_first=True)
train_data.info()
train_data.replace({"Placed":1,"Not Placed":0},inplace=True)

train_data.head()
train_data.drop(["salary"],axis=1,inplace=True)
final_train= pd.concat([train_data,Gender,ssc_b,hsc_b,hsc_s,degree_t,workex,specialisation],axis=1)

final_train.head()
final_train.drop(["gender","ssc_b","hsc_b","hsc_s","degree_t","workex","specialisation"],axis=1,inplace=True)

final_train.head()
print("Final Shape of data: ",final_train.shape)

print("\nFinal Columns of data:\n",final_train.columns)
X = final_train.loc[:,['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p', 'gender_M',

       'ssc_b_Others', 'hsc_b_Others', 'hsc_s_Commerce', 'hsc_s_Science',

       'degree_t_Others', 'degree_t_Sci&Tech', 'workex_Yes',

       'specialisation_Mkt&HR']]

y = final_train.loc[:,['status']]
plt.figure(figsize=(18,18))

sns.heatmap(X.corr(),annot=True,cmap='RdYlGn')



plt.show()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 25)
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)

log_reg = LogisticRegression() 

log_reg.fit(X_train,y_train)

log_pred = log_reg.predict(X_test)
import scikitplot as skplt

skplt.metrics.plot_confusion_matrix(y_test,log_pred)
from sklearn import metrics

print("Logistic Model Accuracy is: ",metrics.accuracy_score(y_test, log_pred))
from xgboost import XGBClassifier

xgb = XGBClassifier()

xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)



print("XGBClassifier Accuracy is: ",metrics.accuracy_score(y_test, xgb_pred))

skplt.metrics.plot_confusion_matrix(y_test,xgb_pred)
from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()

gbk.fit(X_train, y_train)

gbk_pred = gbk.predict(X_test)



print("GradientBoostingClassifier Accuracy is: ",metrics.accuracy_score(y_test, gbk_pred))

skplt.metrics.plot_confusion_matrix(y_test,gbk_pred)
from sklearn import metrics



fig, (ax, ax1) = plt.subplots(nrows = 1, ncols = 2, figsize = (15,5))



probs = xgb.predict_proba(X_test)

preds = probs[:,1]

fprxgb, tprxgb, thresholdxgb = metrics.roc_curve(y_test, preds)

roc_aucxgb = metrics.auc(fprxgb, tprxgb)



ax.plot(fprxgb, tprxgb, 'b', label = 'AUC = %0.2f' % roc_aucxgb)

ax.plot([0, 1], [0, 1],'r--')

ax.set_title('Receiver Operating Characteristic XGBOOST ',fontsize=10)

ax.set_ylabel('True Positive Rate',fontsize=20)

ax.set_xlabel('False Positive Rate',fontsize=15)

ax.legend(loc = 'lower right', prop={'size': 16})



probs = gbk.predict_proba(X_test)

preds = probs[:,1]

fprgbk, tprgbk, thresholdgbk = metrics.roc_curve(y_test, preds)

roc_aucgbk = metrics.auc(fprgbk, tprgbk)



ax1.plot(fprgbk, tprgbk, 'b', label = 'AUC = %0.2f' % roc_aucgbk)

ax1.plot([0, 1], [0, 1],'r--')

ax1.set_title('Receiver Operating Characteristic GRADIENT BOOST ',fontsize=10)

ax1.set_ylabel('True Positive Rate',fontsize=20)

ax1.set_xlabel('False Positive Rate',fontsize=15)

ax1.legend(loc = 'lower right', prop={'size': 16})



plt.subplots_adjust(wspace=1)
