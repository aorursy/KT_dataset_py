# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
print(df.head(10))
df.info()
Non_fraud = df[df['Class']==0]['Class'].count()
Fraud = df[df['Class']==1]['Class'].count()
Fraud_percent = Fraud/(Fraud+Non_fraud)
Non_Fraud_percent = Non_fraud/(Fraud+Non_fraud)
print('Total Number of Non-Fraud Transactions:',Non_fraud)
print('Total Number of Fraud Transcations:',Fraud)
print('Percentage of Non-Fraud Transactions:',Non_Fraud_percent*100,'%')
print('Percentage of Fraud Transactions:',Fraud_percent*100,'%')
import seaborn as sns

sns.countplot(df['Class'],data=df,color='green')
import matplotlib.pyplot as plt

plt.figure(figsize = (15,8))

plt.subplot(1,2,1)

sns.distplot(df['Time'],color='blue')
plt.title('Time Distribution')
plt.xlim([min(df['Time']),max(df['Time'])])

plt.subplot(1,2,2)

sns.distplot(df['Amount'],color='red')
plt.title('Amount Distribution')
plt.xlim([min(df['Amount']),max(df['Amount'])])

plt.show()
from sklearn.preprocessing import StandardScaler

scal = StandardScaler()

df['Scaled_Time'] = scal.fit_transform(df['Time'].values.reshape(-1,1))
df['Scaled_Amount'] = scal.fit_transform(df['Amount'].values.reshape(-1,1))

df.drop(['Time','Amount'],axis=1,inplace=True)
df.head()
plt.figure(figsize=(14,7))
sns.heatmap(df.corr(),cmap='gnuplot_r')
print(df.corr()['Class'].sort_values(ascending = False))
X = df.drop(['Class','V4','V2','V21','V19','V20','V8','V27','V28','Scaled_Amount','V26','V25','V22','V23','V15','V13','V24','Scaled_Time','V6','V5','V9','V1','V18'],axis=1)
y = df['Class']
print(X.head())
print(y.head())
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=0,replacement =True)
rus.fit(X,y)
X_under_sampled,y_under_sampled = rus.fit_resample(X,y)
print("No. of Non-Fraud Transactions in under sampled data: ",y_under_sampled[y_under_sampled==0].value_counts())
print("No. of Fraud Transactions in under sampled data: ",y_under_sampled[y_under_sampled==1].value_counts())
from imblearn.under_sampling import NearMiss

near = NearMiss(sampling_strategy="not minority")
near.fit(X,y)
X_near_sampled,y_near_sampled = near.fit_resample(X,y)
print("No. of Non-Fraud Transactions in under sampled data: ",y_near_sampled[y_near_sampled==0].value_counts())
print("No. of Fraud Transactions in under sampled data: ",y_near_sampled[y_near_sampled==1].value_counts())
from imblearn.under_sampling import TomekLinks

tomek = TomekLinks(sampling_strategy='auto')
tomek.fit(X,y)
X_tomek_sampled,y_tomek_sampled = tomek.fit_resample(X,y)
print("No. of Non-Fraud Transactions in under sampled data: ",y_tomek_sampled[y_tomek_sampled==0].value_counts())
print("No. of Fraud Transactions in under sampled data: ",y_tomek_sampled[y_tomek_sampled==1].value_counts())
from imblearn.under_sampling import ClusterCentroids

clusters = ClusterCentroids(sampling_strategy='auto',random_state = 1)
clusters.fit(X,y)
X_cluster_sampled,y_cluster_sampled = clusters.fit_resample(X,y)
print("No. of Non-Fraud Transactions in under sampled data: ",y_cluster_sampled[y_cluster_sampled==0].value_counts())
print("No. of Fraud Transactions in under sampled data: ",y_cluster_sampled[y_cluster_sampled==1].value_counts())
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=0)
ros.fit(X,y)
X_over_sampled,y_over_sampled = ros.fit_resample(X,y)
print("No. of Non-Fraud Transactions in under sampled data: ",y_over_sampled[y_over_sampled==0].value_counts())
print("No. of Fraud Transactions in under sampled data: ",y_over_sampled[y_over_sampled==1].value_counts())
from imblearn.over_sampling import SMOTE

r_smote = SMOTE(random_state =0)
r_smote.fit(X,y)
X_smote,y_smote = r_smote.fit_resample(X,y)
print("No. of Non-Fraud Transactions in under sampled data: ",y_smote[y_smote==0].value_counts())
print("No. of Fraud Transactions in under sampled data: ",y_smote[y_smote==1].value_counts())
from imblearn.over_sampling import ADASYN

adasyn = ADASYN(random_state =0,sampling_strategy = 'auto')
adasyn.fit(X,y)
X_adasyn,y_adasyn = adasyn.fit_resample(X,y)
print("No. of Non-Fraud Transactions in under sampled data: ",y_adasyn[y_adasyn==0].value_counts())
print("No. of Fraud Transactions in under sampled data: ",y_adasyn[y_adasyn==1].value_counts())
from imblearn.combine import SMOTETomek

smote_tomek = SMOTETomek(random_state =0,sampling_strategy = 'auto')
smote_tomek.fit(X,y)
X_smote_tomek,y_smote_tomek = smote_tomek.fit_resample(X,y)
print("No. of Non-Fraud Transactions in under sampled data: ",y_smote_tomek[y_smote_tomek==0].value_counts())
print("No. of Fraud Transactions in under sampled data: ",y_smote_tomek[y_smote_tomek==1].value_counts())
from sklearn.model_selection import train_test_split

X_under_train,X_under_test,y_under_train,y_under_test = train_test_split(X_under_sampled,y_under_sampled,random_state=0,train_size=0.7)
X_near_train,X_near_test,y_near_train,y_near_test = train_test_split(X_near_sampled,y_near_sampled,random_state=0,train_size=0.7)
X_tomek_train,X_tomek_test,y_tomek_train,y_tomek_test = train_test_split(X_tomek_sampled,y_tomek_sampled,random_state=0,train_size=0.7)
X_cluster_train,X_cluster_test,y_cluster_train,y_cluster_test = train_test_split(X_cluster_sampled,y_cluster_sampled,random_state=0,train_size=0.7)

X_over_train,X_over_test,y_over_train,y_over_test = train_test_split(X_over_sampled,y_over_sampled,random_state=0,train_size=0.7)
X_SMOTE_train,X_SMOTE_test,y_SMOTE_train,y_SMOTE_test = train_test_split(X_smote,y_smote,random_state=0,train_size=0.7)
X_adasyn_train,X_adasyn_test,y_adasyn_train,y_adasyn_test = train_test_split(X_adasyn,y_adasyn,random_state=0,train_size=0.7)
X_SMOTETomek_train,X_SMOTETomek_test,y_SMOTETomek_train,y_SMOTETomek_test = train_test_split(X_smote_tomek,y_smote_tomek,random_state=0,train_size=0.7)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report,mean_absolute_error,accuracy_score
from sklearn.model_selection import GridSearchCV

rfc = RandomForestClassifier(random_state = 1)
param_grid = { 
    'n_estimators': [100,200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}
CV_rfc_under = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 3)
CV_rfc_under.fit(X_under_train, y_under_train)
CV_rfc_under.best_params_
rfc_under = RandomForestClassifier(random_state=1, max_features='auto', n_estimators= 100, max_depth=5, criterion='entropy')
rfc_under.fit(X_under_train, y_under_train)
pred_under = rfc_under.predict(X_under_test)


conf_mat = confusion_matrix(y_true=y_under_test, y_pred=pred_under)
print('Confusion matrix:\n', conf_mat)
labels = ['Non-Fraud', 'Fraud']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf_mat, cmap=plt.cm.winter)
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Expected')
plt.show()
print(classification_report(y_under_test,pred_under))
rfc_near_miss = RandomForestClassifier(random_state=1, max_features='auto', n_estimators= 100, max_depth=5, criterion='entropy')
rfc_near_miss.fit(X_near_train, y_near_train)
pred_near_miss = rfc_near_miss.predict(X_near_test)


conf_mat = confusion_matrix(y_true=y_near_test, y_pred=pred_near_miss)
print('Confusion matrix:\n', conf_mat)
labels = ['Non-Fraud', 'Fraud']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf_mat, cmap=plt.cm.winter)
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Expected')
plt.show()
print(classification_report(y_near_test,pred_near_miss))
rfc_tomek = RandomForestClassifier(random_state=1, max_features='auto', n_estimators= 100, max_depth=5, criterion='entropy')
rfc_tomek.fit(X_tomek_train, y_tomek_train)
pred_tomek = rfc_tomek.predict(X_tomek_test)


conf_mat = confusion_matrix(y_true=y_tomek_test, y_pred=pred_tomek)
print('Confusion matrix:\n', conf_mat)
labels = ['Non-Fraud', 'Fraud']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf_mat, cmap=plt.cm.winter)
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Expected')
plt.show()
print(classification_report(y_tomek_test,pred_tomek))
rfc_cluster = RandomForestClassifier(random_state=1, max_features='auto', n_estimators= 100, max_depth=5, criterion='entropy')
rfc_cluster.fit(X_cluster_train, y_cluster_train)
pred_cluster = rfc_cluster.predict(X_cluster_test)


conf_mat = confusion_matrix(y_true=y_cluster_test, y_pred=pred_cluster)
print('Confusion matrix:\n', conf_mat)
labels = ['Non-Fraud', 'Fraud']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf_mat, cmap=plt.cm.winter)
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Expected')
plt.show()
print(classification_report(y_cluster_test,pred_cluster))
rfc_over = RandomForestClassifier(random_state=1, max_features='log2', n_estimators= 100, max_depth=8, criterion='entropy')
rfc_over.fit(X_over_train, y_over_train)
pred_over = rfc_over.predict(X_over_test)


conf_mat = confusion_matrix(y_true=y_over_test, y_pred=pred_over)
print('Confusion matrix:\n', conf_mat)
labels = ['Non-Fraud', 'Fraud']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf_mat, cmap=plt.cm.winter)
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Expected')
plt.show()
print(classification_report(y_over_test,pred_over))
rfc_smote = RandomForestClassifier(random_state=1, max_features='log2', n_estimators= 100, max_depth=8, criterion='entropy')
rfc_smote.fit(X_SMOTE_train, y_SMOTE_train)
pred_smote = rfc_smote.predict(X_SMOTE_test)


conf_mat = confusion_matrix(y_true=y_SMOTE_test, y_pred=pred_smote)
print('Confusion matrix:\n', conf_mat)
labels = ['Non-Fraud', 'Fraud']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf_mat, cmap=plt.cm.winter)
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Expected')
plt.show()
print(classification_report(y_SMOTE_test,pred_smote))
rfc_adasyn = RandomForestClassifier(random_state=1, max_features='log2', n_estimators= 100, max_depth=8, criterion='entropy')
rfc_adasyn.fit(X_adasyn_train, y_adasyn_train)
pred_adasyn = rfc_adasyn.predict(X_adasyn_test)


conf_mat = confusion_matrix(y_true=y_adasyn_test, y_pred=pred_adasyn)
print('Confusion matrix:\n', conf_mat)
labels = ['Non-Fraud', 'Fraud']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf_mat, cmap=plt.cm.winter)
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Expected')
plt.show()
print(classification_report(y_adasyn_test,pred_adasyn))
rfc_smotetomek = RandomForestClassifier(random_state=1, max_features='log2', n_estimators= 100, max_depth=8, criterion='entropy')
rfc_smotetomek.fit(X_SMOTETomek_train, y_SMOTETomek_train)
pred_smotetomek = rfc_smotetomek.predict(X_SMOTETomek_test)


conf_mat = confusion_matrix(y_true=y_SMOTETomek_test, y_pred=pred_smotetomek)
print('Confusion matrix:\n', conf_mat)
labels = ['Non-Fraud', 'Fraud']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf_mat, cmap=plt.cm.winter)
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Expected')
plt.show()
print(classification_report(y_SMOTETomek_test,pred_smotetomek))