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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
df=pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
df.head()
df.info()
df.describe()
sns.heatmap(df.isnull(),cmap='viridis')
df['Outcome'].value_counts()
labels=['True','False']
explode=[0.03,0.03]
color=['pink','lightgreen']
f,ax = plt.subplots(1,2,figsize = (15, 7))
_=df.Outcome.value_counts().plot.bar(ax=ax[0],cmap='viridis')
_=df.Outcome.value_counts().plot.pie(ax=ax[1],labels=labels,autopct='%.2f%%',colors=color,explode=explode)
sns.pairplot(df,hue='Outcome')
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(),annot=True,cmap='plasma',linecolor='black',linewidths=0.01)
fig, ax = plt.subplots(4,2,figsize=(16,16))
sns.distplot(df.Age,bins=20, ax=ax[0,0]) 
sns.distplot(df.Pregnancies,bins=20,ax=ax[0,1]) 
sns.distplot(df.Glucose,bins=20,ax=ax[1,0]) 
sns.distplot(df.BloodPressure,bins=20,ax=ax[1,1]) 
sns.distplot(df.SkinThickness,bins=20,ax=ax[2,0])
sns.distplot(df.Insulin,bins=20,ax=ax[2,1])
sns.distplot(df.DiabetesPedigreeFunction,bins=20,ax=ax[3,0]) 
sns.distplot(df.BMI,bins=20,ax=ax[3,1]) 
plt.figure(figsize=(15,6))
sns.countplot('Pregnancies',hue='Outcome',data=df,palette='viridis')
plt.legend(loc='upper right',labels=['False','True'])
data=df.copy(deep=True)
data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
      'BMI', 'DiabetesPedigreeFunction']]=data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction']].replace(0,np.NaN)
data.isnull().sum()
data=data.fillna(data.mean())
plt.figure(figsize=(10,6))
sns.heatmap(data.corr(),annot=True,cmap='plasma',linecolor='black',linewidths=0.01)
fig,ax=plt.subplots(4,2,figsize=(16,16))
sns.distplot(data['Pregnancies'],ax=ax[0,0],bins=20)
sns.distplot(data['Glucose'],ax=ax[0,1],bins=20)
sns.distplot(data['BloodPressure'],ax=ax[1,0],bins=20)
sns.distplot(data['SkinThickness'],ax=ax[1,1],bins=20)
sns.distplot(data['Insulin'],ax=ax[2,0],bins=20)
sns.distplot(data['BMI'],ax=ax[2,1],bins=20)
sns.distplot(data['DiabetesPedigreeFunction'],ax=ax[3,0],bins=20)
sns.distplot(data['Age'],ax=ax[3,1],bins=20)
X=data.drop('Outcome',axis=1)
y=data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)
knn=KNeighborsClassifier()
params={'n_neighbors':range(1,21),'p':[1,2,3,4,5,6,7,8,9,10],
        'weights':['distance','uniform'],'leaf_size':range(1,21)}
gs_knn=GridSearchCV(knn,param_grid=params,cv=10,n_jobs=-1)
gs_knn.fit(X_train,y_train)
gs_knn.best_params_
prediction=gs_knn.predict(X_test)
acc_knn=accuracy_score(y_test,prediction)
print(acc_knn)
print(confusion_matrix(y_test,prediction))
probability=gs_knn.predict_proba(X_test)[:,1]
fpr_knn,tpr_knn,thresh=roc_curve(y_test,probability)
plt.figure(figsize=(12,6))
plt.plot(fpr_knn,tpr_knn)
plt.plot([0,1],ls='--')
plt.plot([0,0],[1,0],c='0.5')
plt.plot([1,1],c='0.5')
roc_auc_score(y_test,probability)*100
log_reg=LogisticRegression()
params={'C':[0.01,0.1,1,10],'max_iter':[100,300,600]}
gs_lr=GridSearchCV(log_reg,param_grid=params,n_jobs=-1,cv=10)
gs_lr.fit(X_train,y_train)
gs_lr.best_params_
prediction=gs_lr.predict(X_test)
acc_lr=accuracy_score(y_test,prediction)
print(acc_lr)
print(confusion_matrix(y_test,prediction))
probability=gs_lr.predict_proba(X_test)[:,1]
fpr_lr,tpr_lr,thresh=roc_curve(y_test,probability)
plt.figure(figsize=(14,6))
plt.plot(fpr_lr,tpr_lr)
plt.plot([0,1],ls='--')
plt.plot([0,0],[1,0],c='0.5')
plt.plot([1,1],c='0.5')
roc_auc_score(y_test,probability)*100
dtr=DecisionTreeClassifier()
params={'max_features':["auto", "sqrt", "log2"],'min_samples_leaf':range(1,11),'min_samples_split':range(1,11)}
gs_dtr=GridSearchCV(dtr,param_grid=params,n_jobs=-1,cv=5)
gs_dtr.fit(X_train,y_train)
gs_dtr.best_params_
prediction=gs_dtr.predict(X_test)
acc_dtr=accuracy_score(y_test,prediction)
print(acc_dtr)
print(confusion_matrix(y_test,prediction))
probability=gs_dtr.predict_proba(X_test)[:,1]
fpr_dtr,tpr_dtr,thresh=roc_curve(y_test,probability)
plt.figure(figsize=(14,6))
plt.plot(fpr_dtr,tpr_dtr)
plt.plot([0,1],ls='--')
plt.plot([0,0],[1,0],c='0.5')
plt.plot([1,1],c='0.5')
roc_auc_score(y_test,probability)*100
rfc=RandomForestClassifier()
params={'n_estimators':[100,300,500],'min_samples_leaf':range(1,11)}
gs_rfc=GridSearchCV(rfc,param_grid=params,n_jobs=-1,cv=5)
gs_rfc.fit(X_train,y_train)
gs_rfc.best_params_
prediction=gs_rfc.predict(X_test)
acc_rfc=accuracy_score(y_test,prediction)
print(acc_rfc)
print(confusion_matrix(y_test,prediction))
probability=gs_rfc.predict_proba(X_test)[:,1]
fpr_rfc,tpr_rfc,thresh=roc_curve(y_test,probability)
plt.figure(figsize=(14,6))
plt.plot(fpr_rfc,tpr_rfc)
plt.plot([0,1],ls='--')
plt.plot([0,0],[1,0],c='0.5')
plt.plot([1,1],c='0.5')
roc_auc_score(y_test,probability)*100
report=pd.DataFrame({'Model':['KNeighborsClassifier','LogisticRegression','DecisionTreeClassifier','RandoForestClassifier'],
                    'Score':[acc_knn,acc_lr,acc_dtr,acc_rfc]})
report.sort_values(by='Score',ascending=False)