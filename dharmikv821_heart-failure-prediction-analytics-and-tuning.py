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

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set_style('darkgrid')
data = pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
data.head()
plt.figure(figsize=(8,6))
sns.heatmap(data.isna(),yticklabels=False, cmap='plasma')
plt.show()
plt.figure(figsize=(10,10))
sns.heatmap(data.corr(),annot=True,cmap='coolwarm')
plt.show()
data.describe()
data.info()
len(data.columns)
data.nunique()
data['DEATH_EVENT'].value_counts()
data['DEATH_EVENT'].value_counts()/len(data)
cat_features = ['anaemia','diabetes','high_blood_pressure','sex','smoking']
for cat in cat_features:
    sns.countplot(data[cat], hue=data.DEATH_EVENT)
    plt.title(cat.upper()+' (w.r.t. DEATH_EVENT)')
    plt.show()
    print(data.groupby(cat)['DEATH_EVENT'].value_counts())
sns.distplot(data.age,bins=15)
for cat in cat_features:
    plt.figure(figsize=(15,6))
    plt.subplot(1,2,1)
    sns.distplot(data[data[cat]==0]['age'],label=0,color='blue',bins=15)
    plt.legend()
    plt.title(cat.upper())
    plt.subplot(1,2,2)
    sns.distplot(data[data[cat]==1]['age'],label=1,color='red',bins=15)
    plt.legend()
    plt.title(cat.upper())
    plt.show()
cont_features = ['age','creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium','time']
for col in cont_features:
    plt.figure(figsize=(15,6))
    plt.subplot(1,2,1)
    sns.distplot(data[data['DEATH_EVENT']==0][col],label=0,color='blue',bins=10)
    plt.legend()
    plt.title(col.upper())
    plt.subplot(1,2,2)
    sns.distplot(data[data['DEATH_EVENT']==1][col],label=1,color='red',bins=10)
    plt.legend()
    plt.title(col.upper())
    plt.show()
for col in cont_features:
    plt.boxplot(data[col])
    plt.title(col.upper())
    plt.show()
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100, random_state=0)
rfc.fit(X_train, y_train)
pred = rfc.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
print(confusion_matrix(y_test, pred))
print(accuracy_score(y_test, pred))
print(classification_report(y_test, pred))
from sklearn.model_selection import cross_val_score
val_score = cross_val_score(estimator=rfc,X=X_train,y=y_train,cv=10,n_jobs=-1)
val_score
rfc = RandomForestClassifier()
rf_params = {'n_estimators':[i for i in range(100,1000,10)],
          'criterion':['gini','entropy'],
          'max_features':['auto','sqrt','log2'],
          'max_depth':[i for i in range(10,1000,10)],
          'min_samples_split':[2,4,6,8,10],
          'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10]
}
from sklearn.model_selection import RandomizedSearchCV
rs = RandomizedSearchCV(estimator=rfc,n_jobs=-1,cv=10,n_iter=100,param_distributions=rf_params,verbose=5,random_state=0)
rs.fit(X_train, y_train)
rs
rs.best_params_
rs.best_estimator_
best = rs.best_estimator_
y_pred = best.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))