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
#importing basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import data
data = pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
data.head(10)
data.describe()
#checking number of records and features
data.shape
#checking total NaN values in each column
data.isnull().sum()
data['anaemia'].value_counts()
#value_count_for_respective_columns
list = ['diabetes','ejection_fraction','high_blood_pressure','sex','smoking','DEATH_EVENT']
for col in list:
    print(data[col].value_counts())
sns.countplot(x='DEATH_EVENT',hue='diabetes',data=data,palette='RdBu_r')
sns.countplot(x="ejection_fraction", hue="DEATH_EVENT", data=data)
data['ejection_fraction'].replace(17,25,inplace=True)
data['ejection_fraction'].replace(62,25,inplace=True)
data['ejection_fraction'].replace(65,14,inplace=True)
data['ejection_fraction'].replace(15,14,inplace=True)
data['ejection_fraction'].replace(70,14,inplace=True)
sns.countplot(x="ejection_fraction", hue="DEATH_EVENT", data=data)
sns.countplot(x='DEATH_EVENT',hue="high_blood_pressure",data=data,palette='RdBu_r')
sns.countplot(x='DEATH_EVENT',hue="sex",data=data,palette='RdBu_r')
sns.countplot(x='DEATH_EVENT',hue="smoking",data=data,palette='RdBu_r')
sns.countplot(x='DEATH_EVENT',hue='anaemia',data=data,palette='RdBu_r')
sns.FacetGrid(data,hue='DEATH_EVENT',size=5).map(sns.distplot,"age").add_legend()
sns.FacetGrid(data,hue='DEATH_EVENT',size=5).map(sns.distplot,"creatinine_phosphokinase").add_legend()
sns.FacetGrid(data,hue='DEATH_EVENT',size=5).map(sns.distplot,"serum_creatinine").add_legend()
sns.FacetGrid(data,hue='DEATH_EVENT',size=5).map(sns.distplot,"serum_sodium").add_legend()
#outliers checking and treatment
data.boxplot('serum_sodium')
data['serum_sodium'].quantile(np.arange(0,1,0.01))
data.loc[(data['serum_sodium']<125),'serum_sodium']=125
data.boxplot('serum_sodium')
data.boxplot('serum_creatinine')
data.boxplot('creatinine_phosphokinase')
sns.heatmap(data.corr())
X = data.drop("DEATH_EVENT",axis=1)
y = data["DEATH_EVENT"]
cnames = ["time","serum_sodium","serum_creatinine","platelets","creatinine_phosphokinase","age"]
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
for col in cnames:
    X[col] = sc.fit_transform(X[[col]])
X.shape
#split into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from  sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(X_train,y_train)

y_pred=logreg.predict(X_test)
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,recall_score,precision_score
print("Accuracy:",accuracy_score(y_test, y_pred))
print('f1 score', f1_score(y_test, y_pred,
                              ))
sns.heatmap(confusion_matrix(y_test, y_pred),annot=True)
data.DEATH_EVENT.value_counts()
df_majority = data[data.DEATH_EVENT==0]
df_min = data[data.DEATH_EVENT==1]
import sklearn.utils as ut
df_minority_upsample = ut.resample(df_min,replace=True,n_samples=203,random_state=1)
print(df_majority.shape)
print(df_minority_upsample.shape)
df_upsampled = pd.concat([df_majority,df_minority_upsample])
print(df_upsampled.DEATH_EVENT.value_counts())
X1=df_upsampled.drop("DEATH_EVENT",axis=1)
Y1=df_upsampled["DEATH_EVENT"]
cnames = ["time","serum_sodium","serum_creatinine","platelets","creatinine_phosphokinase","age"]
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
for col in cnames:
    X1[col] = sc.fit_transform(X1[[col]])
#split into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X1,Y1, test_size=0.2, random_state=0)
#Random Forst Classifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
rf=RandomForestClassifier(n_estimators=100,random_state=0)
rf.fit(X_train,y_train)
y_pred=rf.predict(X_test)
print("Accuracy by random forest:",accuracy_score(y_test, y_pred))
print('f1 score ', f1_score(y_test, y_pred,
                              ))
#Applying decision tree
from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier("entropy")
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("Accuracy:",accuracy_score(y_test, y_pred))
print('f1 score', f1_score(y_test, y_pred,
                              ))
cf=confusion_matrix(y_test, y_pred)
sns.heatmap(cf, annot=True)
