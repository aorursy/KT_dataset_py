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
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
data=pd.read_csv("../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv")
data.isnull().sum()
#Correlation between Features
plt.figure(figsize=(12,10))
sns.heatmap(data.corr(),annot=True)
data.head()
plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
sns.distplot(data["age"])
plt.subplot(1,2,2)
sns.boxplot(x="DEATH_EVENT",y="age",data=data)
#As you may notice,The people of higher ages are more likely to face Heartfailure Than Young People
data.groupby("anaemia")["DEATH_EVENT"].describe()
sns.countplot(x="anaemia",data=data,hue="DEATH_EVENT")
# As you can notice people having anaemia are more vulnerable towards Heart_Faliure
plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
sns.distplot(data["creatinine_phosphokinase"])
plt.subplot(1,2,2)
sns.boxplot(x="DEATH_EVENT",y="creatinine_phosphokinase",data=data)
data["creatinine_phosphokinase"]=np.log(data["creatinine_phosphokinase"])
plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
sns.distplot(data["creatinine_phosphokinase"])
plt.subplot(1,2,2)
sns.boxplot(x="DEATH_EVENT",y="creatinine_phosphokinase",data=data)
data.head()
data.groupby("diabetes")["DEATH_EVENT"].describe()
plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
sns.distplot(data["ejection_fraction"])
plt.subplot(1,2,2)
sns.boxplot(x="DEATH_EVENT",y="ejection_fraction",data=data)
data["ejection_fraction"].describe()
for i in data.loc[data["ejection_fraction"]>60].index:
    data.iloc[i,4]=60
plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
sns.distplot(data["ejection_fraction"])
plt.subplot(1,2,2)
sns.boxplot(x="DEATH_EVENT",y="ejection_fraction",data=data)
data.head(1)
data.groupby("high_blood_pressure")["DEATH_EVENT"].describe()
# As you can see the people having high blood pressure has high tendency of Heart_Failure
plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
sns.distplot(data["platelets"])
plt.subplot(1,2,2)
sns.boxplot(x="DEATH_EVENT",y="platelets",data=data)
data["platelets"].describe()
263358-2*97804
for i in data.loc[data["platelets"]>458966].index:
    data.iloc[i,6]=458966
plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
sns.distplot(data["platelets"])
plt.subplot(1,2,2)
sns.boxplot(x="DEATH_EVENT",y="platelets",data=data)
data.head(1)
plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
sns.distplot(data["serum_creatinine"])
plt.subplot(1,2,2)
sns.boxplot(x="DEATH_EVENT",y="serum_creatinine",data=data)
sns.distplot((data["serum_creatinine"]))
plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
sns.distplot(data["serum_creatinine"])
plt.subplot(1,2,2)
sns.boxplot(x="DEATH_EVENT",y="serum_creatinine",data=data)
sns.distplot(data.loc[data["DEATH_EVENT"]==0]["serum_creatinine"])
sns.distplot(data.loc[data["DEATH_EVENT"]==1]["serum_creatinine"])
for i in data.loc[(data["DEATH_EVENT"]==1) & (data["serum_creatinine"]>5.3436)].index:
     data.iloc[i,7]=5.3436
data.loc[data["DEATH_EVENT"]==0]["serum_creatinine"].describe()
for i in data.loc[(data["DEATH_EVENT"]==0) & (data["serum_creatinine"]>2.5)].index:
     data.iloc[i,7]=2.5
plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
sns.distplot(data["serum_creatinine"])
plt.subplot(1,2,2)
sns.boxplot(x="DEATH_EVENT",y="serum_creatinine",data=data)
plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
sns.distplot(data["serum_sodium"])
plt.subplot(1,2,2)
sns.boxplot(x="DEATH_EVENT",y="serum_sodium",data=data)
data.loc[data["DEATH_EVENT"]==1]["serum_sodium"].describe()
for i in data.loc[(data["DEATH_EVENT"]==0) & (data["serum_sodium"]<128)].index:
     data.iloc[i,8]=128
for i in data.loc[(data["DEATH_EVENT"]==1) & (data["serum_sodium"]<120)].index:
     data.iloc[i,8]=120
plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
sns.distplot(data["serum_sodium"])
plt.subplot(1,2,2)
sns.boxplot(x="DEATH_EVENT",y="serum_sodium",data=data)
data.columns
data.groupby("sex")["DEATH_EVENT"].describe()
data.groupby("smoking")["DEATH_EVENT"].describe()
plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
sns.distplot(data["time"])
plt.subplot(1,2,2)
sns.boxplot(x="DEATH_EVENT",y="time",data=data)
from sklearn.model_selection import StratifiedShuffleSplit,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,recall_score,precision_score,roc_auc_score,confusion_matrix,classification_report
X=data.iloc[:,:-1]
y=data.iloc[:,-1]
sf=StratifiedShuffleSplit(n_splits=1,test_size=0.33,random_state=100)
for i,j in sf.split(X,y):
    train_X,train_y=X.iloc[i],y.iloc[i]
    test_X,test_y=X.iloc[j],y.iloc[j]
RandomForestClassifier().get_params().keys()
params={'class_weight':[{0:1,1:2},{0:1,1:1}],
        'max_depth':[2,4,6],'max_leaf_nodes':[6,8,12,10],'min_samples_leaf':[4,6,8],
                      'min_samples_split':[6,8,12],'n_estimators':[50,100,150,200]}
grid=GridSearchCV(RandomForestClassifier(),cv=10,scoring="roc_auc",verbose=2,n_jobs=-1,param_grid=params)
grid.fit(train_X,train_y)
def check_score(test_y,pred):
    print(accuracy_score(test_y,pred))
    print(recall_score(test_y,pred))
    print(precision_score(test_y,pred))
    print(roc_auc_score(test_y,pred))
    print(confusion_matrix(test_y,pred))
    print(classification_report(test_y,pred))
grid.best_score_
grid.best_estimator_
rf=RandomForestClassifier(class_weight={0: 1, 1: 2}, max_depth=6,
                       max_leaf_nodes=12, min_samples_leaf=4,
                       min_samples_split=6, n_estimators=50)
rf.fit(train_X,train_y)
prediction=rf.predict(test_X)
check_score(test_y,prediction)
important_features=rf.feature_importances_
pd.DataFrame(important_features,index=X.columns).sort_values(0).plot(kind="barh")
cols=["time","age","serum_creatinine","ejection_fraction","creatinine_phosphokinase","serum_sodium","platelets"]
X=X[cols]
sf=StratifiedShuffleSplit(n_splits=1,test_size=0.33,random_state=100)
for i,j in sf.split(X,y):
    train_X,train_y=X.iloc[i],y.iloc[i]
    test_X,test_y=X.iloc[j],y.iloc[j]
params={'class_weight':[{0:1,1:2},{0:1,1:1}],
        'max_depth':[2,4,6],'max_leaf_nodes':[6,8,12,10],'min_samples_leaf':[4,6,8],
                      'min_samples_split':[6,8,12],'n_estimators':[30,50,100,150,200]}
grid=GridSearchCV(RandomForestClassifier(),cv=10,scoring="roc_auc",verbose=2,n_jobs=-1,param_grid=params)
grid.fit(train_X,train_y)
grid.best_estimator_
grid.best_score_
def check_score(test_y,pred):
    print(accuracy_score(test_y,pred))
    print(recall_score(test_y,pred))
    print(precision_score(test_y,pred))
    print(roc_auc_score(test_y,pred))
    print(confusion_matrix(test_y,pred))
    print(classification_report(test_y,pred))
rf_se=RandomForestClassifier(class_weight={0: 1, 1: 2}, max_depth=4,
                       max_leaf_nodes=10, min_samples_leaf=6,
                       min_samples_split=8, n_estimators=200)
rf_se.fit(train_X,train_y)
prediction=rf_se.predict(test_X)
check_score(test_y,prediction)

