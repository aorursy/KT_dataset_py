import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
train.head()
train.isnull().sum()
sns.countplot(train.Response);
sns.countplot(train.Gender)
df=train.groupby(['Gender','Response'])['id'].count().to_frame().rename(columns={'id':'count'}).reset_index()
df
sns.catplot(x="Gender", y="count",col="Response",
                data=df, kind="bar");
sns.distplot(train.Age);
train.Age.describe()
sns.jointplot(x='Age',y='Annual_Premium',data=train,kind='scatter');
train.Annual_Premium.describe()
sns.distplot(train.Annual_Premium);
sns.countplot(train.Previously_Insured);
sns.countplot(train.Vehicle_Age);
pd.crosstab(train['Driving_License'],train['Response']).plot(kind='bar');
pd.crosstab(train.Vehicle_Damage,train.Response).plot(kind='bar');
train.head()
train = train.drop('id',axis=1)
train['Gender'] = train['Gender'].map({'Male':0,'Female':1}).astype(int)
train=pd.get_dummies(train,drop_first=True)
train=train.rename(columns={"Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year", "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"})
train['Vehicle_Age_lt_1_Year']=train['Vehicle_Age_lt_1_Year'].astype('int')
train['Vehicle_Age_gt_2_Years']=train['Vehicle_Age_gt_2_Years'].astype('int')
train['Vehicle_Damage_Yes']=train['Vehicle_Damage_Yes'].astype('int')
train.head()
target = train['Response']
train = train.drop('Response',axis=1)
x = train.copy()
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import RandomizedSearchCV 
X_train, X_valid, y_train, y_valid = train_test_split(x,target,test_size=0.3, random_state=17)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
logit = LogisticRegression(random_state=17)
logit.fit(X_train_scaled, y_train)
prediction = logit.predict_proba(X_valid_scaled)[:,1]
roc_auc_score(y_valid,prediction)
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train_scaled,y_train)
prediction = rfc.predict_proba(X_valid)[:,1]
roc_auc_score(y_valid,prediction)
test.head()
test = test.drop('id',axis=1)
test['Gender'] = test['Gender'].map({'Male':0,'Female':1}).astype(int)
test=pd.get_dummies(test,drop_first=True)
test=test.rename(columns={"Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year", "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"})
test['Vehicle_Age_lt_1_Year']=test['Vehicle_Age_lt_1_Year'].astype('int')
test['Vehicle_Age_gt_2_Years']=test['Vehicle_Age_gt_2_Years'].astype('int')
test['Vehicle_Damage_Yes']=test['Vehicle_Damage_Yes'].astype('int')
X_train_scaled = scaler.fit_transform(train)
X_test_scaled = scaler.fit_transform(test)
rfs = RandomForestClassifier(random_state=17)
param_grid = {'n_estimators':[100,150,200],
             'criterion':['gini','entropy'],
             'bootstrap':[True],
             'max_depth':[15,20,25,30],
             'max_features':['auto','sqrt',10],
             'min_samples_leaf':[2,3],
             'min_samples_split':[2,3]}
clf_rfs = RandomizedSearchCV(rfs,param_distributions=param_grid,cv=5,verbose=True,n_jobs=-1)
best_clf_rfs = clf_rfs.fit(X_train_scaled,target)
best_clf_rfs.best_params_
best_clf_rfs.best_score_
predictions = best_clf_rfs.predict(X_test_scaled).astype(int)
output_test = pd.read_csv('data/test.csv')
output = pd.DataFrame({'id':output_test.id,'Response':predictions})
output.to_csv('submission.csv',index=False)
output.head()