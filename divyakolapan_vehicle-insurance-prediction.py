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
tr = pd.read_csv("../input/insurance-train/health_insurance_sell_pre_train.csv")
tr
te = pd.read_csv("../input/insurance-test/health_insurance_sell_pre_test.csv")
te
print(tr.shape)
print(te.shape)
tr.dtypes
te.dtypes
tr.isnull().sum()
te.isnull().sum()
tr.head(20)
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = 12,10
tr.hist();
import seaborn as sns
sns.countplot(tr.Response)
tr.Response.value_counts()
d1 = tr.groupby(['Gender','Response'])['id'].count().to_frame().reset_index()
sns.catplot(x="Gender", y="id",col="Response",
                data=d1, kind="bar",
                height=4, aspect=.7);
d2 = tr.groupby(['Gender','Driving_License'])["id"].count().to_frame().reset_index()
d2
sns.catplot(x="Gender", y="id",col="Driving_License",data=d2, kind="bar",height=4, aspect=.7)
sns.countplot(tr.Vehicle_Age)
d3=tr.groupby(['Vehicle_Age','Response'])['id'].count().to_frame().reset_index()
d3
sns.catplot(x="Vehicle_Age", y="id",col="Response",data=d3, kind="bar",height=4, aspect=.6)
sns.countplot(tr.Vehicle_Damage)
d4=tr.groupby(['Vehicle_Damage','Response'])['id'].count().to_frame().reset_index()
d4
sns.catplot(x="Vehicle_Damage", y="id",col="Response",data=d4, kind="bar",height=4, aspect=.7)
sns.distplot(tr.Age)
sns.distplot(tr.Annual_Premium)
tr.Annual_Premium.mean()
tr.groupby(['Vintage','Response'])['id'].count().to_frame().reset_index()
sns.distplot(tr.Vintage)
tr['Gender'] = tr['Gender'].map( {'Female': 0, 'Male': 1} ).astype(int)
train=pd.get_dummies(tr,drop_first=True)
train
train = train.rename(columns = {"Vehicle_Age_< 1 Year": "veh_less_1_year","Vehicle_Age_> 2 Years":"veh_gre_2_year"})
train['veh_less_1_year']=train['veh_less_1_year'].astype('int')
train['veh_gre_2_year']=train['veh_gre_2_year'].astype('int')
train['Vehicle_Damage_Yes']=train['Vehicle_Damage_Yes'].astype('int')
train.head()
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
mm_scaler = MinMaxScaler()

num = ['Age','Vintage']
cat = ['Gender','Driving_License','Previously_Insured','Region_Code','Policy_Sales_Channel','veh_less_1_year','veh_gre_2_year','Vehicle_Damage_Yes']
train[num] = scaler.fit_transform(train[num])
train[['Annual_Premium']] = mm_scaler.fit_transform(train[['Annual_Premium']])
train
te['Gender'] = te['Gender'].map( {'Female': 0, 'Male': 1} ).astype(int)
test=pd.get_dummies(te,drop_first=True)
test = test.rename(columns = {"Vehicle_Age_< 1 Year": "veh_less_1_year","Vehicle_Age_> 2 Years":"veh_gre_2_year"})
test['veh_less_1_year']=test['veh_less_1_year'].astype('int')
test['veh_gre_2_year']=test['veh_gre_2_year'].astype('int')
test['Vehicle_Damage_Yes']=test['Vehicle_Damage_Yes'].astype('int')
test[num] = scaler.fit_transform(test[num])
test[['Annual_Premium']] = mm_scaler.fit_transform(test[['Annual_Premium']])
test
y = train.Response
x = train.drop(['Response'], axis = 1)
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = .20, random_state = 0)
from sklearn.linear_model import LogisticRegression
l1 = LogisticRegression()
model_1 = l1.fit(xtrain,ytrain)
pre_1 = model_1.predict(xtest)
pre_test_1 = model_1.predict(test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(ytest, pre_1)
cm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
print (classification_report(ytest, pre_1))
pred_1_prob = model_1.predict_proba(xtest)[:, 1]
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(ytest,pred_1_prob)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, linewidth=2)
plt.plot([0,1], [0,1], 'b--' )
from sklearn.metrics import roc_auc_score

score = roc_auc_score(ytest, pred_1_prob)
score
from xgboost.sklearn import XGBClassifier
l2 = XGBClassifier()
model_2 = l2.fit(xtrain,ytrain)
pre_2 = model_2.predict(xtest)
pre_test_2 = model_2.predict(test)
cm = confusion_matrix(ytest, pre_2)
cm
pred_2_prob = model_2.predict_proba(xtest)[:, 1]
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(ytest,pred_2_prob)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, linewidth=2)
plt.plot([0,1], [0,1], 'b--' )
score = roc_auc_score(ytest, pred_2_prob)
score
from scipy import stats
from sklearn.model_selection import KFold
clf_xgb = XGBClassifier(objective = 'binary:logistic')
param_dist = {'n_estimators': stats.randint(150, 1000),
              'learning_rate': stats.uniform(0.01, 0.6),
              'subsample': stats.uniform(0.3, 0.9),
              'max_depth': [3, 4, 5, 6, 7, 8, 9],
              'colsample_bytree': stats.uniform(0.5, 0.9),
              'min_child_weight': [1, 2, 3, 4]
             }

numFolds = 5
kfold_5 = KFold(n_splits = numFolds, shuffle = True)
from sklearn.model_selection import RandomizedSearchCV
clf = RandomizedSearchCV(clf_xgb, 
                         param_distributions = param_dist,
                         cv = kfold_5,  
                         n_iter = 5, # you want 5 here not 25 if I understand you correctly 
                         scoring = 'roc_auc', 
                         error_score = 0, 
                         verbose = 3, 
                         n_jobs = -1)
model_3 = clf.fit(xtrain,ytrain)
pre_3 = model_3.predict(xtest)
cm = confusion_matrix(ytest, pre_3)
cm
pred_3_prob = model_3.predict_proba(xtest)[:, 1]
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(ytest,pred_3_prob)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, linewidth=2)
plt.plot([0,1], [0,1], 'b--' )
score = roc_auc_score(ytest, pred_3_prob)
score
from sklearn.model_selection import KFold, GridSearchCV

estimator = XGBClassifier(
    objective= 'binary:logistic',
    nthread=4,
    seed=42
)
parameters = {
    'max_depth': range (2, 10, 1),
    'n_estimators': range(60, 220, 40),
    'learning_rate': [0.1, 0.01, 0.05]
}
grid_search = GridSearchCV(
    estimator=estimator,
    param_grid=parameters,
    scoring = 'roc_auc',
    n_jobs = 5,
    cv = 3,
    verbose=True
)
model_4 = grid_search.fit(xtrain,ytrain)
model_4.best_params_
pred_4 = model_4.predict(xtest)
cm = confusion_matrix(ytest, pred_4)
cm
pred_4_prob = model_4.predict_proba(xtest)[:, 1]

fpr, tpr, thresholds = roc_curve(ytest,pred_4_prob)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, linewidth=2)
plt.plot([0,1], [0,1], 'b--' )
score = roc_auc_score(ytest, pred_4_prob)
score
pred_4 = model_4.predict(test)
pred_4 = pd.DataFrame(pred_4)
pred_4.value_counts()
test["Response"] = pred_4
test
test.to_csv('submission_veh.csv') 
