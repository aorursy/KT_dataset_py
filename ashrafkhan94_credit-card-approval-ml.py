import warnings
warnings.filterwarnings('ignore')

import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import plotly.express as px
import cufflinks as cf
cf.go_offline(True)
application_data = pd.read_csv('../input/credit-card-approval-prediction/application_record.csv')
credit_data = pd.read_csv('../input/credit-card-approval-prediction/credit_record.csv')

display(application_data.info())
print("\n")
display(credit_data.info())
display(application_data.head())
print('\n')
display(credit_data.head())
defaults = credit_data[['ID','MONTHS_BALANCE']].groupby('ID').agg(min).reset_index()

display(defaults.head())
print("\n")
display(defaults.info())
drop_index = application_data[application_data.duplicated(subset=['ID'], keep=False)].index

application_unique_data = application_data.drop(drop_index)

application_unique_data.shape
data = application_unique_data.merge(defaults, on='ID', how='left')
def risk(x):
    if x >= -3:
        return 'no'
    elif x < -3:
        return 'yes'
    else:
        return 'null'

data['RISK'] = data['MONTHS_BALANCE'].apply(lambda x: risk(x))
data.drop(data[data['RISK'] == 'null'].index, inplace=True)
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()

data['MALE'] = lb.fit_transform(data['CODE_GENDER'])
data['CAR'] = lb.fit_transform(data['FLAG_OWN_CAR'])
data['REALTY'] = lb.fit_transform(data['FLAG_OWN_REALTY'])

data.drop(['CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY'], axis=1, inplace=True)
data['AGE'] = data['DAYS_BIRTH'].apply(lambda x: round(abs(x/365)))
data['YEARS_EMPLOYED'] = data['DAYS_EMPLOYED'].apply(lambda x: round(abs(x/365),2))

data.drop(['DAYS_BIRTH','DAYS_EMPLOYED'], axis=1, inplace=True)
data['INCOME'] = data['AMT_INCOME_TOTAL'].apply(lambda x: x/1000)

data.drop(['AMT_INCOME_TOTAL'], axis=1, inplace=True)
pd.set_option('display.max_columns', None)
data.head()
px.box(data_frame=data, x='OCCUPATION_TYPE', y='AGE')
px.box(data_frame=data, x='OCCUPATION_TYPE', y='INCOME')

#Outliers detected in laborers, Security Staff, 
sns.lmplot(data=data[data['OCCUPATION_TYPE']=='Laborers'], x='AGE', y='INCOME', hue='RISK')

#Notice that the salary is increasing with age
sns.lmplot(data=data[data['OCCUPATION_TYPE']=='Security staff'], x='AGE', y='INCOME', hue='RISK')

#Notice that the salary is increasing with age
data.drop(data.query('OCCUPATION_TYPE=="Laborers" and AGE > 45 and INCOME > 200').index, inplace=True)
#data.drop(data.query('OCCUPATION_TYPE=="Laborers" and AGE > 55').index, inplace=True)
data.drop(data.query('OCCUPATION_TYPE=="Laborers" and INCOME > 400').index, inplace=True)
sns.lmplot(data=data[data['OCCUPATION_TYPE']=='Laborers'], x='AGE', y='INCOME', hue='RISK')
data.drop(data.query('OCCUPATION_TYPE=="Security staff" and AGE > 50 and INCOME > 200').index, inplace=True)
#data.drop(data.query('OCCUPATION_TYPE=="Laborers" and AGE > 55').index, inplace=True)
data.drop(data.query('OCCUPATION_TYPE=="Security staff" and INCOME > 400').index, inplace=True)
sns.lmplot(data=data[data['OCCUPATION_TYPE']=='Security staff'], x='AGE', y='INCOME', hue='RISK')
data['RISK'] = lb.fit_transform(data['RISK'])
data.head()
display(data['NAME_INCOME_TYPE'].value_counts())

display(data['NAME_EDUCATION_TYPE'].value_counts())

display(data['NAME_FAMILY_STATUS'].value_counts())

display(data['NAME_HOUSING_TYPE'].value_counts())
data['MARRIED'] = data['NAME_FAMILY_STATUS'].apply(lambda x: 1 if ((x == 'Married') or (x == 'Civil marriage')) else 0)
DEGREE = pd.get_dummies(data['NAME_EDUCATION_TYPE'], drop_first=True)
OCCUPATION = pd.get_dummies(data['NAME_INCOME_TYPE'], drop_first=True)
#HOUSE = pd.get_dummies(data['NAME_HOUSING_TYPE'], drop_first=True)
data = pd.concat([data, DEGREE, OCCUPATION], axis=1)
data.head()
data_occp = data.groupby(['NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','OCCUPATION_TYPE']).size().rename("count").reset_index()

display(data_occp)
data_occp_final = data_occp[data_occp.groupby(['NAME_INCOME_TYPE','NAME_EDUCATION_TYPE'])['count'].transform(max) == data_occp['count']]

data_occp_final
def fill_occp(values):
    profession = values[0]
    degree = values[1]
    occupation = values[2]
    if pd.isnull(occupation):
        for index,row in data_occp_final.iterrows():
            if ((row['NAME_INCOME_TYPE'] == profession) and (row['NAME_EDUCATION_TYPE'] == degree)):
                return row['OCCUPATION_TYPE']
    else:
        return occupation
            
            
data['OCCUPATION'] = data[['NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','OCCUPATION_TYPE']].apply(fill_occp,axis=1)
OCCUPATION = pd.get_dummies(data['OCCUPATION'], drop_first=True)

OCCUPATION.head()
px.box(data_frame=data, x='OCCUPATION_TYPE', y='INCOME')
OCCUPATION['CLASS 3 WORKERS'] = OCCUPATION[['Cleaning staff','Cooking staff','Drivers','Laborers','Low-skill Laborers','Security staff','Waiters/barmen staff']].sum(axis=1)
OCCUPATION['CLASS 2 WORKERS'] = OCCUPATION[['HR staff','Sales staff','Secretaries','Medicine staff','Private service staff']].sum(axis=1)
OCCUPATION['CLASS 1 WORKERS'] = OCCUPATION[['Managers','Core staff','High skill tech staff','IT staff','Realty agents']].sum(axis=1)


OCCUPATION.drop(['Cleaning staff','Cooking staff','Drivers','Laborers','Low-skill Laborers','Security staff','Waiters/barmen staff','HR staff','Sales staff','Secretaries','Medicine staff','Private service staff','Managers','High skill tech staff','IT staff','Realty agents','Core staff'], axis=1,inplace=True)

OCCUPATION.head()
data = pd.concat([data, OCCUPATION], axis=1)
data['NAME_HOUSING_TYPE'].value_counts()
data['OWN_HOUSE'] = data['NAME_HOUSING_TYPE'].apply(lambda x: 1 if x == 'House / apartment' else 0)
data.drop(['ID','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','OCCUPATION_TYPE','MONTHS_BALANCE','OCCUPATION'], axis=1, inplace=True)

data.rename(columns={'CNT_CHILDREN':'CHILD','FLAG_MOBIL':'MOBIL','FLAG_PHONE':'PHONE','FLAG_EMAIL':'EMAIL','CNT_FAM_MEMBERS':'FAMILY_MEMBERS'}, inplace=True)
data.head()
from imblearn.over_sampling import SMOTE

X = data.drop('RISK', axis=1)
y = data['RISK']


X_bal,y_bal = SMOTE().fit_sample(X,y)

X_bal = pd.DataFrame(X_bal, columns=X.columns)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_bal,y_bal,stratify = y_bal,test_size=0.3, random_state=123)
from sklearn.linear_model import LogisticRegression

logReg = LogisticRegression()

logReg.fit(X_train, y_train)

y_pred = logReg.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score

print("Train Acc")
print(logReg.score(X_train, y_train))
print("\nTest Acc")
print(logReg.score(X_test, y_test))

print("\nAUC")
print(roc_auc_score(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
from sklearn.model_selection import RandomizedSearchCV

C = np.logspace(-3,1,5)
max_iter = [50,80,100,120,150,180,200]
penalty = ['l2']
tol = np.logspace(-8,-3,6)

param = {'C':C,'max_iter':max_iter,'tol':tol,'penalty':penalty}

logReg_cv = RandomizedSearchCV(estimator=logReg, 
                               param_distributions=param, 
                               n_iter=50, 
                               cv=10, 
                               scoring='accuracy', 
                               verbose=3, n_jobs=-1)

logReg_cv.fit(X_train, y_train)
logReg_best = logReg_cv.best_estimator_

y_pred = logReg_best.predict(X_test)
logReg_cv.best_params_
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score

print("Train Acc")
print(logReg_best.score(X_train, y_train))
print("\nTest Acc")
print(logReg_best.score(X_test, y_test))

print("\nAUC")
print(roc_auc_score(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
from sklearn.svm import LinearSVC,SVC

svm = LinearSVC()

svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)
print("Train Acc")
print(svm.score(X_train, y_train))
print("\nTest Acc")
print(svm.score(X_test, y_test))

print("\nAUC")
print(roc_auc_score(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
C = np.logspace(-4,1,6)
tol = np.logspace(-6,-2,5)


param = {'C':C, 'tol':tol}
svm_cv = RandomizedSearchCV(estimator=svm, 
                               param_distributions=param, 
                               n_iter=50, 
                               cv=10, 
                               scoring='accuracy', 
                               verbose=3, n_jobs=-1)

svm_cv.fit(X_train, y_train)

y_pred = svm_cv.predict(X_test)
svm_cv.best_params_
print("Train Acc")
print(svm_cv.score(X_train, y_train))
print("\nTest Acc")
print(svm_cv.score(X_test, y_test))

print("\nAUC")
print(roc_auc_score(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))
from sklearn.neighbors import KNeighborsClassifier
neighbours = np.arange(1,12)

train_acc = np.empty(len(neighbours))
test_acc = np.empty(len(neighbours))

for i,n in enumerate(neighbours):
    knn = KNeighborsClassifier(n_neighbors=n)
    
    knn.fit(X_train, y_train)
    train_acc[i] = knn.score(X_train, y_train)
    test_acc[i] = knn.score(X_test, y_test)
    

plt.plot(neighbours, train_acc, label='tain')
plt.plot(neighbours, test_acc, label='test')
knn_1 = KNeighborsClassifier(n_neighbors=1)

knn_1.fit(X_train, y_train)
y_pred = knn_1.predict(X_test)
print('Train Acc')
print(knn_1.score(X_train, y_train))
print()
print('Test Acc')
print(knn_1.score(X_test, y_test))


print('Confusion_matrix')
print(confusion_matrix(y_test, y_pred))

print('Report')
print(classification_report(y_test, y_pred))

print("\nAUC")
print(roc_auc_score(y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


rf = RandomForestClassifier()

params = {
                'max_depth' : [3,4,5,6],
                'min_samples_leaf' : [0.02,0.04,0.06],
                'max_features' : [0.2,0.4,0.8],
                'n_estimators' : [150,200,250]
                
        }

rf_cv = RandomizedSearchCV(estimator=rf,
                          param_distributions=params,
                           n_iter=100,
                          cv=3,
                          scoring='accuracy',
                          n_jobs=-1,
                           verbose=3,
                           random_state=123
                          )


rf_cv.fit(X_train, y_train)
rf_cv.best_params_
y_pred = rf_cv.predict(X_test)
print('Train Acc')
print(rf_cv.score(X_train, y_train))
print()
print('Test Acc')
print(rf_cv.score(X_test, y_test))


print('Confusion_matrix')
print(confusion_matrix(y_test, y_pred))

print('Report')
print(classification_report(y_test, y_pred))

print("\nAUC")
print(roc_auc_score(y_test, y_pred))
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV



params_xgb = {
                'max_depth' : [3,4,5,6],
                'min_samples_leaf' : [0.02,0.04,0.05,0.06],
                'max_features' : [0.2,0.4,0.6,0.8,0.9],
                'n_estimators' : [150,200,250],
                'subsample' : np.arange(0.05,1.05,0.1),
                'learning_rate' : np.arange(0.05,1.05,0.1),
                'colsample_bytree' : np.arange(0.05,1.05,0.1),
                'gamma' : [0.05,0.1,0.5,1]
                           
            }

gbm = xgb.XGBClassifier()

xgb_cv =     RandomizedSearchCV(estimator=gbm,
                                n_iter=150,
                                param_distributions=params_xgb, 
                                cv=3, scoring='accuracy',
                                n_jobs=-1, verbose=2,
                               random_state=123)
xgb_cv.fit(X_train, y_train)
xgb_cv.best_params_
xgb_pred = xgb_cv.predict(X_test)

print('Train Acc')
print(xgb_cv.score(X_train, y_train))
print("\n")
print('Test Acc')
print(xgb_cv.score(X_test, y_test))

from sklearn.metrics import confusion_matrix, classification_report

print('Confusion_matrix')
print(confusion_matrix(y_test, xgb_pred))

print('Report')
print(classification_report(y_test, xgb_pred))