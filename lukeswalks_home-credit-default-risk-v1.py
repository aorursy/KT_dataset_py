#import libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import make_classification

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import xgboost as xgb

%matplotlib inline

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)
#import data

application_test = pd.read_csv("../input/home-credit-default-risk/application_test.csv")

application_train = pd.read_csv("../input/home-credit-default-risk/application_train.csv")

bureau = pd.read_csv("../input/home-credit-default-risk/bureau.csv")

bureau_balance = pd.read_csv("../input/home-credit-default-risk/bureau_balance.csv")

credit_card_balance = pd.read_csv("../input/home-credit-default-risk/credit_card_balance.csv")

installments_payments = pd.read_csv("../input/home-credit-default-risk/installments_payments.csv")

POS_CASH_balance = pd.read_csv("../input/home-credit-default-risk/POS_CASH_balance.csv")

previous_application = pd.read_csv("../input/home-credit-default-risk/previous_application.csv")

sample_submission = pd.read_csv("../input/home-credit-default-risk/sample_submission.csv")
application_test.head()
#application_train.head()

application_train[application_train["SK_ID_CURR"]==176158]
bureau.head()
bureau_balance.head()
credit_card_balance.head()
installments_payments.head()
POS_CASH_balance.head()
previous_application.head()
sample_submission.head()
# Missing Values Bureau

((bureau.isnull().sum()/len(bureau))*100).sort_values(ascending = False)
# Missing Values Bureau

((bureau.isnull().sum()/len(bureau))*100).sort_values(ascending = False)
# Missing Values Previous_Application 

((previous_application.isnull().sum()/len(previous_application))*100).sort_values(ascending = False)
# Missing Values Credit Card Balance 

((credit_card_balance.isnull().sum()/len(credit_card_balance))*100).sort_values(ascending = False)
#BUREAU_PREVIOUS_LOAN_COUNTS

BUREAU_PREVIOUS_LOAN_COUNTS = bureau.groupby('SK_ID_CURR', as_index=False)['SK_ID_BUREAU'].count().rename(columns = {'SK_ID_BUREAU': 'BUREAU_PREVIOUS_LOAN_COUNTS'})

application_train = application_train.merge(BUREAU_PREVIOUS_LOAN_COUNTS, on = 'SK_ID_CURR', how = 'left')

application_train['BUREAU_PREVIOUS_LOAN_COUNTS'] = application_train['BUREAU_PREVIOUS_LOAN_COUNTS'].fillna(0)
#BUREAU_PREVIOUS_PAID_LOAN_COUNTS

BUREAU_PREVIOUS_PAID_LOAN_COUNTS = bureau[bureau['CREDIT_ACTIVE']=="Closed"].groupby('SK_ID_CURR', as_index=False)['SK_ID_BUREAU'].count().rename(columns = {'SK_ID_BUREAU': 'BUREAU_PREVIOUS_PAID_LOAN_COUNTS'})

application_train = application_train.merge(BUREAU_PREVIOUS_PAID_LOAN_COUNTS, on = 'SK_ID_CURR', how = 'left')

application_train['BUREAU_PREVIOUS_PAID_LOAN_COUNTS'] = application_train['BUREAU_PREVIOUS_PAID_LOAN_COUNTS'].fillna(0)
#PREVIOUS_APPLICATION_PREVIOUS_LOAN_COUNTS

PREVIOUS_APPLICATION_PREVIOUS_LOAN_COUNTS = previous_application.groupby('SK_ID_CURR', as_index=False)['SK_ID_PREV'].count().rename(columns = {'SK_ID_PREV': 'PREVIOUS_APPLICATION_PREVIOUS_LOAN_COUNTS'})

application_train = application_train.merge(PREVIOUS_APPLICATION_PREVIOUS_LOAN_COUNTS, on = 'SK_ID_CURR', how = 'left')

application_train['PREVIOUS_APPLICATION_PREVIOUS_LOAN_COUNTS'] = application_train['PREVIOUS_APPLICATION_PREVIOUS_LOAN_COUNTS'].fillna(0)
#PREVIOUS_APPLICATION_COUNT_OF_MISSED_INSTALLMENTS

installments_payments["Missed"] =  installments_payments['AMT_PAYMENT'] < installments_payments['AMT_INSTALMENT']

PREVIOUS_APPLICATION_COUNT_OF_MISSED_INSTALLMENTS = installments_payments[installments_payments['Missed']==True].groupby('SK_ID_CURR', as_index=False)['SK_ID_PREV'].count().rename(columns = {'SK_ID_PREV': 'PREVIOUS_APPLICATION_COUNT_OF_MISSED_INSTALLMENTS'})

application_train = application_train.merge(PREVIOUS_APPLICATION_COUNT_OF_MISSED_INSTALLMENTS, on = 'SK_ID_CURR', how = 'left')

application_train['PREVIOUS_APPLICATION_COUNT_OF_MISSED_INSTALLMENTS'] = application_train['PREVIOUS_APPLICATION_COUNT_OF_MISSED_INSTALLMENTS'].fillna(0)
#sum of AMT_CREDIT_SUM_DEBT

bureau['AMT_CREDIT_SUM_DEBT'] = bureau['AMT_CREDIT_SUM_DEBT'].fillna(0)

AMT_CREDIT_SUM_DEBT = bureau.groupby('SK_ID_CURR', as_index=False)['SK_ID_BUREAU'].sum().rename(columns = {'SK_ID_BUREAU': 'AMT_CREDIT_SUM_DEBT'})

application_train = application_train.merge(AMT_CREDIT_SUM_DEBT, on = 'SK_ID_CURR', how = 'left')

application_train['AMT_CREDIT_SUM_DEBT'] = application_train['AMT_CREDIT_SUM_DEBT'].fillna(0)
#sum of CREDIT_CARD_BALANCE_DEBT

AMT_BALANCE = credit_card_balance.groupby('SK_ID_CURR', as_index=False)['SK_ID_PREV'].sum().rename(columns = {'SK_ID_PREV': 'CREDIT_CARD_AMT_BALANCE'})

application_train = application_train.merge(AMT_BALANCE, on = 'SK_ID_CURR', how = 'left')

application_train['CREDIT_CARD_AMT_BALANCE'] = application_train['CREDIT_CARD_AMT_BALANCE'].fillna(0)
#sum of CREDIT_DAY_OVERDUE

bureau['CREDIT_DAY_OVERDUE'] = bureau['CREDIT_DAY_OVERDUE'].fillna(0)

credit_day_overdue = bureau[bureau['CREDIT_DAY_OVERDUE']>0].groupby('SK_ID_CURR', as_index=False)['SK_ID_BUREAU'].sum().rename(columns = {'SK_ID_BUREAU': 'CREDIT_DAY_OVERDUE_sum'})

application_train = application_train.merge(credit_day_overdue, on = 'SK_ID_CURR', how = 'left')

application_train['CREDIT_DAY_OVERDUE_sum'] = application_train['CREDIT_DAY_OVERDUE_sum'].fillna(0)
#Categorical Variables

categorical = pd.get_dummies(application_train.select_dtypes('object'))

categorical['CNT_CHILDREN'] = application_train['CNT_CHILDREN']

categorical['CNT_FAM_MEMBERS'] = application_train['CNT_FAM_MEMBERS']

categorical['SK_ID_CURR'] = application_train['SK_ID_CURR']

application_train = application_train.merge(categorical, on = 'SK_ID_CURR', how = 'left')

application_train = application_train.drop(application_train.select_dtypes('object').columns, axis=1)

# did not see any important variables that should be label encoded
application_train.info(verbose=True)
# Missing Values application_train

null = pd.DataFrame(((application_train.isnull().sum()/len(application_train))*100).sort_values(ascending = False))

corr = pd.DataFrame(application_train.corr(method='pearson')['TARGET'].sort_values(ascending=False))
nullandcorr = null.merge(corr,left_index = True,right_index = True)
nullandcorr["TARGET"] = nullandcorr["TARGET"].apply(lambda d: abs(d))

nullandcorr.sort_values(0,ascending=False)

application_train.loc[:,["EXT_SOURCE_1","EXT_SOURCE_3"]] = application_train[["EXT_SOURCE_1","EXT_SOURCE_3"]].apply(lambda x: x.fillna(x.median()),axis=0)

application_train.loc[:,list(nullandcorr[nullandcorr[0]<19][nullandcorr[0]>0].index)]= application_train.loc[:,list(nullandcorr[nullandcorr[0]<19][nullandcorr[0]>0].index)].apply(lambda x: x.fillna(x.median()),axis=0)
g = pd.DataFrame(((application_train.isnull().sum()/len(application_train))*100).sort_values(ascending = False))

application_train = application_train.drop(columns = list(g[g[0]>0].index))
# Multicollinearity

#from statsmodels.stats.outliers_influence import variance_inflation_factor

#vifcalc= application_train.drop(columns="SK_ID_CURR")

#vif = pd.DataFrame()

#vif["VIF Factor"] = [variance_inflation_factor(vifcalc.values, i) for i in range(vifcalc.shape[1])]

#vif["features"] = vifcalc.columns

#takes too long, alternative method needed
corr
y = application_train['TARGET']

x = application_train.set_index("SK_ID_CURR").drop(columns="TARGET")
# Scaling of Data probably with Standard Scalar

scalar=StandardScaler()

x = scalar.fit_transform(x)
param_grid = {'penalty' : ['l1', 'l2'],'C' : [0.05,0.1,0.5,1]}

clf = RandomizedSearchCV(LogisticRegression(solver='saga'), param_distributions = param_grid, cv = 5, verbose=True, n_jobs=-1)

tunedclf = clf.fit(x,y)
tunedclf.best_params_
logreg = LogisticRegression(penalty = "l2", C=0.1, solver="liblinear")

print("Logistic Regression average auc: ",cross_val_score(logreg, x, y, cv=5, scoring='roc_auc', n_jobs=-1))
logreg.fit(x,y)

preds=logreg.predict(x)

print(confusion_matrix(y,preds,labels=[1,0]))
param_grid = {'n_estimators': [10,50,100,200]}

clfRFC = RandomizedSearchCV(RandomForestClassifier(),param_distributions= param_grid,cv=5,verbose=True,n_jobs=-1)

tunedclfRFC = clfRFC.fit(x,y)
tunedclfRFC.best_params_
RFC = RandomForestClassifier(n_estimators = 50)

print("RandomForestClassifier average auc: ",cross_val_score(RFC, x, y, cv=5, scoring='roc_auc',n_jobs=-1))
RFC.fit(x,y)

RFCpreds=RFC.predict(x)
print(confusion_matrix(y,RFCpreds,labels=[1,0]))
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30)

dtrain = xgb.DMatrix(X_train,label=y_train)

dtest = xgb.DMatrix(X_test)



params = {"max_depth":2, "eta":0.1}

model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)



model_xgb = xgb.XGBClassifier(n_estimators=360, max_depth=2, learning_rate=0.1)

model_xgb.fit(X_train, y_train,eval_metric='auc')
print("XGBoostClassifier average auc: ",cross_val_score(model_xgb, X_train, y_train, cv=5,scoring='roc_auc',n_jobs=-1))
XGBpreds=model_xgb.predict(X_test)

print(confusion_matrix(y_test,XGBpreds,labels=[1,0]))