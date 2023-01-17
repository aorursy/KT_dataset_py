import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost

from xgboost import XGBClassifier

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt 

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

from sklearn.preprocessing import StandardScaler 

import time 

%matplotlib inline
df = pd.read_csv("/kaggle/input/Churn_Modelling.csv")

df.shape
df.columns
df.head()
df.describe()
df.isnull().sum()
df.info()
fig, ax = plt.subplots(2,2)

plt.tight_layout()



df['CreditScore'].plot.box(ax=ax[0][0])

df['Age'].plot.box(ax=ax[0][1])

df['Balance'].plot.box(ax=ax[1][0])

df['EstimatedSalary'].plot.box(ax=ax[1][1])
col_names = ['CreditScore','Balance','EstimatedSalary']

features = df[col_names]

scaler = StandardScaler().fit(features.values)

features = scaler.transform(features.values)    

df[col_names] = features
Event_rate_name  = df.groupby('Surname').agg({

  'CustomerId' : 'count',

   'Exited' : 'sum', 

}).rename(columns={'CustomerId':'Total_Customers',

                   'Exited':'Churners'}).reset_index()

Event_rate_name['Event_Rate'] = round(Event_rate_name['Churners'] / Event_rate_name['Total_Customers'] ,2)

df = pd.merge(df,Event_rate_name[['Surname','Total_Customers','Event_Rate']],on='Surname',how='left')
Event_Imbalance = df.groupby('Exited').agg({'Exited':'count'})

Event_Imbalance['PERCENTAGE_CONTRIBUTION']=Event_Imbalance['Exited']/sum(Event_Imbalance['Exited'])

print(Event_Imbalance)

df['Exited'].value_counts().plot(kind='bar')
encode = df[['Geography','Gender']]

ohe = pd.get_dummies(encode)

df = df.drop(encode,axis=1)

df = pd.concat([df, ohe], axis=1)

df.shape
df = df.drop(['RowNumber','CustomerId', 'Surname'],axis=1)
X= df.drop(['Exited'],axis=1)

y= df.Exited

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=0)
clf = xgboost.XGBClassifier()
param_grid = {

        'silent': [False],

        'max_depth': [6, 10, 15, 20],

        'learning_rate': [0.001, 0.01, 0.1, 0.2, 0,3],

        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],

        'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],

        'colsample_bylevel': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],

        'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],

        'gamma': [0, 0.25, 0.5, 1.0],

        'reg_lambda': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0],

        'n_estimators': [100,200,300,500,700,1000],

        'early_stopping_rounds': [10]}



#params = {'eval_metric': 'mlogloss',

 #             'early_stopping_rounds': 10,

  #            'eval_set': [(X_val, y_val)]}
rs_clf = RandomizedSearchCV(clf, param_grid, n_iter=10,

                            n_jobs=1, verbose=2, cv=2,

                            #fit_params= params,

                            scoring='roc_auc', refit=True, random_state=0

                            )
print("Randomized search..")

search_time_start = time.time()

rs_clf.fit(X_train, y_train)

print("Randomized search time:", time.time() - search_time_start)
best_score = rs_clf.best_score_

print(best_score)

best_params = rs_clf.best_params_

print(best_params)
xg_roc_auc = roc_auc_score(y_train, rs_clf.predict(X_train))

fpr, tpr, thresholds = roc_curve(y_train, rs_clf.predict_proba(X_train)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='XgBoost (area = %0.2f)' % xg_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.savefig('XgBoost_ROC')

plt.show()
xg_roc_auc = roc_auc_score(y_val, rs_clf.predict(X_val))

fpr, tpr, thresholds = roc_curve(y_val, rs_clf.predict_proba(X_val)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='XgBoost (area = %0.2f)' % xg_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.savefig('XgBoost_ROC')

plt.show()