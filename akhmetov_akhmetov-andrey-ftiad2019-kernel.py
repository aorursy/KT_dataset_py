# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('/kaggle/input/hse-practical-ml-1/car_loan_train.csv', sep=',')

df.describe()
from datetime import datetime

df['AVERAGE.ACCT.AGE']=df['AVERAGE.ACCT.AGE'].apply(lambda x: (int(x.split(' ')[0][:-3])*356+int(x.split(' ')[1][:-3])*30)/365)

df['CREDIT.HISTORY.LENGTH']=df['CREDIT.HISTORY.LENGTH'].apply(lambda x: (int(x.split(' ')[0][:-3])*356+int(x.split(' ')[1][:-3])*30)/365)

df['age']=df['Date.of.Birth'].apply(lambda x: (datetime.strptime('01-01-2020', '%d-%m-%Y') - datetime.strptime(x[:-2]+'20'+x[-2:], '%d-%m-%Y')).days/365 

                          if x[-2:]<'20' 

                          else (datetime.strptime('01-01-2020', '%d-%m-%Y') - datetime.strptime(x[:-2]+'19'+x[-2:], '%d-%m-%Y')).days/365 ).astype(int)

df.drop('Date.of.Birth', inplace=True, axis=1)

df['Employment.Type']=df['Employment.Type'].astype('category').cat.codes

df['PERFORM_CNS.SCORE.DESCRIPTION']=df['PERFORM_CNS.SCORE.DESCRIPTION'].astype('category').cat.codes

df['DisbursalDate']=df['DisbursalDate'].apply(lambda x: (datetime.strptime('01-01-2020', '%d-%m-%Y') - datetime.strptime(x[:-2]+'20'+x[-2:], '%d-%m-%Y')).days/365 

                          if x[-2:]<'20' 

                          else (datetime.strptime('01-01-2020', '%d-%m-%Y') - datetime.strptime(x[:-2]+'19'+x[-2:], '%d-%m-%Y')).days/365 ).astype(float)

df['diff']=df['PRI.SANCTIONED.AMOUNT'] - df['PRI.DISBURSED.AMOUNT']



UniqueID='UniqueID'

traget='target'

text_cat=['Employment.Type','PERFORM_CNS.SCORE.DESCRIPTION']

real=['DisbursalDate','disbursed_amount', 'asset_cost','ltv', 'PERFORM_CNS.SCORE','age','PRI.NO.OF.ACCTS', 'PRI.ACTIVE.ACCTS', 

'PRI.OVERDUE.ACCTS', 'PRI.CURRENT.BALANCE', 'PRI.SANCTIONED.AMOUNT', 'PRI.DISBURSED.AMOUNT', 'SEC.NO.OF.ACCTS', 'SEC.ACTIVE.ACCTS',

'SEC.OVERDUE.ACCTS', 'SEC.CURRENT.BALANCE', 'SEC.SANCTIONED.AMOUNT', 'SEC.DISBURSED.AMOUNT', 'PRIMARY.INSTAL.AMT',

'SEC.INSTAL.AMT', 'NEW.ACCTS.IN.LAST.SIX.MONTHS', 'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS','AVERAGE.ACCT.AGE', 'CREDIT.HISTORY.LENGTH', 

'NO.OF_INQUIRIES','magic_0', 'magic_1', 'magic_2', 'magic_3', 'magic_4', 'magic_5', 'f1', 'f2', 'diff']

cat=['branch_id', 'supplier_id', 'manufacturer_id', 'Current_pincode_ID', 'Employment.Type', 

 'State_ID', 'Employee_code_ID', 'MobileNo_Avl_Flag', 'Aadhar_flag', 'PAN_flag', 'VoterID_flag', 'Driving_flag',

 'Passport_flag', 'PERFORM_CNS.SCORE.DESCRIPTION']



features=cat+real
import xgboost as xgb

import numpy as np

def prec_xgb(n_trees, max_depth, X_train, y_train, X_test, y_test, learning_rate=0.1):

    clf=xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

                  colsample_bynode=1, colsample_bytree=1, gamma=0,

                  learning_rate=0.1, max_delta_step=0, max_depth=3,

                  min_child_weight=1, missing=None, n_estimators=300, n_jobs=1,

                  nthread=None, objective='binary:logistic', random_state=42,

                  reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

                  silent=None, subsample=1, verbosity=1)

#     clf = xgb.XGBClassifier(n_estimators=n_trees, max_depth=max_depth, objective='binary:logistic',

#             seed=0, silent=True, nthread=-1, learning_rate=learning_rate, num_classes=2)

    eval_set = [(X_test, y_test)]

    clf.fit(X_train, y_train, eval_set=eval_set, eval_metric="auc")

    y_pred = clf.predict(X_test)

    prec = float(np.sum(y_pred == y_test)) / len(y_test)

    ra=roc_auc_score(y_pred,y_test)



    return clf, y_pred, prec, ra
from catboost import CatBoostClassifier

def prec_cat(n_trees, max_depth, X_train, y_train, X_test, y_test, learning_rate=0.1):

    clf=CatBoostClassifier(iterations=n_trees,

                          learning_rate=learning_rate,

                          depth=max_depth)

    eval_set = [(X_test, y_test)]

    clf.fit(X_train, y_train, [X_train.columns.get_loc(i) for i in cat])

    y_pred = clf.predict_proba(X_test)

    prec = float(np.sum(y_pred[:,0] == y_test)) / len(y_test)

    ra=roc_auc_score(y_test, y_pred[:,1])

    print(ra)

    return clf, y_pred, prec, ra
def transform_cat(dd, dd_test):

    for i in [j for j in dd.columns if j!=0]:

        mean_some=dd.groupby(i)['target'].mean()

        dd[i]=dd[i].map(mean_some)

        dd_test[i]=dd_test[i].map(mean_some)

    dd.columns=[i+'_mean_enc' for i in list(dd.columns)]

    dd_test.columns=[i+'_mean_enc' for i in list(dd_test.columns)]

    del dd['target_mean_enc']

    del dd_test['target_mean_enc']

    return dd, dd_test
columns_to_drop = ['UniqueID','MobileNo_Avl_Flag','DisbursalDate','AVERAGE.ACCT.AGE','CREDIT.HISTORY.LENGTH','SEC.OVERDUE.ACCTS']

features2=[i for i in features if i not in columns_to_drop]
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import KFold

kf = KFold(n_splits = 3, shuffle = True, random_state=3)

for tr_ind, val_ind in kf.split(df):

    clf, y_pred, accur, ra =prec_cat(200, 3, 

            df.loc[tr_ind,features], df.loc[tr_ind,'target'], 

            df.loc[val_ind,features], df.loc[val_ind,'target'], learning_rate=0.1)
clf, y_pred, accur, ra =prec_cat(200, 3, 

        df.loc[:,features], df.loc[:,'target'], 

        df.loc[val_ind,features], df.loc[val_ind,'target'], learning_rate=0.1)
df_test=pd.read_csv('/kaggle/input/hse-practical-ml-1/car_loan_test.csv', sep=',')

df_test.head()
from datetime import datetime

df_test['AVERAGE.ACCT.AGE']=df_test['AVERAGE.ACCT.AGE'].apply(lambda x: (int(x.split(' ')[0][:-3])*356+int(x.split(' ')[1][:-3])*30)/365)

df_test['CREDIT.HISTORY.LENGTH']=df_test['CREDIT.HISTORY.LENGTH'].apply(lambda x: (int(x.split(' ')[0][:-3])*356+int(x.split(' ')[1][:-3])*30)/365)

df_test['age']=df_test['Date.of.Birth'].apply(lambda x: (datetime.strptime('01-01-2020', '%d-%m-%Y') - datetime.strptime(x[:-2]+'20'+x[-2:], '%d-%m-%Y')).days/365 

                          if x[-2:]<'20' 

                          else (datetime.strptime('01-01-2020', '%d-%m-%Y') - datetime.strptime(x[:-2]+'19'+x[-2:], '%d-%m-%Y')).days/365 ).astype(int)

df_test.drop('Date.of.Birth', inplace=True, axis=1)

df_test['Employment.Type']=df_test['Employment.Type'].astype('category').cat.codes

df_test['PERFORM_CNS.SCORE.DESCRIPTION']=df_test['PERFORM_CNS.SCORE.DESCRIPTION'].astype('category').cat.codes

df_test['DisbursalDate']=df_test['DisbursalDate'].apply(lambda x: (datetime.strptime('01-01-2020', '%d-%m-%Y') - datetime.strptime(x[:-2]+'20'+x[-2:], '%d-%m-%Y')).days/365 

                          if x[-2:]<'20' 

                          else (datetime.strptime('01-01-2020', '%d-%m-%Y') - datetime.strptime(x[:-2]+'19'+x[-2:], '%d-%m-%Y')).days/365 ).astype(float)

df_test['diff']=df_test['PRI.SANCTIONED.AMOUNT'] - df_test['PRI.DISBURSED.AMOUNT']



y_pred = clf.predict_proba(df_test[features])



res=pd.DataFrame(y_pred[:,1])

res.columns=['Predicted']

submit = pd.DataFrame()

submit['ID'] = [i for i in range(len(df_test['UniqueID']))]

submit['Predicted'] = y_pred[:,1]



submit.to_csv('submission.csv', index=False)



# res=res.reset_index()

# res.columns=['ID', 'Predicted']

# res.head()
#res.to_csv('sub_nis.csv',  index=None)

#res.to_csv('/kaggle/output/sub_nis.csv')