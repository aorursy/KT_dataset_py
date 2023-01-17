# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import sklearn
table=pd.read_csv('../input/application_train.csv')
table_test=pd.read_csv('../input/application_test.csv')
table.head()
import xgboost as xgb
table.loc[:,'ratio_income_amtcredit']=table.loc[:,'AMT_INCOME_TOTAL']/table.loc[:,'AMT_CREDIT']
table_test.loc[:,'ratio_income_amtcredit']=table_test.loc[:,'AMT_INCOME_TOTAL']/table_test.loc[:,'AMT_CREDIT']

clf = xgb.XGBClassifier(n_estimators=1500, objective='binary:logistic', gamma=0.1, subsample=0.5 )
X_train=table.loc[:,('ratio_income_amtcredit','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','DAYS_EMPLOYED','DAYS_BIRTH','FLAG_MOBIL','CNT_FAM_MEMBERS')]
y_train=table.loc[:,"TARGET"]

X_test=table_test.loc[:,('ratio_income_amtcredit','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','DAYS_EMPLOYED','DAYS_BIRTH','FLAG_MOBIL','CNT_FAM_MEMBERS')]

X_train=np.array(X_train)
X_test=np.array(X_test)
y_train=np.array(y_train)
clf.fit(X_train,y_train, eval_set=[(X_train, y_train)], eval_metric='auc', early_stopping_rounds=10)
predictions_train = clf.predict_proba(X_train)
predictions_train=predictions_train[:,1].reshape(predictions_train[:,1].shape[0],1)
predictions_train.shape

sklearn.metrics.roc_auc_score(y_train, predictions_train[:,0])
predictions_test = clf.predict_proba(X_test)
predictions_test.shape
predictions_test=predictions_test[:,1].reshape(predictions_test.shape[0],1)
predictions_test.shape
envio=np.column_stack((table_test.loc[:,"SK_ID_CURR"],predictions_test))
envio.shape
df=pd.DataFrame(envio, columns=['SK_ID_CURR','TARGET'])
df.to_csv('resultado_1.csv', index = False)
