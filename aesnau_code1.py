# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# load data
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

# remove constant columns
remove = []
for col in df_train.columns:
    if df_train[col].std() == 0:
        remove.append(col)
col2 = ('imp_ent_var16_ult1', 'imp_op_var39_comer_ult1',
       'imp_op_var39_comer_ult3', 'imp_op_var40_comer_ult1',
       'imp_op_var40_comer_ult3', 'imp_op_var40_efect_ult1',
       'imp_op_var40_efect_ult3','saldo_medio_var33_hace2', 'saldo_medio_var33_hace3',
       'saldo_medio_var33_ult1', 'saldo_medio_var33_ult3',
       'saldo_medio_var44_hace2', 'saldo_medio_var44_hace3',
       'saldo_medio_var44_ult1', 'saldo_medio_var44_ult3','imp_op_var40_ult1', 'imp_op_var41_comer_ult1',
       'imp_op_var41_comer_ult3', 'imp_op_var41_efect_ult1',
       'imp_op_var41_efect_ult3', 'imp_op_var41_ult1',
       'imp_op_var39_efect_ult1', 'saldo_medio_var17_hace2', 'saldo_medio_var17_hace3',
       'saldo_medio_var17_ult1', 'saldo_medio_var17_ult3',
       'saldo_medio_var29_hace2', 'saldo_medio_var29_hace3',
       'saldo_medio_var29_ult1', 'saldo_medio_var29_ult3',  'saldo_medio_var13_corto_ult3', 'saldo_medio_var13_largo_hace2',
       'saldo_medio_var13_largo_hace3', 'saldo_medio_var13_largo_ult1',
       'saldo_medio_var13_largo_ult3', 'saldo_medio_var13_medio_hace2',
       'saldo_medio_var13_medio_ult1', 'saldo_medio_var13_medio_ult3', 'imp_op_var39_efect_ult3', 'imp_op_var39_ult1',
       'imp_sal_var16_ult1','saldo_medio_var8_ult3', 'saldo_medio_var12_hace2',
       'saldo_medio_var12_hace3', 'saldo_medio_var12_ult1',
       'saldo_medio_var12_ult3', 'saldo_medio_var13_corto_hace2',
       'saldo_medio_var13_corto_hace3', 'saldo_medio_var13_corto_ult1', 'saldo_medio_var5_hace2', 'saldo_medio_var5_hace3',
       'saldo_medio_var5_ult1', 'saldo_medio_var5_ult3',
       'saldo_medio_var8_hace2', 'saldo_medio_var8_hace3',
       'saldo_medio_var8_ult1','num_trasp_var33_in_ult1', 'num_trasp_var33_out_ult1',
       'num_venta_var44_hace3', 'num_venta_var44_ult1', 'num_var45_hace2',
       'num_var45_hace3', 'num_var45_ult1', 'num_var45_ult3','num_sal_var16_ult1', 'num_var43_emit_ult1', 'num_var43_recib_ult1',
       'num_trasp_var11_ult1', 'num_trasp_var17_in_hace3',
       'num_trasp_var17_in_ult1', 'num_trasp_var17_out_ult1',
       'num_trasp_var33_in_hace3','num_op_var41_efect_ult1', 'num_op_var41_efect_ult3',
       'num_op_var39_efect_ult1', 'num_op_var39_efect_ult3',
       'num_reemb_var13_ult1', 'num_reemb_var17_hace3', 'num_reemb_var17_ult1',
       'num_reemb_var33_ult1',   'num_op_var39_comer_ult1', 'num_op_var39_comer_ult3',
       'num_op_var40_comer_ult1', 'num_op_var40_comer_ult3',
       'num_op_var40_efect_ult1', 'num_op_var40_efect_ult3',
       'num_op_var41_comer_ult1', 'num_op_var41_comer_ult3',)        
for col in col2:
        remove.append(col)
df_train.drop(remove, axis=1, inplace=True)
df_test.drop(remove, axis=1, inplace=True)

# remove duplicated columns
remove = []
c = df_train.columns
for i in range(len(c)-1):
    v = df_train[c[i]].values
    for j in range(i+1,len(c)):
        if np.array_equal(v,df_train[c[j]].values):
            remove.append(c[j])

df_train.drop(remove, axis=1, inplace=True)
df_test.drop(remove, axis=1, inplace=True)

y_train = df_train['TARGET'].values
X_train = df_train.drop(['ID','TARGET'], axis=1).values

id_test = df_test['ID']
X_test = df_test.drop(['ID'], axis=1).values

# length of dataset
len_train = len(X_train)
len_test  = len(X_test)

# classifier
clf = xgb.XGBClassifier(missing=np.nan,objective = "reg:logistic", max_depth=6, n_estimators=325, learning_rate=0.05, nthread=4, subsample=1, colsample_bytree=1, seed=0, base_score = 0.01)

X_fit, X_eval, y_fit, y_eval= train_test_split(X_train, y_train, test_size=0.1)

# fitting
clf.fit(X_train, y_train, early_stopping_rounds=20, eval_metric="auc", eval_set=[(X_eval, y_eval)])

print('Overall AUC:', roc_auc_score(y_train, clf.predict_proba(X_train)[:,1]))

# predicting
y_pred= clf.predict_proba(X_test)[:,1]

submission = pd.DataFrame({"ID":id_test, "TARGET":y_pred})
submission.to_csv("submission8.csv", index=False)

print('Completed!')
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# load data
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

# remove constant columns
remove = []
for col in df_train.columns:
    if df_train[col].std() == 0:
        remove.append(col)
col2 = ('imp_ent_var16_ult1', 'imp_op_var39_comer_ult1',
       'imp_op_var39_comer_ult3', 'imp_op_var40_comer_ult1',
       'imp_op_var40_comer_ult3', 'imp_op_var40_efect_ult1',
       'imp_op_var40_efect_ult3','saldo_medio_var33_hace2', 'saldo_medio_var33_hace3',
       'saldo_medio_var33_ult1', 'saldo_medio_var33_ult3',
       'saldo_medio_var44_hace2', 'saldo_medio_var44_hace3',
       'saldo_medio_var44_ult1', 'saldo_medio_var44_ult3','imp_op_var40_ult1', 'imp_op_var41_comer_ult1',
       'imp_op_var41_comer_ult3', 'imp_op_var41_efect_ult1',
       'imp_op_var41_efect_ult3', 'imp_op_var41_ult1',
       'imp_op_var39_efect_ult1', 'saldo_medio_var17_hace2', 'saldo_medio_var17_hace3',
       'saldo_medio_var17_ult1', 'saldo_medio_var17_ult3',
       'saldo_medio_var29_hace2', 'saldo_medio_var29_hace3',
       'saldo_medio_var29_ult1', 'saldo_medio_var29_ult3',  'saldo_medio_var13_corto_ult3', 'saldo_medio_var13_largo_hace2',
       'saldo_medio_var13_largo_hace3', 'saldo_medio_var13_largo_ult1',
       'saldo_medio_var13_largo_ult3', 'saldo_medio_var13_medio_hace2',
       'saldo_medio_var13_medio_ult1', 'saldo_medio_var13_medio_ult3', 'imp_op_var39_efect_ult3', 'imp_op_var39_ult1',
       'imp_sal_var16_ult1','saldo_medio_var8_ult3', 'saldo_medio_var12_hace2',
       'saldo_medio_var12_hace3', 'saldo_medio_var12_ult1',
       'saldo_medio_var12_ult3', 'saldo_medio_var13_corto_hace2',
       'saldo_medio_var13_corto_hace3', 'saldo_medio_var13_corto_ult1', 'saldo_medio_var5_hace2', 'saldo_medio_var5_hace3',
       'saldo_medio_var5_ult1', 'saldo_medio_var5_ult3',
       'saldo_medio_var8_hace2', 'saldo_medio_var8_hace3',
       'saldo_medio_var8_ult1','num_trasp_var33_in_ult1', 'num_trasp_var33_out_ult1',
       'num_venta_var44_hace3', 'num_venta_var44_ult1', 'num_var45_hace2',
       'num_var45_hace3', 'num_var45_ult1', 'num_var45_ult3','num_sal_var16_ult1', 'num_var43_emit_ult1', 'num_var43_recib_ult1',
       'num_trasp_var11_ult1', 'num_trasp_var17_in_hace3',
       'num_trasp_var17_in_ult1', 'num_trasp_var17_out_ult1',
       'num_trasp_var33_in_hace3','num_op_var41_efect_ult1', 'num_op_var41_efect_ult3',
       'num_op_var39_efect_ult1', 'num_op_var39_efect_ult3',
       'num_reemb_var13_ult1', 'num_reemb_var17_hace3', 'num_reemb_var17_ult1',
       'num_reemb_var33_ult1',   'num_op_var39_comer_ult1', 'num_op_var39_comer_ult3',
       'num_op_var40_comer_ult1', 'num_op_var40_comer_ult3',
       'num_op_var40_efect_ult1', 'num_op_var40_efect_ult3',
       'num_op_var41_comer_ult1', 'num_op_var41_comer_ult3',)        
for col in col2:
        remove.append(col)

df_train.drop(remove, axis=1, inplace=True)
df_test.drop(remove, axis=1, inplace=True)

# remove duplicated columns
remove = []
c = df_train.columns

print(c)