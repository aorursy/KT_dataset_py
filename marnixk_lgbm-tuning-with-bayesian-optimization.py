# Data prep kernel of reference: https://www.kaggle.com/kailex/tidy-xgb-0-778/code

import numpy as np
import pandas as pd
import gc
import lightgbm as lgb

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.plots import plot_convergence

skip = range(1, 300000)

print("Loading data...\n")
lb=LabelEncoder()
def LabelEncoding_Cat(df):
    df=df.copy()
    Cat_Var=df.select_dtypes('object').columns.tolist()
    for col in Cat_Var:
        df[col]=lb.fit_transform(df[col].astype('str'))
    return df

def Fill_NA(df):
    df=df.copy()
    Num_Features=df.select_dtypes(['float64','int64']).columns.tolist()
    df[Num_Features]= df[Num_Features].fillna(-999)
    return df

bureau= (pd.read_csv("../input/bureau.csv", skiprows=skip)
         .pipe(LabelEncoding_Cat))

cred_card_bal=(pd.read_csv("../input/credit_card_balance.csv", skiprows=skip)
               .pipe(LabelEncoding_Cat))

pos_cash_bal=(pd.read_csv("../input/POS_CASH_balance.csv", skiprows=skip)
               .pipe(LabelEncoding_Cat))

prev =(pd.read_csv("../input/previous_application.csv", skiprows=skip)
               .pipe(LabelEncoding_Cat))

print("Preprocessing...\n")
Label_1=[s+'_'+l for s in bureau.columns.tolist() if s!='SK_ID_CURR' for l in ['mean','count','median','max']]
avg_bureau=bureau.groupby('SK_ID_CURR').agg(['mean','count','median','max']).reset_index()
avg_bureau.columns=['SK_ID_CURR']+Label_1

Label_2=[s+'_'+l for s in cred_card_bal.columns.tolist() if s!='SK_ID_CURR' for l in ['mean','count','median','max']]
avg_cred_card_bal=cred_card_bal.groupby('SK_ID_CURR').agg(['mean','count','median','max']).reset_index()
avg_cred_card_bal.columns=['SK_ID_CURR']+Label_2

Label_3=[s+'_'+l for s in pos_cash_bal.columns.tolist() if s not  in ['SK_ID_PREV','SK_ID_CURR'] for l in ['mean','count','median','max']]
avg_pos_cash_bal=pos_cash_bal.groupby(['SK_ID_PREV','SK_ID_CURR'])\
            .agg(['mean','count','median','max']).groupby(level='SK_ID_CURR')\
            .agg('mean').reset_index()
avg_pos_cash_bal.columns=['SK_ID_CURR']+Label_3

Label_4=[s+'_'+l for s in prev.columns.tolist() if s!='SK_ID_CURR' for l in ['mean','count','median','max']]
avg_prev=prev.groupby('SK_ID_CURR').agg(['mean','count','median','max']).reset_index()
avg_prev.columns=['SK_ID_CURR']+Label_4

del(Label_1,Label_2,Label_3,Label_4)
tr = pd.read_csv("../input/application_train.csv", skiprows=skip)
te = pd.read_csv("../input/application_test.csv")

tri=tr.shape[0]
y = tr.TARGET.copy()

tr_te=(tr.drop(labels=['TARGET'],axis=1).append(te)
         .pipe(LabelEncoding_Cat)
         .pipe(Fill_NA)
         .merge(avg_bureau,on='SK_ID_CURR',how='left')
         .merge(avg_cred_card_bal,on='SK_ID_CURR',how='left')
         .merge(avg_pos_cash_bal,on='SK_ID_CURR',how='left')
         .merge(avg_prev,on='SK_ID_CURR',how='left'))

del(tr,te,bureau,cred_card_bal,pos_cash_bal,prev, avg_prev,avg_bureau,avg_cred_card_bal,avg_pos_cash_bal)
gc.collect()

print("Preparing data...\n")
tr_te.drop(labels=['SK_ID_CURR'],axis=1,inplace=True)
tr=tr_te.iloc[:tri,:].copy()
te=tr_te.iloc[tri:,:].copy()

train_x, valid_x, train_y, valid_y = train_test_split(tr, y, test_size=0.2, shuffle=True)
del(tr_te)


search_space = [Real(0.1, 0.5, name='learning_rate'),
                Integer(3, 8, name='max_depth'),
                Integer(6, 12, name='num_leaves'),
                Integer(1, 10, name='scale_pos_weight')]

def BayesianOptimization(values):
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': values[0],
        'max_depth': values[1],
        'nthread': -1,
        'num_leaves': values[2],
        'scale_pos_weight': values[3]
    }

    print('\nTesting next set of paramaters...', params)

    train_data = lgb.Dataset(train_x, label=train_y)
    valid_data = lgb.Dataset(valid_x, label=valid_y)

    lgbm = lgb.train(params, train_data, 10, valid_sets=valid_data, early_stopping_rounds=50,
                 verbose_eval=0)

    neg_auc = round(-roc_auc_score(valid_y, lgbm.predict(valid_x)), 6)

    print('AUC: ', -neg_auc, ' of boosting iteration ', lgbm.current_iteration())
    return neg_auc
res_gp = gp_minimize(BayesianOptimization, search_space, random_state=26, n_jobs=-1, verbose=True, n_random_starts=10)
import lightgbm as lgb
plot_convergence(res_gp)