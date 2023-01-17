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
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.preprocessing import StandardScaler
from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.metrics import log_loss
from keras.models import model_from_json
from rgf.sklearn import FastRGFClassifier
import catboost as cb
import xgboost as xgb

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from tqdm import tqdm_notebook as tqdm
import datetime as dt

import lightgbm as lgb
from lightgbm import LGBMClassifier

from sklearn.model_selection import RandomizedSearchCV
from scipy import stats
from statistics import mean, median, variance, stdev

from sklearn.preprocessing import quantile_transform
epsilon = 1e-7
#input_path='C:/data/jupyter_notebook/DataRobot/1_data/homework-for-students3'
input_path='/kaggle/input/homework-for-students3'
df_train = pd.read_csv(input_path+'/train.csv')
df_test = pd.read_csv(input_path+'/test.csv')
df_statelatlong = pd.read_csv(input_path+'/statelatlong.csv')
df_gdp = pd.read_csv(input_path+'/US_GDP_by_State.csv')
df_gdp = df_gdp[df_gdp.year==2015]
df_zipcode = pd.read_csv(input_path+'/free-zipcode-database.csv', dtype={'Zipcode':np.int64})
df_zipcode = df_zipcode[['Zipcode','Xaxis','Yaxis','Zaxis']]
len(df_train)
#zipcode結合
df_zipcode = df_zipcode.rename(columns={'Zipcode':'zip_code'})
df_zipcode['zip_code'] = df_zipcode.zip_code.astype(str).str[:3].astype(np.int64)
df_zipcode = df_zipcode[~df_zipcode.duplicated('zip_code')]
df_train['zip_code'] = df_train.zip_code.str[:3].astype(np.int64)
df_test['zip_code'] = df_test.zip_code.str[:3].astype(np.int64)
df_train = pd.merge(df_train,df_zipcode,on='zip_code',how='left')
df_test = pd.merge(df_test,df_zipcode,on='zip_code',how='left')
len(df_train)
#YYYY列追加
#df_train['earliest_cr_lineYYYY']=df_train.earliest_cr_line.str[-4:]
#df_test['earliest_cr_lineYYYY']=df_test.earliest_cr_line.str[-4:]

# State、GDP結合
df_statelatlong = df_statelatlong.rename(columns={'State':'addr_state'})
df_gdp = df_gdp.rename(columns={'State':'City'})
df_train = pd.merge(df_train,df_statelatlong,on='addr_state',how='left')
df_test = pd.merge(df_test,df_statelatlong,on='addr_state',how='left')
#df_train = pd.merge(df_train,df_gdp,on='City',how='left')
#df_test = pd.merge(df_test,df_gdp,on='City',how='left')

len(df_train)
X_train = df_train[pd.to_datetime(df_train.issue_d) >= dt.datetime(2014,1,1)].copy()
#X_train = df_train
y_train = X_train.loan_condition
#X_train.drop('loan_condition', axis=1, inplace=True)
X_test = df_test
#GA2Mによる交互作用
cols1 = ['loan_amnt','installment','installment','loan_amnt','loan_amnt']
cols2 = ['revol_bal','revol_bal','revol_util','open_acc','revol_util']

#add_cols = []
for col1,col2 in zip(cols1,cols2):
    col12m = col1+'&'+col2
    X_train[col12m]=X_train[col1]*X_train[col2]
    X_test[col12m]=X_test[col1]*X_test[col2]
#    add_cols.append(col12m)
X_test.describe()
X_train.to_csv('/kaggle/working/X_train.2015.csv')
X_test.to_csv('/kaggle/working/X_test.2016.csv')

## 総債務支払い額

#X_train['annual_inc*dti'] = X_train.annual_inc * X_train.dti
#X_test['annual_inc*dti'] = X_test.annual_inc * X_test.dti
#############
#### add 3/30
#############


# rank gauss
cols = []
cols = ['annual_inc','dti','revol_bal']
all_X = pd.concat([X_train, X_test], axis=0)

for col in cols:
    all_X[col] = quantile_transform(np.array(all_X[col].values).reshape(-1,1),
                                    n_quantiles=200,
                                    random_state=71,
                                    output_distribution='normal')

# log

cols = []
cols = ['delinq_2yrs','acc_now_delinq','tot_coll_amt','tot_cur_bal']

for col in cols:
    all_X[col] = np.log1p(all_X[col])

X_train = all_X.iloc[:X_train.shape[0],:].reset_index(drop=True)
X_test = all_X.iloc[X_train.shape[0]:,:].reset_index(drop=True)
#loan_amnt/installment

X_train['loan_amnt/installment']=X_train.loan_amnt / (X_train.installment + epsilon)
X_test['loan_amnt/installment']=X_test.loan_amnt / (X_test.installment + epsilon)

X_train['loan_amnt/tot_coll_amt']=X_train.loan_amnt / (X_train.tot_coll_amt + epsilon)
X_test['loan_amnt/tot_coll_amt']=X_test.loan_amnt / (X_test.tot_coll_amt + epsilon)

X_train['annulal_inc/loan_amntd']=X_train['annual_inc']/(X_train['loan_amnt'] + epsilon)
X_test['annulal_inc/loan_amntd']=X_test['annual_inc']/(X_test['loan_amnt'] + epsilon)

#X_train['delinq_acc_tot']=X_train['delinq_2yrs'] * X_train['acc_now_delinq'] * X_train['tot_coll_amt']
#X_test['delinq_acc_tot']=X_test['delinq_2yrs'] * X_test['acc_now_delinq'] * X_test['tot_coll_amt']

X_train['acc_now_delinq*mths_since_last_delinq']=X_train['mths_since_last_delinq'] * X_train['acc_now_delinq']
X_test['acc_now_delinq*mths_since_last_delinq']=X_test['mths_since_last_delinq'] * X_test['acc_now_delinq']

#zero for NaN
cols = ['mths_since_last_delinq','mths_since_last_major_derog','mths_since_last_record','tot_coll_amt','acc_now_delinq']

for col in cols:
    X_train[col].fillna(0, inplace=True)
    X_test[col].fillna(0, inplace=True)
# 不要列削除
X_train.drop('ID', axis=1, inplace=True)
X_test.drop('ID', axis=1, inplace=True)
X_train.drop('issue_d', axis=1, inplace=True)
X_test.drop('issue_d', axis=1, inplace=True)
#X_train.drop('earliest_cr_line',axis=1, inplace=True)
#X_test.drop('earliest_cr_line',axis=1, inplace=True)

#X_train.drop('title',axis=1, inplace=True)
#X_test.drop('title',axis=1, inplace=True)
X_train.drop('City',axis=1, inplace=True)
X_test.drop('City',axis=1, inplace=True)
#X_train.drop('year',axis=1, inplace=True)
#X_test.drop('year',axis=1, inplace=True)
#数値型列名取得
num_col_name = []
for col in X_train.columns:
    if X_train[col].dtype in ['float64','int64']:
        num_col_name.append(col)
        
        print(col, X_train[col].nunique())

#num_col_name.remove('loan_condition')
#欠損値フラグ
havenullcol = []
havnullcol = ['emp_title','emp_length']
#havenullcol = X_train.columns[X_train.isnull().sum()!=0].values

for col in havenullcol:
    X_train[col+'null'] = 0 
    X_train[col+'null'][X_train[col].isnull()] = 1
    X_test[col+'null'] = 0 
    X_test[col+'null'][X_test[col].isnull()] = 1
#onehot_cols = ['grade','sub_grade','emp_length','purpose','tltle','addr_state','home_ownership']
#for col in cols:
#    X_train[col+'_onehot'] = X_train[col]
#    X_test[col+'_onehot'] = X_test[col]
#オブジェクト型列名取得
obj_col_name = []
for col in X_train.columns:
    if X_train[col].dtype == 'object':
        obj_col_name.append(col)
        
        print(col, X_train[col].nunique())
#欠損値補完
X_train['emp_title'].fillna(X_train['emp_title'].fillna('etc'),inplace=True)
#X_train['title'].fillna(X_train['title'].mode()[0],inplace=True)
#X_train['earliest_cr_line'].fillna(X_train['earliest_cr_line'].mode()[0],inplace=True)
X_test['emp_title'].fillna(X_test['emp_title'].fillna('etc'),inplace=True)
#X_test['title'].fillna(X_test['title'].mode()[0],inplace=True)
#X_test['earliest_cr_line'].fillna(X_test['earliest_cr_line'].mode()[0],inplace=True)


# 最小値で補完
X_train['emp_length'].fillna(X_train['emp_length'].fillna(0),inplace=True)
X_test['emp_length'].fillna(X_test['emp_length'].fillna(0),inplace=True)
#cols = ['delinq_2yrs','mths_since_last_delinq','mths_since_last_record','tot_cur_bal','tot_coll_amt']

#for col in cols:
#    X_train[col+'_des'] = X_train[col] / (stdev(X_train[col]) + epsilon)
#    X_test[col+'_des'] = X_test[col] / (stdev(X_test[col]) + epsilon)
X_train.head()
#count_encoding

cnt_cols = ['grade','sub_grade','emp_title','purpose','addr_state','home_ownership','acc_now_delinq','mths_since_last_delinq','tot_coll_amt','zip_code','title']

for col in cnt_cols:
    train_cnt = X_test[col].value_counts()
    test_cnt = X_test[col].value_counts()
    
    X_train[col+'_cnt'] = X_train[col].map(train_cnt)
    X_test[col+'_cnt'] = X_test[col].map(test_cnt)
#カテゴリ型の変換
oe = OrdinalEncoder(cols=obj_col_name, return_df=False)

#X_train[obj_col_name] = oe.fit_transform(X_train[obj_col_name])
#X_test[obj_col_name] = oe.fit_transform(X_test[obj_col_name])

all_X = pd.concat([X_train,X_test])

all_X[obj_col_name] = oe.fit_transform(all_X[obj_col_name])

X_train = all_X.iloc[:X_train.shape[0],:].reset_index(drop=True)
X_test = all_X.iloc[X_train.shape[0]:,:].reset_index(drop=True)
#特徴量追加 交互作用

cols1 = ['grade','grade','grade','grade','grade','grade']
cols3 = ['sub_grade','sub_grade','sub_grade','sub_grade','sub_grade','sub_grade']
cols2 = ['dti','annual_inc','loan_amnt','installment','tot_cur_bal','tot_coll_amt']

cols1.extend(cols3)
cols2.extend(cols2)

add_cols = []
for col1,col2 in zip(cols1,cols2):
    col12m = col1+col2
    X_train[col12m]=X_train[col1]*X_train[col2]
    X_test[col12m]=X_test[col1]*X_test[col2]
    add_cols.append(col12m)


X_temp = pd.concat([X_train, y_train], axis=1)
te_col = 'sub_grade'
target = ['acc_now_delinq','annual_inc','loan_amnt','mths_since_last_delinq']
X_temp = pd.concat([X_train, y_train], axis=1)

for col in target:
    # X_testはX_trainでエンコーディングする
    summary = X_temp.groupby([te_col])[col].mean()
    X_test[te_col+col+'_mean'] = X_test[te_col].map(summary) 
    
    # X_trainのカテゴリ変数をoofでエンコーディングする
    skf = StratifiedKFold(n_splits=8, random_state=71, shuffle=True)


    enc_train = Series(np.zeros(len(X_train)), index=X_train.index)

    for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):
        X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]
        X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]

        summary = X_train_.groupby([te_col])[col].mean()
        enc_train.iloc[val_ix] = X_val[te_col].map(summary)
    
        X_train[te_col+col+'_mean'] = enc_train
X_train.head()
# Traget Encoding
te_col = ['purpose','home_ownership','grade','sub_grade','addr_state','acc_now_delinq','mths_since_last_delinq','tot_coll_amt','zip_code','emp_title','title']
#te_col = ['purpose','home_ownership','tot_coll_amt']
target = 'loan_condition'
X_temp = pd.concat([X_train, y_train], axis=1)

for col in te_col:
    # X_testはX_trainでエンコーディングする
    summary = X_temp.groupby([col])[target].mean()
    X_test[col+'_te'] = X_test[col].map(summary) 
    
    # X_trainのカテゴリ変数をoofでエンコーディングする
    skf = StratifiedKFold(n_splits=8, random_state=71, shuffle=True)


    enc_train = Series(np.zeros(len(X_train)), index=X_train.index)

    for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):
        X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]
        X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]

        summary = X_train_.groupby([col])[target].mean()
        enc_train.iloc[val_ix] = X_val[col].map(summary)
    
        X_train[col+'_te'] = enc_train
X_test[te_col]
X_train.fillna(X_train.median(), inplace=True)
X_test.fillna(X_test.median(), inplace=True)
#数値型の標準化
#num_col_name.extend(add_cols)
#num_col_name.extend(te_col)
#num_col_name.extend(cnt_cols)
#for col in num_col_name:
#    ss = StandardScaler()
#    X_train[col] = ss.fit_transform(np.array(X_train[col].values).reshape(-1,1))
#    X_test[col] = ss.fit_transform(np.array(X_test[col].values).reshape(-1,1))
#不要列削除

del_cols = ['pub_rec','application_type','collections_12_mths_ex_med']

for col in del_cols:
    X_train.drop(col, axis=1, inplace=True)
    X_test.drop(col, axis=1, inplace=True)
#################
### 2020/3/30 修正
#################


#X_train.drop('emp_title', axis=1, inplace=True)
#X_test.drop('emp_title', axis=1, inplace=True)
# add knc,dnn

knc_train = X_train.copy()
knc_test = X_test.copy()
# add knc,dnn
ss_cols = []
#onehot_cols = ['grade_onehot','sub_grade_onehot','addr_state',
#               'earliest_cr_lineYYYY','emp_length_onehot',
#              'purpose_onehot']

onehot_cols = ['grade','sub_grade','emp_length','purpose','title','addr_state','home_ownership','earliest_cr_lineYYYY']

for col in knc_train.columns:
    ss_cols.append(col)

for col in onehot_cols:
    ss_cols.remove(col)
# add knc,dnn
#数値型の標準化

for col in ss_cols:
    ss = StandardScaler()
    knc_train[col] = ss.fit_transform(np.array(X_train[col].values).reshape(-1,1))
    knc_test[col] = ss.fit_transform(np.array(X_test[col].values).reshape(-1,1))
#add knc,dnn

dnn_train = knc_train.copy()
dnn_test = knc_test.copy()
# add knc, dnn
# one-hot encording

all_X = pd.concat([dnn_train,dnn_test])
all_X = pd.get_dummies(all_X, columns=onehot_cols)

dnn_train = all_X.iloc[:dnn_train.shape[0],:].reset_index(drop=True)
dnn_test = all_X.iloc[dnn_train.shape[0]:,:].reset_index(drop=True)
X_test.head()
X_test.head()
fig=plt.figure(figsize=[40,80])

i = 0
for col in X_train.columns:
    i=i+1
#    print(col)
#    print(i)
    ax_name = 'ax'+str(i)
    ax_name = fig.add_subplot(30,3,i)
    ax_name.hist(X_train[col],bins=40,density=True, alpha=0.5,color = 'r')
    ax_name.hist(X_test[col],bins=40,density=True, alpha=0.5, color = 'b')
    ax_name.set_title(col)
# catBoost

cb_clf = cb.CatBoostClassifier(iterations=500,
                               learning_rate=0.01,
                               l2_leaf_reg=9,
                               depth=10,
                               loss_function='Logloss')
#scores = []
#kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=71)

#for i,(idx_train, idx_validate) in tqdm(enumerate(kf.split(X_train,y_train))):
#    cv_X_train, cv_X_validate = X_train.iloc[idx_train],X_train.iloc[idx_validate]
#    cv_y_train, cv_y_validate = y_train.iloc[idx_train],y_train.iloc[idx_validate] 

#    cb_clf.fit(cv_X_train, cv_y_train, early_stopping_rounds=20, eval_set=[(cv_X_validate, cv_y_validate)])
    
#    pred_validate = cb_clf.predict_proba(cv_X_validate)[:,1]
#    score = roc_auc_score(cv_y_validate, pred_validate)
#    scores.append(score)

#clf.fit(X_train, y_train)

#pred_test_rgf = clf.predict_proba(X_test)[:,1]
# dnn

nn_model = Sequential()
nn_model.add(Dense(300, activation='relu', input_shape=(dnn_train.shape[1],)))
nn_model.add(BatchNormalization())
nn_model.add(Dropout(0.2))
nn_model.add(Dense(300, activation='relu'))
nn_model.add(BatchNormalization())
nn_model.add(Dropout(0.2))
nn_model.add(Dense(100, activation='relu'))
nn_model.add(BatchNormalization())
nn_model.add(Dropout(0.2))
nn_model.add(Dense(1, activation='sigmoid'))

nn_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=0)

#kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=71)
#for i,(idx_train, idx_validate) in tqdm(enumerate(kf.split(X_train,y_train))):
#    cv_X_train, cv_X_validate = dnn_train.iloc[idx_train],dnn_train.iloc[idx_validate]
#    cv_y_train, cv_y_validate = y_train.iloc[idx_train],y_train.iloc[idx_validate] 
    
#    history = nn_model.fit(cv_X_train, cv_y_train,
#                       batch_size=256, epochs=30,
#                       verbose=1, validation_data=(cv_X_validate, cv_y_validate),callbacks=[es])
    
#    pred_validate = nn_model.predict(cv_X_validate)
#    score = log_loss(cv_y_validate, pred_validate, eps=1e-7)
#    print(f'logloss: {score:.4f}')

#history = nn_model.fit(dnn_train, y_train,
#                       batch_size=128, epochs=100,
#                       verbose=1, validation_split=0.3,callbacks=[es])

#history = nn_model.fit(dnn_train, y_train,
#                       batch_size=256, epochs=30,
#                       verbose=1, validation_split=0.3)

#LightGBM

lgb_clf = LGBMClassifier(boosting_type='gbdt', class_weight=None,
               colsample_bytree=0.27968666496016553, importance_type='split',
               learning_rate=0.05, max_depth=6, min_child_samples=20,
               min_child_weight=15.796627047202797, min_split_gain=0.0,
               n_estimators=350, n_jobs=-1, num_leaves=73, objective=None,
               random_state=71, reg_alpha=0.1, reg_lambda=0.1, silent=True,
               subsample=0.6692221371880118, subsample_for_bin=500000,
               subsample_freq=0)
cb_clf.fit(X_train, y_train)
history = nn_model.fit(dnn_train, y_train,
                     batch_size=128, epochs=100,
                    verbose=1, validation_split=0.3,callbacks=[es])
lgb_clf.fit(X_train,y_train)
pred_test_cb = cb_clf.predict_proba(X_test)[:,1]
pred_test_dnn = nn_model.predict(dnn_test)
pred_test_gbdt = lgb_clf.predict_proba(X_test)[:,1]
#アンサンブル

#def ensemble(y_len, X_train, dnn_train, y_train, X_test, dnn_test, seed=1):
    #学習
#    cb_clf.fit(X_train, y_train)
#    history = nn_model.fit(dnn_train, y_train,
#                           batch_size=128, epochs=100,
#                           verbose=1, validation_split=0.3,callbacks=[es])
#    lgb_clf.fit(X_train,y_train)
    #予測
#    pred = np.zeros((2,y_len))
#    pred[0] = cb_clf.predict_proba(X_test)[:,1]
#    pred[1] = nn_model.predict(dnn_test)
#    pred[2] = clf.predict_proba(X_test)[:,1]
#予測
#y_len = len(y_test)
#pred=np.zeros((2,9,y_len))
#x=0
#y=0
#for i in np.arange(18):
#    x=i//9
#    y=i%9
#    pred[x][y] = cb_clf.predict_proba(X_test)[:,1]
#    pred[x][y] = nn_model.predict(dnn_test)
#    pred[x][y] = clf.predict_proba(X_test)[:,1]
# RGF

#scores = []
#clf = FastRGFClassifier(max_leaf=400,opt_algorithm='rgf',loss='LOGISTIC',verbose=True)

#kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=71)

#for i,(idx_train, idx_validate) in tqdm(enumerate(kf.split(X_train,y_train))):
#    cv_X_train, cv_X_validate = X_train.iloc[idx_train],X_train.iloc[idx_validate]
#    cv_y_train, cv_y_validate = y_train.iloc[idx_train],y_train.iloc[idx_validate] 

#    clf.fit(cv_X_train, cv_y_train)
    
#    pred_validate = clf.predict_proba(cv_X_validate)[:,1]
#    score = roc_auc_score(cv_y_validate, pred_validate)
#    scores.append(score)

#clf.fit(X_train, y_train)

#pred_test_rgf = clf.predict_proba(X_test)[:,1]
#pred_test_rgf
# KNN

#clf = KNeighborsClassifier(algorithm='auto',n_neighbors=10,weights='distance',leaf_size=30)
#clf.fit(X_train, y_train)
#pred_test_knc = clf.predict_proba(X_test)[:,1]

#pred_test_knc
#print('save model')
#json_string = nn_model.to_json()
#open('/kaggle/working/dnn_model.json', 'w').write(json_string)
#print('save weights')
#nn_model.save_weights('/kaggle/working/dnn_model_weights.hdf5')
#json_string = open('/kaggle/working/dnn_model.json').read()
#nn_model = model_from_json(json_string)
#nn_model.summary()

#nn_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

#nn_model.load_weights('/kaggle/working/dnn_model_weights.hdf5')
#pred_test_dnn = nn_model.predict(dnn_test)
#clf = LGBMClassifier(boosting_type='gbdt', class_weight=None,
#                                importance_type='split', learning_rate=0.05,
#                                min_child_samples=20, min_split_gain=0.0,
#                                n_estimators=500, n_jobs=-1, objective=None,
#                                random_state=71, silent=True, reg_lambda=0.1, reg_alpha=0.1,
#                                subsample_for_bin=500000, subsample_freq=0)

#kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=71)

#random_params = {'max_depth': stats.randint(3,15),
#                 'num_leaves': stats.randint(10,80),
#                 'min_child_weight': stats.uniform(0.01, 30),
#                 'colsample_bytree': stats.uniform(0.2,0.7),
#                 'subsample': stats.uniform(0.2,0.7)}

#fit_params = {'early_stopping_rounds':20}
#rs = RandomizedSearchCV(clf, random_params, cv=kf, verbose=2, scoring='roc_auc', n_iter=270)
#rs.fit(X_train, y_train)

#print(rs.best_estimator_)
#print(rs.best_params_)
#clf = LGBMClassifier(boosting_type='goss', class_weight=None,
#               colsample_bytree=0.27968666496016553, importance_type='split',
#               learning_rate=0.05, max_depth=6, min_child_samples=20,
#               min_child_weight=15.796627047202797, min_split_gain=0.0,
#               n_estimators=400, n_jobs=-1, num_leaves=73, objective=None,
#               random_state=71, reg_alpha=0.1, reg_lambda=0.1, silent=True,
#               subsample=0.6692221371880118, subsample_for_bin=500000,
#               subsample_freq=0)
#clf.fit(X_train,y_train)
pred_test = (0.4*pred_test_gbdt + 0.3*pred_test_cb.reshape(-1,) + 0.3*(pred_test_dnn.reshape(-1,)))
#pred_test = (0.5*pred_test_gbdt +  0.5*(pred_test_dnn.reshape(-1,)))
pred_test
#imp = DataFrame(clf.booster_.feature_importance(importance_type='gain'), index = X_train.columns, columns=['importance']).sort_values(['importance'], ascending=False)
#imp
submission = pd.read_csv(input_path+'/sample_submission.csv', index_col=0)

submission.loan_condition = pred_test
submission.to_csv('submission.csv')