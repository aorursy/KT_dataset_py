# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
TRANS_tr1 = pd.read_csv("../input/ieee-fraud-detection/train_transaction.csv")

TRANS_te1 = pd.read_csv("../input/ieee-fraud-detection/test_transaction.csv")

ID_tr1 = pd.read_csv("../input/ieee-fraud-detection/train_identity.csv")

ID_te1 = pd.read_csv("../input/ieee-fraud-detection/test_identity.csv")
TRANS_tr = TRANS_tr1.merge(ID_tr1,how="left")

TRANS_te = TRANS_te1.merge(ID_te1,how="left")
print(TRANS_tr.shape)

print(TRANS_te.shape)
print(TRANS_tr.columns.tolist())

print(TRANS_te.columns.tolist())
col_TBC = ['TransactionID', 'isFraud', 'TransactionDT', 'TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'dist1', 'dist2', 'P_emaildomain', 'R_emaildomain', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9','id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_07', 'id_08', 'id_09', 'id_10', 'id_11', 'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29', 'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo']

col_TBC_for_test = ['TransactionID', 'TransactionDT', 'TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'dist1', 'dist2', 'P_emaildomain', 'R_emaildomain', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9','id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_07', 'id_08', 'id_09', 'id_10', 'id_11', 'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29', 'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo']
TRANS_tr = TRANS_tr[col_TBC]

TRANS_te = TRANS_te[col_TBC_for_test]
col_to_be_droped = []

for k in TRANS_tr.columns:

    if (100*(TRANS_tr[k].isnull().sum())/TRANS_tr[k].size).round(2) > 90.0 :

        col_to_be_droped.append(k)
print(len(col_to_be_droped))

print(col_to_be_droped)
TRANS_tr_new = TRANS_tr.drop(col_to_be_droped,axis=1)

TRANS_te_new = TRANS_te.drop(col_to_be_droped,axis=1)
print(TRANS_tr_new.columns.tolist())
tmp_list = ['addr1', 'addr2', 'dist1','id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_09', 'id_10', 'id_11', 'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_19', 'id_20', 'id_28', 'id_29', 'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38']



for col in tmp_list:

    TRANS_tr_new[col] = TRANS_tr_new[col].astype("object")

    TRANS_te_new[col] = TRANS_te_new[col].astype("object")
for k in TRANS_tr_new.keys():

    if "object" not in str(TRANS_tr_new[k].dtype):

        TRANS_tr_new[k].fillna(TRANS_tr_new[k].median(),inplace=True)
for k in TRANS_te_new.keys():

    if "object" not in str(TRANS_te_new[k].dtype):

        TRANS_te_new[k].fillna(TRANS_te_new[k].median(),inplace=True)
for k in ["card4","card6","addr1","addr2","dist1","M1","M2","M3","M4","M5","M6","M7","M8","M9","P_emaildomain","R_emaildomain",'id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_09', 'id_10', 'id_11', 'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_19', 'id_20', 'id_28', 'id_29', 'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38','DeviceType','DeviceInfo']:

    TRANS_te_new[k].fillna(TRANS_te_new[k].mode()[0],inplace=True)
for k in ["card4","card6","addr1","addr2","dist1","M1","M2","M3","M4","M5","M6","M7","M8","M9","P_emaildomain","R_emaildomain",'id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_09', 'id_10', 'id_11', 'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_19', 'id_20', 'id_28', 'id_29', 'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38','DeviceType','DeviceInfo']:

    TRANS_tr_new[k].fillna(TRANS_tr_new[k].mode()[0],inplace=True)
d_types = ["int8","int16","int32","int64","float16","float32","float64"]

for k in TRANS_tr_new.columns:

    if TRANS_tr_new[k].dtype in d_types:

        if TRANS_tr_new[k].min() > np.iinfo("int8").min and TRANS_tr_new[k].max()  < np.iinfo("int8").max:

            TRANS_tr_new[k] = TRANS_tr_new[k].astype("int8")

        if TRANS_tr_new[k].min() > np.iinfo("int16").min and TRANS_tr_new[k].max()  < np.iinfo("int16").max:

            TRANS_tr_new[k] = TRANS_tr_new[k].astype("int16")

        if TRANS_tr_new[k].min() > np.iinfo("int32").min and TRANS_tr_new[k].max()  < np.iinfo("int32").max:

            TRANS_tr_new[k] = TRANS_tr_new[k].astype("int32")

        if TRANS_tr_new[k].min() > np.iinfo("int64").min and TRANS_tr_new[k].max()  < np.iinfo("int64").max:

            TRANS_tr_new[k] = TRANS_tr_new[k].astype("int64")

        if TRANS_tr_new[k].min() > np.finfo("float16").min and TRANS_tr_new[k].max()  < np.finfo("float16").max:

            TRANS_tr_new[k] = TRANS_tr_new[k].astype("float16")

        if TRANS_tr_new[k].min() > np.finfo("float32").min and TRANS_tr_new[k].max()  < np.finfo("float32").max:

            TRANS_tr_new[k] = TRANS_tr_new[k].astype("float32")

        if TRANS_tr_new[k].min() > np.finfo("float64").min and TRANS_tr_new[k].max()  < np.finfo("float64").max:

            TRANS_tr_new[k] = TRANS_tr_new[k].astype("float64")
d_types = ["int8","int16","int32","int64","float16","float32","float64"]

for k in TRANS_te_new.columns:

    if TRANS_te_new[k].dtype in d_types:

        if TRANS_te_new[k].min() > np.iinfo("int8").min and TRANS_te_new[k].max()  < np.iinfo("int8").max:

            TRANS_te_new[k] = TRANS_te_new[k].astype("int8")

        if TRANS_te_new[k].min() > np.iinfo("int16").min and TRANS_te_new[k].max()  < np.iinfo("int16").max:

            TRANS_te_new[k] = TRANS_te_new[k].astype("int16")

        if TRANS_te_new[k].min() > np.iinfo("int32").min and TRANS_te_new[k].max()  < np.iinfo("int32").max:

            TRANS_te_new[k] = TRANS_te_new[k].astype("int32")

        if TRANS_te_new[k].min() > np.iinfo("int64").min and TRANS_te_new[k].max()  < np.iinfo("int64").max:

            TRANS_te_new[k] = TRANS_te_new[k].astype("int64")

        if TRANS_te_new[k].min() > np.finfo("float16").min and TRANS_te_new[k].max()  < np.finfo("float16").max:

            TRANS_te_new[k] = TRANS_te_new[k].astype("float16")

        if TRANS_te_new[k].min() > np.finfo("float32").min and TRANS_te_new[k].max()  < np.finfo("float32").max:

            TRANS_te_new[k] = TRANS_te_new[k].astype("float32")

        if TRANS_te_new[k].min() > np.finfo("float64").min and TRANS_te_new[k].max()  < np.finfo("float64").max:

            TRANS_te_new[k] = TRANS_te_new[k].astype("float64")
# Convert isFraud dtype to int8

TRANS_tr_new.isFraud = TRANS_tr_new.isFraud.astype("int8")
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,precision_score,roc_auc_score

from sklearn.preprocessing import OneHotEncoder,LabelBinarizer,LabelEncoder



from xgboost import XGBClassifier

from catboost import CatBoostClassifier

from lightgbm import LGBMClassifier



# Load scikit plotting library

import scikitplot as skp



import warnings

warnings.simplefilter("ignore")
tmp_list = ['addr1', 'addr2', 'dist1','id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_09', 'id_10', 'id_11', 'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_19', 'id_20', 'id_28', 'id_29', 'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38']



for col in tmp_list:

    TRANS_tr_new[col] = TRANS_tr_new[col].astype("object")

    TRANS_te_new[col] = TRANS_te_new[col].astype("object")
for k in TRANS_tr_new.keys():

    if "object" in str(TRANS_tr_new[k].dtype):

        print(k,end=",")
# label encoding to be done on 

labels_TBE = ["ProductCD","card4","card6","addr1","addr2","dist1","P_emaildomain","R_emaildomain","M1","M2","M3","M4","M5","M6","M7","M8","M9","id_01","id_02","id_03","id_04","id_05","id_06","id_09","id_10","id_11","id_12","id_13","id_14","id_15","id_16","id_17","id_19","id_20","id_28","id_29","id_30","id_31","id_32","id_33","id_34","id_35","id_36","id_37","id_38","DeviceType","DeviceInfo"]
le = LabelEncoder()
for la in labels_TBE:

    le.fit(TRANS_tr_new[la])

    TRANS_tr_new[la] = le.transform(TRANS_tr_new[la])
for la in labels_TBE:

    le.fit(TRANS_te_new[la])

    TRANS_te_new[la] = le.transform(TRANS_te_new[la])
d = TRANS_tr_new.iloc[:,2:]

t = TRANS_tr_new.iloc[:,1]
from sklearn.model_selection import KFold

import lightgbm as lgb

import gc
# NFOLDS = 12

# folds = KFold(n_splits=NFOLDS)



# X = d

# y = t

# X_test = TRANS_te_new.iloc[:,1:]

# columns = X.columns

# splits = folds.split(X, y)

# y_preds = np.zeros(X_test.shape[0])

# y_oof = np.zeros(X.shape[0])

# score = 0







# params = {'num_leaves': 491,

#           'min_child_weight': 0.03454472573214212,

#           'feature_fraction': 0.3797454081646243,

#           'bagging_fraction': 0.4181193142567742,

#           'min_data_in_leaf': 106,

#           'objective': 'binary',

#           'max_depth': -1,

#           'learning_rate': 0.006883242363721497,

#           "boosting_type": "gbdt",

#           "bagging_seed": 11,

#           "metric": 'auc',

#           "verbosity": -1,

#           'reg_alpha': 0.3899927210061127,

#           'reg_lambda': 0.6485237330340494,

#           'random_state': 47,

#          }





# feature_importances = pd.DataFrame()

# feature_importances['feature'] = columns

  

# for fold_n, (train_index, valid_index) in enumerate(splits):

#     X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]

#     y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    

#     dtrain = lgb.Dataset(X_train, label=y_train)

#     dvalid = lgb.Dataset(X_valid, label=y_valid)



#     clf = lgb.train(params, dtrain, 10000, valid_sets = [dtrain, dvalid], verbose_eval=200, early_stopping_rounds=500)

    

#     feature_importances[f'fold_{fold_n + 1}'] = clf.feature_importance()

    

#     y_pred_valid = clf.predict(X_valid)

#     y_oof[valid_index] = y_pred_valid

#     print(f"Fold {fold_n + 1} | AUC: {roc_auc_score(y_valid, y_pred_valid)}")

    

#     score += roc_auc_score(y_valid, y_pred_valid) / NFOLDS

#     y_preds += clf.predict(X_test) / NFOLDS



#     if roc_auc_score(y_valid, y_pred_valid) >= .98:

#         del X_train, X_valid, y_train, y_valid

#         gc.collect()

#         break





    

# print(f"\nMean AUC = {score}")

# print(f"Out of folds AUC = {roc_auc_score(y, y_oof)}")
# Submission4_xgb = pd.DataFrame()

# Submission4_xgb["TransactionID"] = TRANS_te_new.TransactionID

# Submission4_xgb["isFraud"] = y_preds

# Submission4_xgb.to_csv("Submission8_lgb_7fold_new2.csv")

# Submission4_xgb.head(2)
# from xgboost import XGBClassifier
# xgb = XGBClassifier(

#     n_estimators=1200, # 500

#     max_depth=25,

#     learning_rate=0.05,

#     subsample=0.9,

#     colsample_bytree=0.9,

#     missing=-999,

#     random_state=2019,

#     tree_method='gpu_hist'  # THE MAGICAL PARAMETER

# )
d = TRANS_tr_new.iloc[:,2:]

t = TRANS_tr_new.iloc[:,1]
X_train, X_test, y_train, y_test = train_test_split(d,t,test_size=0.2, random_state=5)
# xgb.fit(X_train,y_train)

# y_pred = xgb.predict(X_test)

# print(y_pred)

# print(accuracy_score(y_test,y_pred)*100)

# skp.metrics.plot_confusion_matrix(y_test,y_pred) # 2868 1-1
# y_prob = xgb.predict_proba(X_test)

# y_prob
# skp.metrics.plot_roc(y_test,y_prob,plot_macro=False,plot_micro=False,figsize=(10,5)) # ROC=98%
# y_prob_actual = xgb.predict_proba(TRANS_te_new.iloc[:,1:])

# y_prob_actual
# Submission4_xgb = pd.DataFrame()

# Submission4_xgb["TransactionID"] = TRANS_te_new.TransactionID

# Submission4_xgb["isFraud"] = y_prob_actual[:,1]

# Submission4_xgb.to_csv("Submission8_xgb_new1.csv")

# Submission4_xgb.head(2)
from tpot import TPOTClassifier
tpc = TPOTClassifier(verbosity=2,scoring="roc_auc",max_time_mins=300,random_state=6)

tpc
tpc.fit(X_train,y_train)