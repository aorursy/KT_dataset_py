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

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import MinMaxScaler,LabelEncoder,OneHotEncoder

from sklearn.metrics import f1_score

from sklearn.model_selection import GridSearchCV,StratifiedKFold,KFold,train_test_split

from sklearn.ensemble import RandomForestClassifier

import lightgbm as lgb

import xgboost as xgb

import catboost as cbt

import datetime

import time

import warnings

import gc

warnings.filterwarnings("ignore")

sns.set(style="whitegrid",color_codes=True)

sns.set(font_scale=1)

%time
#reduce mem cost

%time

def reduce_mem_cost(df,verbose = True):

    num_type = ['int16','int32','int64','float32','float64','object']

    start_mem = df.memory_usage().sum() / 1024 ** 2

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in num_type:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)

            elif str(col_type)[:5] == 'float':

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose:

        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df       
%time

train_data = pd.read_csv("/kaggle/input/cat-in-the-dat/train.csv")

test_data = pd.read_csv("/kaggle/input/cat-in-the-dat/test.csv")

train_target = train_data.pop("target")

train_data = reduce_mem_cost(train_data)

test_data = reduce_mem_cost(test_data)
#查看数据

train_data.info()
%time

label_encoder_cols = []

no_label_encoder_cols = []

for col in train_data.columns:

    if train_data[col].dtypes == "int32" or train_data[col].dtypes == "int8":

        no_label_encoder_cols.append(col)

    else:

        label_encoder_cols.append(col)
%time

# label encoder on some cols

train_id = train_data.pop("id")

test_id = test_data.pop("id")

all_data = pd.concat([train_data,test_data],axis=0)

all_data[label_encoder_cols]
%time

#generate  count feats

bin_3_cnt = all_data["bin_3"].value_counts().to_dict()

all_data["bin_3_cnt"] = all_data["bin_3"].map(bin_3_cnt)



bin_4_cnt = all_data["bin_4"].value_counts().to_dict()

all_data["bin_4_cnt"] = all_data["bin_4"].map(bin_4_cnt)



nom_0_cnt = all_data["nom_0"].value_counts().to_dict()

all_data["nom_0_cnt"] = all_data["nom_0"].map(nom_0_cnt)



nom_1_cnt = all_data["nom_1"].value_counts().to_dict()

all_data["nom_1_cnt"] = all_data["nom_1"].map(nom_1_cnt)



nom_2_cnt = all_data["nom_2"].value_counts().to_dict()

all_data["nom_2_cnt"] = all_data["nom_2"].map(nom_2_cnt)



nom_3_cnt = all_data["nom_3"].value_counts().to_dict()

all_data["nom_3_cnt"] = all_data["nom_3"].map(nom_3_cnt)



nom_4_cnt = all_data["nom_4"].value_counts().to_dict()

all_data["nom_4_cnt"] = all_data["nom_4"].map(nom_4_cnt)



nom_5_cnt = all_data["nom_5"].value_counts().to_dict()

all_data["nom_5_cnt"] = all_data["nom_5"].map(nom_5_cnt)



nom_6_cnt = all_data["nom_6"].value_counts().to_dict()

all_data["nom_6_cnt"] = all_data["nom_6"].map(nom_6_cnt)



nom_7_cnt = all_data["nom_7"].value_counts().to_dict()

all_data["nom_7_cnt"] = all_data["nom_7"].map(nom_7_cnt)



nom_8_cnt = all_data["nom_8"].value_counts().to_dict()

all_data["nom_8_cnt"] = all_data["nom_8"].map(nom_8_cnt)



nom_9_cnt = all_data["nom_9"].value_counts().to_dict()

all_data["nom_9_cnt"] = all_data["nom_9"].map(nom_9_cnt)



ord_1_cnt = all_data["ord_1"].value_counts().to_dict()

all_data["ord_1_cnt"] = all_data["ord_1"].map(ord_1_cnt)



ord_2_cnt = all_data["ord_2"].value_counts().to_dict()

all_data["ord_2_cnt"] = all_data["ord_2"].map(ord_2_cnt)



ord_3_cnt = all_data["ord_3"].value_counts().to_dict()

all_data["ord_3_cnt"] = all_data["ord_3"].map(ord_3_cnt)



ord_4_cnt = all_data["ord_4"].value_counts().to_dict()

all_data["ord_4_cnt"] = all_data["ord_4"].map(ord_4_cnt)



ord_5_cnt = all_data["ord_5"].value_counts().to_dict()

all_data["ord_5_cnt"] = all_data["ord_5"].map(ord_5_cnt)



all_data
%time

#label encoder on some cols

for le in label_encoder_cols:

    le_feat = LabelEncoder()

    le_feat.fit(all_data[le])

    all_data[le] = le_feat.transform(all_data[le])
#split all data into train and test

train_data = all_data[:train_data.shape[0]]

test_data = all_data[train_data.shape[0]:]
%time

train_x,train_y,testX = train_data.values,train_target.values,test_data.values
start = time.time()

model = lgb.LGBMClassifier(boosting_type="gbdt",num_leaves=48, max_depth=-1, learning_rate=0.05,

                               n_estimators=3000, subsample_for_bin=50000,objective="binary",min_split_gain=0, min_child_weight=5, min_child_samples=30, #10

                               subsample=0.8,subsample_freq=1, colsample_bytree=1, reg_alpha=3,reg_lambda=5,

                               feature_fraction= 0.9, bagging_fraction = 0.9,

                               seed= 2019,n_jobs=10,slient=True,num_boost_round=3000)

n_splits = 3

random_seed = 2019

skf = StratifiedKFold(shuffle = True, random_state = random_seed, n_splits = n_splits)

cv_pred = []

val_score = []

for idx, (tra_idx,val_idx) in enumerate(skf.split(train_x,train_y)):

    startTime = time.time()

    print("============================================fold_{}===================================================".format(str(idx+1)))

    X_train,Y_train = train_x[tra_idx],train_y[tra_idx]

    X_val,Y_val = train_x[val_idx], train_y[val_idx]

    lgb_model = model.fit(X_train,Y_train,eval_names=["train","valid"],eval_metric=["logloss"],eval_set=[(X_train, Y_train),(X_val,Y_val)],early_stopping_rounds=200)

    val_pred = lgb_model.predict(X_val,num_iteration = lgb_model.best_iteration_)

    val_score.append(f1_score(Y_val,val_pred))

    print("f1_score:",f1_score(Y_val, val_pred))

    test_pred = lgb_model.predict(testX, num_iteration = lgb_model.best_iteration_).astype(int)

    cv_pred.append(test_pred)

    endTime = time.time()

    print("fold_{} finished in {}".format(str(idx+1), datetime.timedelta(seconds= endTime-startTime)))

    

end = time.time()

print('-'*60)

print("Training has finished.")

print("Total training time is {}".format(str(datetime.timedelta(seconds=end-start))))

print(val_score)

print("mean f1:",np.mean(val_score))

print('-'*60)



submit = []

for line in np.array(cv_pred).transpose():

    submit.append(np.argmax(np.bincount(line)))

final_result = pd.DataFrame(columns=["id","target"])

final_result["id"] = list(test_id.unique())

final_result["target"] = submit

final_result.to_csv("submitLGB{0}.csv".format(datetime.datetime.now().strftime("%Y%m%d%H%M")),index = False)

print(final_result.head())       