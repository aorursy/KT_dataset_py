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
import sys

import lightgbm as lgb

from sklearn import preprocessing

train_df = pd.read_csv("../input/ds2019uec-task1/train.csv")

test_df = pd.read_csv("../input/ds2019uec-task1/test.csv")

#train_df.head()



# preprocessing

train = train_df.copy()

test = test_df.copy()



# info

print(train.shape)

print(test.shape)
# Excelで生データを確認

# durationについてtestは全て-1であり不適切な説明変数とわかる



# 欠損 , 負値の確認

#print(train.isnull().sum())

#print(test.isnull().sum())

#print(train.describe())

print(test.describe())



# result

# duration -> all -1

# pdays -> -1 あり

# balance -> 負値あり
# データ分析(category)

train["y"] = (train_df["y"] == "yes").astype(np.int)

print(train.groupby(["job"]).agg(["count","mean"])["y"])

print(train.groupby(["marital"]).agg(["count","mean"])["y"])

print(train.groupby(["education"]).agg(["count","mean"])["y"])

print(train.groupby(["default"]).agg(["count","mean"])["y"])

print(train.groupby(["housing"]).agg(["count","mean"])["y"])

print(train.groupby(["loan"]).agg(["count","mean"])["y"])

print(train.groupby(["contact"]).agg(["count","mean"])["y"])

print(train.groupby(["month"]).agg(["count","mean"])["y"])

print(train.groupby(["poutcome"]).agg(["count","mean"])["y"])



# result

# month-> feb,mar,aprが高い(oct,decは母数が少ない) (※trainのみsepがない)

# job-> unknownはdrop?

# poutcome-> success or notで分ける？

# poutcome,contact -> 欠損が多いので列削除



# 欠損 -> job(unknown),education(unknown),contact(unknown),poutcome(unknown)

print(test.groupby(["job"]).agg("count"))
import matplotlib.pyplot as plt

train["job"].value_counts().plot(kind="bar")

plt.show()

train["marital"].value_counts().plot(kind="bar")

plt.show()

train["education"].value_counts().plot(kind="bar")

plt.show()

train["default"].value_counts().plot(kind="bar")

plt.show()

train["housing"].value_counts().plot(kind="bar")

plt.show()

train["loan"].value_counts().plot(kind="bar")

plt.show()

train["contact"].value_counts().plot(kind="bar")

plt.show()

train["month"].value_counts().plot(kind="bar")

plt.show()

train["poutcome"].value_counts().plot(kind="bar")

plt.show()

train["y"].value_counts().plot(kind="bar")

plt.show()
# データ分析(dummy)

import matplotlib.pyplot as plt

train["age"].hist(by=train["y"])

plt.show()

train["balance"].hist(by=train["y"])

plt.show()

# balance-1

plt.hist(train.balance)

plt.title("balance")

plt.xlabel("balance")

plt.ylabel("count")

#plt.xticks([-8000,-5000,-1000,0,100,200,300,400,500,1000,5000,10000,50000,1000000])

plt.xticks([-1000,0,100,1000])

plt.show()



# balance-2

plt.hist(train.balance)

plt.hist(train.balance, range = (0,1000), bins=20)

plt.show()



train["day"].hist(by=train["y"])

plt.show()

train["duration"].hist(by=train["y"])

plt.show()

train["campaign"].hist(by=train["y"])

plt.show()

train["pdays"].hist(by=train["y"])

plt.show()

train["previous"].hist(by=train["y"])

plt.xlabel("previous")

plt.show()

train["y"].hist(by=train["job"])

plt.ylabel("y")

plt.show()

# データ分析(test)

import matplotlib.pyplot as plt

plt.hist(test.previous)

plt.show()
# 負値補完

# pdays:-1

# to 0

print(train.shape)

# train

print(train.describe())

train["pdays"] = train["pdays"].replace(-1,0)

print(train.describe())



# test

#print(test.describe())

test["pdays"] = test["pdays"].replace(-1,0)

#print(test.describe())
# preprocessing

# campaign+previous=contact2

print(train.columns)

print(train.head(20))

train["contact2"] = train["campaign"] + train["previous"].astype(int)

print(train.columns)

print(train.head(20))

test["contact2"] = test["campaign"] + test["previous"].astype(int)



# age_range()

train["age_range"] = 1

train.loc[train["age"] < 20,"age_range"] = 1

train.loc[(train["age"] >= 20) & (train["age"] < 30),"age_range"] = 2

train.loc[(train["age"] >= 30) & (train["age"] < 40),"age_range"] = 3

train.loc[(train["age"] >= 40) & (train["age"] < 50),"age_range"] = 4

train.loc[(train["age"] >= 50) & (train["age"] < 50),"age_range"] = 5

train.loc[(train["age"] >= 60) & (train["age"] < 50),"age_range"] = 6

train.loc[(train["age"] >= 70) & (train["age"] < 50),"age_range"] = 7

train.loc[(train["age"] >= 80) & (train["age"] < 50),"age_range"] = 8

train.loc[train["age"] >= 90,"age_range"] = 9

print(train.head(20))



# age_range()

test["age_range"] = 1

test.loc[test["age"] < 20,"age_range"] = 1

test.loc[(test["age"] >= 20) & (test["age"] < 30),"age_range"] = 2

test.loc[(test["age"] >= 30) & (test["age"] < 40),"age_range"] = 3

test.loc[(test["age"] >= 40) & (test["age"] < 50),"age_range"] = 4

test.loc[(test["age"] >= 50) & (test["age"] < 50),"age_range"] = 5

test.loc[(test["age"] >= 60) & (test["age"] < 50),"age_range"] = 6

test.loc[(test["age"] >= 70) & (test["age"] < 50),"age_range"] = 7

test.loc[(test["age"] >= 80) & (test["age"] < 50),"age_range"] = 8

test.loc[test["age"] >= 90,"age_range"] = 9

print(test.head(20))



# balance_range

train["balance_range"] = 1

train.loc[train["balance"] < 0, "balance_range"] = 0

train.loc[(train["balance"] >= 0) & (train["balance"] < 10), "balance_range"] = 1

train.loc[(train["balance"] >= 10) & (train["balance"] < 100), "balance_range"] = 2

train.loc[(train["balance"] >= 100) & (train["balance"] < 300), "balance_range"] = 3

train.loc[(train["balance"] >= 300) & (train["balance"] < 600), "balance_range"] = 4

train.loc[(train["balance"] >= 600) & (train["balance"] < 1000), "balance_range"] = 5

train.loc[(train["balance"] >= 1000) & (train["balance"] < 1800), "balance_range"] = 6

train.loc[(train["balance"] >= 1800) & (train["balance"] < 3000), "balance_range"] = 7

train.loc[(train["balance"] >= 3000) & (train["balance"] < 5000), "balance_range"] = 8

train.loc[train["balance"] >= 5000, "balance_range"] = 9

print(train.head(20))



test["balance_range"] = 1

test.loc[test["balance"] < 0, "balance_range"] = 0

test.loc[(test["balance"] >= 0) & (test["balance"] < 10), "balance_range"] = 1

test.loc[(test["balance"] >= 10) & (test["balance"] < 100), "balance_range"] = 2

test.loc[(test["balance"] >= 100) & (test["balance"] < 300), "balance_range"] = 3

test.loc[(test["balance"] >= 300) & (test["balance"] < 600), "balance_range"] = 4

test.loc[(test["balance"] >= 600) & (test["balance"] < 1000), "balance_range"] = 5

test.loc[(test["balance"] >= 1000) & (test["balance"] < 1800), "balance_range"] = 6

test.loc[(test["balance"] >= 1800) & (test["balance"] < 3000), "balance_range"] = 7

test.loc[(test["balance"] >= 3000) & (test["balance"] < 5000), "balance_range"] = 8

test.loc[test["balance"] >= 5000, "balance_range"] = 9

print(test.head(20))



# day_range(1-10,11-20,21-31)

train["day_range"] = 1

train.loc[train["day"] <= 10, "day_range"] = 1

train.loc[(train["day"] > 10) & (train["day"] <= 20), "day_range"] = 2

train.loc[train["day"] > 20, "day_range"] = 3

print(train.head(20))



test["day_range"] = 1

test.loc[test["day"] <= 10, "day_range"] = 1

test.loc[(test["day"] > 10) & (test["day"] <= 20), "day_range"] = 2

test.loc[test["day"] > 20, "day_range"] = 3

print(test.head(20))

# month+day_range -> month_BME(int)

print(train.head(20))

train["month_BME"] = 1

train.loc[(train["month"] == "jan") & (train["day_range"] == 1), "month_BME"] = 3

train.loc[(train["month"] == "jan") & (train["day_range"] == 2), "month_BME"] = 2

train.loc[(train["month"] == "jan") & (train["day_range"] == 3), "month_BME"] = 1

train.loc[(train["month"] == "feb") & (train["day_range"] == 1), "month_BME"] = 27

train.loc[(train["month"] == "feb") & (train["day_range"] == 2), "month_BME"] = 26

train.loc[(train["month"] == "feb") & (train["day_range"] == 3), "month_BME"] = 25

train.loc[(train["month"] == "mar") & (train["day_range"] == 1), "month_BME"] = 33

train.loc[(train["month"] == "mar") & (train["day_range"] == 2), "month_BME"] = 32

train.loc[(train["month"] == "mar") & (train["day_range"] == 3), "month_BME"] = 31

train.loc[(train["month"] == "apr") & (train["day_range"] == 1), "month_BME"] = 30

train.loc[(train["month"] == "apr") & (train["day_range"] == 2), "month_BME"] = 29

train.loc[(train["month"] == "apr") & (train["day_range"] == 3), "month_BME"] = 28

train.loc[(train["month"] == "may") & (train["day_range"] == 1), "month_BME"] = 6

train.loc[(train["month"] == "may") & (train["day_range"] == 2), "month_BME"] = 5

train.loc[(train["month"] == "may") & (train["day_range"] == 3), "month_BME"] = 4

train.loc[(train["month"] == "jun") & (train["day_range"] == 1), "month_BME"] = 9

train.loc[(train["month"] == "jun") & (train["day_range"] == 2), "month_BME"] = 8

train.loc[(train["month"] == "jun") & (train["day_range"] == 3), "month_BME"] = 7

train.loc[(train["month"] == "jul") & (train["day_range"] == 1), "month_BME"] = 15

train.loc[(train["month"] == "jul") & (train["day_range"] == 2), "month_BME"] = 14

train.loc[(train["month"] == "jul") & (train["day_range"] == 3), "month_BME"] = 13

train.loc[(train["month"] == "aug") & (train["day_range"] == 1), "month_BME"] = 12

train.loc[(train["month"] == "aug") & (train["day_range"] == 2), "month_BME"] = 11

train.loc[(train["month"] == "aug") & (train["day_range"] == 3), "month_BME"] = 10

train.loc[(train["month"] == "sep") & (train["day_range"] == 1), "month_BME"] = 24

train.loc[(train["month"] == "sep") & (train["day_range"] == 2), "month_BME"] = 23

train.loc[(train["month"] == "sep") & (train["day_range"] == 3), "month_BME"] = 22

train.loc[(train["month"] == "oct") & (train["day_range"] == 1), "month_BME"] = 36

train.loc[(train["month"] == "oct") & (train["day_range"] == 2), "month_BME"] = 35

train.loc[(train["month"] == "oct") & (train["day_range"] == 3), "month_BME"] = 34

train.loc[(train["month"] == "nov") & (train["day_range"] == 1), "month_BME"] = 18

train.loc[(train["month"] == "nov") & (train["day_range"] == 2), "month_BME"] = 17

train.loc[(train["month"] == "nov") & (train["day_range"] == 3), "month_BME"] = 16

train.loc[(train["month"] == "dec") & (train["day_range"] == 1), "month_BME"] = 21

train.loc[(train["month"] == "dec") & (train["day_range"] == 2), "month_BME"] = 20

train.loc[(train["month"] == "dec") & (train["day_range"] == 3), "month_BME"] = 19

print(train.head(20))

print(train[30300:30360])

print(train.corr())



print(test.head(20))

test["month_BME"] = 1

test.loc[(test["month"] == "jan") & (test["day_range"] == 1), "month_BME"] = 3

test.loc[(test["month"] == "jan") & (test["day_range"] == 2), "month_BME"] = 2

test.loc[(test["month"] == "jan") & (test["day_range"] == 3), "month_BME"] = 1

test.loc[(test["month"] == "feb") & (test["day_range"] == 1), "month_BME"] = 27

test.loc[(test["month"] == "feb") & (test["day_range"] == 2), "month_BME"] = 26

test.loc[(test["month"] == "feb") & (test["day_range"] == 3), "month_BME"] = 25

test.loc[(test["month"] == "mar") & (test["day_range"] == 1), "month_BME"] = 33

test.loc[(test["month"] == "mar") & (test["day_range"] == 2), "month_BME"] = 32

test.loc[(test["month"] == "mar") & (test["day_range"] == 3), "month_BME"] = 31

test.loc[(test["month"] == "apr") & (test["day_range"] == 1), "month_BME"] = 30

test.loc[(test["month"] == "apr") & (test["day_range"] == 2), "month_BME"] = 29

test.loc[(test["month"] == "apr") & (test["day_range"] == 3), "month_BME"] = 28

test.loc[(test["month"] == "may") & (test["day_range"] == 1), "month_BME"] = 6

test.loc[(test["month"] == "may") & (test["day_range"] == 2), "month_BME"] = 5

test.loc[(test["month"] == "may") & (test["day_range"] == 3), "month_BME"] = 4

test.loc[(test["month"] == "jun") & (test["day_range"] == 1), "month_BME"] = 9

test.loc[(test["month"] == "jun") & (test["day_range"] == 2), "month_BME"] = 8

test.loc[(test["month"] == "jun") & (test["day_range"] == 3), "month_BME"] = 7

test.loc[(test["month"] == "jul") & (test["day_range"] == 1), "month_BME"] = 15

test.loc[(test["month"] == "jul") & (test["day_range"] == 2), "month_BME"] = 14

test.loc[(test["month"] == "jul") & (test["day_range"] == 3), "month_BME"] = 13

test.loc[(test["month"] == "aug") & (test["day_range"] == 1), "month_BME"] = 12

test.loc[(test["month"] == "aug") & (test["day_range"] == 2), "month_BME"] = 11

test.loc[(test["month"] == "aug") & (test["day_range"] == 3), "month_BME"] = 10

test.loc[(test["month"] == "sep") & (test["day_range"] == 1), "month_BME"] = 24

test.loc[(test["month"] == "sep") & (test["day_range"] == 2), "month_BME"] = 23

test.loc[(test["month"] == "sep") & (test["day_range"] == 3), "month_BME"] = 22

test.loc[(test["month"] == "oct") & (test["day_range"] == 1), "month_BME"] = 36

test.loc[(test["month"] == "oct") & (test["day_range"] == 2), "month_BME"] = 35

test.loc[(test["month"] == "oct") & (test["day_range"] == 3), "month_BME"] = 34

test.loc[(test["month"] == "nov") & (test["day_range"] == 1), "month_BME"] = 18

test.loc[(test["month"] == "nov") & (test["day_range"] == 2), "month_BME"] = 17

test.loc[(test["month"] == "nov") & (test["day_range"] == 3), "month_BME"] = 16

test.loc[(test["month"] == "dec") & (test["day_range"] == 1), "month_BME"] = 21

test.loc[(test["month"] == "dec") & (test["day_range"] == 2), "month_BME"] = 20

test.loc[(test["month"] == "dec") & (test["day_range"] == 3), "month_BME"] = 19

print(test.head(20))

print(test.corr())
# month+day_range -> month_BME(category)

#print(train.head(20))

# poutcome_success

print(train.head(20))

train["poutcome_success"] = 0

train.loc[train["poutcome"] == "success", "poutcome_success"] = 1

print(train.head(20))

print(train.corr())



print(test.head(20))

test["poutcome_success"] = 0

test.loc[test["poutcome"] == "success", "poutcome_success"] = 1

print(test.head(20))

print(test.corr())
# month_fma -> feb,mar,apr

print(train.head(20))

train["month_fma"] = 0

train.loc[train["month"] == "feb", "month_fma"] = 1

train.loc[train["month"] == "mar", "month_fma"] = 1

train.loc[train["month"] == "apr", "month_fma"] = 1

print(train.head(20))

print(train.corr())



print(test.head(20))

test["month_fma"] = 0

test.loc[test["month"] == "feb", "month_fma"] = 1

test.loc[test["month"] == "mar", "month_fma"] = 1

test.loc[test["month"] == "apr", "month_fma"] = 1

print(test.head(20))

print(test.corr())
# month_fmaod -> feb,mar,apr,oct,dec

print(train.head(20))

train["month_fmaod"] = 0

train.loc[train["month"] == "feb", "month_fmaod"] = 1

train.loc[train["month"] == "mar", "month_fmaod"] = 2

train.loc[train["month"] == "apr", "month_fmaod"] = 3

train.loc[train["month"] == "oct", "month_fmaod"] = 4

train.loc[train["month"] == "dec", "month_fmaod"] = 5

print(train.head(20))

print(train.corr())



print(test.head(20))

test["month_fmaod"] = 0

test.loc[test["month"] == "feb", "month_fmaod"] = 1

test.loc[test["month"] == "mar", "month_fmaod"] = 2

test.loc[test["month"] == "apr", "month_fmaod"] = 3

test.loc[test["month"] == "oct", "month_fmaod"] = 4

test.loc[test["month"] == "dec", "month_fmaod"] = 5

print(test.head(20))

print(test.corr())
# labelencoding

encoders = dict()

#for col in ["job", "marital", "education", "default", "housing", "loan", "month"]:

for col in ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]:

    le = preprocessing.LabelEncoder()

    le.fit(pd.concat([train_df[col], test_df[col]], axis=0))

    train[col] = le.transform(train_df[col])

    test[col] = le.transform(test_df[col])

    encoders[col] = le

#train.head()

print(test.head(20))

# trainのyをラベル化

train["y"] = (train_df["y"] == "yes").astype(np.int)

#train.head()

print(train.info())



# 相関

corr_matrix = train.corr()

corr_y = pd.DataFrame({"features":train.columns, "corr_y":corr_matrix["y"]}, index=None)

corr_y = corr_y.reset_index(drop=True)

#corr_y.style.background_gradient()

print(corr_y)

# 相関を確認

#print(train[['month','y']].groupby(['month'],as_index=False).mean()) 

#print(train[['loan','y']].groupby(['loan'],as_index=False).mean()) 
# データ解析ライブラリ

import pandas_profiling as pdp

#pdp.ProfileReport(train_df)
# train,testの列削除

#print(train.columns)

#train_data = train.drop("index",axis=1)



# case1 -> job,marital,education,default,housing,loan,contact,pdays,contact2,age_range,balance_range,month_BME,poutcome_success

train_data = train.drop(["index","age","balance","day","month","duration","campaign","previous","poutcome","day_range","month_fma","month_fmaod"],axis=1)

print(train_data.columns)

test_data = test.drop(["index","age","balance","day","month","duration","campaign","previous","poutcome","day_range","month_fma","month_fmaod"],axis=1)

print(test_data.columns)



# case2 



print(train_data.shape)

print(test_data.shape)

# Hold-out

from sklearn.model_selection import train_test_split

train_x = train_data.drop(["y"], axis=1)

train_y = train_data.y

train_x,test_x,train_y,test_y = train_test_split(train_x,train_y,test_size=0.2,random_state=0)
# old

#train_data = lgb.Dataset(train.drop("y", axis=1), label=train["y"])
# new

lgb_train = lgb.Dataset(train_x,train_y)

lgb_eval = lgb.Dataset(test_x,test_y,reference=lgb_train)

#print(test_lgb.head())
# old

#param = {'num_leaves': 31, 'objective': 'binary'}

#param['verbose'] = -1

#param['metric'] = 'auc'
# new

param = {'learning_rate':0.08,

             'num_iterations':5000,

             'task':'train',

             'boosting_type':'gbdt',

             'num_leaves':31,

             'objective':'binary',

             'metric':{'l2'},

             'max_depth':-1,

             'max_bin':380,

             'min_data_in_leaf':30,

             'feature_fraction':0.50,

             'bagging_fraction':0.80,

             'bagging_freq':7

            }
gbm = lgb.train(params=param,

                train_set=lgb_train,

                num_boost_round=30000,

                valid_sets=lgb_eval,

                early_stopping_rounds=2000,

                verbose_eval=True)
# validation

#import sckit-learn

from sklearn.metrics import roc_auc_score

from sklearn.metrics import accuracy_score

pred_test = gbm.predict(test_x, num_iteration=gbm.best_iteration)

#sklearn.metrics.auc(test_y,pred_test)

roc_auc_score(test_y,pred_test)

#accuracy_score(test_y,pred_test)

# old

#cv_result = lgb.cv(param, train_data, 50, nfold=5, verbose_eval=False)

# old

#np.array(cv_result["auc-mean"]).argmax()

#bst = lgb.train(param, train_data, 42)

#ypred = bst.predict(test)

#print(ypred)
# new

pred_x = gbm.predict(test_data, num_iteration=gbm.best_iteration)

print(pred_x)
# old

#sub = pd.read_csv("../input/ds2019uec-task1/sample_submission.csv")

#sub["pred"] = ypred

#sub.to_csv("submit_lgbm.csv", index=False)
# new

submit = pd.read_csv("../input/ds2019uec-task1/sample_submission.csv")

submit["pred"] = pred_x

submit.to_csv("submit_lgbm.csv", index=False)

# 重要度を表示

import lightgbm

lightgbm.plot_importance(gbm, max_num_features = 10)