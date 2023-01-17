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
df_train = pd.read_csv("/kaggle/input/exam-for-students20200129/train.csv")

df_test = pd.read_csv("/kaggle/input/exam-for-students20200129/test.csv")

df_country = pd.read_csv("/kaggle/input/exam-for-students20200129/country_info.csv")

df_survay = pd.read_csv("/kaggle/input/exam-for-students20200129/survey_dictionary.csv")

df_sample = pd.read_csv("/kaggle/input/exam-for-students20200129/sample_submission.csv")
#データカラム数、行数(学習データ、予測データ)

print(df_train.columns.size, len(df_train),df_test.columns.size, len(df_test),)
#データ可視化　給与のトレンドを見てみる

import matplotlib

matplotlib.pyplot.rcParams['figure.figsize'] = (20, 10)

df_train["ConvertedSalary"].hist(bins=2000)
#数値項目だけのデータセット



#Nullを0に変換し、数値のみにする

df_train_num = df_train.select_dtypes(include=[int,float]).fillna(0)

df_test_num  = df_test.select_dtypes(include=[int,float]).fillna(0)

print(len(df_train_num),len(df_test_num))
#学習データをX,Yに分ける

X_train = df_train_num[df_train_num.columns[df_train_num.columns != "Respondent"]]

y_train = df_train_num["ConvertedSalary"].values



print(X_train.shape)

print(y_train.shape)
#文字列項目を数値化する





df_train_str = df_train.select_dtypes(include=object)

df_test_str = df_test.select_dtypes(include=object)

print(df_train_str.columns.size)

print(df_test_str.columns.size)

#'MilitaryUS'がtestにはないのでtrainから削除

del df_train_str["MilitaryUS"]

print(df_train_str.columns.size)

print(df_test_str.columns.size)

#nullを-にする

df_train_str2 = df_train_str.fillna("-")

df_test_str2 = df_test_str.fillna("-")
#エンコーディング

from sklearn import preprocessing

df_train_str2 = df_train_str2.apply(preprocessing.LabelEncoder().fit_transform)

df_test_str2 = df_test_str2.apply(preprocessing.LabelEncoder().fit_transform)
#学習データをX,Yに分ける

df_train_num2 = df_train_num[df_train_num.columns[df_train_num.columns != "Respondent"]]

X_train = pd.concat([df_train_num2, df_train_str2],axis=1)

y_train = df_train_num2["ConvertedSalary"].values



print(X_train.shape)

print(y_train.shape)
#予測データを学習データのXと項目を合わせる

df_test_num2 = df_test_num[df_test_num.columns[df_test_num.columns != "Respondent"]]

X_test = pd.concat([df_test_num2, df_test_str2],axis=1)



#X_test = df_test_num[df_test_num.columns[df_test_num.columns != "Respondent"]]

#X_test = X_test.values



print(X_test.shape)



#X_test = np.c_[X_test,df_le_test.values]

#X_test = df_le_test.values
#機械学習ライブラリインポート

import sklearn

from sklearn.linear_model import *



#LightGBMモデル

import lightgbm as lgbm

model = lgbm.LGBMRegressor()
#評価指標RMSLEを定義



from sklearn.metrics import make_scorer



def rmsle(predicted, actual, size):

    return np.sqrt(np.nansum(np.square(np.log(predicted + 1) - np.log(actual + 1)))/float(size))



scorer = make_scorer(rmsle, greater_is_better=False, size=10)



#Grid Search

from sklearn.model_selection import GridSearchCV

#grid = GridSearchCV(model,{},cv=2, verbose=2)





grid = GridSearchCV(

    model,

    {},

    scoring=scorer,

    #    scoring="neg_mean_squared_log_error",

    cv=2,

    verbose=2,

    #n_jobs=2

)



grid.fit(

    X_train,

    y_train

)

pred = grid.predict(X_test)

best = grid.best_estimator_

best.fit(X_train,y_train)

pred = best.predict(X_test)
grid.best_estimator_.feature_importances_
ids = df_test["Respondent"].values

submit = np.c_[ids,pred]

print(submit)
df_sub = pd.DataFrame(

    data=submit,

    columns=[

        "Respondent",

        "ConvertedSalary"

    ]

)



df_sub["Respondent"] = df_sub["Respondent"].astype(int)

df_sub = df_sub.set_index("Respondent")

df_sub.head()
df_sub.to_csv(

    "/kaggle/working/submission.csv",

    columns=[

         "ConvertedSalary"

    ],

    index=True

)