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
import time

import warnings

warnings.simplefilter('ignore', FutureWarning)

import numpy as np

import pandas as pd
# 1.加工前の元データを表示させるだけ

# ごちゃごちゃしているのを確認して、どう整理するかを考えるため



train=pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

#1000行目までを読み込む、環境によっては10万行全て読み込むとconcatなど所々でエラーが発生する

submit=pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")



# print(train.head()) #→ <class 'sklearn.utils.Bunch'>

# print(submit.head())

print(train.dtypes)

# ここでは、まだ sklearn.utils.Bunch 型でも pandas.core.frame.DataFrame 型でも構わない
# ラベルエンコーダーを使って、オブジェクトをカテゴリーの数値に変換する

# 問答無用に数値化しているので、名前の意味は考えないやり方

from sklearn.preprocessing import LabelEncoder



for i in range(train.shape[1]):

    if train.iloc[:,i].dtypes == object:

        lbl = LabelEncoder()

        lbl.fit(list(train.iloc[:,i].values) + list(submit.iloc[:,i].values))

        train.iloc[:,i]  = lbl.transform(list(train.iloc[:,i].values))

        submit.iloc[:,i] = lbl.transform(list(submit.iloc[:,i].values))



print(train.dtypes)
# submitのデータ

#submit_df.info()

print("train の行数・・・{}行".format(len(train )))

print("submitの行数・・・{}行".format(len(submit)))

#わかったこと・・・trainもtestも1400行
# もらったデータの確認

# Q 全てのカラムに名前がついているか

print(len(train.columns))

print(len(submit.columns))

print(train.columns)

print(submit.columns)

# A ok

# なければ、元データにカラム名をつける

#train = pd.DataFrame(train, columns=None)
# データフレーム同士のカラム名の差分を取得・比較したい

# pandas.core.indexes.base.Index型をlistに変換すればよい

# trainにあって、testにないものを表示する

print(list(filter(lambda x: x not in submit.columns, train.columns)))

# testにあって、trainにないものを表示する

print(list(filter(lambda x: x not in train.columns,  submit.columns)))

#trainデータとtestデータの列の差分は、trainデータにSalePriceの列が存在するということ
train['WhatIsData']  = 'Train' #列を追加して、データ元を明示する

submit['WhatIsData'] = 'Test'  #列を追加して、データ元を明示する

target = 'SalePrice'

submit[target]  = 10      #テストにSurvivedカラムを仮置き

all = pd.concat([train, submit], axis=0, sort=False).reset_index(drop=True)

all.head()                     #WhatIsData列が追加されていることを確認する
# LandContour  ・・・土地の輪郭

# LotConfig　　 ・・・ロット構成

def changeFloat(s):

    try:

        float(s)

    except:

        return False

    return True



def CheckDataByColumn(mylist):

    for i, param in enumerate(mylist):

        #if(i > 20 ):continue

#         if (not param in submit.columns):

#             # 目的変数はsubmitにない。trainにあるのにsubmitにないような列はuse_columnsに追加しないためcontinue

#             print("{}はsubmitに存在しない列なので、use_columnsに追加しない".format(param))

#             continue

        if(str(train.loc[0,param]).isnumeric() == False):

            #文字列の場合

            #print("b")

            if(changeFloat(str(train.loc[0,param])) == True):

                myBool = True

            else:

                myBool = False

        else:

            # 数値の場合

            #print("aa・・・{}".format(param))

            myBool = True

#             if(len(train[param].unique()) < 20):

#                 print("ac・・・{}".format(param))

#                 myBool = False

#             else:

#                 myBool = True

        if(myBool == True):

            #if(train[param].isnull().sum() == 0 and submit[param].isnull().sum() == 0):

            print("カラム名:{}".format(param))

            print("trainの欠損値の数{}".format(train[param].isnull().sum()))

            print("test の欠損値の数{}".format(submit[param].isnull().sum()))

            print("{}  train   min={}  max={}".format(param, train[param].min(), train[param].max()))

            print("{}  submit  min={}  max={}".format(param, submit[param].min(), submit[param].max()))

            print("")

            use_columns.append(param)

        else:

            no_use_columns.append(param)

#             print("カラム名:{}".format(param))

#             print(train[param].sort_values().unique())

#             print(submit[param].sort_values().unique())



use_columns = []

no_use_columns = []

CheckDataByColumn(train.columns.values.tolist())

use_columns.remove("Id")

use_columns.remove(target)

print(use_columns)

print(no_use_columns)
# 4 統計量を出力

print(train.describe())

print("-------------------------------------------------------------------")

print(submit.describe())

#titanicのデータは、trainのAgeに欠損、testのAgeとFareに欠損
# 4-2.ここでも、Ageに欠損があることを調査する

# print(train.isnull().sum())   # 欠損があるところをsumする

def LossRate2(df):

    print("↓ここから")

    for i, param in enumerate(df):

        if df[param].isnull().sum() > 0:

            print("【{}】 の欠損率は・・・{:.2%}".format(param, df[param].isnull().sum() / len(df) ))

    print("↑ここまで")



print("trainデータのうち欠損のあるカラムの確認")

LossRate2(train)

print("")

print("submitデータのうち欠損のあるカラム確認")

LossRate2(submit)
# 欠損値を平均値で補完

df = train

#df = df.drop(['LotFrontage','MasVnrArea','GarageYrBlt'], axis=1)

print(len(df.columns))

print(df['LotFrontage'].mean())

df = df.fillna(round(df.mean(),1))

print("trainデータのうち欠損のあるカラムの確認")

LossRate2(df)

print(df['LotFrontage'].head(10))

# print(df['MasVnrArea'].head(10))

# print(df['GarageYrBlt'].head(10))

train = df



df = submit

df = df.fillna(round(df.mean(),1))

print("submitデータのうち欠損のあるカラムの確認")

LossRate2(df)

submit = df
train[use_columns].columns
# 9.元データを加工する

# 2-end.説明変数を取得してXとする

# 2-end目的変数を取得してyとする

# df = pd.DataFrame(data=[[1, 2, 3], [4, 5, 6]], columns=['a', 'b', 'c'])

# print(type(df))   # df → <class 'pandas.core.frame.DataFrame'>



#print(type(df_wine_all))   # → <class 'pandas.core.frame.DataFrame'>

#X = df.iloc[:, 2:4]         # X → <class 'pandas.core.frame.DataFrame'>

#X = df.iloc[:, 2:4].values  # X → <class 'numpy.ndarray'>



# ↓各問題に合わせて修正する

# x_nameには欠損値がなくなった列のみ追加する

# x_nameにはカテゴリ値の列を追加してはならない

# min ~ max で表示されたもののうち、欠損のないものをリストしていけば、機械的に抽出できる

#x_name = ('LotArea','YearBuilt','YearRemodAdd','1stFlrSF','2ndFlrSF')

#x_name = useList

#y_name = target



#x = train[[x_name[0],x_name[1],x_name[2],x_name[3],x_name[4]]].values  # X → <class 'numpy.ndarray'>

x = train[use_columns].values



#X = df[[X_name[0],X_name[1],X_name[2]]].values  # X → <class 'numpy.ndarray'>

# 説明変数を2つから3つに増やした方が予測の精度があがる

# 3つに増やすと精度が下がる場合もある

#X = df[['LipidAbnormalityScore','age','sex(man=0)']].values

print(x[:3])



#X = iris.data[:, 2:4]     # X→ <class 'numpy.ndarray'>

#y = df.target.values  # X → <class 'numpy.ndarray'>

y = train[target].values  # X → <class 'numpy.ndarray'>

print(y[:5]) # y → <class 'numpy.ndarray'>



#submit_ch1 = submit[[x_name[0],x_name[1],x_name[2],x_name[3],x_name[4]]]

submit_ch1 = submit[use_columns]

print("¥n↓submit_ch1")

print(submit_ch1.head())
# normarity check for the terget

import matplotlib.pyplot as plt

import seaborn as sns

y = train[target]

ax = sns.distplot(y)

plt.show()
#logを使って疑似的に正規分布にする

y_log = np.log(train[target])

ax = sns.distplot(y_log)

plt.show()
# 11.学習データとテストデータに分ける

from sklearn.model_selection import train_test_split

# ホールドアウト法の場合（データのテスト手法）

x_train, x_test, y_train, y_test = train_test_split(x, y_log, test_size=0.2, random_state=0)

#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)



# X_train = X

# y_train = y

# X_test  = X_test

# #y_test  = test_data.Survived



print(x_train[0:3])  # X_train → <class 'numpy.ndarray'>

print(x_test[0:3])   # X_test  → <class 'numpy.ndarray'>

print(y_train[0:5])  # y_train → <class 'numpy.ndarray'>

print(y_test[0:5])   # y_test  → <class 'numpy.ndarray'>
# 12.標準化させる

# 1回目はランダムフォレスト使うなら標準化させる必要なしか

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()    # 標準化

# calculate mean and standard deviation

sc.fit(x_train)

#sc.fit(x_test)

# standardization

x_train_std = sc.transform(x_train)

#y_train_std = sc.transform(y_train)

x_test_std = sc.transform(x_test)

#y_test_std = sc.transform(y_test)

print('標準化前\n', x_train[:3])     # X_test      <class 'numpy.ndarray'>

print('feature mean: {:.2f}, std: {:.2f}'.format(x_train.mean(), x_train.std()))

print('feature mean: {:.2f}, std: {:.2f}'.format(x_test.mean(), x_test.std()))

print('標準化後\n', x_train_std[:3]) # X_test_std  <class 'numpy.ndarray'>

print('↓標準化すると、平均が0、分散が1になる')

print('feature mean: {:.2f}, std: {:.2f}'.format(x_train_std.mean(), x_train_std.std()))

print('feature mean: {:.2f}, std: {:.2f}'.format(x_test_std.mean(), x_test_std.std()))
print(x_train[:1])

print(y_train[:1])

# print(x_train[:1])

# print(x_train_std[:1])

# print(submit[:1])

# print(submit_ch1[:1])
#RandomForestで学習させる

from sklearn.ensemble import RandomForestRegressor

rdf = RandomForestRegressor(

    n_estimators=200, 

    max_depth=5, 

    max_features=0.5, 

    random_state=449,

    n_jobs=-1

)

clf = rdf.fit(x_train, y_train)

# #print("経過時間：{}".format(time.time()-start_time))     # 経過時間を表示

print('trainの正解率 (Accuracy): {:.4%}'.format(clf.score(x_train, y_train))) 

print('test の正解率 (Accuracy): {:.4%}'.format(clf.score(x_test, y_test))) 



y_pred_train = rdf.predict(x_train)

y_pred_test = rdf.predict(x_test)

# 回帰の場合の評価

from sklearn.metrics import r2_score

print("trainのR2              : {:.4%}".format(r2_score(y_train, y_pred_train)))

print("test のR2              : {:.4%}".format(r2_score(y_test , y_pred_test)))



# 平均絶対誤差（MAE）

from sklearn.metrics import mean_absolute_error

#print("{}   {}".format(y_train[1], round(y_pred_train[1],0)))

print("trainの平均絶対誤差(MAE) : {:.4f}".format(mean_absolute_error(y_train, y_pred_train)))

print("test の平均絶対誤差(MAE) : {:.4f}".format(mean_absolute_error(y_test , y_pred_test )))



# 平方根平均二乗誤差（RMSE）

from sklearn.metrics import mean_squared_error

print("trainの平均二乗誤差（RMSE）: {:.4f}".format(np.sqrt(mean_squared_error(y_train, y_pred_train))))

print("test の平均二乗誤差（RMSE）: {:.4f}".format(np.sqrt(mean_squared_error(y_test , y_pred_test ))))



# RMSLE (Root Mean Squared Logarithmic Error)

# 外れ値に強い

from sklearn.metrics import mean_squared_log_error

print("trainのRMSLE           : {:.4f}".format(np.sqrt(mean_squared_log_error(y_train, y_pred_train))))

print("test のRMSLE           : {:.4f}".format(np.sqrt(mean_squared_log_error(y_test , y_pred_test ))))
# 15.ランダムフォレストで特徴量の重要度を評価する（ランダムフォレストで特徴選定）

#学習させたいデータ項目がたくさんある場合、

#全てのデータを学習に使うのではなく、まずランダムフォレストで特徴量を評価して、

#重要でない特徴量を削除すると精度が上がることが多い。



#特徴量の重要度

feature = rdf.feature_importances_



#特徴量の重要度を上から順に出力する

f = pd.DataFrame({'number': range(0, len(feature)), 'feature': feature[:]})

#f2 = f.sort_values('feature',ascending=False)

#f3 = f2.loc[:, 'number']

feature_ranking_df = f.sort_values('feature',ascending=False)

f3 = feature_ranking_df.loc[:, 'number']

feature_name = train[use_columns].columns[0:]      #特徴量の名前

indices = np.argsort(feature)[::-1]                #特徴量の重要度順（降順）



for i in range(len(feature)):

    print(str(i + 1) + "   " + str(feature_name[indices[i]]) + "   " + str(feature[indices[i]]))



#グラフ表示

import matplotlib.pyplot as plt

plt.title('Feature Importance')

plt.bar(range(len(feature)),feature[indices], color='lightblue', align='center')

plt.xticks(range(len(feature)), feature_name[indices], rotation=90)

plt.xlim([-1, len(feature)])

plt.tight_layout()

plt.show()
print(type(f2))
submit_ch1[use_columns]
# テスト値を再読み込み

df = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

#y_submit_pred = rdf.predict(submit_ch1[use_columns])         # yをそのまま表示している場合

y_submit_pred = np.exp(rdf.predict(submit_ch1[use_columns]))  # yをlog表示している場合

#print(len(y_pred))

#print(len(submit_df))

submission = pd.DataFrame({

        "Id": df["Id"],

        target: y_submit_pred

    })

submission.to_csv("submission.csv", index=False)

submission