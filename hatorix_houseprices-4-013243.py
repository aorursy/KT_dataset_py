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
# LandContour  ・・・土地の輪郭

# LotConfig　　 ・・・ロット構成
import time

import warnings

#warnings.simplefilter('ignore', FutureWarning)

import numpy as np

import pandas as pd
# 1.加工前の元データを表示させるだけ

# データを確認して、どう整理するかを考える



train=pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

#1000行目までを読み込む、環境によっては10万行全て読み込むとconcatなど所々でエラーが発生する

submit=pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")



# print(train.head()) #→ <class 'sklearn.utils.Bunch'>

# print(submit.head())

print(train.dtypes)

# ここでは、まだ sklearn.utils.Bunch 型でも pandas.core.frame.DataFrame 型でも構わない
# ラベルエンコーダーを使って、オブジェクトをカテゴリーの数値に変換する

# 問答無用に数値化している

from sklearn.preprocessing import LabelEncoder



for i in range(train.shape[1]):

    if train.iloc[:,i].dtypes == object:

        lbl = LabelEncoder()

        lbl.fit(list(train.iloc[:,i].values) + list(submit.iloc[:,i].values))

        train.iloc[:,i]  = lbl.transform(list(train.iloc[:,i].values))

        submit.iloc[:,i] = lbl.transform(list(submit.iloc[:,i].values))



print(train.dtypes)
# もらったデータの確認、行数の確認

#submit_df.info()

print("train の行数・・・{}行".format(len(train )))

print("submitの行数・・・{}行".format(len(submit)))
# もらったデータの確認、全てのカラムに名前がついているかどうかの確認

# 大抵の場合、submitにターゲットの列がない

print("train データのカラムの列数・・・{}".format(len(train.columns )))

print("submitデータのカラムの列数・・・{}".format(len(submit.columns)))

print(train.columns)

print(submit.columns)

# なければ、元データにカラム名をつける

#train = pd.DataFrame(train, columns=None)
# データフレーム同士のカラム名の差分を取得・比較したい

# 大抵の場合、ターゲットが差となる

# pandas.core.indexes.base.Index型をlistに変換すればよい

# trainにあって、testにないものを表示する

print(list(filter(lambda x: x not in submit.columns, train.columns)))

# testにあって、trainにないものを表示する

print(list(filter(lambda x: x not in train.columns,  submit.columns)))
# 

train['WhatIsData']  = 'Train'   #列を追加して、データ元を明示する

submit['WhatIsData'] = 'Submit'  #列を追加して、データ元を明示する

target = 'SalePrice'

submit[target] = 10      #テストにSurvivedカラムを仮置き

all = pd.concat([train, submit], axis=0, sort=False).reset_index(drop=True)

all.head()                     #WhatIsData列が追加されていることを確認する
# 欠損値の把握

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
# 欠損値の存在率の確認

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



x = train[use_columns].values        # X → <class 'numpy.ndarray'>



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

print(y.head())

ax = sns.distplot(y)

plt.show()
#logを使って疑似的に正規分布にする

# 以下の違いに注意する

# y_log         　← Series

# y_log.values    ← numpy.ndarray

y_log = np.log(train[target])

print(y_log.head())

ax = sns.distplot(y_log)

plt.show()
# 11.学習データとテストデータに分ける

from sklearn.model_selection import train_test_split

# ホールドアウト法の場合（データのテスト手法）

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

x_train_log, x_test_log, y_train_log, y_test_log = train_test_split(x, y_log, test_size=0.2, random_state=0)





print(x_train[0:1])  # X_train → <class 'numpy.ndarray'>

print(x_train[:,:])  # X_train → <class 'numpy.ndarray'>

print("")

print("↓ x_trainはlog化の影響を受けない")

print(x_train_log[0:1])  # X_train → <class 'numpy.ndarray'>

print(x_train_log[:,:])  # X_train → <class 'numpy.ndarray'>

print("")

print("y_train\n{}".format(y_train[:3]))  # y_train → <class 'numpy.ndarray'>

print("↓ y_train_logはlog化の影響でSeries型に変わってしまう")

print("y_train_log\n{}".format(y_train_log[:3]))  # pandas.core.series.Series

print("")

print("↓ y_train_logを.valuesでndarray化する\n{}".format(y_train_log[:3].values))  # y_train → <class 'numpy.ndarray'>

print("")
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

print(type(x_train[1,1]))

# print("")

#print(x_train)

# print(x_train[0,0].astype(np.int64))

# # print(x_train_std[:1])

# # print(submit[:1])

# # print(submit_ch1[:1])

# x_train.astype(np.int64)

# #x_train.min()
#RandomForestで学習させる

from sklearn.ensemble import RandomForestRegressor

rdf = RandomForestRegressor(n_estimators=200, max_depth=5, max_features=0.5, random_state=449, n_jobs=-1)

clf = rdf.fit(x_train, y_train)



rdf_y_log = RandomForestRegressor(n_estimators=200, max_depth=5, max_features=0.5, random_state=449, n_jobs=-1)

clf_y_log = rdf_y_log.fit(x_train, y_train_log)



# #print("経過時間：{}".format(time.time()-start_time))     # 経過時間を表示

print("                            y         y_log")

print('trainの正解率 (Accuracy): {:.4%}    {:.4%}'.format(clf.score(x_train, y_train), clf_y_log.score(x_train, y_train_log))) 

print('test の正解率 (Accuracy): {:.4%}    {:.4%}'.format(clf.score(x_test, y_test), clf_y_log.score(x_test, y_test_log))) 



y_pred_train = rdf.predict(x_train)

y_pred_test = rdf.predict(x_test)

y_log_pred_train = rdf_y_log.predict(x_train)

y_log_pred_test = rdf_y_log.predict(x_test)

# 回帰の場合の評価

from sklearn.metrics import r2_score

print("trainのR2              : {:.4%}    {:.4%}".format(r2_score(y_train, y_pred_train), r2_score(y_train_log, y_log_pred_train)))

print("test のR2              : {:.4%}    {:.4%}".format(r2_score(y_test , y_pred_test), r2_score(y_test_log , y_log_pred_test)))



# 平均絶対誤差（MAE）

from sklearn.metrics import mean_absolute_error

print("trainの平均絶対誤差(MAE) : {:.4f}  {:.4f}".format(mean_absolute_error(y_train, y_pred_train), mean_absolute_error(y_train_log, y_log_pred_train)))

print("test の平均絶対誤差(MAE) : {:.4f}  {:.4f}".format(mean_absolute_error(y_test , y_pred_test ), mean_absolute_error(y_test_log , y_log_pred_test )))



# 平方根平均二乗誤差（RMSE）

from sklearn.metrics import mean_squared_error

print("trainの平均二乗誤差（RMSE）: {:.4f}  {:.4f}".format(np.sqrt(mean_squared_error(y_train, y_pred_train)), np.sqrt(mean_squared_error(y_train_log, y_log_pred_train))))

print("test の平均二乗誤差（RMSE）: {:.4f}  {:.4f}".format(np.sqrt(mean_squared_error(y_test , y_pred_test )), np.sqrt(mean_squared_error(y_test_log , y_log_pred_test ))))



# RMSLE (Root Mean Squared Logarithmic Error)

# 外れ値に強い

from sklearn.metrics import mean_squared_log_error

print("trainのRMSLE           : {:.4f}      {:.4f}".format(np.sqrt(mean_squared_log_error(y_train, y_pred_train)), np.sqrt(mean_squared_log_error(np.exp(y_train_log), np.exp(y_log_pred_train)))))

print("test のRMSLE           : {:.4f}      {:.4f}".format(np.sqrt(mean_squared_log_error(y_test , y_pred_test )), np.sqrt(mean_squared_log_error(np.exp(y_test_log) , np.exp(y_log_pred_test )))))
#前回

#                            y          y_log

# trainの正解率 (Accuracy): 91.4618%    90.5080%

# test の正解率 (Accuracy): 83.6505%    86.0260%

# trainのR2              : 91.4618%    90.5080%

# test のR2              : 83.6505%    86.0260%

# trainの平均絶対誤差(MAE) : 16175.1998  0.0893

# test の平均絶対誤差(MAE) : 19230.9896  0.0990

# trainの平均二乗誤差（RMSE）: 22927.9516  0.1238

# test の平均二乗誤差（RMSE）: 33601.6305  0.1455

# trainのRMSLE           : 0.1432      0.1238

# test のRMSLE           : 0.1511      0.1455
# 15.ランダムフォレストで特徴量の重要度を評価する（ランダムフォレストで特徴選定）

#学習させたいデータ項目がたくさんある場合、

#全てのデータを学習に使うのではなく、まずランダムフォレストで特徴量を評価して、

#重要でない特徴量を削除すると精度が上がることが多い。



#特徴量の重要度

#feature = rdf.feature_importances_

feature = rdf_y_log.feature_importances_



#特徴量の重要度を上から順に出力する

f = pd.DataFrame({'number': range(0, len(feature)), 'feature': feature[:]})

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
# 特徴量のデータフレームを作る

df = pd.DataFrame({'number': range(0, len(feature)), 'feature': feature[:]})

s = pd.Series(train[use_columns].columns.to_list(), name='feature_name')

df = pd.concat([df, s], axis=1)

df = df.drop('number', axis=1)

df = df.reset_index(drop=True)

feature_ranking_df = df.sort_values('feature',ascending=False)

print(feature_ranking_df)
# 特徴量の多いものから30番目までを選択

use_columns_limit = feature_ranking_df[:30]['feature_name'].to_list()

print(len(use_columns_limit))



# interaction between the top 2

train["Interaction"]  = train['OverallQual']  * train['GrLivArea']

submit["Interaction"] = submit['OverallQual'] * submit['GrLivArea']

use_columns_limit.append("Interaction")

print(len(use_columns_limit))

#train.columnsにInteractionが加わっている

train.columns

# 利用する列を修正する

x = train[use_columns_limit].values   # x → <class 'numpy.ndarray'>

print(type(x))

print(x[:1])



y = train[target].values  # X → <class 'numpy.ndarray'>

print(type(y))

print(y[:5]) # y → <class 'numpy.ndarray'>



submit_ch2 = submit[use_columns_limit]

print("¥n↓submit_ch2")

print(submit_ch2.head(3))

# submitは標準化させたほうが良いのかどうか
# 学習データとテストデータに分ける

# 特徴量のランキングが出たことから、利用する列が削減されたので、もう一度実行する

from sklearn.model_selection import train_test_split

# ホールドアウト法の場合（データのテスト手法）

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

x_train_log, x_test_log, y_train_log, y_test_log = train_test_split(x, y_log, test_size=0.2, random_state=0)



# もう少し表示の方法を考える

#print(x_train[0,0], x_train_log[0,0])  # x_train → <class 'numpy.ndarray'>

#print(x_test[0,0], x_test_log[0,0])   # x_test  → <class 'numpy.ndarray'>

#print("{} ・・・\nlog {}".format(y_train[0], y_train_log[:1]))  # y_train → <class 'numpy.ndarray'>

#print("{} ・・・\nlog {}".format(y_test[0], y_test_log[:1]))   # y_test  → <class 'numpy.ndarray'>



print("↓ x_train\n{}".format(x_train[0:1]))  # X_train → <class 'numpy.ndarray'>

print(x_train[:,:])  # X_train → <class 'numpy.ndarray'>

print("")

print("↓ x_train_log　・・・x_trainに比べて、内容に変化がないことを確認する\n{}".format(x_train_log[0:1]))  # X_train → <class 'numpy.ndarray'>

print(x_train_log[:,:])  # X_train → <class 'numpy.ndarray'>

#print(x_test[0:1])   # X_test  → <class 'numpy.ndarray'>

print("")

print("y_train\n{}".format(y_train[:1]))  # y_train → <class 'numpy.ndarray'>

print("y_train_log\n{}".format(y_train_log[:1]))  # y_train → <class 'numpy.ndarray'>

print("")

#print(y_test[0:1])   # y_test  → <class 'numpy.ndarray'>

print("")
type(submit_ch2.values)
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

submit_ch2_std = sc.transform(submit_ch2.values)

print('標準化前\n', x_train[:3])     # X_test      <class 'numpy.ndarray'>

print(' x_train mean: {:.2f}, std: {:.2f}'.format(x_train.mean(), x_train.std()))

print(' x_test  mean: {:.2f}, std: {:.2f}'.format(x_test.mean(), x_test.std()))

print(' submit  mean: {:.2f}, std: {:.2f}'.format(submit_ch2.values.mean(), submit_ch2.values.std()))

print('標準化後\n', x_train_std[:3]) # X_test_std  <class 'numpy.ndarray'>

print('↓標準化すると、平均が0、分散が1になることを確認する')

print(' x_train_std mean: {:.2f}, std: {:.2f}'.format(x_train_std.mean(), x_train_std.std()))

print(' x_test_std  mean: {:.2f}, std: {:.2f}'.format(x_test_std.mean(), x_test_std.std()))

print(' submit_std  mean: {:.2f}, std: {:.2f}'.format(submit_ch2_std.mean(), submit_ch2_std.std()))
#relation to the target

fig = plt.figure(figsize=(12,7))

for i in np.arange(30):

    ax = fig.add_subplot(5,6,i+1)

    sns.regplot(x=train[use_columns_limit].iloc[:,i], y=train[target])



plt.tight_layout()

plt.show()
#外れ値を除く

# y_log_pred_train = reg_xgb_log.predict(x_train)

# y_log_pred_test = reg_xgb_log.predict(x_test)

# print("trainのR2              : {:.4%}".format(r2_score(y_train_log, y_log_pred_train)))

# print("test のR2              : {:.4%}".format(r2_score(y_test_log , y_log_pred_test)))



# # 平均絶対誤差（MAE）

# from sklearn.metrics import mean_absolute_error

# 回帰の評価を表示する関数

def ShowRegressionAssessment(xx_train, xx_test, yy_train, yy_test, reg, bool_ylog):       

    from sklearn.metrics import r2_score

    y_pred_train = reg.predict(xx_train)

    y_pred_test  = reg.predict(xx_test)

    if bool_ylog == True:

        print("trainのR2              : {:.4%}".format(r2_score(np.log(yy_train), np.log(y_pred_train))))

        print("test のR2              : {:.4%}".format(r2_score(yy_test , y_pred_test)))

    else:

        print("trainのR2              : {:.4%}".format(r2_score(yy_train, y_pred_train)))

        print("test のR2              : {:.4%}".format(r2_score(yy_test , y_pred_test)))



    # 平均絶対誤差（MAE）

    from sklearn.metrics import mean_absolute_error

    if bool_ylog == True:

        print("trainの平均絶対誤差(MAE) : {:.4f}".format(mean_absolute_error(yy_train, y_pred_train)))

        print("test の平均絶対誤差(MAE) : {:.4f}".format(mean_absolute_error(yy_test , y_pred_test )))

    else:

        print("trainの平均絶対誤差(MAE) : {:.4f}".format(mean_absolute_error(yy_train, y_pred_train)))

        print("test の平均絶対誤差(MAE) : {:.4f}".format(mean_absolute_error(yy_test , y_pred_test )))



    # 平方根平均二乗誤差（RMSE）

    from sklearn.metrics import mean_squared_error

    if bool_ylog == True:

        print("trainの平均二乗誤差（RMSE）: {:.4f}".format(np.sqrt(mean_squared_error(yy_train, y_pred_train))))

        print("test の平均二乗誤差（RMSE）: {:.4f}".format(np.sqrt(mean_squared_error(yy_test , y_pred_test ))))

    else:

        print("trainの平均二乗誤差（RMSE）: {:.4f}".format(np.sqrt(mean_squared_error(yy_train, y_pred_train))))

        print("test の平均二乗誤差（RMSE）: {:.4f}".format(np.sqrt(mean_squared_error(yy_test , y_pred_test ))))



    # RMSLE (Root Mean Squared Logarithmic Error)・・・外れ値に強い

    from sklearn.metrics import mean_squared_log_error

    if bool_ylog == True:

        print("trainのRMSLE           : {:.4f}".format(np.sqrt(mean_squared_log_error(np.exp(yy_train), np.exp(y_pred_train)))))

        print("test のRMSLE           : {:.4f}".format(np.sqrt(mean_squared_log_error(np.exp(yy_test) , np.exp(y_pred_test )))))

    else:

        print("trainのRMSLE           : {:.4f}".format(np.sqrt(mean_squared_log_error(yy_train, y_pred_train))))

        print("test のRMSLE           : {:.4f}".format(np.sqrt(mean_squared_log_error(yy_test , y_pred_test ))))
# XGBoost xは標準化しない、yはlog化しない

import xgboost as xgb

from sklearn.model_selection import GridSearchCV

print("Parameter optimization")

xgb_model = xgb.XGBRegressor()

reg_xgb = GridSearchCV(xgb_model,

                   {'max_depth': [2,4,6],

                    'n_estimators': [50,100,200]}, verbose=1)

reg_xgb.fit(x_train, y_train)

# 回帰の場合の評価

ShowRegressionAssessment(x_train, x_test, y_train, y_test, reg_xgb, False)
# trainのR2              : 93.9084%

# test のR2              : 88.9262%

# trainの平均絶対誤差(MAE) : 13592.6981

# test の平均絶対誤差(MAE) : 17269.8565

# trainの平均二乗誤差（RMSE）: 19366.3732

# test の平均二乗誤差（RMSE）: 27653.9083

# trainのRMSLE           : 0.1141

# test のRMSLE           : 0.1368
# XGBoost

import xgboost as xgb

from sklearn.model_selection import GridSearchCV

print("Parameter optimization")

xgb_model = xgb.XGBRegressor()

reg_xgb = GridSearchCV(xgb_model,

                   {'max_depth': [2,4,6],

                    'n_estimators': [50,100,200]}, verbose=1)

reg_xgb.fit(x_train_std, y_train)

# 回帰の場合の評価

ShowRegressionAssessment(x_train_std, x_test_std, y_train, y_test, reg_xgb, False)
# trainのR2              : 93.9084%

# test のR2              : 88.9262%

# trainの平均絶対誤差(MAE) : 13592.6981

# test の平均絶対誤差(MAE) : 17269.8565

# trainの平均二乗誤差（RMSE）: 19366.3732

# test の平均二乗誤差（RMSE）: 27653.9083

# trainのRMSLE           : 0.1141

# test のRMSLE           : 0.1368
# XGBoost yをlog化した場合

import xgboost as xgb

from sklearn.model_selection import GridSearchCV

print("Parameter optimization")

xgb_model = xgb.XGBRegressor()

reg_xgb_log = GridSearchCV(xgb_model,

                   {'max_depth': [2,4,6],

                    'n_estimators': [50,100,200]}, verbose=1)

reg_xgb_log.fit(x_train_std, y_train_log)

# 回帰の場合の評価

ShowRegressionAssessment(x_train_std, x_test_std, y_train_log, y_test_log, reg_xgb_log, True)
# trainのR2              : 94.7093%

# test のR2              : 88.2225%

# trainの平均絶対誤差(MAE) : 0.0672

# test の平均絶対誤差(MAE) : 0.0940

# trainの平均二乗誤差（RMSE）: 0.0915

# test の平均二乗誤差（RMSE）: 0.1335

# trainのRMSLE           : 0.0915

# test のRMSLE           : 0.1335
type(y_train)

type(y_train_log)
from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasRegressor



def create_model(optimizer='adam'):

    model = Sequential()

    model.add(Dense(x_train.shape[1], input_dim=x_train.shape[1], kernel_initializer='normal', activation='relu'))

    model.add(Dense(16, kernel_initializer='normal', activation='relu'))

    model.add(Dense(1, kernel_initializer='normal'))

    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model



model = KerasRegressor(build_fn=create_model, verbose=0)

# define the grid search parameters

optimizer = ['SGD','Adam']

batch_size = [10, 30, 50]

epochs = [10, 50, 100]

param_grid = dict(optimizer=optimizer, batch_size=batch_size, epochs=epochs)

reg_dl = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)

reg_dl.fit(x_train, y_train)

# 回帰の場合の評価

ShowRegressionAssessment(x_train, x_test, y_train, y_test, reg_dl, False)
# trainのR2              : 82.3971%

# test のR2              : 50.7468%

# trainの平均絶対誤差(MAE) : 20286.9805

# test の平均絶対誤差(MAE) : 23666.2774

# trainの平均二乗誤差（RMSE）: 32921.1390

# test の平均二乗誤差（RMSE）: 58321.1389

# trainのRMSLE           : 0.1659

# test のRMSLE           : 0.1951
from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasRegressor



def create_model(optimizer='adam'):

    model = Sequential()

    model.add(Dense(x_train.shape[1], input_dim=x_train.shape[1], kernel_initializer='normal', activation='relu'))

    model.add(Dense(16, kernel_initializer='normal', activation='relu'))

    model.add(Dense(1, kernel_initializer='normal'))

    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model



model = KerasRegressor(build_fn=create_model, verbose=0)

# define the grid search parameters

optimizer  = ['SGD','Adam']

batch_size = [10, 30, 50]

epochs     = [10, 50, 100]

param_grid = dict(optimizer=optimizer, batch_size=batch_size, epochs=epochs)

reg_dl_log = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)

reg_dl_log.fit(x_train_std, y_train_log)

# 回帰の場合の評価

ShowRegressionAssessment(x_train_std, x_test_std, y_train_log, y_test_log, reg_dl_log, True)
# # ランダムを固定してないので毎回変化する

# trainのR2              : 95.3751%

# test のR2              : 75.7272%

# trainの平均絶対誤差(MAE) : 0.0627

# test の平均絶対誤差(MAE) : 0.1162

# trainの平均二乗誤差（RMSE）: 0.0856

# test の平均二乗誤差（RMSE）: 0.1917

# trainのRMSLE           : 0.0856

# test のRMSLE           : 0.1917
#RandomForestで学習させる

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV

#print("経過時間：{}".format(time.time()-start_time))     # 経過時間を表示

rdf = RandomForestRegressor(random_state=1)

clf = GridSearchCV(rdf, {'max_depth': [2, 4, 6, 10],

                         'n_estimators': [50, 100, 200],

                         'max_features': [0.5, 1, 3, 5]},

                   verbose=1)

clf = rdf.fit(x_train_std, y_train)



rdf_log = RandomForestRegressor(random_state=1)

clf_log = GridSearchCV(rdf_log, {'max_depth': [2, 4, 6, 10],

                         'n_estimators': [50, 100, 200],

                         'max_features': [0.5, 1, 3, 5]},

                   verbose=1)

clf_log = rdf_log.fit(x_train_std, y_train_log)

# 回帰の場合の評価

print("　↓そのまま場合")

ShowRegressionAssessment(x_train_std, x_test_std, y_train, y_test, rdf, False)

print("　↓yをlog化した場合")

ShowRegressionAssessment(x_train_std, x_test_std, y_train_log, y_test_log, rdf_log, True)
# 　↓そのまま場合

# trainのR2              : 97.1931%

# test のR2              : 80.1022%

# trainの平均絶対誤差(MAE) : 7639.1691

# test の平均絶対誤差(MAE) : 18982.2127

# trainの平均二乗誤差（RMSE）: 13145.9858

# test の平均二乗誤差（RMSE）: 37068.9778

# trainのRMSLE           : 0.0684

# test のRMSLE           : 0.1521

# 　↓yをlog化した場合

# trainのR2              : 97.5367%

# test のR2              : 85.2891%

# trainの平均絶対誤差(MAE) : 0.0411

# test の平均絶対誤差(MAE) : 0.0990

# trainの平均二乗誤差（RMSE）: 0.0622

# test の平均二乗誤差（RMSE）: 0.1492

# trainのRMSLE           : 0.0622

# test のRMSLE           : 0.1492
# ニューラルネットワークを学習させる

from sklearn.neural_network import MLPClassifier

nn = MLPClassifier(solver="sgd",random_state=0,max_iter=1000)

# ↓y_train_logを入れるとエラー

nn.fit(x_train, y_train)

print(nn)

# 回帰の場合の評価

from sklearn.metrics import r2_score

ShowRegressionAssessment(x_train, x_test, y_train, y_test, nn, False)

# ｘを標準化しなかった場合・・・これはひどい

#trainのR2              : -62.0491%

#test のR2              : -56.3300%

#trainの平均絶対誤差(MAE) : 69521.2671

# test の平均絶対誤差(MAE) : 69078.7397

# trainの平均二乗誤差（RMSE）: 99886.4364

# test の平均二乗誤差（RMSE）: 103903.3912

# trainのRMSLE           : 0.5241

# test のRMSLE           : 0.5167
# ニューラルネットワークを学習させる

from sklearn.neural_network import MLPClassifier

nn = MLPClassifier(solver="sgd",random_state=0,max_iter=1000)

# ↓y_train_logを入れるとエラー、なんでだろ？

nn.fit(x_train_std, y_train)

print(nn)

# 回帰の場合の評価

ShowRegressionAssessment(x_train_std, x_test_std, y_train, y_test, nn, False)
# trainのR2              : 71.4153%

# test のR2              : 61.0542%

# trainの平均絶対誤差(MAE) : 23539.6336

# test の平均絶対誤差(MAE) : 29283.6815

# trainの平均二乗誤差（RMSE）: 41951.7694

# test の平均二乗誤差（RMSE）: 51860.7504

# trainのRMSLE           : 0.1902

# test のRMSLE           : 0.2100
# SVR

from sklearn.svm import SVR



reg_svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,

                   param_grid={"C": [1e0, 1e1, 1e2, 1e3],

                               "gamma": np.logspace(-2, 2, 5)})

reg_svr.fit(x_train, y_train)

#reg_svr.fit(x_train, y_train_log)



# 回帰の場合の評価

ShowRegressionAssessment(x_train, x_test, y_train, y_test, reg_svr, False)

# 標準化しないとやばい

# trainのR2              : -3.4359%

# test のR2              : -4.9570%

# trainの平均絶対誤差(MAE) : 54526.3865

# test の平均絶対誤差(MAE) : 55553.4578

# trainの平均二乗誤差（RMSE）: 79802.9241

# test の平均二乗誤差（RMSE）: 85136.2181

# trainのRMSLE           : 0.3977

# test のRMSLE           : 0.3900
# SVR

from sklearn.svm import SVR



reg_svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,

                   param_grid={"C": [1e0, 1e1, 1e2, 1e3],

                               "gamma": np.logspace(-2, 2, 5)})

reg_svr.fit(x_train_std, y_train)

#reg_svr.fit(x_train, y_train_log)



# 回帰の場合の評価

ShowRegressionAssessment(x_train_std, x_test_std, y_train, y_test, reg_svr, False)
# trainのR2              : 62.8899%

# test のR2              : 55.8853%

# trainの平均絶対誤差(MAE) : 26110.1031

# test の平均絶対誤差(MAE) : 26580.4888

# trainの平均二乗誤差（RMSE）: 47800.1774

# test の平均二乗誤差（RMSE）: 55195.0422

# trainのRMSLE           : 0.2133

# test のRMSLE           : 0.2021
# SVR

from sklearn.svm import SVR



reg_svr_log = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,

                   param_grid={"C": [1e0, 1e1, 1e2, 1e3],

                               "gamma": np.logspace(-2, 2, 5)})

reg_svr_log.fit(x_train_std, y_train_log)



# 回帰の場合の評価

ShowRegressionAssessment(x_train_std, x_test_std, y_train_log, y_test_log, reg_svr_log, True)
# trainのR2              : 93.7566%

# test のR2              : 89.1932%

# trainの平均絶対誤差(MAE) : 0.0724

# test の平均絶対誤差(MAE) : 0.0909

# trainの平均二乗誤差（RMSE）: 0.0985

# test の平均二乗誤差（RMSE）: 0.1279

# trainのRMSLE           : 0.0984

# test のRMSLE           : 0.1279
# second feature matrix

x_train2_std = pd.DataFrame( 

    {'XGB': reg_xgb_log.predict(x_train_std),

     'DL' : reg_dl_log.predict(x_train_std).ravel(),

     'SVR': reg_svr_log.predict(x_train_std),

    })

x_test2_std = pd.DataFrame( 

    {'XGB': reg_xgb_log.predict(x_test_std),

     'DL' : reg_dl_log.predict(x_test_std).ravel(),

     'SVR': reg_svr_log.predict(x_test_std),

    })

x_submit2_std = pd.DataFrame( 

    {'XGB': reg_xgb_log.predict(submit_ch2_std),

     'DL' : reg_dl_log.predict(submit_ch2_std).ravel(),

     'SVR': reg_svr_log.predict(submit_ch2_std),

    })

print(x_train2_std.head(2))

print(x_test2_std.head(2))

print(x_submit2_std.head(2))
# second-feature modeling using linear regression

from sklearn import linear_model



reg = linear_model.LinearRegression()

reg.fit(x_train2_std, y_train)

reg_log = linear_model.LinearRegression()

reg_log.fit(x_train2_std, y_train_log)



#clf_log = rdf_log.fit(x_train_std, y_train_log)

# 回帰の場合の評価

print("　↓そのまま場合")

# log化したやつでpredictしているためのエラーなので問題ないかと思う

#ShowRegressionAssessment(x_train2_std, x_test2_std, y_train, y_test, reg, False)

print("　↓yをlog化した場合")

ShowRegressionAssessment(x_train2_std, x_test2_std, y_train_log, y_test_log, reg_log, True)

# ↓そのまま場合

# 　↓yをlog化した場合

# trainのR2              : 95.3462%

# test のR2              : 84.7895%

# trainの平均絶対誤差(MAE) : 0.0624

# test の平均絶対誤差(MAE) : 0.0980

# trainの平均二乗誤差（RMSE）: 0.0858

# test の平均二乗誤差（RMSE）: 0.1518

# trainのRMSLE           : 0.0858

# test のRMSLE           : 0.1518
type(x_submit2_std)

x_submit2_std
# テスト値を再読み込み

df = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

#y_submit_pred = rdf.predict(submit_ch1[use_columns])         # yをそのまま表示している場合

y_submit_pred = np.exp(reg_log.predict(x_submit2_std))  # yをlog表示している場合

#print(len(y_pred))

#print(len(submit_df))

submission = pd.DataFrame({

        "Id": df["Id"],

        target: y_submit_pred

    })

submission.to_csv("submission.csv", index=False)

print(submission)
#         Id      SalePrice

# 0     1461  120177.658201

# 1     1462  159530.281683

# 2     1463  182488.068159

# 3     1464  189127.072801

# 4     1465  177505.079005

# ...    ...            ...

# 1454  2915   81569.641805

# 1455  2916   82198.481560

# 1456  2917  189403.913302

# 1457  2918  105608.693374

# 1458  2919  228085.250355