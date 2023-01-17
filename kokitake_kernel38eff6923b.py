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
%matplotlib inline

from mpl_toolkits.mplot3d import Axes3D #3D散布図の描画

from sklearn.linear_model import LinearRegression, SGDClassifier

from sklearn.linear_model import Ridge, Lasso, ElasticNet

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # 回帰問題における性能評価に関する関数

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix # 回帰問題における性能評価に関する関数

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, KFold, GridSearchCV # 検証法

from sklearn.svm import SVR



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

# データ確認

df = pd.read_excel("../input/measurements2.xlsx")



df.head()

# df
df.info()



# distance      走行距離 (km)

# consume       燃料の消費量 (l/100km)

# speed         平均速度 (km/h)

# temp_inside   車内？の温度 (℃)

# temp_outside  車外？の温度 (℃)

# specials      特記事項 暑い日、雨、晴れ

# gas_type      ガスのタイプ (SP98とE10 E10のほうが粗悪？)

# AC            特記事項[暑い日] t/f 

# rain          特記事項[雨] t/f

# sun           特記事項[晴れ] t/f

# refill liters 補給量？ (l)

# refill gas    補給したガス？
# ユニーク値の確認

# df.nunique(dropna=False)

for i in range(len(df.dtypes)):

    if df.dtypes[i] == "object":

        obj = df.columns[i]

        print(obj, df[obj].nunique(dropna=False))

        print(df[obj].unique().tolist())
# 欠損値の確認

df.isnull().sum()
# 欠損値率

def df_detail_info(df):

    for i in df.columns:

            print("■  " + i )

            print("NULL数:" + str(df[i].isnull().sum() )+

                     "　　　　NULL率:" + str((df[i].isnull().sum()/len(df)).round(3)) +

                     "    データの種類数:" + str(df[i].value_counts().count()))

            

df_detail_info(df)
# 欠損率の高い列は削除

 # refill liters

 # refill gas

 # 時系列データだった場合、

df_copy = df

for i in df_copy.columns:

    if (df_copy[i].isnull().sum()/len(df_copy)).round(3) > 0.9:

        df = df.drop(i, axis=1)



# temp_inside の欠損値を平均値で埋める 

# 時系列データであった場合、前行で埋めるのが好ましい？

mean_temp_inside = round(df["temp_inside"].mean(), 0)

df["temp_inside"] = df["temp_inside"].fillna(mean_temp_inside)

        

# specialsは AC\rain\sun に記載があるので削除

df = df.drop(["specials"], axis=1)



# gas_type [E10,SP98]=>[0,1] ダミー変数作成

df = df.replace({

    "E10": 0,

    "SP98": 1

})
# 運転時間

df["drive_time"] = df["distance"] / df["speed"]



# 外気との温度差

df["diff_temperture"] = df["temp_inside"] - df["temp_outside"]



# なんか追加？
df_detail_info(df)
# 散布図

pd.plotting.scatter_matrix(df, figsize=(20,20))

plt.show()
# 相関係数　ヒートマップ化

fig = plt.figure(figsize=(12,10))

sns.heatmap(df.corr(), annot=True)

plt.show()
df.corr()
y = df["consume"].values

# X = df.drop(columns="consume").values

# X_list = ["distance", "speed", "temp_inside", "temp_outside", "gas_type"]

X_list = ["distance", "speed", "drive_time", "diff_temperture"]

X= df[X_list].values

print(len(X[0]))



# ホールドアウト法

test_size = 0.2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1234)



regr = LinearRegression(fit_intercept=False)

regr.fit(X_train, y_train)



# 重みを取り出す

w = [regr.intercept_]

for i in range(len(X_list)):

    w.append(regr.coef_[i])

    

x = []

for i in range(len(df.columns)): 

    if df.columns[i] in X_list :

        x.append(df[df.columns[i]].values)

    

def calculate_y_est(w, x):

    y_est = w[0]

    for i in range(len(x)):

        y_est += (w[i+1]*x[i])

    return y_est



# 重みと二乗誤差の確認

y_est = calculate_y_est(w,x)

squared_error = 0.5 * np.sum((y - y_est) ** 2)

# print('w[0] = {:.3f}, w[1] = {:.3f}, w[2] = {:.3f}, w[3] = {:.3f}, w[4] = {:.3f}, w[5] = {:.3f}, w[6] = {:.3f}'.format(w0, w1, w2, w3, w4, w5, w6))

print('二乗誤差 = {:.3f}'.format(squared_error))
# 学習データに対する予測

y_pred_train = regr.predict(X_train)

print("### 学習データ")



# MSEを計算

mse = mean_squared_error(y_train, y_pred_train) 

print("MSE = %s"%round(mse,3) )  



# MAEを計算

mae = mean_absolute_error(y_train, y_pred_train) 

print("MAE = %s"%round(mae,3) )



# RMSEを計算

rmse = np.sqrt(mse)

print("RMSE = %s"%round(rmse, 3) )



# 決定係数を計算

# r2score = r2_score(y_train, y_pred_train)

# print("r2score = %s"%round(r2score,3))



print('')

# テストデータに対する予測を実行

y_pred_test = regr.predict(X_test)

print("### テストデータ")



# MSEを計算

mse = mean_squared_error(y_test, y_pred_test) 

print("MSE = %s"%round(mse,3) )  



# MAEを計算

mae = mean_absolute_error(y_test, y_pred_test) 

print("MAE = %s"%round(mae,3) )



# RMSEを計算

rmse = np.sqrt(mse)

print("RMSE = %s"%round(rmse, 3) )



# 決定係数を計算

# r2score = r2_score(y_test, y_pred_test)

# print("r2score = %s"%round(r2score,3))
# print(x)



# X = x.reshape(-1,1) # scikit-learnに入力するために整形

n_split = 5 # グループ数を設定（今回は5分割）



cross_valid_mae = 0

split_num = 1



# テスト役を交代させながら学習と評価を繰り返す

for train_idx, test_idx in KFold(n_splits=n_split, random_state=1234).split(X, y):

    X_train, y_train = X[train_idx], y[train_idx] #学習用データ

    X_test, y_test = X[test_idx], y[test_idx]     #テスト用データ

    

    # 学習用データを使って線形回帰モデルを学習

    regr = LinearRegression(fit_intercept=False)

    regr.fit(X_train, y_train)



    # テストデータに対する予測を実行

    y_pred_test = regr.predict(X_test)

    

    # テストデータを評価

    mse = mean_squared_error(y_test, y_pred_test) 

    mae = mean_absolute_error(y_test, y_pred_test)

    rmse = np.sqrt(mse)

    print("Fold %s"%split_num)

    print("MSE = %s"%round(mse, 3))

    print("MAE = %s"%round(mae, 3))

    print("RMSE = %s"%round(rmse, 3))

    print()

    

    cross_valid_mae += mae #後で平均を取るためにMAEを加算

    split_num += 1



# MAEの平均値を最終的な汎化誤差値とする

final_mae = cross_valid_mae / n_split

print("Cross Validation MAE = %s"%round(final_mae, 3))
# 正則化

# ElasticNet



# Grid Search

alpha_params = np.logspace(-3, 3, 7)

l1_ratio_params = np.arange(0.1, 1, 0.1)

param = {

    'alpha': alpha_params,

    'l1_ratio': l1_ratio_params

}



model_tuning = GridSearchCV(ElasticNet(), 

                   param, 

                   cv=4, scoring='r2', 

                   return_train_score=True)

model_tuning.fit(X_train, y_train)



print(model_tuning.best_params_)

print('r2_fit = %s'%round(model_tuning.best_score_,3))



elastic_net_tuned = model_tuning.best_estimator_

elastic_net_tuned.fit(X_train, y_train)

y_pred = elastic_net_tuned.predict(X_test)



print(len(X[0]))

x_size = len(X[0])

plt.bar(np.linspace(0, x_size, x_size), elastic_net_tuned.coef_.reshape(x_size))

df = pd.DataFrame(model_tuning.cv_results_)[['param_alpha','param_l1_ratio','mean_test_score']]

#display(df)



sns.set()

sns.set_style('whitegrid')

fig = plt.figure(figsize=(10,5))

sns.pointplot(x = 'param_l1_ratio',

              y = 'mean_test_score',

              hue = 'param_alpha',

              data=df)
# 学習データに対する予測

y_pred_train = elastic_net_tuned.predict(X_train)

print("### 学習データ")



# MSEを計算

mse = mean_squared_error(y_train, y_pred_train) 

print("MSE = %s"%round(mse,3) )  



# MAEを計算

mae = mean_absolute_error(y_train, y_pred_train) 

print("MAE = %s"%round(mae,3) )



# RMSEを計算

rmse = np.sqrt(mse)

print("RMSE = %s"%round(rmse, 3) )



# 決定係数を計算

r2score = elastic_net_tuned.score(X_train, y_pred_train)

print("r2score = %s"%round(r2score,3))



print('')

# テストデータに対する予測を実行

y_pred_test = elastic_net_tuned.predict(X_test)

print("### テストデータ")



# MSEを計算

mse = mean_squared_error(y_test, y_pred_test) 

print("MSE = %s"%round(mse,3) )  



# MAEを計算

mae = mean_absolute_error(y_test, y_pred_test) 

print("MAE = %s"%round(mae,3) )



# RMSEを計算

rmse = np.sqrt(mse)

print("RMSE = %s"%round(rmse, 3) )



# 決定係数を計算

r2score = elastic_net_tuned.score(X_test, y_pred_test)

print("r2score = %s"%round(r2score,3))