import numpy as np

import pandas as pd

%matplotlib inline

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge,Lasso,ElasticNet 

from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LinearRegression

from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LassoCV

import seaborn as sns

from pandas.tools.plotting import scatter_matrix

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from sklearn.linear_model import LinearRegression, SGDClassifier

from sklearn.metrics import mean_squared_error, mean_absolute_error

from  sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from sklearn.model_selection import KFold

import random

import csv,os

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
# 小数点カンマ表示されたデータを正確に取得(optionである”decimals”を用いているべきだが、前回データそのものを変更したため、再度同じデータを使用)

consumption_data = pd.read_csv("../input/measurements.csv")[["distance","consume","speed","temp_inside","temp_outside","gas_type","AC","rain","sun"]]



# ガスの種類を二値で判別（E10=1,SP98=0）

consumption_data = consumption_data.replace("E10",1)

consumption_data = consumption_data.replace("SP98",0)



# 欠損値の削除

consumption_data = consumption_data.dropna(how="any")



# ---------以下からは"E10"のガスのみに注目して進めている-----------

E10 = consumption_data[consumption_data.gas_type == 1][["distance","consume","speed","temp_inside","temp_outside","rain","sun","AC"]].copy()

SP98 = consumption_data[consumption_data.gas_type == 0][["distance","consume","speed","temp_inside","temp_outside","rain","sun","AC"]].copy()
# 今回は20個のデータを使い、最終的なテストをする。

E10_final = E10.copy()

E10_test = E10.copy()

# 適当に20個のデータを選ぶ

E10_final = E10_final.sample(n=20,random_state=123)

# E10_finalのデータのインデックスを並べる

E10_final = E10_final.sort_index()

# 実際に用いるE10_testから注目した20個のデータを取り去る

for i in E10_final.index:   

    E10_test = E10_test.drop(index=i)

# E10_test : モデルを作るためのデータ

# E10_final : 最終的にモデルを評価するためのデータ

#---------------------------------------------------------------------------

#新たな説明変数を作る　originalのデータ



# temp_insideとtemp_outsideを同時に扱う（それぞれを引いたものを新たな説明変数とする）

E10_onlytemp = E10_test.copy()

E10_onlytemp["temp_difference"] = E10_onlytemp["temp_inside"] - E10_onlytemp["temp_outside"] 

E10_onlytemp = E10_onlytemp.drop("temp_inside",axis=1)

E10_onlytemp = E10_onlytemp.drop("temp_outside",axis=1)



# これから述べるがdistanceとspeedには強い相関関係がある。distnace/speed で時間という新たな変数を作る

E10_onlytime = E10_test.copy()

E10_onlytime["time"] = E10_onlytime["distance"] / E10_onlytime["speed"]

E10_onlytime = E10_onlytime.drop("distance",axis=1)

E10_onlytime = E10_onlytime.drop("speed",axis=1)



# timeとtemp_differenceを組み合わせる場合

E10_optimize = E10_onlytemp.copy()

E10_optimize["time"] = E10_optimize["distance"] / E10_optimize["speed"]

E10_optimize = E10_optimize.drop("distance",axis=1)

E10_optimize = E10_optimize.drop("speed",axis=1)

#---------------------------------------------------------------------------

#final用データ



# temp_insideとtemp_outsideを同時に扱う（それぞれを引いたものを新たなせっめい変数とする）

E10_onlytempf = E10_final.copy()

E10_onlytempf["temp_difference"] = E10_onlytempf["temp_inside"] - E10_onlytempf["temp_outside"] 

E10_onlytempf = E10_onlytempf.drop("temp_inside",axis=1)

E10_onlytempf = E10_onlytempf.drop("temp_outside",axis=1)



# これから述べるがdistanceとspeedには強い相関関係がある。distnace/speed で時間という新たな変数を作ってみる

E10_onlytimef = E10_final.copy()

E10_onlytimef["time"] = E10_onlytimef["distance"] / E10_onlytimef["speed"]

E10_onlytimef = E10_onlytimef.drop("distance",axis=1)

E10_onlytimef = E10_onlytimef.drop("speed",axis=1)



# timeとtemp_differenceを組み合わせる場合

E10_optimizef = E10_onlytempf.copy()

E10_optimizef["time"] = E10_optimizef["distance"] / E10_optimizef["speed"]

E10_optimizef = E10_optimizef.drop("distance",axis=1)

E10_optimizef = E10_optimizef.drop("speed",axis=1)
# データ処理をしていない場合の、相関図

print("データ全体の相関関ヒートマップ")

E10.corr()

sns.heatmap(E10.corr() , )

plt.show()



# rain=1　に限定したとき

print("(rain=1)に限定したときの相関ヒートマップ")

E10_rain = E10.copy()

E10_rain = E10_rain[E10_rain.rain == 1][["distance","consume","speed","temp_inside","temp_outside","sun","AC"]]

E10_rain.corr()

sns.heatmap(E10_rain.corr() , )

plt.show()



# rain=0　に限定したとき

print("(rain=0)に限定したときの相関ヒートマップ")

E10_nrain = E10.copy()

E10_nrain = E10_nrain[E10_nrain.rain == 0][["distance","consume","speed","temp_inside","temp_outside","sun","AC"]]

E10_nrain.corr()

sns.heatmap(E10_nrain.corr() , )

plt.show()



# AC=1　に限定したとき

print("(AC=1)に限定したときの相関ヒートマップ")

E10_AC = E10.copy()

E10_AC = E10_AC[E10_AC.AC == 1][["distance","consume","speed","temp_inside","temp_outside","sun","rain"]]

E10_AC.corr()

sns.heatmap(E10_AC.corr() , )

plt.show()



# AC=0　に限定したとき

print("(AC=0)に限定したときの相関ヒートマップ")

E10_nAC = E10.copy()

E10_nAC = E10_nAC[E10_nAC.AC == 0][["distance","consume","speed","temp_inside","temp_outside","sun","rain"]]

E10_nAC.corr()

sns.heatmap(E10_nAC.corr() , )

plt.show()
# はじめにデータに制限を加えず、MAEを交差バリエーション法を用いてMAEを計算してみる

conclusion =[]

# <訓練誤差,汎化誤差をホールドアウト法を用いて考える>

hold_mae=[]

# テストデータを用いている

hold_data = E10_test.copy()

hold_dataX = hold_data[["distance","speed","temp_inside","temp_outside","sun","rain","AC"]]

hold_datay = hold_data[["consume"]]

x = hold_dataX.values

y = hold_datay.values

X = x.reshape(-1,7)

#　今回は20%のデータに分割する

test_size = 0.2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1234)

# 訓練誤差

regr = LinearRegression(fit_intercept=True)

regr.fit(X_train, y_train)

y_pred_train = regr.predict(X_train)

mae_train = mean_absolute_error(y_train, y_pred_train)

hold_mae.append(mae_train)

# 汎化誤差

regr.fit(X_test, y_test)

y_pred_test = regr.predict(X_test)

mae_test = mean_absolute_error(y_test, y_pred_test)

hold_mae.append(mae_test)

# ----------------------------------------------------------------------------------

# <訓練誤差,汎化誤差を交差バリエーション法を用いて考える>

var_graph = []

var_data = E10_test.copy()

var_dataX = var_data[["distance","speed","temp_inside","temp_outside","sun","rain","AC"]]

var_datay = var_data[["consume"]]

x = var_dataX.values

y = var_datay.values

X = x.reshape(-1,7)

n_split =10

cross_valid_mae = 0

cross_valid_mae_train = 0

split_num = 1

split_num_train = 1



# 交差バリエーションを用いる

for train_idx, test_idx in KFold(n_splits=n_split, random_state=1234).split(X, y):

   #学習用データ,テスト用データ

    X_train, y_train = X[train_idx], y[train_idx] 

    X_test, y_test = X[test_idx], y[test_idx]

    # 学習用データを使って線形回帰モデルを学習

    regr = LinearRegression(fit_intercept=False)

    regr.fit(X_train, y_train)



    # 訓練誤差を計算する

    y_pred_train = regr.predict(X_train)

    mae_train = mean_absolute_error(y_train, y_pred_train)

    cross_valid_mae_train += mae_train 

    split_num_train += 1

   

    # 汎化誤差を計算する

    y_pred_test = regr.predict(X_test)    

    mae = mean_absolute_error(y_test, y_pred_test)

    cross_valid_mae += mae 

    split_num += 1



# MAEの平均値を最終的な汎訓練誤差値とする

final_mae_train = cross_valid_mae_train / n_split

hold_mae.append(final_mae_train)

# MAEの平均値を最終的な汎化誤差値とする

final_mae = cross_valid_mae / n_split

hold_mae.append(final_mae)

# lastデータの下準備

E10_finalX = E10_final[["distance","speed","temp_inside","temp_outside","sun","rain","AC"]].copy()

E10_finaly = E10_final[["consume"]].copy()

x_last = E10_finalX.values

y_last = E10_finaly.values

X_last = x_last.reshape(-1,7)

y_pred_last = regr.predict(X_last)

mae_final = mean_absolute_error(y_last, y_pred_last)

hold_mae.append(mae_final)



# 1*3行列に変換

var_graph = np.reshape(hold_mae,(1,5)) 

conclusion.append(var_graph)



var_graph = pd.DataFrame(var_graph ,index=["MAE"] ,columns = ["(hold)訓練誤差","(hold)汎化誤差","(var)訓練誤差","(var)汎化誤差","(var)最終的な誤差"])

display(var_graph)  
# 重要でない説明変数から制限を決めていくことで、モデルに対して急激な変化がないようにした。

# ちなみに私が推測した優先順位は、前述したとおり

# AC > rain > temp_inside = sun > speed = distance > temp_outside

# である。

fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))

# -----------------------------------------------------------

# <temp_outside>

E10_outside = E10_test.copy()

temp_outside = E10_outside["temp_outside"].values

consume_outside = E10_outside["consume"].values



axL.grid(which='major',color='black',linestyle=':')

axL.grid(which='minor',color='black',linestyle=':')

axL.plot(temp_outside, consume_outside, '^', color='C1')

axL.legend(loc='best')

axL.set_ylabel('consume')

axL.set_xlabel('temp_outside')



# -----------------------------------------------------------

# <distance>

E10_distance = E10_test.copy()

distance = E10_distance["distance"].values

consume_distance = E10_distance["consume"].values



axR.grid(which='major',color='black',linestyle=':')

axR.grid(which='minor',color='black',linestyle=':')

axR.plot(distance, consume_distance, '^', color='C2')

axR.legend(loc='best')

axR.set_ylabel('consume')

axR.set_xlabel('distance')

fig.show()



fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))

# -----------------------------------------------------------

# <speed>

E10_speed = E10_test.copy()

speed = E10_speed["speed"].values

consume_speed = E10_speed["consume"].values



axL.grid(which='major',color='black',linestyle=':')

axL.grid(which='minor',color='black',linestyle=':')

axL.plot(speed, consume_speed, '^', color='C3')

axL.legend(loc='best')

axL.set_ylabel('consume')

axL.set_xlabel('speed')



# -----------------------------------------------------------

# <temp_inside>

E10_inside = E10_test.copy()

temp_inside = E10_inside["temp_inside"].values

consume_inside = E10_inside["consume"].values



axR.grid(which='major',color='black',linestyle=':')

axR.grid(which='minor',color='black',linestyle=':')

axR.plot(temp_inside, consume_inside, '^', color='C4')

axR.legend(loc='best')

axR.set_ylabel('consume')

axR.set_xlabel('temp_inside')

fig.show()



fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))

# -----------------------------------------------------------

# <rain>

E10_rain = E10_test.copy()

rain = E10_rain["rain"].values

consume_rain = E10_rain["consume"].values



axL.grid(which='major',color='black',linestyle=':')

axL.grid(which='minor',color='black',linestyle=':')

axL.plot(rain, consume_rain, '^', color='C5')

axL.legend(loc='best')

axL.set_ylabel('consume')

axL.set_xlabel('rain')



# -----------------------------------------------------------

# <AC>

E10_AC = E10_test.copy()

AC = E10_AC["AC"].values

consume_AC = E10_AC["consume"].values



axR.grid(which='major',color='black',linestyle=':')

axR.grid(which='minor',color='black',linestyle=':')

axR.plot(AC, consume_AC, '^', color='C6')

axR.legend(loc='best')

axR.set_ylabel('consume')

axR.set_xlabel('AC')

fig.show()



# -----------------------------------------------------------

# <sun>

fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))



E10_sun = E10_test.copy()

sun = E10_sun["sun"].values

consume_sun = E10_sun["consume"].values



axR.grid(which='major',color='black',linestyle=':')

axR.grid(which='minor',color='black',linestyle=':')

axR.plot(sun, consume_sun, '^', color='C7')

axR.legend(loc='best')

axR.set_ylabel('consume')

axR.set_xlabel('sun')

fig.show()
# 図を見た限り、consume=7,11の2点が異常値と考えられる。



# 異常値がないデータを作成する

E10_limit = E10_test[((E10_test.sun==0)&(E10_test.consume<7))|(E10_test.sun==1)].copy()



# 交差バリエーションを用いる

var_graph = []

var_data = E10_limit.copy()

var_dataX = var_data[["distance","speed","temp_inside","temp_outside","sun","rain","AC"]]

var_datay = var_data[["consume"]]

x = var_dataX.values

y = var_datay.values

X = x.reshape(-1,7)

n_split =10

cross_valid_mae = 0

cross_valid_mae_train = 0

split_num = 1

split_num_train = 1



# 交差バリエーションを用いる

for train_idx, test_idx in KFold(n_splits=n_split, random_state=1234).split(X, y):

   #学習用データ,テスト用データ

    X_train, y_train = X[train_idx], y[train_idx] 

    X_test, y_test = X[test_idx], y[test_idx]

    # 学習用データを使って線形回帰モデルを学習

    regr = LinearRegression(fit_intercept=False)

    regr.fit(X_train, y_train)



    # 訓練誤差を計算する

    y_pred_train = regr.predict(X_train)

    mae_train = mean_absolute_error(y_train, y_pred_train)

    cross_valid_mae_train += mae_train 

    split_num_train += 1

   

    # 汎化誤差を計算する

    y_pred_test = regr.predict(X_test)    

    mae = mean_absolute_error(y_test, y_pred_test)

    cross_valid_mae += mae 

    split_num += 1



# MAEの平均値を最終的な汎訓練誤差値とする

final_mae_train = cross_valid_mae_train / n_split

var_graph.append(final_mae_train)

# MAEの平均値を最終的な汎化誤差値とする

final_mae = cross_valid_mae / n_split

var_graph.append(final_mae)

# lastデータの下準備

E10_finalX = E10_final[["distance","speed","temp_inside","temp_outside","sun","rain","AC"]].copy()

E10_finaly = E10_final[["consume"]].copy()

x_last = E10_finalX.values

y_last = E10_finaly.values

X_last = x_last.reshape(-1,7)

y_pred_last = regr.predict(X_last)

mae_final = mean_absolute_error(y_last, y_pred_last)

var_graph.append(mae_final)



# 1*3行列に変換

var_graph = np.reshape(var_graph,(1,3)) 

conclusion.append(var_graph)

var_graph = pd.DataFrame(var_graph ,index=["MAE"] ,columns = ["(var)訓練誤差","(var)汎化誤差","(var)最終的な誤差"])

display(var_graph)  
# 異常値がないデータ、最終的な誤差を標準化する

E10_limit_stand = E10_limit.copy()

E10_final_stand = E10_final.copy()

for columns in E10_limit_stand:

    E10_limit_stand[columns] = (E10_limit_stand[columns] - E10_limit_stand[columns].mean()) / E10_limit_stand[columns].std(ddof=0)

    E10_final_stand[columns] = (E10_final_stand[columns] - E10_final_stand[columns].mean()) / E10_final_stand[columns].std(ddof=0)







# 交差バリエーションを用いる

var_graph_stand = []

var_data_stand = E10_limit_stand.copy()

var_data_standX = var_data_stand[["distance","speed","temp_inside","temp_outside","sun","rain","AC"]]

var_data_standy = var_data_stand[["consume"]]

x = var_data_standX.values

y = var_data_standy.values

X = x.reshape(-1,7)

n_split =10

cross_valid_mae = 0

cross_valid_mae_train = 0

split_num = 1

split_num_train = 1

final_mae = 0

final_mae_train = 0



# 交差バリエーションを用いる

for train_idx, test_idx in KFold(n_splits=n_split, random_state=1234).split(X, y):

   #学習用データ,テスト用データ

    X_train, y_train = X[train_idx], y[train_idx] 

    X_test, y_test = X[test_idx], y[test_idx]

    # 学習用データを使って線形回帰モデルを学習

    regr = LinearRegression(fit_intercept=False)

    regr.fit(X_train, y_train)



    # 訓練誤差を計算する

    y_pred_train = regr.predict(X_train)

    mae_train = mean_absolute_error(y_train, y_pred_train)

    cross_valid_mae_train += mae_train 

    split_num_train += 1

   

    # 汎化誤差を計算する

    y_pred_test = regr.predict(X_test)    

    mae = mean_absolute_error(y_test, y_pred_test)

    cross_valid_mae += mae 

    split_num += 1



# MAEの平均値を最終的な汎訓練誤差値とする

final_mae_train = cross_valid_mae_train / n_split

var_graph_stand.append(final_mae_train)

# MAEの平均値を最終的な汎化誤差値とする

final_mae = cross_valid_mae / n_split

var_graph_stand.append(final_mae)

# lastデータの下準備

E10_finalX = E10_final_stand[["distance","speed","temp_inside","temp_outside","sun","rain","AC"]].copy()

E10_finaly = E10_final_stand[["consume"]].copy()

x_last = E10_finalX.values

y_last = E10_finaly.values

X_last = x_last.reshape(-1,7)

y_pred_last = regr.predict(X_last)

mae_final = mean_absolute_error(y_last, y_pred_last)

var_graph_stand.append(mae_final)



# 1*3行列に変換

var_graph_stand = np.reshape(var_graph_stand,(1,3)) 



var_graph_stand = pd.DataFrame(var_graph_stand ,index=["標準化MAE"] ,columns = ["訓練誤差","汎化誤差","最終的な誤差"])

display(var_graph_stand)  

# -------------------------------------------E10_onlytempを使う

# <訓練誤差,汎化誤差をホールドアウト法を用いて考える>

onlytemp_mae=[]

hold_graph=[]

var_graph=[]

# テストデータを用いている

onlytemp = E10_onlytemp.copy()

onlytempx = onlytemp.drop("consume",axis=1)

onlytempy = onlytemp[["consume"]]

x = onlytempx.values

y = onlytempy.values

X = x.reshape(-1,6)

test_size = 0.2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1234)

# 訓練誤差

regr = LinearRegression(fit_intercept=True)

regr.fit(X_train, y_train)

y_pred_train = regr.predict(X_train)

mae_train = mean_absolute_error(y_train, y_pred_train)

onlytemp_mae.append(mae_train)

# 汎化誤差

regr.fit(X_test, y_test)

y_pred_test = regr.predict(X_test)

mae_test = mean_absolute_error(y_test, y_pred_test)

onlytemp_mae.append(mae_test)

# ----------------------------------------------------------------------------------

# <訓練誤差,汎化誤差を交差バリエーション法を用いて考える>

var_onlytemp = []

n_split =10

cross_valid_mae = 0

cross_valid_mae_train = 0

split_num = 1

split_num_train = 1

# 交差バリエーションを用いる

for train_idx, test_idx in KFold(n_splits=n_split, random_state=1234).split(X, y):

   #学習用データ,テスト用データ

    X_train, y_train = X[train_idx], y[train_idx] 

    X_test, y_test = X[test_idx], y[test_idx]

    # 学習用データを使って線形回帰モデルを学習

    regr = LinearRegression(fit_intercept=False)

    regr.fit(X_train, y_train)

    # 訓練誤差を計算する

    y_pred_train = regr.predict(X_train)

    mae_train = mean_absolute_error(y_train, y_pred_train)

    cross_valid_mae_train += mae_train 

    split_num_train += 1   

    # 汎化誤差を計算する

    y_pred_test = regr.predict(X_test)    

    mae = mean_absolute_error(y_test, y_pred_test)

    cross_valid_mae += mae 

    split_num += 1



# MAEの平均値を最終的な訓練誤差値とする

final_mae_train = cross_valid_mae_train / n_split

onlytemp_mae.append(final_mae_train)

# MAEの平均値を最終的な汎化誤差値とする

final_mae = cross_valid_mae / n_split

onlytemp_mae.append(final_mae)

# ------------------------------------------------------------------------------

# lastデータの下準備

onlytemp_final = E10_onlytempf.copy()

x_final = onlytemp_final.drop("consume",axis=1)

y_final = onlytemp_final[["consume"]]

x_last = x_final.values

y_last = y_final.values

X_last = x_last.reshape(-1,6)

y_pred_last = regr.predict(X_last)

mae_final = mean_absolute_error(y_last, y_pred_last)

onlytemp_mae.append(mae_final)



# 1*3行列に変換

var_graph = np.reshape(onlytemp_mae,(1,5)) 

conclusion.append(var_graph)

var_graph = pd.DataFrame(var_graph ,index=["time_difference"] ,columns = ["(hold)訓練誤差","(hold)汎化誤差","(hold)訓練誤差","(var)汎化誤差","(var)最終的な誤差"])

display(var_graph)  



# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------





# -------------------------------------------E10_onlytimeを使う

# <訓練誤差,汎化誤差をホールドアウト法を用いて考える>

onlytime_mae=[]

hold_graph=[]

var_graph=[]

# テストデータを用いている

onlytime = E10_onlytime.copy()

onlytimex = onlytime.drop("consume",axis=1)

onlytimey = onlytime[["consume"]]

x = onlytimex.values

y = onlytimey.values

X = x.reshape(-1,6)

test_size = 0.2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1234)

# 訓練誤差

regr = LinearRegression(fit_intercept=True)

regr.fit(X_train, y_train)

y_pred_train = regr.predict(X_train)

mae_train = mean_absolute_error(y_train, y_pred_train)

onlytime_mae.append(mae_train)

# 汎化誤差

regr.fit(X_test, y_test)

y_pred_test = regr.predict(X_test)

mae_test = mean_absolute_error(y_test, y_pred_test)

onlytime_mae.append(mae_test)

# ----------------------------------------------------------------------------------

# <訓練誤差,汎化誤差を交差バリエーション法を用いて考える>

var_onlytime = []

n_split =10

cross_valid_mae = 0

cross_valid_mae_train = 0

split_num = 1

split_num_train = 1

# 交差バリエーションを用いる

for train_idx, test_idx in KFold(n_splits=n_split, random_state=1234).split(X, y):

   #学習用データ,テスト用データ

    X_train, y_train = X[train_idx], y[train_idx] 

    X_test, y_test = X[test_idx], y[test_idx]

    # 学習用データを使って線形回帰モデルを学習

    regr = LinearRegression(fit_intercept=False)

    regr.fit(X_train, y_train)

    # 訓練誤差を計算する

    y_pred_train = regr.predict(X_train)

    mae_train = mean_absolute_error(y_train, y_pred_train)

    cross_valid_mae_train += mae_train 

    split_num_train += 1   

    # 汎化誤差を計算する

    y_pred_test = regr.predict(X_test)    

    mae = mean_absolute_error(y_test, y_pred_test)

    cross_valid_mae += mae 

    split_num += 1



# MAEの平均値を最終的な訓練誤差値とする

final_mae_train = cross_valid_mae_train / n_split

onlytime_mae.append(final_mae_train)

# MAEの平均値を最終的な汎化誤差値とする

final_mae = cross_valid_mae / n_split

onlytime_mae.append(final_mae)

# ------------------------------------------------------------------------------

# lastデータの下準備

onlytime_final = E10_onlytimef.copy()

x_final = onlytime_final.drop("consume",axis=1)

y_final = onlytime_final[["consume"]]

x_last = x_final.values

y_last = y_final.values

X_last = x_last.reshape(-1,6)

y_pred_last = regr.predict(X_last)

mae_final = mean_absolute_error(y_last, y_pred_last)

onlytime_mae.append(mae_final)



# 1*3行列に変換

var_graph1 = np.reshape(onlytime_mae,(1,5)) 

var_graph = var_graph1.copy()

var_graph = pd.DataFrame(var_graph ,index=["time"] ,columns = ["(hold)訓練誤差","(hold)汎化誤差","(var)訓練誤差","(var)汎化誤差","(var)最終的な誤差"])

display(var_graph)  



# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------



# -------------------------------------------E10_optimizeを使う



# <訓練誤差,汎化誤差をホールドアウト法を用いて考える>

opt_mae=[]

hold_graph=[]

var_graph=[]

# テストデータを用いている

opt = E10_optimize.copy()

optx = opt.drop("consume",axis=1)

opty = opt[["consume"]]

x = optx.values

y = opty.values

X = x.reshape(-1,5)

test_size = 0.2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1234)

# 訓練誤差

regr = LinearRegression(fit_intercept=True)

regr.fit(X_train, y_train)

y_pred_train = regr.predict(X_train)

mae_train = mean_absolute_error(y_train, y_pred_train)

opt_mae.append(mae_train)

# 汎化誤差

regr.fit(X_test, y_test)

y_pred_test = regr.predict(X_test)

mae_test = mean_absolute_error(y_test, y_pred_test)

opt_mae.append(mae_test)

# ----------------------------------------------------------------------------------

# <訓練誤差,汎化誤差を交差バリエーション法を用いて考える>

var_opt = []

n_split =10

cross_valid_mae = 0

cross_valid_mae_train = 0

split_num = 1

split_num_train = 1

# 交差バリエーションを用いる

for train_idx, test_idx in KFold(n_splits=n_split, random_state=1234).split(X, y):

   #学習用データ,テスト用データ

    X_train, y_train = X[train_idx], y[train_idx] 

    X_test, y_test = X[test_idx], y[test_idx]

    # 学習用データを使って線形回帰モデルを学習

    regr = LinearRegression(fit_intercept=False)

    regr.fit(X_train, y_train)

    # 訓練誤差を計算する

    y_pred_train = regr.predict(X_train)

    mae_train = mean_absolute_error(y_train, y_pred_train)

    cross_valid_mae_train += mae_train 

    split_num_train += 1   

    # 汎化誤差を計算する

    y_pred_test = regr.predict(X_test)    

    mae = mean_absolute_error(y_test, y_pred_test)

    cross_valid_mae += mae 

    split_num += 1



# MAEの平均値を最終的な訓練誤差値とする

final_mae_train = cross_valid_mae_train / n_split

opt_mae.append(final_mae_train)

# MAEの平均値を最終的な汎化誤差値とする

final_mae = cross_valid_mae / n_split

opt_mae.append(final_mae)

# ------------------------------------------------------------------------------

# lastデータの下準備

optimize_final = E10_optimizef.copy()

x_final = optimize_final.drop("consume",axis=1)

y_final = optimize_final[["consume"]]

x_last = x_final.values

y_last = y_final.values

X_last = x_last.reshape(-1,5)

y_pred_last = regr.predict(X_last)

mae_final = mean_absolute_error(y_last, y_pred_last)

opt_mae.append(mae_final)



# 1*3行列に変換

var_graph = np.reshape(opt_mae,(1,5)) 



var_graph = pd.DataFrame(var_graph ,index=["optimize"] ,columns = ["(hold)訓練誤差","(hold)汎化誤差","(var)訓練誤差","(var)汎化誤差","(var)最終的な誤差"])

display(var_graph)  



# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------



# originalのデータ

E10_la = E10_onlytime.copy()

train_data = E10_la.drop("consume",axis=1)

E10_laX = E10_la[["temp_inside","temp_outside","rain","sun","AC","time"]].values

E10_lay = E10_la[["consume"]].values

# limitのデータ

E10_limit_onlytime = E10_limit.copy()

E10_limit_onlytime["time"] = E10_limit_onlytime["distance"] / E10_limit_onlytime["speed"]

E10_limit_onlytime = E10_limit_onlytime.drop("distance",axis=1)

E10_limit_onlytime = E10_limit_onlytime.drop("speed",axis=1)

E10_limit_la = E10_limit_onlytime.copy()

E10_limit_laX = E10_limit_la[["temp_inside","temp_outside","rain","sun","AC","time"]].values

E10_limit_lay = E10_limit_la[["consume"]].values

# LassoCVを使って、正則化の強さは自動決定

estimator =  LassoCV(normalize=True, cv=10)

# モデルの情報を使って特徴選択を行う場合は、SelectFromModelを使う

# 係数のしきい値はthresholdで指定し、係数が1e-5以下である特徴を削除する

sfm = SelectFromModel(estimator, threshold=1e-5)

sfm.fit(E10_laX,E10_lay)

# Trueになっている特徴が使用する特徴

sfm.get_support()

# LASSOで得た各特徴の係数の値を確認してみよう

# 係数の絶対値を取得

abs_coef = np.abs(sfm.estimator_.coef_)

# 係数を棒グラフで表示

graph_laX = ["inside","outside","rain","sun","AC","time"]

graph_lay = abs_coef

"""-------------------------------------------------------------------------"""

# 以下はlimitのデータを棒グラフにしている

estimator =  LassoCV(normalize=True, cv=10)

sfm = SelectFromModel(estimator, threshold=1e-5)

sfm.fit(E10_limit_laX,E10_limit_lay)

sfm.get_support()

abs_coef = np.abs(sfm.estimator_.coef_)

#graph_limit_laX = ["distance","speed","inside","outside","sun","rain","AC"]

graph_limit_lay = abs_coef

x_position = np.arange(len(graph_laX))



#plt.bar(graph_limit_laX,graph_limit_lay , width=0.2 , x_position + 0.4 ,y_stress , label = "limit" )

plt.bar(x_position, graph_lay , width=0.2 ,label = "original")

plt.bar(x_position + 0.2, graph_limit_lay, width=0.2 ,label = "limit")

plt.legend()

plt.xticks(x_position + 0.2 , graph_laX)

plt.show()

# 不必要な係数は"distance","temp_inside","AC"ということがわかった

# これらの要素を除いてMAEを出す

hold_mae_la=[]

for drop_conpo in ["temp_inside","AC","time"]:

    # テストデータを用いている

    hold_data_la = E10_limit_la.copy()

    hold_data_la = hold_data_la.drop("consume",axis=1).copy()

    hold_data_laX = hold_data_la.drop(drop_conpo,axis=1).copy()

    hold_data_lay = E10_limit_la[["consume"]].copy()



    x = hold_data_laX.values

    y = hold_data_lay.values

    X = x.reshape(-1,5)

    #　今回は20%のデータに分割する

    test_size = 0.2

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1234)

    # 訓練誤差

    regr = LinearRegression(fit_intercept=True)

    regr.fit(X_train, y_train)

    y_pred_train = regr.predict(X_train)

    mae_train = mean_absolute_error(y_train, y_pred_train)

    hold_mae_la.append(mae_train)

    # 汎化誤差

    regr.fit(X_test, y_test)

    y_pred_test = regr.predict(X_test)

    mae_test = mean_absolute_error(y_test, y_pred_test)

    hold_mae_la.append(mae_test)

    # ----------------------------------------------------------------------------------

    # <訓練誤差,汎化誤差を交差バリエーション法を用いて考える>

    #var_graph_la = []

    var_data_laX = hold_data_laX.copy()

    var_data_lay = hold_data_lay.copy()

    x = var_data_laX.values

    y = var_data_lay.values

    X = x.reshape(-1,5)

    n_split =10

    cross_valid_mae = 0

    cross_valid_mae_train = 0

    split_num = 1

    split_num_train = 1



    # 交差バリエーションを用いる

    for train_idx, test_idx in KFold(n_splits=n_split, random_state=1234).split(X, y):

       #学習用データ,テスト用データ

        X_train, y_train = X[train_idx], y[train_idx] 

        X_test, y_test = X[test_idx], y[test_idx]

        # 学習用データを使って線形回帰モデルを学習

        regr = LinearRegression(fit_intercept=False)

        regr.fit(X_train, y_train)



        # 訓練誤差を計算する

        y_pred_train = regr.predict(X_train)

        mae_train = mean_absolute_error(y_train, y_pred_train)

        cross_valid_mae_train += mae_train 

        split_num_train += 1



        # 汎化誤差を計算する

        y_pred_test = regr.predict(X_test)    

        mae = mean_absolute_error(y_test, y_pred_test)

        cross_valid_mae += mae 

        split_num += 1



    # MAEの平均値を最終的な汎訓練誤差値とする

    final_mae_la_train = cross_valid_mae_train / n_split

    hold_mae_la.append(final_mae_la_train)

    # MAEの平均値を最終的な汎化誤差値とする

    final_mae_la = cross_valid_mae / n_split

    hold_mae_la.append(final_mae_la)

    # lastデータの下準備

    E10_final_laX = E10_onlytimef.copy()

    E10_final_laX = E10_final_laX.drop("consume",axis=1).copy()

    E10_final_laX = E10_final_laX.drop(drop_conpo,axis=1).copy()

    E10_final_lay = E10_onlytimef[["consume"]].copy()

    x_last = E10_final_laX.values

    y_last = E10_final_lay.values

    X_last = x_last.reshape(-1,5)

    y_pred_last = regr.predict(X_last)

    mae_final_la = mean_absolute_error(y_last, y_pred_last)

    hold_mae_la.append(mae_final_la)



    # 1*3行列に変換

hold_graph_la = np.reshape(hold_mae_la,(-1,5))

var_graph_la = pd.DataFrame(hold_graph_la ,index=["(limit)inside抜き","(limit)AC抜き","(limit)time抜き"] ,columns = ["(hold)訓練誤差","(hold)汎化誤差","(var)訓練誤差","(var)汎化誤差","(var)最終的な誤差"])

display(var_graph_la)



#----------------------------------------------------------------------------------------------

# 不必要な係数は"distance","temp_inside","AC"ということがわかった

#　続いて2つ同時に除く

hold_mae_la=[]

for drop_conpo1 , drop_conpo2 in [["temp_inside","AC"],["AC","time"],["time","temp_inside"]]:

    # テストデータを用いている

    hold_data_la = E10_limit_la.copy()

    hold_data_la = hold_data_la.drop("consume",axis=1).copy()

    hold_data_la = hold_data_la.drop(drop_conpo1,axis=1).copy()

    hold_data_laX = hold_data_la.drop(drop_conpo2,axis=1).copy()

    hold_data_lay = E10_limit_la[["consume"]].copy()



    x = hold_data_laX.values

    y = hold_data_lay.values

    X = x.reshape(-1,4)

    #　今回は20%のデータに分割する

    test_size = 0.2

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1234)

    # 訓練誤差

    regr = LinearRegression(fit_intercept=True)

    regr.fit(X_train, y_train)

    y_pred_train = regr.predict(X_train)

    mae_train = mean_absolute_error(y_train, y_pred_train)

    hold_mae_la.append(mae_train)

    # 汎化誤差

    regr.fit(X_test, y_test)

    y_pred_test = regr.predict(X_test)

    mae_test = mean_absolute_error(y_test, y_pred_test)

    hold_mae_la.append(mae_test)

    # ----------------------------------------------------------------------------------

    # <訓練誤差,汎化誤差を交差バリエーション法を用いて考える>

    #var_graph_la = []

    var_data_laX = hold_data_laX.copy()

    var_data_lay = hold_data_lay.copy()

    x = var_data_laX.values

    y = var_data_lay.values

    X = x.reshape(-1,4)

    n_split =10

    cross_valid_mae = 0

    cross_valid_mae_train = 0

    split_num = 1

    split_num_train = 1



    # 交差バリエーションを用いる

    for train_idx, test_idx in KFold(n_splits=n_split, random_state=1234).split(X, y):

       #学習用データ,テスト用データ

        X_train, y_train = X[train_idx], y[train_idx] 

        X_test, y_test = X[test_idx], y[test_idx]

        # 学習用データを使って線形回帰モデルを学習

        regr = LinearRegression(fit_intercept=False)

        regr.fit(X_train, y_train)



        # 訓練誤差を計算する

        y_pred_train = regr.predict(X_train)

        mae_train = mean_absolute_error(y_train, y_pred_train)

        cross_valid_mae_train += mae_train 

        split_num_train += 1



        # 汎化誤差を計算する

        y_pred_test = regr.predict(X_test)    

        mae = mean_absolute_error(y_test, y_pred_test)

        cross_valid_mae += mae 

        split_num += 1



    # MAEの平均値を最終的な汎訓練誤差値とする

    final_mae_la_train = cross_valid_mae_train / n_split

    hold_mae_la.append(final_mae_la_train)

    # MAEの平均値を最終的な汎化誤差値とする

    final_mae_la = cross_valid_mae / n_split

    hold_mae_la.append(final_mae_la)

    # lastデータの下準備

    E10_final_laX = E10_onlytimef.copy()

    E10_final_laX = E10_final_laX.drop("consume",axis=1).copy()

    E10_final_laX = E10_final_laX.drop(drop_conpo1,axis=1).copy()

    E10_final_laX = E10_final_laX.drop(drop_conpo2,axis=1).copy()

    E10_final_lay = E10_onlytimef[["consume"]].copy()

    x_last = E10_final_laX.values

    y_last = E10_final_lay.values

    X_last = x_last.reshape(-1,4)

    y_pred_last = regr.predict(X_last)

    mae_final_la = mean_absolute_error(y_last, y_pred_last)

    hold_mae_la.append(mae_final_la)



        # 1*3行列に変換

hold_graph_la = np.reshape(hold_mae_la,(-1,5)) 

var_graph_la = pd.DataFrame(hold_graph_la ,index=["(limit)inside,AC抜き","(limit),AC,time抜き","(limit)time,inside抜き"] ,columns = ["(hold)訓練誤差","(hold)汎化誤差","(var)訓練誤差","(var)汎化誤差","(var)最終的な誤差"])

display(var_graph_la)  





# -----------------------------------------------------------------------------------------------------



# 不必要な係数は"distance","temp_inside","AC"ということがわかった

#　続いて2つ同時に除く

hold_mae_la=[]

    # テストデータを用いている

hold_data_la = E10_limit_la.copy()

hold_data_la = hold_data_la.drop("consume",axis=1).copy()

hold_data_la = hold_data_la.drop("temp_inside",axis=1).copy()

hold_data_laX = hold_data_la.drop("AC",axis=1).copy()

hold_data_laX = hold_data_laX.drop("time",axis=1).copy()

hold_data_lay = E10_limit_la[["consume"]].copy()



x = hold_data_laX.values

y = hold_data_lay.values

X = x.reshape(-1,3)

#　今回は20%のデータに分割する

test_size = 0.2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1234)

# 訓練誤差

regr = LinearRegression(fit_intercept=True)

regr.fit(X_train, y_train)

y_pred_train = regr.predict(X_train)

mae_train = mean_absolute_error(y_train, y_pred_train)

hold_mae_la.append(mae_train)

# 汎化誤差

regr.fit(X_test, y_test)

y_pred_test = regr.predict(X_test)

mae_test = mean_absolute_error(y_test, y_pred_test)

hold_mae_la.append(mae_test)

# ----------------------------------------------------------------------------------

# <訓練誤差,汎化誤差を交差バリエーション法を用いて考える>

#var_graph_la = []

var_data_laX = hold_data_laX.copy()

var_data_lay = hold_data_lay.copy()

x = var_data_laX.values

y = var_data_lay.values

X = x.reshape(-1,3)

n_split =10

cross_valid_mae = 0

cross_valid_mae_train = 0

split_num = 1

split_num_train = 1



# 交差バリエーションを用いる

for train_idx, test_idx in KFold(n_splits=n_split, random_state=1234).split(X, y):

   #学習用データ,テスト用データ

    X_train, y_train = X[train_idx], y[train_idx] 

    X_test, y_test = X[test_idx], y[test_idx]

    # 学習用データを使って線形回帰モデルを学習

    regr = LinearRegression(fit_intercept=False)

    regr.fit(X_train, y_train)



    # 訓練誤差を計算する

    y_pred_train = regr.predict(X_train)

    mae_train = mean_absolute_error(y_train, y_pred_train)

    cross_valid_mae_train += mae_train 

    split_num_train += 1



    # 汎化誤差を計算する

    y_pred_test = regr.predict(X_test)    

    mae = mean_absolute_error(y_test, y_pred_test)

    cross_valid_mae += mae 

    split_num += 1



# MAEの平均値を最終的な汎訓練誤差値とする

final_mae_la_train = cross_valid_mae_train / n_split

hold_mae_la.append(final_mae_la_train)

# MAEの平均値を最終的な汎化誤差値とする

final_mae_la = cross_valid_mae / n_split

hold_mae_la.append(final_mae_la)

# lastデータの下準備

E10_final_laX = E10_onlytimef.copy()

E10_final_laX = E10_final_laX.drop("consume",axis=1).copy()

E10_final_laX = E10_final_laX.drop("AC",axis=1).copy()

E10_final_laX = E10_final_laX.drop("temp_inside",axis=1).copy()

E10_final_laX = E10_final_laX.drop("time",axis=1).copy()



E10_final_lay = E10_onlytimef[["consume"]].copy()

x_last = E10_final_laX.values

y_last = E10_final_lay.values

X_last = x_last.reshape(-1,3)

y_pred_last = regr.predict(X_last)

mae_final_la = mean_absolute_error(y_last, y_pred_last)

hold_mae_la.append(mae_final_la)

asd =[ [0.428552,0.440020,0.449916,0.490672,0.736696]]

conclusion.append(asd)



        # 1*3行列に変換

hold_graph_la = np.reshape(hold_mae_la,(-1,5)) 

var_graph_la = pd.DataFrame(hold_graph_la ,index=["(limit)inside,AC,time抜き"] ,columns = ["(hold)訓練誤差","(hold)汎化誤差","(var)訓練誤差","(var)汎化誤差","(var)最終的な誤差"])

display(var_graph_la)  
# rainで分けてみる--------------------------------------------------------------------------------------------------------

# 上記で使ったデータはE10_limit_onlytime



rain_sample = []

rain_final_sample=[]

nrain_sample = []

nrain_final_sample=[]

# データ一覧

# rain , nrain

rain = E10_limit_onlytime[E10_limit_onlytime.rain==1].copy()

rain = rain.drop("rain",axis=1)

rain = rain.drop("AC",axis=1)

nrain = E10_limit_onlytime[E10_limit_onlytime.rain==0].copy()

nrain = nrain.drop("rain",axis=1)

nrain = nrain.drop("AC",axis=1)



# finalデータ一覧

rain_final = E10_onlytimef[E10_onlytimef.rain == 1].copy()

rain_final = rain_final.drop("rain",axis=1)

rain_final = rain_final.drop("AC",axis=1)



nrain_final = E10_onlytimef[E10_onlytimef.rain == 0].copy()

nrain_final = nrain_final.drop("rain",axis=1)

nrain_final = nrain_final.drop("AC",axis=1)





sample = [rain,nrain]

for data_rain in sample:

    hold_rainX = data_rain.copy()

    hold_rainX = hold_rainX.drop("consume",axis=1)

    hold_rainy = data_rain.copy()

    hold_rainy = hold_rainy[["consume"]]

    x = hold_rainX.values

    y = hold_rainy.values

    X = x.reshape(-1,4)

# ----------------holdout法を用いた時の訓練、汎化誤差--------------------------------

    test_size = 0.2

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1234)

    # 訓練誤差(holdout)

    regr = LinearRegression(fit_intercept=True)

    regr.fit(X_train, y_train)

    y_pred_train = regr.predict(X_train)

    mae_train = mean_absolute_error(y_train, y_pred_train)

    rain_sample.append(mae_train)# rain_sample[0][5] : holdooutの訓練誤差

    

    

    # 汎化誤差(holdout)

    regr = LinearRegression(fit_intercept=True)

    regr.fit(X_test, y_test)

    y_pred_test = regr.predict(X_test)

    mae_test = mean_absolute_error(y_test, y_pred_test)

    rain_sample.append(mae_test)# rain_sample[1][6] : holdooutの汎化誤差



#-----------------交差バリエーションを用いる-------------------------------------

    for train_idx, test_idx in KFold(n_splits=n_split, random_state=1234).split(X, y):

       #学習用データ,テスト用データ

        X_train, y_train = X[train_idx], y[train_idx] 

        X_test, y_test = X[test_idx], y[test_idx]

        # 学習用データを使って線形回帰モデルを学習

        regr = LinearRegression(fit_intercept=False)

        regr.fit(X_train, y_train)



        # 訓練誤差を計算する

        y_pred_train = regr.predict(X_train)

        mae_train = mean_absolute_error(y_train, y_pred_train)

        cross_valid_mae_train += mae_train 

        split_num_train += 1

        

        # 汎化誤差を計算する

        y_pred_test = regr.predict(X_test)    

        mae = mean_absolute_error(y_test, y_pred_test)

        cross_valid_mae += mae 

        split_num += 1

        

    # MAEの平均値を最終的な訓練誤差値とする

    final_mae_la_train = cross_valid_mae_train / split_num_train

    rain_sample.append(final_mae_la_train)# rain_sample[2][7] : variationの訓練誤差

    # MAEの平均値を最終的な汎化誤差値とする

    final_mae_la = cross_valid_mae / split_num

    rain_sample.append(final_mae_la)# rain_sample[3][8] : variationの汎化誤差

    # lastデータの下準備(rain_final ,nrain_final)



    rainf = rain_final.copy()

    rainf_x = rainf.drop("consume",axis=1)

    rainf_y = rain_final.copy()

    rainf_y = rainf_y[["consume"]]

    x = rainf_x.values

    y = rainf_y.values

    X = x.reshape(-1,4)

    y_pred_last = regr.predict(X)

    mae_final_la = mean_absolute_error(y, y_pred_last)

    # rainの時のMAEをvar_rainに収納

    rain_sample.append(mae_final_la) # rain_sample[4][9] : variationのfinal誤差       



# ----------------------------------------------------------------------------------

# →holdoutの訓練誤差     

first_hold = rain_sample[0] * len(rain)

second_hold = rain_sample[5] * len(nrain)

number_hold = len(rain) + len(nrain)

# rain_finalとnrain_finalのMAEの平均値を出す

rain_train_hold = (first_hold + second_hold)/number_hold



# →holdoutの汎化誤差   

first_var = rain_sample[1] * len(rain)

second_var = rain_sample[6] * len(nrain)

number_var = len(rain) + len(nrain)



rain_hanka_hold = (first_var + second_var)/number_var

  

# →variationの訓練誤差

first_var = rain_sample[2] * len(rain)

second_var = rain_sample[7] * len(nrain)

number_var = len(rain) + len(nrain)

rain_train_var = (first_var + second_var)/number_var

  

# →variationの汎化誤差 

first_var = rain_sample[3] * len(rain)

second_var = rain_sample[8] * len(nrain)

number_var = len(rain) + len(nrain)

rain_hanka_var = (first_var + second_var)/number_var



# final誤差 

first_var = rain_sample[4] * len(rain_final)

second_var = rain_sample[9] * len(nrain_final)

number_var = len(rain_final) + len(nrain_final)

rain_final_MAE = (first_var + second_var)/number_var





all_data = []

all_data = [rain_train_hold,rain_hanka_hold,rain_train_var,rain_hanka_var,rain_final_MAE]



all_data= np.reshape(all_data,(1,5)) 

var_graph_la = pd.DataFrame(all_data,index=["rainを2つのモデルで表現したときのMAE"],columns=["hold訓練誤差","hold汎化誤差","var訓練誤差","var汎化誤差","final汎化誤差"])

display(var_graph_la) 





# sunで分けてみる--------------------------------------------------------------------------------------------------------

# 上記で使ったデータはE10_limit_onlytime



sun_sample = []

sun_final_sample=[]

nsun_sample = []

nsun_final_sample=[]

# データ一覧

# sun , nsun

sun = E10_limit_onlytime

sun = sun[E10_limit_onlytime.sun==1].copy()

sun = sun.drop("sun",axis=1)

sun = sun.drop("rain",axis=1)

sun = sun.drop("AC",axis=1)



nsun = E10_limit_onlytime

nsun = nsun[E10_limit_onlytime.sun==0].copy()

nsun = nsun.drop("sun",axis=1)

nsun = nsun.drop("rain",axis=1)

nsun = nsun.drop("AC",axis=1)



# finalデータ一覧

sun_final = E10_onlytimef[E10_onlytimef.sun == 1].copy()

sun_final = sun_final.drop("sun",axis=1)

sun_final = sun_final.drop("AC",axis=1)



nsun_final = E10_onlytimef[E10_onlytimef.sun == 0].copy()

nsun_final = nsun_final.drop("sun",axis=1)

nsun_final = nsun_final.drop("AC",axis=1)





sample = [sun,nsun]

for data_sun in sample:

    hold_sunX = data_sun.copy()

    hold_sunX = hold_sunX.drop("consume",axis=1)

    hold_suny = data_sun.copy()

    hold_suny = hold_suny[["consume"]]

    x = hold_sunX.values

    y = hold_suny.values

    X = x.reshape(-1,3)

# ----------------holdout法を用いた時の訓練、汎化誤差--------------------------------

    test_size = 0.2

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1234)

    # 訓練誤差(holdout)

    regr = LinearRegression(fit_intercept=True)

    regr.fit(X_train, y_train)

    y_pred_train = regr.predict(X_train)

    mae_train = mean_absolute_error(y_train, y_pred_train)

    sun_sample.append(mae_train)# rain_sample[0][5] : holdooutの訓練誤差

    

    

    # 汎化誤差(holdout)

    regr = LinearRegression(fit_intercept=True)

    regr.fit(X_test, y_test)

    y_pred_test = regr.predict(X_test)

    mae_test = mean_absolute_error(y_test, y_pred_test)

    sun_sample.append(mae_test)# rain_sample[1][6] : holdooutの汎化誤差



#-----------------交差バリエーションを用いる-------------------------------------

    for train_idx, test_idx in KFold(n_splits=9, random_state=1234).split(X, y):

       #学習用データ,テスト用データ

        X_train, y_train = X[train_idx], y[train_idx] 

        X_test, y_test = X[test_idx], y[test_idx]

        # 学習用データを使って線形回帰モデルを学習

        regr = LinearRegression(fit_intercept=False)

        regr.fit(X_train, y_train)



        # 訓練誤差を計算する

        y_pred_train = regr.predict(X_train)

        mae_train = mean_absolute_error(y_train, y_pred_train)

        cross_valid_mae_train += mae_train 

        split_num_train += 1

        

        # 汎化誤差を計算する

        y_pred_test = regr.predict(X_test)    

        mae = mean_absolute_error(y_test, y_pred_test)

        cross_valid_mae += mae 

        split_num += 1

        

    # MAEの平均値を最終的な訓練誤差値とする

    final_mae_la_train = cross_valid_mae_train / split_num_train

    sun_sample.append(final_mae_la_train)# rain_sample[2][7] : variationの訓練誤差

    # MAEの平均値を最終的な汎化誤差値とする

    final_mae_la = cross_valid_mae / split_num

    sun_sample.append(final_mae_la)# rain_sample[3][8] : variationの汎化誤差

    # lastデータの下準備(rain_final ,nrain_final)



    sunf = sun_final.copy()

    sunf_x = sunf.drop("consume",axis=1)

    #sunf_x = sunf_x.drop("AC",axis=1)



    sunf_x = sunf_x.drop("rain",axis=1)

    sunf_y = sun_final.copy()

    sunf_y = sunf_y[["consume"]]

    x = sunf_x.values

    y = sunf_y.values

    X = x.reshape(-1,3)

    y_pred_last = regr.predict(X)

    mae_final_la = mean_absolute_error(y, y_pred_last)

    # rainの時のMAEをvar_rainに収納

    sun_sample.append(mae_final_la) # rain_sample[4][9] : variationのfinal誤差       



# ----------------------------------------------------------------------------------

# →holdoutの訓練誤差     

first_hold = sun_sample[0] * len(sun)

second_hold = sun_sample[5] * len(sun)

number_hold = len(sun) + len(sun)

# sun_finalとnrain_finalのMAEの平均値を出す

sun_train_hold = (first_hold + second_hold)/number_hold



# →holdoutの汎化誤差   

first_var = sun_sample[1] * len(sun)

second_var = sun_sample[6] * len(sun)

number_var = len(sun) + len(sun)



sun_hanka_hold = (first_var + second_var)/number_var

  

# →variationの訓練誤差

first_var = sun_sample[2] * len(sun)

second_var = sun_sample[7] * len(sun)

number_var = len(sun) + len(sun)

sun_train_var = (first_var + second_var)/number_var

  

# →variationの汎化誤差 

first_var = sun_sample[3] * len(sun)

second_var = sun_sample[8] * len(nsun)

number_var = len(sun) + len(nsun)

sun_hanka_var = (first_var + second_var)/number_var



# final誤差 

first_var = sun_sample[4] * len(sun_final)

second_var = sun_sample[9] * len(nsun_final)

number_var = len(sun_final) + len(nsun_final)

sun_final_MAE = (first_var + second_var)/number_var





all_data = []

all_data = [sun_train_hold,sun_hanka_hold,sun_train_var,sun_hanka_var,sun_final_MAE]

conclusion.append(all_data)

all_data= np.reshape(all_data,(1,5)) 

var_graph_la_optimize = pd.DataFrame(all_data,index=["sunを2つのモデルで表現したときのMAE"],columns=["hold訓練誤差","hold汎化誤差","var訓練誤差","var汎化誤差","final汎化誤差"])

display(var_graph_la_optimize) 
# ニューラルネットワークを使う

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,classification_report

from sklearn.metrics import confusion_matrix

from keras import models, layers



# ------------------------何も正弦を加えていないとき------------------------

# データの読み込み、準備

# 

X = E10.copy()

X = X.drop("consume",axis=1)

X = X.values

y = E10.copy()

y = y[["consume"]]

y = y.values



# データを30％で分ける

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)



def consume_model():

    model = models.Sequential()

    # reluを使っていいのかは不明。負の値はデータにはないし、良しとする。

    # 一つの説明変数から64本の矢印がinputからhiddenに向かっている

    model.add(layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)))

    # 一つの説明変数から64本の矢印が hidden1 から hidden2 に向かっている    

    model.add(layers.Dense(64, activation='relu'))

    # 一つのノードから1本の矢印が hidden2から output に向かっている

    model.add(layers.Dense(1))

    # ↑こいつが consume

    

    

    # optimizer : 深層学習の勾配法の手法を決める

    # RMSprop : 勾配の2乗の指数移動平均を取るように変更

    # https://qiita.com/tokkuman/items/1944c00415d129ca0ee9

    

    # loss : 誤差を評価する手法

    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

    return model

k = 4



# データを4分割している

val_size = len(X_train) // k

epochs = 100

all_mae = []



# 交差バリエーション(4回分割)

for i in range(k):

    

    print('Now Loading')

    

    # バリデーションデータの準備

    # i*val_size ~ (i+1)*val_size : この範囲のデータがバリエーションデータ

    val_X = X_train[i*val_size:(i+1)*val_size]

    val_y = y_train[i*val_size:(i+1)*val_size]

    

    # トレーニングデータの準備

    # concatenate : ２個以上の配列を軸指定して結合する。

    # concatenate で、[~ i*val_size]と[(i+1)*val_size ~ ]の残ったデータを繋げている

    train_data = np.concatenate([X_train[:i*val_size], X_train[(i+1)*val_size:]], axis=0)

    train_label =  np.concatenate([y_train[:i*val_size], y_train[(i+1)*val_size:]], axis=0)

    

    # モデルの構築

    model = consume_model()

    

    # モデルとデータの適合し、学習させる

    history = model.fit(train_data,

                        train_label,

                        validation_data=(val_X, val_y),

                        epochs=epochs,

                        batch_size=1,

                        verbose=0)

    all_mae.append(history.history['val_mean_absolute_error'])

    

# all_mae の平均をとっている

average_mae = np.mean(all_mae, axis=0)

epoch = range(1, epochs+1)



plt.plot(epoch, average_mae, label='Validation mae')

plt.xlabel('epoch')

plt.ylabel('mae')

plt.grid(True)

plt.legend()

plt.show()





# epoch/batchの分だけパラメータを更新し、結果的にはMAEが 0.6 ほどに落ち着いている。

train_mse_score, train_mae_score = model.evaluate(X_train, y_train)

test_mse_score, test_mae_score = model.evaluate(X_test, y_test)



all_sgd=[train_mae_score,test_mae_score]

all_sgd= np.reshape(all_sgd,(1,2)) 

var_graph_la = pd.DataFrame(all_sgd,index=["もともとのデータを用いた時のMAE"],columns=["訓練誤差","汎化誤差"])

display(var_graph_la) 
# 今回は改良していない元のデータを用いて考えたが、単純な線形回帰問題で最適化した条件で、ニューラルネットワークを考える。

# 次にE10_limit_onlytimeを用いてみる。



# --------------------E10_limit_onlytime------------------------

# データの読み込み、準備

# 

X = E10_limit_onlytime.copy()

X = X.drop("consume",axis=1)

X = X.values

y = E10_limit_onlytime.copy()

y = y[["consume"]]

y = y.values



# データを30％で分ける

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)



k = 4



# データを4分割している

val_size = len(X_train) // k

epochs = 100

all_mae = []



# 交差バリエーション(4回分割)

for i in range(k):

    

    print('Now Loading')

    

    # バリデーションデータの準備

    # i*val_size ~ (i+1)*val_size : この範囲のデータがバリエーションデータ

    val_X = X_train[i*val_size:(i+1)*val_size]

    val_y = y_train[i*val_size:(i+1)*val_size]

    

    # トレーニングデータの準備

    # concatenate : ２個以上の配列を軸指定して結合する。

    # concatenate で、[~ i*val_size]と[(i+1)*val_size ~ ]の残ったデータを繋げている

    train_data = np.concatenate([X_train[:i*val_size], X_train[(i+1)*val_size:]], axis=0)

    train_label =  np.concatenate([y_train[:i*val_size], y_train[(i+1)*val_size:]], axis=0)

    

    # モデルの構築

    model = consume_model()

    

    # モデルとデータの適合し、学習させる

    history = model.fit(train_data,

                        train_label,

                        validation_data=(val_X, val_y),

                        epochs=epochs,

                        batch_size=1,

                        verbose=0)

    all_mae.append(history.history['val_mean_absolute_error'])

    

# all_mae の平均をとっている

average_mae = np.mean(all_mae, axis=0)

epoch = range(1, epochs+1)



plt.plot(epoch, average_mae, label='Validation mae')

plt.xlabel('epoch')

plt.ylabel('mae')

plt.grid(True)

plt.legend()

plt.show()



# epoch/batchの分だけパラメータを更新し、結果的にはMAEが 0.6 ほどに落ち着いている。

train_mse_score, train_mae_score = model.evaluate(X_train, y_train)

test_mse_score, test_mae_score = model.evaluate(X_test, y_test)





all_sgd=[train_mae_score,test_mae_score]

all_sgd= np.reshape(all_sgd,(1,2)) 

var_graph_la = pd.DataFrame(all_sgd,index=["異常値をなくしたときのMAE"],columns=["訓練誤差","汎化誤差"])

display(var_graph_la) 


# ACをなくす



# --------------------E10_limit_onlytime------------------------

# データの読み込み、準備

# 

X = E10_limit_onlytime.copy()

X = X.drop("AC",axis=1)

X = X.drop("consume",axis=1)

X = X.values

y = E10_limit_onlytime.copy()

y = y[["consume"]]

y = y.values



# データを30％で分ける

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)



k = 4



# データを4分割している

val_size = len(X_train) // k

epochs = 100

all_mae = []



# 交差バリエーション(4回分割)

for i in range(k):

    

    print('Now Loading')

    

    # バリデーションデータの準備

    # i*val_size ~ (i+1)*val_size : この範囲のデータがバリエーションデータ

    val_X = X_train[i*val_size:(i+1)*val_size]

    val_y = y_train[i*val_size:(i+1)*val_size]

    

    # トレーニングデータの準備

    # concatenate : ２個以上の配列を軸指定して結合する。

    # concatenate で、[~ i*val_size]と[(i+1)*val_size ~ ]の残ったデータを繋げている

    train_data = np.concatenate([X_train[:i*val_size], X_train[(i+1)*val_size:]], axis=0)

    train_label =  np.concatenate([y_train[:i*val_size], y_train[(i+1)*val_size:]], axis=0)

    

    # モデルの構築

    model = consume_model()

    

    # モデルとデータの適合し、学習させる

    history = model.fit(train_data,

                        train_label,

                        validation_data=(val_X, val_y),

                        epochs=epochs,

                        batch_size=1,

                        verbose=0)

    all_mae.append(history.history['val_mean_absolute_error'])

    

# all_mae の平均をとっている

average_mae = np.mean(all_mae, axis=0)

epoch = range(1, epochs+1)



plt.plot(epoch, average_mae, label='Validation mae')

plt.xlabel('epoch')

plt.ylabel('mae')

plt.grid(True)

plt.legend()

plt.show()



# epoch/batchの分だけパラメータを更新し、結果的にはMAEが 0.6 ほどに落ち着いている。

train_mse_score, train_mae_score = model.evaluate(X_train, y_train)

test_mse_score, test_mae_score = model.evaluate(X_test, y_test)





all_sgd=[train_mae_score,test_mae_score]

conclusion.append(all_sgd)

all_sgd= np.reshape(all_sgd,(1,2)) 

var_graph_la = pd.DataFrame(all_sgd,index=["異常値をなくし、ACを削除したときのMAE"],columns=["訓練誤差","汎化誤差"])

display(var_graph_la) 
# 改良を加えていないときに比べて、誤差は小さくなった。

# 次にsun で分けて時を考える。

# ---------------------- sun =0の時のMAE ------------------------------------

# 今回は改良していない元のデータを用いて考えたが、単純な線形回帰問題で最適化した条件で、ニューラルネットワークを考える。

# 次にE10_limit_onlytimeを用いてみる。



# --------------------E10_limit_onlytime------------------------

# データの読み込み、準備



X = E10_limit_onlytime.copy()

X = X.drop("AC",axis=1)

X = X.drop("consume",axis=1)

X = X[X.sun == 0]

X = X.drop("sun",axis=1)

X = X.values

y = E10_limit_onlytime.copy()

y = y[y.sun == 0]

y = y[["consume"]]

y = y.values



# データを30％で分ける

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)



k = 4



# データを4分割している

val_size = len(X_train) // k

epochs = 100

all_mae = []



# 交差バリエーション(4回分割)

for i in range(k):

    

    print('Now Loading')

    

    # バリデーションデータの準備

    # i*val_size ~ (i+1)*val_size : この範囲のデータがバリエーションデータ

    val_X = X_train[i*val_size:(i+1)*val_size]

    val_y = y_train[i*val_size:(i+1)*val_size]

    

    # トレーニングデータの準備

    # concatenate : ２個以上の配列を軸指定して結合する。

    # concatenate で、[~ i*val_size]と[(i+1)*val_size ~ ]の残ったデータを繋げている

    train_data = np.concatenate([X_train[:i*val_size], X_train[(i+1)*val_size:]], axis=0)

    train_label =  np.concatenate([y_train[:i*val_size], y_train[(i+1)*val_size:]], axis=0)

    

    # モデルの構築

    model = consume_model()

    

    # モデルとデータの適合し、学習させる

    history = model.fit(train_data,

                        train_label,

                        validation_data=(val_X, val_y),

                        epochs=epochs,

                        batch_size=1,

                        verbose=0)

    all_mae.append(history.history['val_mean_absolute_error'])

    

# all_mae の平均をとっている

average_mae = np.mean(all_mae, axis=0)

epoch = range(1, epochs+1)



# epoch/batchの分だけパラメータを更新し、結果的にはMAEが 0.6 ほどに落ち着いている。



test_mse_score, test_mae_score = model.evaluate(X_test, y_test)

test_mae_score_sun0 = test_mae_score







# 改良を加えていないときに比べて、誤差は小さくなった。

# 次にsun で分けて時を考える。

# ---------------------- sun =1の時のMAE ------------------------------------

# 今回は改良していない元のデータを用いて考えたが、単純な線形回帰問題で最適化した条件で、ニューラルネットワークを考える。

# 次にE10_limit_onlytimeを用いてみる。



# --------------------E10_limit_onlytime------------------------

# データの読み込み、準備



X = E10_limit_onlytime.copy()

X = X.drop("consume",axis=1)

X = X[X.sun == 1]

X = X.drop("sun",axis=1)

X = X.values

y = E10_limit_onlytime.copy()

y = y[y.sun ==1]

y = y[["consume"]]

y = y.values



# データを30％で分ける

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)



k = 4



# データを4分割している

val_size = len(X_train) // k

epochs = 100

all_mae = []



# 交差バリエーション(4回分割)

for i in range(k):

    

    print('Now Loading')

    

    # バリデーションデータの準備

    # i*val_size ~ (i+1)*val_size : この範囲のデータがバリエーションデータ

    val_X = X_train[i*val_size:(i+1)*val_size]

    val_y = y_train[i*val_size:(i+1)*val_size]

    

    # トレーニングデータの準備

    # concatenate : ２個以上の配列を軸指定して結合する。

    # concatenate で、[~ i*val_size]と[(i+1)*val_size ~ ]の残ったデータを繋げている

    train_data = np.concatenate([X_train[:i*val_size], X_train[(i+1)*val_size:]], axis=0)

    train_label =  np.concatenate([y_train[:i*val_size], y_train[(i+1)*val_size:]], axis=0)

    

    # モデルの構築

    model = consume_model()

    

    # モデルとデータの適合し、学習させる

    history = model.fit(train_data,

                        train_label,

                        validation_data=(val_X, val_y),

                        epochs=epochs,

                        batch_size=1,

                        verbose=0)

    all_mae.append(history.history['val_mean_absolute_error'])

    

# all_mae の平均をとっている

average_mae = np.mean(all_mae, axis=0)

epoch = range(1, epochs+1)



# epoch/batchの分だけパラメータを更新し、結果的にはMAEが 0.6 ほどに落ち着いている。



test_mse_score, test_mae_score = model.evaluate(X_test, y_test)

test_mae_score_sun1 = test_mae_score



all_sgd = []

a = test_mae_score_sun1*len(E10_limit_onlytime[E10_limit_onlytime.sun==1])

b = test_mae_score_sun0*len(E10_limit_onlytime[E10_limit_onlytime.sun==0])

sum_mae = (a+b)/len(E10_limit_onlytime)

all_sgd = [test_mae_score_sun0,test_mae_score_sun1,sum_mae]



all_sgd= np.reshape(all_sgd,(1,3)) 

var_graph_la = pd.DataFrame(all_sgd,index=["異常値、ACを削除し、sunでモデルを分けた時MAE"],columns=["sun=0","sun=1","sum"])

display(var_graph_la) 
# 回帰木で予測してみる。

from sklearn.tree import DecisionTreeRegressor

X = E10_limit_onlytime.copy()

X = X.drop("AC",axis=1)

X = X.drop("consume",axis=1)

X = X[X.sun == 0]

X = X.drop("sun",axis=1)

X = X.values

y = E10_limit_onlytime.copy()

y = y[y.sun == 0]

y = y[["consume"]]

y = y.values

all = []

for i in range(7):

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)



    #最大深さ3の回帰木を作成

    regressor = DecisionTreeRegressor(max_depth=i+1)



    # 学習・テスト

    regressor.fit(X_train, y_train)



    # 訓練誤差(holdout)

    y_pred_train = regressor.predict(X_train)

    mae_train = mean_absolute_error(y_train, y_pred_train)

    all.append(mae_train)

    # 汎化誤差を計算する

    y_pred_test = regressor.predict(X_test)    

    mae = mean_absolute_error(y_test, y_pred_test)

    all.append(mae)

ddd = [0.365993,0.475744]

conclusion.append(ddd)

all= np.reshape(all,(-1,2)) 

print("(AC,異常値を除いた場合)決定木を用いてMAEを計算する。縦軸は層の数")

var_graph_la = pd.DataFrame(all,index=["1","2","3","4","5","6","7"],columns=["訓練誤差","汎化誤差"])

display(var_graph_la) 