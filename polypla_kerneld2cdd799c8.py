#ライブラリのインポート

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import KFold # 交差検証法に関する関数

from sklearn.metrics import mean_squared_error, mean_absolute_error # 回帰問題における性能評価に関する関数

#データセットのインポート

car_fuel = pd.read_csv("../input/car-consume/measurements.csv")
car_fuel.head()
car_fuel.info()
for column_name in (['distance','consume','temp_inside']):    

    car_fuel_new = car_fuel.loc[:,column_name]

    car_fuel_new = car_fuel_new.str.replace(",",".").astype(float)

    car_fuel.loc[:,column_name] = car_fuel_new

car_fuel.head()



#car_fuel_dis = car_fuel.loc[:,"distance"]

#car_fuel_dis = car_fuel_dis.str.replace(",",".").astype(float)

#car_fuel.loc[:,"distance"] = car_fuel_dis

#car_fuel_dis

#car_fuel.head()
car_fuel.info()
# 散布図行列を書いてみる

pd.plotting.scatter_matrix(car_fuel, figsize=(10,10))

plt.show()

#目的変数consume に有効そうな変数はdistance と spped?
#外れ値の除外("distance")



# 列を抽出する

col = car_fuel.loc[:,"distance"]



# 平均と標準偏差

average = np.mean(col)

sd = np.std(col)



# 外れ値の基準点

outlier_min = average - (sd) * 2

outlier_max = average + (sd) * 2



# 範囲から外れている値を除く

col[col < outlier_min] = None

col[col > outlier_max] = None



car_fuel.loc[:,"distance"] = col



#外れ値の除外("consume")

# 列を抽出する

col = car_fuel.loc[:,"consume"]



# 平均と標準偏差

average = np.mean(col)

sd = np.std(col)



# 外れ値の基準点

outlier_min = average - (sd) * 2

outlier_max = average + (sd) * 2



# 範囲から外れている値を除く

col[col < outlier_min] = None

col[col > outlier_max] = None



car_fuel.loc[:,"consume"] = col
car_fuel.info()
# 散布図行列を書いてみる

pd.plotting.scatter_matrix(car_fuel, figsize=(10,10))

plt.show()

#目的変数consume に有効そうな変数はdistance と spped?
#distance 正規分布にする。logをとる、または べき変換をする

#consume 正規分布にする。　logをとる、または べき変換をする→外れ値を除外したことで、分布がきれいになったので、このままにする。

  # 学習後はexpを掛けてスケールを元に戻す。



car_fuel_dis = car_fuel.loc[:,"distance"]

car_fuel_dis = np.log(car_fuel_dis)

car_fuel.loc[:,"distance"] = car_fuel_dis



#car_fuel_cons = car_fuel.loc[:,"consume"]

#car_fuel_cons = np.log(car_fuel_cons)

#car_fuel.loc[:,"consume"] = car_fuel_cons
## temp_inside 欠損値をtemp_insideの平均値で置き換える

#car_fuel.loc[(car_fuel.temp_inside.isnull()),"temp_inside"] = car_fuel.loc[:,"temp_inside"].mean()

#car_fuel.info()

# （変更点）temp_inside の欠損値はOFFの状態なのでtemp_outsideの値で置き換える

car_fuel.loc[(car_fuel.temp_inside.isnull()),"temp_inside"] = car_fuel.loc[:,"temp_outside"]

#temp_insideとtemp_outsideの差を新しいカラムに追加し、説明変数にする

car_fuel_temp = abs(car_fuel.loc[:,"temp_outside"] - car_fuel.loc[:,"temp_inside"])

car_fuel["temp_delt"] = car_fuel_temp

car_fuel.head()



#speedとdistanceの積を走行時間として、新しいカラムに追加し、説明変数にする

# gas_type を数字に置き換える

car_fuel.loc[car_fuel["gas_type"]=="E10","gas_type"]=0

car_fuel.loc[car_fuel["gas_type"]=="SP98","gas_type"]=1

car_fuel.head()
# 不要なデータを削除する

# specials データ量不足＋扱いにくい

# refill liters データ量の不足

# refill gas gas_typeとの重複

drop_columns = ["specials","refill liters","refill gas"]

car_fuel = car_fuel.drop(drop_columns, axis = 1)

car_fuel.head()

car_fuel = car_fuel.dropna() #欠損値を１つでも含む行を削除する

car_fuel.info()
# 散布図行列を書いてみる

pd.plotting.scatter_matrix(car_fuel, figsize=(10,10))

plt.show()

#目的変数consume に有効そうな変数はdistance と spped?
#異常値の確認

df = car_fuel.boxplot(column=['distance', 'consume', 'speed', 'temp_delt'])



#speed とtemp_deltのスケールが異なるので、標準化により統一する
# speed の標準化

# 平均を引いて、標準偏差で割る操作

from sklearn.preprocessing import StandardScaler

stdsc = StandardScaler()



car_fuel_speed = car_fuel.loc[:,"speed"]

car_fuel_speed = stdsc.fit_transform(car_fuel[["speed"]].values)

car_fuel.loc[:,"speed"] = car_fuel_speed



# temp_delt の標準化

car_fuel_tdelt = car_fuel.loc[:,"temp_delt"]

car_fuel_tdelt = stdsc.fit_transform(car_fuel[["temp_delt"]].values)

car_fuel.loc[:,"temp_delt"] = car_fuel_tdelt



#df = car_fuel.boxplot(column=['distance', 'consume', 'speed', 'temp_inside', 'temp_outside', 'gas_type', 'AC', 'rain', 'sun'])

df = car_fuel.boxplot(column=['distance', 'consume', 'speed', 'temp_delt'])
# 相関係数を確認

car_fuel.corr()
# 係数を求める

y = car_fuel["consume"].values

X = car_fuel[['distance', 'speed', 'temp_delt', 'gas_type', 'AC', 'rain', 'sun']].values

regr = LinearRegression(fit_intercept=True)

regr.fit(X, y)



# 重みを取り出す

w0 = regr.intercept_

w1 = regr.coef_[0]

w2 = regr.coef_[1]

w3 = regr.coef_[2]

w4 = regr.coef_[3]

w5 = regr.coef_[4]

w6 = regr.coef_[5]

w7 = regr.coef_[6]

#w8 = regr.coef_[7]
x1 = car_fuel['distance'].values

x2 = car_fuel['speed'].values

x3 = car_fuel['temp_delt'].values

x4 = car_fuel['gas_type'].values

x5 = car_fuel['AC'].values

x6 = car_fuel['rain'].values

x7 = car_fuel['sun'].values



# 重みと二乗誤差の確認

y_est = w0 + w1 * x1 + w2 * x2 + w3 * x3 + w4 * x4 + w5 * x5 + w6 * x6 + w7 * x7

squared_error = 0.5 * np.sum((y - y_est) ** 2)

print('w0 = {:.3f}, w1 = {:.3f}, w2 = {:.3f}, w3 = {:.3f}, w4 = {:.3f}, w5 = {:.3f}, w6 = {:.3f}, w7 = {:.3f}'.format(w0, w1, w2, w3, w4, w5, w6, w7))

print('二乗誤差 = {:.3f}'.format(squared_error))
# 値を予測

#y_pred_exp = np.exp(regr.predict(X))

#y_exp = np.exp(car_fuel["consume"].values)

y_pred = regr.predict(X)

y = car_fuel["consume"].values





# MSEを計算

mse = mean_squared_error(y, y_pred) 

print("MSE = %s"%round(mse,3) )  



# MAEを計算

mae = mean_absolute_error(y, y_pred) 

print("MAE = %s"%round(mae,3) )



# RMSEを計算

rmse = np.sqrt(mse)

print("RMSE = %s"%round(rmse, 3) )
#グラフの作成

fig = plt.scatter(y,y_pred)

plt.xlabel("y")

plt.ylabel("y_predition")

#plt.xlim(3,13)

#plt.ylim(3,13)

plt.grid()
#交差検証法による汎化誤差の評価



y = car_fuel["consume"].values

x = car_fuel[['distance', 'speed', 'temp_delt', 'gas_type', 'AC', 'rain', 'sun']].values



X = x.reshape(-1,7) # scikit-learnに入力するために整形 行数任意(-1)、列数(1)の行列に変換



n_split = 5 # グループ数を設定（今回は5分割）



cross_valid_mae = 0

split_num = 1



# テスト役を交代させながら学習と評価を繰り返す

for train_idx, test_idx in KFold(n_splits=n_split, shuffle=True, random_state=1234).split(X, y):

    X_train, y_train = X[train_idx], y[train_idx] #学習用データ

    X_test, y_test = X[test_idx], y[test_idx]     #テスト用データ

    

    # 学習用データを使って線形回帰モデルを学習

    regr = LinearRegression(fit_intercept=True)

    regr.fit(X_train, y_train)



    # テストデータに対する予測を実行

#    y_pred_test = np.exp(regr.predict(X_test))

#    y_test = np.exp(y_test)

    y_pred_test = regr.predict(X_test)

    y_test = y_test

    

    # テストデータに対するMAEを計算

    mae = mean_absolute_error(y_test, y_pred_test)

    print("Fold %s"%split_num)

    print("MAE = %s"%round(mae, 3))

    print()

    

    cross_valid_mae += mae #後で平均を取るためにMAEを加算

    split_num += 1



# MAEの平均値を最終的な汎化誤差値とする

final_mae = cross_valid_mae / n_split

print("Cross Validation MAE = %s"%round(final_mae, 3))
#グリッドサーチの実装

from sklearn.model_selection import GridSearchCV



alphas = [0.0, 1e-8, 1e-5, 1e-1]#alpha(数式ではλ)

#alpha_params = np.logspace(-3, 3, 7)

#l1_ratio_params = np.arange(0.1, 1.0, 0.1)

#estimator = ElasticNet()

#paramters = {'alpha': alpha_params,'l1_ratio': l1_ratio_params}



grid_search = GridSearchCV(LogisticRegression(), alphas, cv=5)

#grid_search = GridSearchCV(estimator = ElasticNet(), alphas, cv=5)





import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



def plot_approximation(est, ax, label=None):

    """Plot the approximation of ``est`` on axis ``ax``. """

    ax.plot(x_plot, f(x_plot), color='green')

    ax.scatter(X, y, s=10)

    ax.plot(x_plot, est.predict(x_plot[:, np.newaxis]), color='red', label=label)

    ax.set_ylim((-2, 2))

    ax.set_xlim((0, 1))

    ax.set_ylabel('y')

    ax.set_xlabel('x')

    ax.legend(loc='upper right')  #, fontsize='small')



def plot_coefficients(est, ax, label=None, yscale='log'):

    coef = est.steps[-1][1].coef_.ravel() #coef_ 偏回帰係数 ravel() 一次元のリストで返す

    if yscale == 'log':

        ax.semilogy(np.abs(coef), marker='o', label=label)

        ax.set_ylim((1e-1, 1e8))

    else:

        ax.plot(np.abs(coef), marker='o', label=label)

    ax.set_ylabel('abs(coefficient)')

    ax.set_xlabel('coefficients')

    ax.set_xlim((0, 9))


