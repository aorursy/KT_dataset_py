#ライブラリの読み込み
import pandas as pd
from IPython.display import display
from dateutil.parser import parse
import matplotlib.pyplot as plt
import itertools
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
#データの読み込み
df_data = pd.read_csv("../input/kc_house_data.csv")

print("")
print("データセットの頭出し")
display(df_data.head())
# date列の変換（日付の形に変更） (説明変数として使わないため、実行しない。)
#df_data["date"] = [ parse(i[:-7]).date() for i in df_data["date"]]
#display(df_data.head())
# 欠損値のデータが含まれているかどうか確認する
pd.DataFrame(df_data.isnull().sum(), columns=["num of missing"])
#不要な列の削除
df_data_main = df_data.drop(["id","date","zipcode"], axis=1)
df1 = df_data_main.iloc[:,:9]
display(df1.head())
df2 = df_data_main.iloc[:,[0]+list(range(9,18))]
display(df2.head())
# describe（記述統計量の算出）
df_data.describe()
# 散布図行列
pd.plotting.scatter_matrix(df1,figsize=(10,10))
plt.show()
pd.plotting.scatter_matrix(df2,figsize=(10,10))
plt.show()
li_combi = list(itertools.combinations(df_data_main.columns[0:], 2))
for X,Y in li_combi:
    if X=='price':
        print("X=%s"%X,"Y=%s"%Y)
        df_data_main.plot(kind="scatter",x=X,y=Y,alpha=0.7,s=10)#散布図の作成
        plt.xlabel(X)
        plt.ylabel(Y)
        plt.tight_layout()
        plt.show()#グラフをここで描画させるための行
df_data_main.corr()
for col in df_data_main.columns:
    print(col)
    df_data_main[col].hist()
    plt.xlabel(col)
    plt.ylabel("num")
    plt.show()
for col in df_data_main.columns:
    print(col)
    df_data_main.boxplot(column=col)
    plt.xlabel(col)
    plt.ylabel("num")
    plt.show()
#異常値を除外
def drop_outlier(df):
  for i, col in df.iteritems():
    #四分位数
    q1 = col.describe()['25%']
    q3 = col.describe()['75%']
    iqr = q3 - q1 #四分位範囲

    #外れ値の基準点
    outlier_min = q1 - (iqr) * 1.5
    outlier_max = q3 + (iqr) * 1.5

    #範囲から外れている値を除く
    col[col < outlier_min] = None
    col[col > outlier_max] = None

if __name__ == '__main__':
    drop_outlier(df_data_main)
    
    for col in df_data_main.columns:
        print(col)
        df_data_main.boxplot(column=col)
        plt.xlabel(col)
        plt.ylabel("num")
        plt.show()
#どの程度、異常値を除外したのか、確認を行う。
df_data_main.isnull().any(axis=0)
#全体の平均で、欠損値を埋める。
for col in df_data_main.columns:
    mean_all = df_data_main[col].mean()
    df_data_main[col] = df_data_main[col].fillna(mean_all)
    #df_data_main.loc[parse(col):parse(col + 'after')]
df_data_main
#分析に用いるデータのみを取得。
X_var = ["sqft_living","grade","sqft_above","bathrooms"]
y_var = ["price"]
df = df_data[y_var+ X_var]

# scikit learnの入力形式に変換する
X = df[X_var].as_matrix()
y = df[y_var].values

# 学習
regr = linear_model.LinearRegression(fit_intercept=True)
regr.fit(X, y)

print("決定係数=",regr.score(X,y))
from sklearn.model_selection import train_test_split
# 利用できるデータのうち、学習用を8割、テスト用を2割にする
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regr = linear_model.LinearRegression(fit_intercept=True)
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("MSE=%s"%round(mse,3) )
print("RMSE=%s"%round(np.sqrt(mse), 3) )
print("MAE=%s"%round(mae,3) )
# 標準化
stdsc = StandardScaler()
X_train_transform = stdsc.fit_transform(X_train)
X_test_transform = stdsc.transform(X_test)

print(X_train_transform)
print(X_test_transform)

# SVMの実行
clf = SVR(C=5, kernel="linear")
clf.fit(X_train_transform, y_train)

# 未知のデータを識別する
clf.predict(X_test_transform)
