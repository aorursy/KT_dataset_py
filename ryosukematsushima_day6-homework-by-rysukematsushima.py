%matplotlib inline 
#グラフをnotebook内に描画させるための設定
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.decomposition import PCA #主成分分析用ライブラリ
from sklearn.metrics import mean_squared_error, mean_absolute_error
from IPython.display import display
import seaborn as sns
from scipy.stats import norm
from sklearn.linear_model import LinearRegression #線形回帰のライブラリ
import math
from sklearn.model_selection import train_test_split
df_data = pd.read_csv("../input/kc_house_data.csv")
df_data["price"] = df_data["price"] / 10**6 #単位を100万ドルにしておく
print(df_data.columns)
display(df_data.head())
display(df_data.tail())
print("")
print("カラム名の確認")
print(df_data.columns)

print("")
print("データセットの頭出し")
display(df_data.head())

print("")
print("目的変数となるクラスラベルの内訳")
display(df_data.groupby(["price"])["price"].count())

print("")
print("説明変数の要約")
display(df_data.iloc[:,2:].describe())
X, y = df_data.iloc[:,1:], df_data["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
display(X_train)
display(X_test)
display(y_train)
display(y_test)
df_data = X_train
ex_ver = ["bedrooms","bathrooms","sqft_living","grade","sqft_above","sqft_living15"]
for ver in ex_ver:
    sns.jointplot(x=ver, y="price", data=df_data,kind = 'reg', size = 10)
    plt.show()
coord_df_data = pd.DataFrame([])

for ver in ex_ver:
    X = np.array(df_data[[ver,"price"]])

    pca = PCA(n_components=2) #主成分分析用のオブジェクトをつくる。削減後の次元数を引数で指定する。2次元データなので、3以上にするとエラーになる
    pca.fit(X) #主成分分析の実行
    #Y = np.dot((X - pca.mean_), pca.components_.T)
    Y = np.dot((X), pca.components_.T)

    dataframe_value = pd.DataFrame(Y)
    dataframe_value.columns = [ver + '_a', ver + '_b']
    
    X = np.array(dataframe_value[ver + '_a'])
    coord_df_data[ver + '_a'] = X
    X = np.array(dataframe_value[ver + '_b'])
    coord_df_data[ver + '_b'] = X
    
    sns.jointplot(x=ver+"_a", y=ver+"_b", data=dataframe_value,kind = 'reg', size = 10)
    plt.show()
ver = ex_ver[5]
n = -3

X = np.array(coord_df_data[ver + '_a'])
X = np.round(X,n)
coord_df_data[ver + '_round_a'] = X

#opp = coord_df_data.assign(round_A=lambda coord_df_data: coord_df_data.bedrooms_a.round())
sns.jointplot(x=ver + '_round_a', y=ver + '_b', data=coord_df_data,kind = 'reg', size = 10)
plt.show()
for ver in ex_ver:
    sns.jointplot(x=ver+"_round_a", y=ver+"_b", data=coord_df_data,kind = 'reg', size = 10)
    plt.show()
ver = ex_ver[1]

df = coord_df_data[coord_df_data[ver + "_round_a"] == 5]
param = norm.fit(coord_df_data[ver + '_b'])
print(param)
x = np.linspace(-2,3,100)
pdf_fitted = norm.pdf(x,loc=param[0], scale=param[1])
pdf = norm.pdf(x)
plt.figure
plt.title('Normal distribution')
plt.plot(x, pdf_fitted, 'r-')
plt.hist(coord_df_data[ver + '_b'], bins=15, normed=1, alpha=0.3)
plt.show()
df_regr = pd.DataFrame([["b0"],["b1"]],columns=["coef"])
ver = ex_ver[4]

sns.jointplot(x=ver+"_round_a", y=ver+"_b", data=coord_df_data,kind = 'reg', size = 10)
plt.show()

order = 10**3

n = 2

n = n * order

x = []
df_param_loc = []
df_param_scale= []

while n != 7 * order:
    df = coord_df_data[coord_df_data[ver + "_round_a"] == n]
    param = norm.fit(df[ver + '_b'])
    #print(param)
    x += [n]
    df_param_loc += [param[0]]
    df_param_scale+= [param[1]]
    
    n += 1 * order
    
#plt.scatter(x,df_param_loc)
plt.scatter(x,df_param_scale)
x = np.array(x)
X = x.reshape(-1,1)
y = df_param_scale

regr = linear_model.LinearRegression(fit_intercept=True)
regr.fit(X, y)

b0 = regr.intercept_
b1 = regr.coef_

plt.plot(x, y, 'o')
plt.plot(x, b0+b1*x)
plt.show()
df_regr[ver + "_regr"] = [regr.intercept_, regr.coef_]
test_dff = X_test
df_result = pd.DataFrame()

for index,row in test_dff.iterrows():
    
    test_df = row
    
    df_test = pd.DataFrame({"price":np.linspace(0,10,10**6)})
    df_norm = pd.DataFrame({"price":np.linspace(0,10,10**6)})

    for ver in ex_ver:
        df_test[ver] = test_df[ver]
        

    for ver in ex_ver:
        X = np.array(df_data[[ver,"price"]])

        pca = PCA(n_components=2) #主成分分析用のオブジェクトをつくる。削減後の次元数を引数で指定する。2次元データなので、3以上にするとエラーになる
        pca.fit(X) #主成分分析の実行

        X = np.array(df_test[[ver,"price"]])
        Y = np.dot((X), pca.components_.T)

        dataframe_value = pd.DataFrame(Y)
        dataframe_value.columns = ['a', 'b']
        

        x = np.array(dataframe_value["a"])
        y = np.array(dataframe_value["b"])

        b0 = df_regr.at[0,ver + "_regr"]
        b1 = df_regr.at[1,ver + "_regr"]
        
        
        sig = b0 + b1 * x

        norm = (np.exp(-(y)**2/(2*sig**2))) / np.sqrt(2*math.pi*sig**2)

        df_norm[ver] = norm

    X = np.array(df_norm[ex_ver])
    
    df_norm["sum_norm"] = X.sum(axis=1)
    df = df_norm[df_norm['sum_norm'] == max(df_norm['sum_norm'])]

    price = np.array(df["price"])
    price = price[0]
    
    y_pred = price
    y_test = test_df["price"]
    
    
    df = pd.DataFrame({"y_test":[y_test], "y_pred":[y_pred]})
    df_result = pd.concat([df_result, df], axis=0)
# 評価指標の算出
y_test = df_result["y_test"]
y_pred = df_result["y_pred"]
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("MSE=%s"%round(mse,3) )
print("RMSE=%s"%round(np.sqrt(mse), 3) )