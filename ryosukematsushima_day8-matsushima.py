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
from sklearn.preprocessing import MinMaxScaler
df_data = pd.read_csv("../input/kc_house_data.csv")
df_data["price"] = df_data["price"] / 10**6 #単位を100万ドルにしておく
print(df_data.columns)
print(df_data.columns)
display(df_data.head())
display(df_data.tail())
ex_ver = ["bedrooms","bathrooms","sqft_living","grade","sqft_above","sqft_living15"]
X, y = df_data.iloc[:,1:], df_data["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=0)
display(X_train)
display(X_test)
display(y_train)
display(y_test)
df_data = X_train
for ver in ex_ver:
    sns.jointplot(x=ver, y="price", data=df_data,kind = 'reg', size = 10)
    plt.show()
coord_df_data = pd.DataFrame([])
PCA_data = pd.DataFrame([])

for ver in ex_ver:
    X = np.array(df_data[[ver,"price"]])

    pca = PCA(n_components=2) #主成分分析用のオブジェクトをつくる。削減後の次元数を引数で指定する。2次元データなので、3以上にするとエラーになる
    pca.fit(X) #主成分分析の実行
    Y = np.dot((X), pca.components_.T)
    
    PCA_data[ver] = [pca.components_]


    dataframe_value = pd.DataFrame(Y)
    dataframe_value.columns = [ver + '_a', ver + '_b']
    
    X = np.array(dataframe_value[ver + '_a'])
    coord_df_data[ver + '_a'] = X
    X = np.array(dataframe_value[ver + '_b'])
    coord_df_data[ver + '_b'] = X
    
    sns.jointplot(x=ver+"_a", y=ver+"_b", data=dataframe_value,kind = 'reg', size = 10)
    plt.show()
for ver in ex_ver:
    
    n = 0
    while n <= 10:
        coord_df_data = coord_df_data[coord_df_data[ver + '_a'] != max(coord_df_data[ver + '_a'])]
        n += 1
        
    n = 0
    while n <= 10:
        coord_df_data = coord_df_data[coord_df_data[ver + '_a'] != min(coord_df_data[ver + '_a'])]
        n += 1
df_regr = pd.DataFrame([["b0_scale"],["b1_scale"],["b0_loc"],["b1_loc"]],columns=["coef"])

for ver in ex_ver:
    
    #df_data["normalized"] = (coord_df_data[ver + '_a'] - coord_df_data[ver + '_a'].min()) / (coord_df_data[ver + '_a'].max() - coord_df_data[ver + '_a'].min())
    
    X = np.array((coord_df_data[ver + '_a']) / (coord_df_data[ver + '_a'].max()))
    X = np.round(X,1)
    coord_df_data[ver + '_round_a'] = X #* (df_data[ver].max() - df_data[ver].min()) + df_data[ver].min()

    
    sns.jointplot(x=ver + '_round_a', y=ver + '_b', data=coord_df_data,kind = 'reg', size = 10)
    plt.show()
    
    
    x = []
    df_param_loc = []
    df_param_scale= []
    n = 0

    while n <= 1:
        n = np.round(n,1)
        df = coord_df_data[coord_df_data[ver + "_round_a"] == n]
        
        param = norm.fit(df[ver + '_b'])

        
        t = n * coord_df_data[ver + '_a'].max()
        
        r = len(df)/100
        r = np.round(r,0)
        r = int(r)
        
        if param[0] != np.nan:

            for i in range(0,r):
                x += [t]
                df_param_loc += [param[0]]
                df_param_scale += [param[1]]

        n += 0.1
        
    x = np.array(x)
    X = x.reshape(-1,1)
    y = df_param_scale

    regr = linear_model.LinearRegression(fit_intercept=True)
    regr.fit(X, y)

    b0_scale = regr.intercept_
    b1_scale = regr.coef_

    plt.plot(x, y, 'o')
    plt.plot(x, b0_scale+b1_scale*x)
    plt.show()
    
    
    x = np.array(x)
    X = x.reshape(-1,1)
    y = df_param_loc

    regr = linear_model.LinearRegression(fit_intercept=True)
    regr.fit(X, y)

    b0_loc = regr.intercept_
    b1_loc = regr.coef_

    plt.plot(x, y, 'o')
    plt.plot(x, b0_loc+b1_loc*x)
    plt.show()
    
    
    df_regr[ver + "_regr"] = [b0_scale, b1_scale,b0_loc,b1_loc]

   
test_dff = X_test
df_result = pd.DataFrame()

for index,row in test_dff.iterrows():
    
    test_df = row
    
    df_test = pd.DataFrame({"price":np.linspace(0,10,10**6)})
    df_norm = pd.DataFrame({"price":np.linspace(0,10,10**6)})

    for ver in ex_ver:
        df_test[ver] = test_df[ver]
        

    for ver in ex_ver:
        
        pca_components_ = PCA_data[ver]
        
        X = np.array(df_test[[ver,"price"]])
        Y = np.dot((X), pca_components_[0].T)

        dataframe_value = pd.DataFrame(Y)
        dataframe_value.columns = ['a', 'b']
        

        x = np.array(dataframe_value["a"])
        y = np.array(dataframe_value["b"])

        b0 = df_regr.at[0,ver + "_regr"]
        b1 = df_regr.at[1,ver + "_regr"]
        
        sig = b0 + b1 * x
        
        
        b0 = df_regr.at[2,ver + "_regr"]
        b1 = df_regr.at[3,ver + "_regr"]
        
        myu = b0 + b1 * x
        

        norm = (np.exp(-(y - myu)**2/(2*sig**2))) / np.sqrt(2*math.pi*sig**2)

        df_norm[ver] = norm

    X = np.array(df_norm[ex_ver])
    
    df_norm["sum_norm"] = X.sum(axis=1)
    df = df_norm[df_norm['sum_norm'] == max(df_norm['sum_norm'])]

    price = np.array(df["price"])
    price = price[0]
    
    y_pred = price
    y_test = test_df["price"]
    
    
    print(y_test)
    print(y_pred)
    
    
    df = pd.DataFrame({"y_test":[y_test], "y_pred":[y_pred]})
    df_result = pd.concat([df_result, df], axis=0)
# 評価指標の算出
y_test = df_result["y_test"]
y_pred = df_result["y_pred"]
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("MSE=%s"%round(mse,3) )
print("RMSE=%s"%round(np.sqrt(mse), 3) )
