###############################
### マルチコの検出 VIFの計算
###############################
def fc_vif(dfxxx):
    from sklearn.linear_model import LinearRegression
    df_vif = dfxxx.drop(["price"],axis=1)
    for cname in df_vif.columns:
        y=df_vif[cname]
        X=df_vif.drop(cname, axis=1)
        regr = LinearRegression(fit_intercept=True)
        regr.fit(X, y)
        rsquared = regr.score(X,y)
        #print(cname,":" ,1/(1-np.power(rsquared,2)))
        if rsquared == 1:
            print(cname,X.columns[(regr.coef_> 0.5) | (regr.coef_ < -0.5)])
        
###############################
### 変数の選択 MAE:AIC
###############################
def fc_var(X, y):
    from sklearn import linear_model
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.feature_selection import SelectKBest,f_regression
    
    N = len(X)
    
    for k in range(1,len(X.columns)):
        skb = SelectKBest(f_regression,k=k).fit(X,y)
        sup = skb.get_support()
        X_selected = X.transpose()[sup].transpose()
        regr = linear_model.LinearRegression()
        model = regr.fit(X_selected,y)
        met = mean_absolute_error(model.predict(X_selected),y)
        aic = N*np.log((met**2).sum()/N) + 2*k
        print('k:',k,'MAE:',met,'AIC:',aic,X.columns[k])
        
# モジュールの読み込み
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.options.display.max_rows = 10 # 常に10行だけ表示
# データの読み込み
df000 = pd.read_csv("../input/kc_house_data.csv") 
display(df000.head())
df600 = df000.drop(['date'],axis=1) #dataの削除
#相関係数表示
df600.corr().style.background_gradient().format("{:.2f}") # わかりやすく色付け表示
# マルチコの検出 VIFの計算
rc = fc_vif(df600)
df700 = df600.drop(['sqft_basement','yr_renovated','zipcode','id'],axis=1)

for c in df700.columns: # 列の分だけ繰り返す
    if (c != "price") & (c != "date"): # ただし、price自身と日付は除く
        df000[[c,"price"]].plot(kind="scatter",x=c,y="price") # priceとの散布図
# マルチコの検出 VIFの計算（再度）→　
rc = fc_vif(df700)
df800 = df700
X = df800.drop(['price'],axis=1)
y = df800['price']

#V変数の選択
rc = fc_var(X, y)
from sklearn.linear_model import LinearRegression
regr = LinearRegression(fit_intercept=True).fit(X,y)
pd.Series(regr.coef_,index=X.columns).sort_values()\
  .plot(kind='barh',figsize=(6,8))
from sklearn.ensemble import RandomForestClassifier # ランダムフォレスト
from sklearn.model_selection import GridSearchCV,train_test_split # グリッドサーチ
from sklearn.metrics import confusion_matrix,classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

param_grid = [{'n_estimators':[10,20]}]
RFC = RandomForestClassifier()
cv = GridSearchCV(RFC,param_grid,verbose=0,cv=5)
# cv.fit(X_train,y_train) # 訓練してモデル作成
# テスト
# confusion_matrix(y_test,cv.predict(X_test))
# pd.Series(cv.best_estimator_.feature_importances_,index=df000.columns[24:]).sort_values().plot(kind='barh')
# データをリセット
df800 = df700
X = df800.drop(['price'],axis=1)
y = df800['price']
from sklearn.linear_model import Lasso                       # Lasso回帰用
from sklearn.metrics import mean_squared_error, mean_absolute_error #MAE,MAE用
from sklearn.model_selection import KFold                           # 交差検証用
from sklearn.model_selection import train_test_split                # データ分割用

#--------------------------------------------
# データの整形——説明変数xの各次元を正規化
#--------------------------------------------
from sklearn import preprocessing # 正規化用
sc = preprocessing.StandardScaler()
sc.fit(X)
X = sc.transform(X)
#--------------------------------------------

# 学習データとテストデータに分割
X_train,X_test,y_train,y_test = train_test_split(np.array(X),np.array(y),test_size=0.2,random_state=42)

kf = KFold(n_splits=5, random_state=1234, shuffle=True)

df_result = pd.DataFrame()
models = []

for i,(train_index, val_index) in enumerate(kf.split(X_train, y_train)):
    X_train_train, X_train_val = X_train[train_index], X_train[val_index]
    y_train_train, y_train_val = y_train[train_index], y_train[val_index]

    regr = Lasso(alpha=1.0) #  Lasso Regressorを適用
    regr.fit(X_train_train, y_train_train)
    models.append(regr)
    y_pred = regr.predict(X_train_val)
    df999 = pd.DataFrame({"y_val":y_train_val, "y_pred":y_pred})
    df_result = pd.concat([df_result, df999], axis=0)
    
# validation dataによる評価指標の算出
    y_val = df_result["y_val"]
    y_pred = df_result["y_pred"]
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    print("**** Training set score( {} ):  MSE={:.3f}  RMSE={:.3f}  MAE={:.3f}  Score={:.3f} ****".format(i,round(mse,3),round(np.sqrt(mse), 3),round(mae,3),regr.score(X_train, y_train)))

#--------------------------------------------
# 交差検証：テスト実施
#--------------------------------------------
z = 2 # 訓練で一番良かったものをセット
y_pred = models[z].predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("**** Test     set score( {} ):  MSE={:.3f}  RMSE={:.3f}  MAE={:.3f}  Score={:.3f} ****".format(z,round(mse,3),round(np.sqrt(mse), 3),round(mae,3),regr.score(X_test, y_test)))

# データをリセット
df800 = df700
X = df800.drop(['price'],axis=1)
y = df800['price']
from sklearn.metrics import mean_squared_error, mean_absolute_error # MAE,MAE用
from sklearn.model_selection import KFold                           # 交差検証用
from sklearn.model_selection import train_test_split                # データ分割用

from sklearn.ensemble import BaggingRegressor                       # バギング 用
from sklearn.tree import DecisionTreeRegressor

# 学習データとテストデータに分割
X_train,X_test,y_train,y_test = train_test_split(np.array(X),np.array(y),test_size=0.2,random_state=42)

kf = KFold(n_splits=4, random_state=1234, shuffle=True)

df_result = pd.DataFrame()
models = []

for i,(train_index, val_index) in enumerate(kf.split(X_train, y_train)):
    X_train_train, X_train_val = X_train[train_index], X_train[val_index]
    y_train_train, y_train_val = y_train[train_index], y_train[val_index]

    regr = BaggingRegressor(DecisionTreeRegressor(), n_estimators=100, max_samples=0.3) # バギング（決定木）
    
    regr.fit(X_train_train, y_train_train)
    models.append(regr)
    y_pred = regr.predict(X_train_val)
    df000 = pd.DataFrame({"y_val":y_train_val, "y_pred":y_pred})
    df_result = pd.concat([df_result, df000], axis=0)
    
# validation dataによる評価指標の算出
    y_val = df_result["y_val"]
    y_pred = df_result["y_pred"]
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    print("**** Training set score( {} ):  MSE={:.3f}  RMSE={:.3f}  MAE={:.3f}  Score={:.3f} ****".format(i,round(mse,3),round(np.sqrt(mse), 3),round(mae,3),regr.score(X_train, y_train)))
    
#--------------------------------------------
# 交差検証：テスト実施
#--------------------------------------------
z = 3 # 訓練で一番良かったものをセット
y_pred = models[z].predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("**** Test     set score( {} ):  MSE={:.3f}  RMSE={:.3f}  MAE={:.3f}  Score={:.3f} ****".format(z,round(mse,3),round(np.sqrt(mse), 3),round(mae,3),regr.score(X_test, y_test)))

