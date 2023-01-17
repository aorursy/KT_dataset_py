import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#ライブラリ読み込み
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PowerTransformer
#データ読み込み
train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
#trainのカラム数確認
len(train.columns)
#trainの先頭5行を確認
pd.options.display.max_columns = 81
train.head()
#目的変数となる家の価格のヒストグラムを表示
plt.figure(figsize=(20, 10))
sns.distplot(train['SalePrice'])
#物件の広さを合計した変数を作成
train["TotalSF"] = train["1stFlrSF"] + train["2ndFlrSF"] + train["TotalBsmtSF"]
test["TotalSF"] = test["1stFlrSF"] + test["2ndFlrSF"] + test["TotalBsmtSF"]

#物件の広さと物件価格の散布図を作成
plt.figure(figsize=(20, 10))
plt.scatter(train["TotalSF"],train["SalePrice"])
plt.xlabel("TotalSF")
plt.ylabel("SalePrice")
#外れ値を除外
train = train.drop(train[(train['TotalSF']>7500) & (train['SalePrice']<300000)].index)

plt.figure(figsize=(20, 10))
plt.scatter(train["TotalSF"],train["SalePrice"])
plt.xlabel("TotalSF")
plt.ylabel("SalePrice")
#築年数と物件価格の箱ひげ図を作成
data = pd.concat([train["YearBuilt"],train["SalePrice"]],axis=1)

plt.figure(figsize=(20, 10))
plt.xticks(rotation='90')
sns.boxplot(x="YearBuilt",y="SalePrice",data=data)
#外れ値を除外する
train = train.drop(train[(train['YearBuilt']<2000) & (train['SalePrice']>600000)].index)

data = pd.concat([train["YearBuilt"],train["SalePrice"]],axis=1)

plt.figure(figsize=(20, 10))
plt.xticks(rotation='90')
sns.boxplot(x="YearBuilt",y="SalePrice",data=data)
#家の材質・完成度と物件価格の散布図を作成
plt.figure(figsize=(20, 10))
plt.scatter(train["OverallQual"],train["SalePrice"])
plt.xlabel("OverallQual")
plt.ylabel("SalePrice")
#外れ値を除外する
train = train.drop(train[(train['OverallQual']<5) & (train['SalePrice']>200000)].index)
train = train.drop(train[(train['OverallQual']<10) & (train['SalePrice']>500000)].index)

plt.figure(figsize=(20, 10))
plt.scatter(train["OverallQual"],train["SalePrice"])
plt.xlabel("OverallQual")
plt.ylabel("SalePrice")
#学習データを目的変数とそれ以外に分ける
train_x = train.drop("SalePrice",axis=1)
train_y = train["SalePrice"]

#学習データとテストデータを統合
all_data = pd.concat([train_x,test],axis=0,sort=True)

#IDのカラムは不必要なので別の変数に格納
train_ID = train['Id']
test_ID = test['Id']

all_data.drop("Id", axis = 1, inplace = True)

#それぞれのデータのサイズを確認
print("train_x: "+str(train_x.shape))
print("train_y: "+str(train_y.shape))
print("test: "+str(test.shape))
print("all_data: "+str(all_data.shape))
#データの欠損値を確認
all_data_na = all_data.isnull().sum()[all_data.isnull().sum()>0].sort_values(ascending=False)

plt.figure(figsize=(20,10))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
# 欠損値があるカラムをリスト化
na_col_list = all_data.isnull().sum()[all_data.isnull().sum()>0].index.tolist()

#欠損があるカラムのデータ型を確認
all_data[na_col_list].dtypes.sort_values()
#試しにfloatの入ったカラムのデータを確認してみる
all_data['GarageArea'].value_counts()
#試しにobjectの入ったカラムのデータを確認してみる
all_data['GarageType'].value_counts()
#欠損値が存在するかつfloat型のリストを作成
float_list = all_data[na_col_list].dtypes[all_data[na_col_list].dtypes == "float64"].index.tolist()

#欠損値が存在するかつobject型のリストを作成
obj_list = all_data[na_col_list].dtypes[all_data[na_col_list].dtypes == "object"].index.tolist()

#float型の場合は欠損値を0で置換
all_data[float_list] = all_data[float_list].fillna(0)

#object型の場合は欠損値を"None"で置換
all_data[obj_list] = all_data[obj_list].fillna("None")
#欠損値の数を確認
all_data.isnull().values.sum()
#SalesPriceが正規分布でなかったため、分散を少なくするために対数変換を行う

#目的変数の対数log(x+1)をとる
train_y = np.log1p(train_y)

#分布を可視化
plt.figure(figsize=(20, 10))
sns.distplot(train_y)
#数値の説明変数のリストを作成
num_feats = all_data.dtypes[all_data.dtypes != "object" ].index

#各説明変数の歪度を計算
#歪度とは、分布が正規分布からどれだけ歪んでいるかを表す統計量のこと
skewed_feats = all_data[num_feats].apply(lambda x: x.skew()).sort_values(ascending = False)

#グラフ化
plt.figure(figsize=(20,10))
plt.xticks(rotation='90')
sns.barplot(x=skewed_feats.index, y=skewed_feats)
#歪度の絶対値が0.5より大きい変数だけに絞る
skewed_feats_over = skewed_feats[abs(skewed_feats) > 0.5].index

#各変数の最小値を表示
for i in skewed_feats_over:
    print(min(all_data[i]))
#Yeo-Johnson変換
pt = PowerTransformer()
pt.fit(all_data[skewed_feats_over])
#変換後のデータで各列を置換
all_data[skewed_feats_over] = pt.transform(all_data[skewed_feats_over])
#特徴量に1部屋あたりの面積を追加
#all_data["FeetPerRoom"] =  all_data["TotalSF"]/all_data["TotRmsAbvGrd"]

#建築した年とリフォームした年の合計
#all_data['YearBuiltAndRemod']=all_data['YearBuilt']+all_data['YearRemodAdd']

#バスルームの合計面積
#all_data['Total_Bathrooms'] = (all_data['FullBath'] + (0.5 * all_data['HalfBath']) +
#                               all_data['BsmtFullBath'] + (0.5 * all_data['BsmtHalfBath']))

#縁側の合計面積
#all_data['Total_porch_sf'] = (all_data['OpenPorchSF'] + all_data['3SsnPorch'] +
#                              all_data['EnclosedPorch'] + all_data['ScreenPorch'] +
#                              all_data['WoodDeckSF'])

#プールの有無
#all_data['haspool'] = all_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

#2階の有無
#all_data['has2ndfloor'] = all_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

#ガレージの有無
#all_data['hasgarage'] = all_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

#地下室の有無
#all_data['hasbsmt'] = all_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

#暖炉の有無
#all_data['hasfireplace'] = all_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
#各カラムのデータ型を確認
all_data.dtypes.value_counts()
#カテゴリ変数となっているカラムを取り出す
cal_list = all_data.dtypes[all_data.dtypes=="object"].index.tolist()
#学習データにおけるカテゴリ変数のデータ数を確認
train_x[cal_list].info()
#カテゴリ変数をget_dummiesによるone-hot-encodingを行う
#one-hot-encodingとは、各カテゴリ変数の二値（0,1）変数を作成する処理のこと
all_data = pd.get_dummies(all_data,columns=cal_list)

#サイズを確認
all_data.shape
#データセットの先頭5行を確認
all_data.head()
#学習データとテストデータに再分割
train_x = all_data.iloc[:train_x.shape[0],:].reset_index(drop=True)
test = all_data.iloc[train_x.shape[0]:,:].reset_index(drop=True)

#サイズを確認
print("train_x: "+str(train_x.shape))
print("test: "+str(test.shape))
#ライブラリのインポート
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV,Ridge,Lasso,ElasticNet
from sklearn.metrics import mean_squared_error, make_scorer
# Cross Validation用にデータを分割
X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.3, random_state=42)
# RMSEの評価関数を作成
scorer = make_scorer(mean_squared_error, greater_is_better = False)

def rmse_cv_train(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring = scorer, cv = 10))
    return(rmse)

def rmse_cv_test(model):
    rmse= np.sqrt(-cross_val_score(model, X_test, y_test, scoring = scorer, cv = 10))
    return(rmse)
#パラメータ調整付きのRidge regressionモデルを作成
ridge = RidgeCV(alphas = [0.01, 0.04, 0.08, 0.1, 0.4, 0.8, 1, 4, 8, 10, 40, 80])
ridge.fit(X_train, y_train)
alpha= ridge.alpha_
print("Best alpha :", alpha)

print("Try again for more precision with alphas centered around " + str(alpha))
ridge = RidgeCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 
                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4], 
                cv = 10)
ridge.fit(X_train, y_train)
alpha_ridge = ridge.alpha_
print("Best alpha :", alpha_ridge)

print("Ridge RMSE on Training set :", rmse_cv_train(ridge).mean())
print("Ridge RMSE on Test set :", rmse_cv_test(ridge).mean())

#Scoring
ridge_=Ridge(alpha=alpha_ridge)
ridge_.fit(X_train,y_train)
ridge_predict=ridge_.predict(X_test)
ridge_RMSE=np.sqrt(mean_squared_error(ridge_predict,y_test))
print("Ridge regression RMSE :",ridge_RMSE)
#Ridge regression予測と正解ラベルの散布図を作成
plt.figure(figsize=(20, 10))
plt.scatter(ridge_predict,y_test)
plt.xlabel("ridge_predict")
plt.ylabel("y_test")
#パラメータ調整付きのLasso regressionモデルを作成
lasso = LassoCV(alphas = [0.0001, 0.0004, 0.0008, 0.001, 0.004, 0.008, 0.01, 0.04, 0.08, 0.1, 
                          0.4, 0.8, 1], 
                max_iter = 50000, cv = 10)
lasso.fit(X_train, y_train)
alpha = lasso.alpha_
print("Best alpha :", alpha)

print("Try again for more precision with alphas centered around " + str(alpha))
lasso = LassoCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, 
                          alpha * .85, alpha * .9, alpha * .95, alpha, alpha * 1.05, 
                          alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, alpha * 1.35, 
                          alpha * 1.4], 
                max_iter = 50000, cv = 10)
lasso.fit(X_train, y_train)
alpha_lasso = lasso.alpha_
print("Best alpha :", alpha_lasso)

print("Lasso RMSE on Training set :", rmse_cv_train(lasso).mean())
print("Lasso RMSE on Test set :", rmse_cv_test(lasso).mean())

#Scoring
lasso_=Lasso(alpha=alpha_lasso)
lasso_.fit(X_train,y_train)
lasso_predict=lasso_.predict(X_test)
lasso_RMSE=np.sqrt(mean_squared_error(lasso_predict,y_test))
print("Lasso regression RMSE :",lasso_RMSE)
#Lasso regression予測と正解ラベルの散布図を作成
plt.figure(figsize=(20, 10))
plt.scatter(lasso_predict,y_test)
plt.xlabel("lasso_predict")
plt.ylabel("y_test")
#精度が最も良いモデルを選択
Final_model=Lasso(alpha=alpha_lasso)

Final_model.fit(train_x, train_y)
submission=Final_model.predict(test)

#対数変換した目的変数を元に戻す
submission=np.expm1(submission)

#Submission用のDataFrame作成
df_submission=pd.DataFrame({'Id':test_ID,'SalePrice':submission})

df_submission.to_csv('submission.csv',index=False)
df_submission
