# Data file
import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Import
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns 

from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor #  KneighborsRegressorではない
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost.sklearn import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
alldata = pd.read_csv('/kaggle/input/avocado-prices/avocado.csv')
alldata
# Check 'year' coloum
alldata['year'].value_counts()

# Convert each 'year' column to Int
alldata['year'] = alldata['year'].replace("2015",0).replace("2016",1).replace("2017",2).replace("2018",3)

# Check 'region' coloum
alldata['region'].value_counts()

# Convert each 'region' column to Int
alldata['region'] = alldata['region'].replace("Syracuse",0).replace("Seattle",1).replace("BuffaloRochester",2).replace("Orlando",3).replace("LasVegas",4).replace("Tampa",5).replace("Columbus",6).replace("Denver",7).replace("TotalUS",8).replace("RichmondNorfolk",9).replace("RaleighGreensboro",10).replace("NorthernNewEngland",11).replace("Northeast",12).replace("Boston",13).replace("MiamiFtLauderdale",14).replace("Midsouth",15).replace("Charlotte",16).replace("Nashville",17).replace("LosAngeles",18).replace("Portland",18).replace("Plains",19).replace("SouthCentral",20).replace("Philadelphia",21).replace("California",22).replace("CincinnatiDayton",23).replace("GrandRapids",24).replace("Louisville",25).replace("Spokane",26).replace("StLouis",27).replace("Detroit",28).replace("HartfordSpringfield",29).replace("Atlanta",30).replace("Indianapolis",31).replace("West",32).replace("SanDiego",33).replace("Houston",34).replace("GreatLakes",35).replace("Pittsburgh",36).replace("HarrisburgScranton",37).replace("Albany",38).replace("DallasFtWorth",39).replace("Roanoke",40).replace("Boise",41).replace("Chicago",42).replace("Sacramento",43).replace("NewYork",44).replace("Jacksonville",45).replace("Southeast",46).replace("PhoenixTucson",47).replace("NewOrleansMobile",48).replace("SanFrancisco",49).replace("SouthCarolina",50).replace("BaltimoreWashington",51).replace("WestTexNewMexico",52)

# Check 'type' coloum
alldata['type'].value_counts()

# Convert each 'type' column to Int
alldata['type'] = alldata['type'].replace("conventional",0).replace("organic",1)

# Delete anyway
del alldata['XLarge Bags']
del alldata['Small Bags']
del alldata['Large Bags']
del alldata['4046']
del alldata['4225']
del alldata['4770']
del alldata['Total Bags']
del alldata['Unnamed: 0']

alldata["Open Date"] = pd.to_datetime(alldata["Date"])
alldata["Year"] = alldata["Open Date"].apply(lambda x:x.year)
alldata["Month"] = alldata["Open Date"].apply(lambda x:x.month)
alldata["Day"] = alldata["Open Date"].apply(lambda x:x.day)
alldata["kijun"] = "2015-04-27"
alldata["kijun"] = pd.to_datetime(alldata["kijun"])
alldata["BusinessPeriod"] = (alldata["kijun"] - alldata["Open Date"]).apply(lambda x: x.days)

alldata = alldata.drop('Open Date', axis=1)
alldata = alldata.drop('kijun', axis=1)

del alldata['Date']

alldata
train_feature = alldata.drop(columns='AveragePrice')
train_target = alldata['AveragePrice']
# Searching for Effective feature（SelectKBest）
from sklearn.feature_selection import SelectKBest, f_regression
# Setting as search for especially 4 features
selector = SelectKBest(score_func=f_regression, k=4) 
selector.fit(train_feature, train_target)
mask_SelectKBest = selector.get_support()

# Searching for Effective feature（SelectPercentile）
from sklearn.feature_selection import SelectPercentile, f_regression
# Setting as search for 40% features
selector = SelectPercentile(score_func=f_regression, percentile=40) 
selector.fit(train_feature, train_target)
mask_SelectPercentile = selector.get_support()

# Searching for Effective feature（SelectFromModel）
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
# Setting as search for above median
selector = SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42), threshold="median")    
selector.fit(train_feature, train_target)
mask_SelectFromModel = selector.get_support()

# Searching for Effective feature（RFE：n_features_to_select）
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
# Setting as search for only 2 features
selector = RFE(RandomForestRegressor(n_estimators=100, random_state=42), n_features_to_select=2)
selector.fit(train_feature, train_target)
mask_RFE = selector.get_support()

print(train_feature.columns)
print(mask_SelectKBest)
print(mask_SelectPercentile)
print(mask_SelectFromModel)
print(mask_RFE)
X_train, X_test, y_train, y_test = train_test_split(train_feature, train_target, test_size=0.2, random_state=0, shuffle=True)
import warnings
warnings.filterwarnings('ignore')

# RandomForest==============

rf = RandomForestRegressor(n_estimators=200, max_depth=5, max_features=0.5,  verbose=True, random_state=0, n_jobs=-1) # RandomForest のオブジェクトを用意する
rf.fit(X_train, y_train)
print('='*20)
print('RandomForestRegressor')
print(f'accuracy of train set: {rf.score(X_train, y_train)}')
print(f'accuracy of test set: {rf.score(X_test, y_test)}')
# SVR（Support Vector Regression）==============

svr = SVR(verbose=True)
svr.fit(X_train, y_train)
print('='*20)
print('SVR')
print(f'accuracy of train set: {svr.score(X_train, y_train)}')
print(f'accuracy of test set: {svr.score(X_test, y_test)}')


# LinearSVR==============

lsvr = LinearSVR(verbose=True, random_state=0)
lsvr.fit(X_train, y_train)
print('='*20)
print('LinearSVR')
print(f'accuracy of train set: {lsvr.score(X_train, y_train)}')
print(f'accuracy of test set: {lsvr.score(X_test, y_test)}')


# SGDRegressor==============

sgd = SGDRegressor(verbose=0, random_state=0)
sgd.fit(X_train, y_train)
print('='*20)
print('SGDRegressor')
print(f'accuracy of train set: {sgd.score(X_train, y_train)}')
print(f'accuracy of test set: {sgd.score(X_test, y_test)}')


# k-近傍法（k-NN）==============

knn = KNeighborsRegressor()
knn.fit(X_train, y_train)
print('='*20)
print('KNeighborsRegressor')
print(f'accuracy of train set: {knn.score(X_train, y_train)}')
print(f'accuracy of test set: {knn.score(X_test, y_test)}')


# 決定木==============

decisiontree = DecisionTreeRegressor(max_depth=3, random_state=0)
decisiontree.fit(X_train, y_train)
print('='*20)
print('DecisionTreeRegressor')
print(f'accuracy of train set: {decisiontree.score(X_train, y_train)}')
print(f'accuracy of test set: {decisiontree.score(X_test, y_test)}')


# LinearRegression (線形回帰)==============

lr = LinearRegression()
lr.fit(X_train, y_train)
print('='*20)
print('LinearRegression')
print(f'accuracy of train set: {lr.score(X_train, y_train)}')
print(f'accuracy of test set: {lr.score(X_test, y_test)}')
# 回帰係数とは、回帰分析において座標平面上で回帰式で表される直線の傾き。 原因となる変数x（説明変数）と結果となる変数y（目的変数）の平均的な関係を、一次式y＝ax＋bで表したときの、係数aを指す。
print("回帰係数:",lr.coef_)
print("切片:",lr.intercept_)


# RANSACRegressor==============

# ロバスト回帰を行う（自然界のデータにはたくさんノイズがある。ノイズなどの外れ値があると、法則性をうまく見つけられないことがある。そんなノイズをうまく無視してモデルを学習させるのがRANSAC）
#線形モデルをRANSACでラッピング　（外れ値の影響を抑える）
from sklearn.linear_model import RANSACRegressor
 
ransac=RANSACRegressor(lr,#基本モデルは、LinearRegressionを流用
                       max_trials=100,#イテレーションの最大数100
                       min_samples=50,#ランダムに選択されるサンプル数を最低50に設定
                       loss="absolute_loss",#学習直線に対するサンプル店の縦の距離の絶対数を計算
                       residual_threshold=5.0,#学習直線に対する縦の距離が5以内のサンプルだけを正常値
                       random_state=0)
 
ransac.fit(X_train, y_train)
print('='*20)
print('RANSACRegressor')
print(f'accuracy of train set: {lr.score(X_train, y_train)}')
print(f'accuracy of test set: {lr.score(X_test, y_test)}')
print("RANSAC回帰係数:",ransac.estimator_.coef_[0])
print("RANSAC切片:",ransac.estimator_.intercept_)


# RIDGE回帰==============

ridge = Ridge(random_state=0)
ridge.fit(X_train, y_train)
print('='*20)
print('Ridge')
print(f'accuracy of train set: {ridge.score(X_train, y_train)}')
print(f'accuracy of test set: {ridge.score(X_test, y_test)}')


# LASSO回帰==============

lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005], verbose=True, random_state=0)
lasso.fit(X_train, y_train)
print('='*20)
print('LassoCV')
print(f'accuracy of train set: {lasso.score(X_train, y_train)}')
print(f'accuracy of test set: {lasso.score(X_test, y_test)}')


# ElasticNet==============

en = ElasticNet(random_state=0)
en.fit(X_train, y_train)
print('='*20)
print('ElasticNet')
print(f'accuracy of train set: {en.score(X_train, y_train)}')
print(f'accuracy of test set: {en.score(X_test, y_test)}')