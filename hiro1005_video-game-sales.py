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
vg = pd.read_csv('/kaggle/input/videogamesales/vgsales.csv', header=0)

vg
# Missing data

def Missing_table(df):

    # null_val = df.isnull().sum()

    null_val = df.isnull().sum()[df.isnull().sum()>0].sort_values(ascending=False)

    percent = 100 * null_val/len(df)

    na_col_list = df.isnull().sum()[df.isnull().sum()>0].index.tolist() # 欠損を含むカラムをリスト化

    list_type = df[na_col_list].dtypes.sort_values(ascending=False) #データ型

    Missing_table = pd.concat([null_val, percent, list_type], axis = 1)

    missing_table_len = Missing_table.rename(

    columns = {0:'Missing data', 1:'%', 2:'type'})

    return missing_table_len.sort_values(by=['Missing data'], ascending=False)



Missing_table(vg)
# Delete rows and columns containing missing values

vg.dropna()
Platform = pd.get_dummies(vg['Platform'])

Genre = pd.get_dummies(vg['Genre'])

Publisher = pd.get_dummies(vg['Publisher'])

del vg['Rank']



vg['Year'].value_counts()

vg["kijun"] = 1980.0

vg["release year"] = vg["Year"] - vg["kijun"]



del vg['Year']

del vg['kijun']

del vg['Name']



total_vg = pd.concat([vg, Platform, Genre, Publisher], axis=1)



del total_vg['Platform']

del total_vg['Genre']

del total_vg['Publisher']
def clean_dataset(df):

    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"

    df.dropna(inplace=True)

    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)

    return df[indices_to_keep].astype(np.float64)

  

clean_dataset(total_vg)
train_feature = total_vg.drop(columns='Global_Sales')

train_target = total_vg['Global_Sales']



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


