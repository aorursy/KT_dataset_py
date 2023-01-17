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

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost.sklearn import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import VotingRegressor

salary = pd.read_csv('/kaggle/input/placement-data-full-class/Placement_data_full_class.csv', header=0)
salary
salary.dropna()
del salary['sl_no']

salary['gender'] = salary['gender'].replace("M",0).replace("F",1)

salary['ssc_b'].value_counts()

salary['ssc_b'] = salary['ssc_b'].replace("Central",0).replace("Others",1)
salary['hsc_b'].value_counts()

salary['hsc_b'] = salary['hsc_b'].replace("Central",0).replace("Others",1)

salary['degree_t'].value_counts()

salary['degree_t'] = salary['degree_t'].replace("Comm&Mgmt",0).replace("Sci&Tech",1).replace("Others",2)

salary['workex'].value_counts()
salary['workex'] = salary['workex'].replace("No",0).replace("Yes",1)

salary['specialisation'].value_counts()
salary['specialisation'] = salary['specialisation'].replace("Mkt&Fin",0).replace("Mkt&HR",1)

salary['status'].value_counts()
salary['status'] = salary['status'].replace("Placed",0).replace("Not Placed",1)

salary['hsc_s'].value_counts()
salary['hsc_s'] = salary['hsc_s'].replace("Commerce",0).replace("Science",1).replace("Arts",2)


# Missing value
def kesson_table(df): 
        null_val = df.isnull().sum()
        percent = 100 * df.isnull().sum()/len(df)
        kesson_table = pd.concat([null_val, percent], axis=1)
        kesson_table_ren_columns = kesson_table.rename(
        columns = {0 : 'Number of Missing value', 1 : '%'})
        return kesson_table_ren_columns
 
kesson_table(salary)
# Insert median to 'Fare'
salary.salary.fillna(salary.salary.median(), inplace=True)

train_feature = salary.drop(columns='salary')
train_target = salary['salary']
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