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
train = pd.read_csv('/kaggle/input/prudential-life-insurance-assessment/train.csv.zip', header=0)
train
test = pd.read_csv('/kaggle/input/prudential-life-insurance-assessment/test.csv.zip', header=0)
test
submission = pd.read_csv('/kaggle/input/prudential-life-insurance-assessment/sample_submission.csv.zip', header=0)
submission
train_mid = train.copy()
train_mid['train_or_test'] = 'train'

test_mid = test.copy()
test_mid['train_or_test'] = 'test'

test_mid['Response'] = 9
alldata = pd.concat([train_mid, test_mid], sort=False, axis=0).reset_index(drop=True) 

print('The size of the train data:' + str(train.shape))
print('The size of the test data:' + str(test.shape))
print('The size of the submission data:' + str(submission.shape))
print('The size of the alldata data:' + str(alldata.shape))
def Missing_table(df):
    # null_val = df.isnull().sum()
    null_val = df.isnull().sum()[df.isnull().sum()>0].sort_values(ascending=False)
    percent = 100 * null_val/len(df)
    na_col_list = df.isnull().sum()[df.isnull().sum()>0].index.tolist() # 欠損を含むカラムをリスト化
    list_type = df[na_col_list].dtypes.sort_values(ascending=False) #データ型
    Missing_table = pd.concat([null_val, percent, list_type], axis = 1)
    missing_table_len = Missing_table.rename(
    columns = {0:'Missing Data', 1:'%', 2:'type'})
    return missing_table_len.sort_values(by=['Missing Data'], ascending=False)

Missing_table(alldata)
# Insert median to NaN
alldata.fillna(alldata.median(), inplace=True)

hot_product_info_2 = pd.get_dummies(alldata['Product_Info_2'])
del alldata['Product_Info_2']
alldata = pd.concat([alldata, hot_product_info_2], axis=1)

train = alldata.query('train_or_test == "train"')
test = alldata.query('train_or_test == "test"')

target_col = 'Response'
drop_col = ['Id', 'Response', 'train_or_test']

train_feature = train.drop(columns=drop_col)
train_target = train[target_col]
test_feature = test.drop(columns=drop_col)
submission_id = test['Id'].values

X_train, X_test, y_train, y_test = train_test_split(train_feature, train_target, test_size=0.2, random_state=0)
# RandomForest==============

rf = RandomForestRegressor(n_estimators=200, max_depth=5, max_features=0.5,  verbose=True, random_state=0, n_jobs=-1) # RandomForest のオブジェクトを用意する
rf.fit(X_train, y_train)
print('='*20)
print('RandomForestRegressor')
print(f'accuracy of train set: {rf.score(X_train, y_train)}')
print(f'accuracy of test set: {rf.score(X_test, y_test)}')

rf_prediction = rf.predict(test_feature)
rf_prediction
