# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

# print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from scipy import stats

from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体

mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

%matplotlib inline

sns.set_style('darkgrid')

sns.set_palette('bone')

from tqdm import tqdm

warnings.filterwarnings('ignore')

import gc, sys

gc.enable()

INPUT_DIR = "../input/"
def fillInf(df, val):  # 删除inf值

    numcols = df.select_dtypes(include='number').columns

    cols = numcols[numcols != 'winPlacePerc']

    df[df == np.Inf] = np.NaN

    df[df == np.NINF] = np.NaN

    for c in cols: df[c].fillna(val, inplace=True)
def feature_engineering(is_train=True):

    if is_train: 

        print("processing train.csv")

        df = pd.read_csv(INPUT_DIR + 'train_V2.csv')



        df = df[df['maxPlace'] > 1]

    else:

        print("processing test.csv")

        df = pd.read_csv(INPUT_DIR + 'test_V2.csv')

    df.dropna(inplace=True)

    df['totalDistance'] = df['rideDistance'] + df["walkDistance"] + df["swimDistance"]

    match = df.groupby('matchId')

    df['killPlacePerc'] = match['kills'].rank(pct=True).values

    df['walkDistancePerc'] = match['walkDistance'].rank(pct=True).values

    

    df['_totalDistance'] = df['rideDistance'] + df['walkDistance'] + df['swimDistance']

    df['zombi'] = ((df['_totalDistance'] == 0) | (df['kills'] == 0)

                     | (df['weaponsAcquired'] == 0)

                     | (df['matchType'].str.contains('solo'))).astype(int)

    df['cheater'] = ((df['kills'] / df['_totalDistance'] >= 1)

                       | (df['kills'] > 30) | (df['roadKills'] > 10)).astype(int)

    pd.concat([df['zombi'].value_counts(), df['cheater'].value_counts()], axis=1).T

    df['_healthAndBoosts'] = df['heals'] + df['boosts']

    df['_killDamage'] = df['kills'] * 100 + df['damageDealt']

    # all_data['_headshotKillRate'] = all_data['headshotKills'] / all_data['kills']

    df['_killPlaceOverMaxPlace'] = df['killPlace'] / df['maxPlace']

    df['_killsOverWalkDistance'] = df['kills'] / df['walkDistance']

    # all_data['_killsOverDistance'] = all_data['kills'] / all_data['_totalDistance']

    df['_walkDistancePerSec'] = df['walkDistance'] / df['matchDuration']

    # suicide: solo and teamKills > 0

    # all_data['_suicide'] = ((all_data['players'] == 1) & (all_data['teamKills'] > 0)).astype(int)

    fillInf(df, 0)

    mapper = lambda x: 'solo' if ('solo' in x) else 'duo' if ('duo' in x) or ('crash' in x) else 'squad'

    # mapper = lambda x: 'solo' if ('solo' in x) else 'team'

    df['matchType'] = df['matchType'].map(mapper)

    df['matchType'] = df['matchType'].map(mapper)

    # 设置哑变量

    a = pd.get_dummies(df['matchType'], prefix='matchType')

    df = pd.concat([df, a], axis=1)

    df.drop(['headshotKills','teamKills','roadKills','vehicleDestroys'], axis=1, inplace=True)

    df.drop(['rideDistance','swimDistance','matchDuration'], axis=1, inplace=True)

    df.drop(['rankPoints','killPoints','winPoints'], axis=1, inplace=True)

    df.drop(['matchType'], axis=1, inplace=True)

    

    print("remove some columns")

    target = 'winPlacePerc'

    features = list(df.columns)

    features.remove("Id")

    features.remove("matchId")

    features.remove("groupId")

    

    y = None

    

    print("get target")

    if is_train: 

        y = np.array(df.groupby(['matchId','groupId'])[target].agg('mean'), dtype=np.float64)

        features.remove(target)



    print("get group mean feature")

    agg = df.groupby(['matchId','groupId'])[features].agg('mean')

    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()

    

    if is_train: df_out = agg.reset_index()[['matchId','groupId']]

    else: df_out = df[['matchId','groupId']]



    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])

    df_out = df_out.merge(agg_rank, suffixes=["_mean", "_mean_rank"], how='left', on=['matchId', 'groupId'])

    del agg, agg_rank

    gc.collect()

    print("get group max feature")

    agg = df.groupby(['matchId','groupId'])[features].agg('max')

    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()

    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])

    df_out = df_out.merge(agg_rank, suffixes=["_max", "_max_rank"], how='left', on=['matchId', 'groupId'])

    del agg, agg_rank

    gc.collect()

    print("get group min feature")

    agg = df.groupby(['matchId','groupId'])[features].agg('min')

    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()

    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])

    df_out = df_out.merge(agg_rank, suffixes=["_min", "_min_rank"], how='left', on=['matchId', 'groupId'])

    del agg, agg_rank

    gc.collect()

    print("get group size feature")

    agg = df.groupby(['matchId','groupId']).size().reset_index(name='group_size')

    df_out = df_out.merge(agg, how='left', on=['matchId', 'groupId'])



    print("get match mean feature")

    agg = df.groupby(['matchId'])[features].agg('mean').reset_index()

    df_out = df_out.merge(agg, suffixes=["", "_match_mean"], how='left', on=['matchId'])

    del agg

    gc.collect()

    print("get match size feature")

    agg = df.groupby(['matchId']).size().reset_index(name='match_size')

    df_out = df_out.merge(agg, how='left', on=['matchId'])

    gc.collect()

    df_out.drop(["matchId", "groupId"], axis=1, inplace=True)



    X = np.array(df_out, dtype=np.float64)

    

    feature_names = list(df_out.columns)



    del df, df_out, agg

    gc.collect()

    return X, y, feature_names
# transform feature

from sklearn.preprocessing import MinMaxScaler

x_train, y, feature_names = feature_engineering(True)

scaler = MinMaxScaler(feature_range=(-1, 1), copy=False).fit(x_train)

scaler.transform(x_train)
x_prediction, _, _ = feature_engineering(False)

scaler = MinMaxScaler(feature_range=(-1, 1), copy=False).fit(x_prediction)

scaler.transform(x_prediction)
# submit_elasticNet.to_csv(r'sample_submission_elasticNet.csv', index=False)
from sklearn.model_selection import train_test_split

X_train,X_test, y_train, y_test =train_test_split(x_train,y,test_size=0.3, random_state=0)
%%time

from sklearn.linear_model import LinearRegression

linreg = LinearRegression()

linreg.fit(X_train, y_train)

print (linreg.intercept_)

print (linreg.coef_)
# %%time

# result_test = linreg.predict(X_test)
# from sklearn.metrics import mean_absolute_error

# mean_absolute_error(y_test, result_test)
# from sklearn.metrics import mean_squared_error

# mean_squared_error(y_test, result_test)
%%time

result = linreg.predict(x_prediction)
%%time

test_data = pd.read_csv(INPUT_DIR+'test_V2.csv')

print("fix winPlacePerc")

for i in range(len(test_data)):

    winPlacePerc = result[i]

    maxPlace = int(test_data.iloc[i]['maxPlace'])

    if maxPlace == 0:

        winPlacePerc = 0.0

    elif maxPlace == 1:

        winPlacePerc = 1.0

    else:

        gap = 1.0 / (maxPlace - 1)

        winPlacePerc = round(winPlacePerc / gap) * gap

    

    if winPlacePerc < 0: winPlacePerc = 0.0

    if winPlacePerc > 1: winPlacePerc = 1.0    

    result[i] = winPlacePerc
f3=open(INPUT_DIR+'sample_submission_V2.csv')

submit=pd.read_csv(f3)

sample_result = pd.DataFrame(result,columns = ['winPlacePerc'])

submit['winPlacePerc'] = sample_result

submit.to_csv(r'sample_submission_lineregression.csv', index=False)

del f3,result,submit

gc.collect()
# %%time

# from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV   # Lasso回归,LassoCV交叉验证实现alpha的选取，LassoLarsCV基于最小角回归交叉验证实现alpha的选取

# # ========Lasso回归========

# # model = Lasso(alpha=0.01)  # 调节alpha可以实现对拟合的程度

# # model = LassoCV()  # LassoCV自动调节alpha可以实现选择最佳的alpha。

# alpha = np.logspace(-3,2,10)

# model = LassoCV(alphas=alpha,cv=5)  # LassoLarsCV自动调节alpha可以实现选择最佳的alpha

# model.fit(x_train, y)   # 线性回归建模

# print('系数矩阵:\n',model.coef_)

# print('线性回归模型:\n',model)

# print('最佳的alpha：',model.alpha_)  # 只有在使用LassoCV、LassoLarsCV时才有效

%%time

from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV 

model_lasso = Lasso(alpha=0.001) 

model_lasso.fit(X_train, y_train)

print (model_lasso.intercept_)

print (model_lasso.coef_)
# %%time

# result_test_lasso = model_lasso.predict(X_test)
# from sklearn.metrics import mean_absolute_error

# from sklearn.metrics import mean_squared_error

# mean_absolute_error(y_test, result_test_lasso)
# mean_squared_error(y_test, result_test_lasso)
%%time

# 使用模型预测

predicted_lasso = model_lasso.predict(x_prediction)
%%time

test_data = pd.read_csv(INPUT_DIR+'test_V2.csv')

print("fix winPlacePerc")

for i in range(len(test_data)):

    winPlacePerc = predicted_lasso[i]

    maxPlace = int(test_data.iloc[i]['maxPlace'])

    if maxPlace == 0:

        winPlacePerc = 0.0

    elif maxPlace == 1:

        winPlacePerc = 1.0

    else:

        gap = 1.0 / (maxPlace - 1)

        winPlacePerc = round(winPlacePerc / gap) * gap

    

    if winPlacePerc < 0: winPlacePerc = 0.0

    if winPlacePerc > 1: winPlacePerc = 1.0    

    predicted_lasso[i] = winPlacePerc


# 存储文件

f4=open(INPUT_DIR+'sample_submission_V2.csv')

submit_lasso=pd.read_csv(f4)

sample_result_lasso = pd.DataFrame(predicted_lasso,columns = ['winPlacePerc'])

submit_lasso['winPlacePerc'] = sample_result_lasso

submit_lasso.to_csv(r'sample_submission_lasso.csv', index=False)

del f4,submit_lasso,sample_result_lasso

gc.collect()
# from sklearn.linear_model import RidgeCV,LassoCV#用这个自带交叉验证参数

# from sklearn.model_selection import GridSearchCV#如果使用RidgeCV就不用GridSearchCV这个API了

# #使用RidgeCV来建立参数

# alpha = np.logspace(-3,2,10)#生成超参数，10的-3次方到10的2次方的等差数列

# ridge_model = RidgeCV(alpha,cv=5)

# ridge_model.fit(x_train,y)

# ridge_model.alphas #输出超参数的值
# ridge_model.alpha_
%%time

from sklearn.linear_model import Ridge

model_ridge = Ridge(alpha=0.5994842503189409) 

model_ridge.fit(X_train, y_train)

print (model_ridge.intercept_)

print (model_ridge.coef_)
# %%time

# result_test_ridge = model_ridge.predict(X_test)
# mean_absolute_error(y_test, result_test_ridge)
# mean_squared_error(y_test, result_test_ridge)
%%time

# 使用模型预测

predicted_ridge = model_ridge.predict(x_prediction)
%%time

test_data = pd.read_csv(INPUT_DIR+'test_V2.csv')

print("fix winPlacePerc")

for i in range(len(test_data)):

    winPlacePerc = predicted_ridge[i]

    maxPlace = int(test_data.iloc[i]['maxPlace'])

    if maxPlace == 0:

        winPlacePerc = 0.0

    elif maxPlace == 1:

        winPlacePerc = 1.0

    else:

        gap = 1.0 / (maxPlace - 1)

        winPlacePerc = round(winPlacePerc / gap) * gap

    

    if winPlacePerc < 0: winPlacePerc = 0.0

    if winPlacePerc > 1: winPlacePerc = 1.0    

    predicted_ridge[i] = winPlacePerc
# 存储文件

f5=open(INPUT_DIR+'sample_submission_V2.csv')

submit_ridge=pd.read_csv(f5)

sample_result_ridge = pd.DataFrame(predicted_ridge,columns = ['winPlacePerc'])

submit_ridge['winPlacePerc'] = sample_result_ridge

submit_ridge.to_csv(r'sample_submission_ridge.csv', index=False)

del f5,submit_ridge,sample_result_ridge

gc.collect()
# from sklearn.linear_model import ElasticNetCV#用这个自带交叉验证参数

# from sklearn.model_selection import GridSearchCV#如果使用RidgeCV就不用GridSearchCV这个API了

# #ElasticNetCV

# alpha = np.logspace(-3,2,10)#生成超参数，10的-3次方到10的2次方的等差数列

# elasticNet_model = ElasticNetCV(alpha,cv=10)

# elasticNet_model.fit(x_train,y)

# elasticNet_model.alpha_#输出超参数的值
%%time

from sklearn.linear_model import ElasticNet

model_elasticnet = ElasticNet(alpha=1.6152516038498196e-06, copy_X=True, fit_intercept=True, l1_ratio=0.5,

      max_iter=1000, normalize=False, positive=False, precompute=False,

      random_state=0, selection='cyclic', tol=0.0001, warm_start=False)

model_elasticnet.fit(X_train, y_train)

print (model_elasticnet.intercept_)

print (model_elasticnet.coef_)
# %%time

# result_test_elasticnet = model_elasticnet.predict(X_test)
# mean_absolute_error(y_test, result_test_elasticnet)
# mean_squared_error(y_test, result_test_elasticnet)
%%time

# 使用模型预测

predicted_elasticNet = model_elasticnet.predict(x_prediction)
%%time

test_data = pd.read_csv(INPUT_DIR+'test_V2.csv')

print("fix winPlacePerc")

for i in range(len(test_data)):

    winPlacePerc = predicted_elasticNet[i]

    maxPlace = int(test_data.iloc[i]['maxPlace'])

    if maxPlace == 0:

        winPlacePerc = 0.0

    elif maxPlace == 1:

        winPlacePerc = 1.0

    else:

        gap = 1.0 / (maxPlace - 1)

        winPlacePerc = round(winPlacePerc / gap) * gap

    

    if winPlacePerc < 0: winPlacePerc = 0.0

    if winPlacePerc > 1: winPlacePerc = 1.0    

    predicted_elasticNet[i] = winPlacePerc
# 存储文件

f6=open(INPUT_DIR+'sample_submission_V2.csv')

submit_elasticNet=pd.read_csv(f6)

sample_result_elasticNet = pd.DataFrame(predicted_elasticNet,columns = ['winPlacePerc'])

submit_elasticNet['winPlacePerc'] = sample_result_elasticNet

submit_elasticNet.to_csv(r'sample_submission_elasticNet.csv', index=False)

del f6,submit_elasticNet,sample_result_elasticNet

gc.collect()