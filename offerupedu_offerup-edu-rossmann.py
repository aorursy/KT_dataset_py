# 载入必要的库

import pandas as pd

import numpy as np

import xgboost as xgb



import missingno as msno

import seaborn as sns

import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline
# 载入数据

train = pd.read_csv('../input/rossmann-store-sales/train.csv')

test = pd.read_csv('../input/rossmann-store-sales/test.csv')

store = pd.read_csv('../input/rossmann-store-sales/store.csv')
train.info(), test.info(), store.info()
fig = plt.figure(figsize=(16,6))



ax1 = fig.add_subplot(121)

ax1.set_xlabel('Sales')

ax1.set_ylabel('Count')

ax1.set_title('Sales of Closed Stores')

plt.xlim(-1,1)

train.loc[train.Open==0].Sales.hist(align='left')



ax2 = fig.add_subplot(122)

ax2.set_xlabel('Sales')

ax2.set_ylabel('PDF')

ax2.set_title('Sales of Open Stores')

sns.distplot(train.loc[train.Open!=0].Sales)



print('The skewness of Sales is {}'.format(train.loc[train.Open!=0].Sales.skew()))
train = train.loc[train.Open != 0]

train = train.loc[train.Sales > 0].reset_index(drop=True)
# train的缺失信息：无缺失

train[train.isnull().values==True]
# test的缺失信息

test[test.isnull().values==True]
# store的缺失信息

msno.matrix(store)
# 默认test中的店铺全部正常营业

test.fillna(1,inplace=True)



# 对CompetitionDistance中的缺失值采用中位数进行填补

store.CompetitionDistance = store.CompetitionDistance.fillna(store.CompetitionDistance.median())



# 对其它缺失值全部补0

store.fillna(0,inplace=True)
# 特征合并

train = pd.merge(train, store, on='Store')

test = pd.merge(test, store, on='Store')
def build_features(features, data):



    # 直接使用的特征

    features.extend(['Store','CompetitionDistance','CompetitionOpenSinceMonth','StateHoliday','StoreType','Assortment',

                     'SchoolHoliday','CompetitionOpenSinceYear', 'Promo', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear'])

    

    # 以下特征处理方式参考：https://blog.csdn.net/aicanghai_smile/article/details/80987666

    

    # 时间特征，使用dt进行处理

    features.extend(['Year','Month','Day','DayOfWeek','WeekOfYear'])

    data['Year'] = data.Date.dt.year

    data['Month'] = data.Date.dt.month

    data['Day'] = data.Date.dt.day

    data['DayOfWeek'] = data.Date.dt.dayofweek

    data['WeekOfYear'] = data.Date.dt.weekofyear

    

    # 'CompetitionOpen'：竞争对手的已营业时间

    # 'PromoOpen'：竞争对手的已促销时间

    # 两个特征的单位均为月

    features.extend(['CompetitionOpen','PromoOpen'])

    data['CompetitionOpen'] = 12*(data.Year-data.CompetitionOpenSinceYear) + (data.Month-data.CompetitionOpenSinceMonth)

    data['PromoOpen'] = 12*(data.Year-data.Promo2SinceYear) + (data.WeekOfYear-data.Promo2SinceWeek)/4.0

    data['CompetitionOpen'] = data.CompetitionOpen.apply(lambda x: x if x > 0 else 0)        

    data['PromoOpen'] = data.PromoOpen.apply(lambda x: x if x > 0 else 0)

    

    # 'IsPromoMonth'：该天店铺是否处于促销月，1表示是，0表示否

    features.append('IsPromoMonth')

    month2str = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}

    data['monthStr'] = data.Month.map(month2str)

    data.loc[data.PromoInterval==0, 'PromoInterval'] = ''

    data['IsPromoMonth'] = 0

    for interval in data.PromoInterval.unique():

        if interval != '':

            for month in interval.split(','):

                data.loc[(data.monthStr == month) & (data.PromoInterval == interval), 'IsPromoMonth'] = 1

    

    # 字符特征转换为数字

    mappings = {'0':0, 'a':1, 'b':2, 'c':3, 'd':4}

    data.StoreType.replace(mappings, inplace=True)

    data.Assortment.replace(mappings, inplace=True)

    data.StateHoliday.replace(mappings, inplace=True)

    data['StoreType'] = data['StoreType'].astype(int)

    data['Assortment'] = data['Assortment'].astype(int)

    data['StateHoliday'] = data['StateHoliday'].astype(int)
# 处理Date方便特征提取

train.Date = pd.to_datetime(train.Date, errors='coerce')

test.Date = pd.to_datetime(test.Date, errors='coerce')



# 使用features数组储存使用的特征

features = []



# 对train与test特征提取

build_features(features, train)

build_features([], test)



# 打印使用的特征

print(features)
# 评价函数Rmspe

# 参考：https://www.kaggle.com/justdoit/xgboost-in-python-with-rmspe



def ToWeight(y):

    w = np.zeros(y.shape, dtype=float)

    ind = y != 0

    w[ind] = 1./(y[ind]**2)

    return w



def rmspe(yhat, y):

    w = ToWeight(y)

    rmspe = np.sqrt(np.mean(w * (y-yhat)**2))

    return rmspe



def rmspe_xg(yhat, y):

    y = y.get_label()

    y = np.expm1(y)

    yhat = np.expm1(yhat)

    w = ToWeight(y)

    rmspe = np.sqrt(np.mean(w * (y-yhat)**2))

    return "rmspe", rmspe



def neg_rmspe(yhat, y):

    y = np.expm1(y)

    yhat = np.expm1(yhat)

    w = ToWeight(y)

    rmspe = np.sqrt(np.mean(w * (y-yhat)**2))

    return -rmspe
from sklearn.model_selection import GridSearchCV, ShuffleSplit

from sklearn.metrics import make_scorer



from sklearn.tree import DecisionTreeRegressor



regressor = DecisionTreeRegressor(random_state=2)



cv_sets = ShuffleSplit(n_splits=5, test_size=0.2)    

params = {'max_depth':range(10,40,2)}

scoring_fnc = make_scorer(neg_rmspe)



grid = GridSearchCV(regressor,params,scoring_fnc,cv=cv_sets)

grid = grid.fit(train[features], np.log1p(train.Sales))



DTR = grid.best_estimator_
# 显示最佳超参数

DTR.get_params()
# 生成上传文件

submission = pd.DataFrame({"Id": test["Id"], "Sales": np.expm1(DTR.predict(test[features]))})

submission.to_csv("benchmark.csv", index=False)
# 在此进行参数调节

params = {'objective': 'reg:linear',

          'eta': 0.01,

          'max_depth': 11,

          'subsample': 0.5,

          'colsample_bytree': 0.5,

          'silent': 1,

          'seed': 1

          }

num_trees = 10000
# 随机划分训练集与验证集

from sklearn.model_selection import train_test_split



X_train, X_test = train_test_split(train, test_size=0.2, random_state=2)



dtrain = xgb.DMatrix(X_train[features], np.log1p(X_train.Sales))

dvalid = xgb.DMatrix(X_test[features], np.log1p(X_test.Sales))

dtest = xgb.DMatrix(test[features])



watchlist = [(dtrain, 'train'),(dvalid, 'eval')]

gbm = xgb.train(params, dtrain, num_trees, evals=watchlist, early_stopping_rounds=50, feval=rmspe_xg, verbose_eval=False)
# 生成提交文件

test_probs = gbm.predict(xgb.DMatrix(test[features]), ntree_limit=gbm.best_ntree_limit)

indices = test_probs < 0

test_probs[indices] = 0

submission = pd.DataFrame({"Id": test["Id"], "Sales": np.expm1(test_probs)})

submission.to_csv("xgboost.csv", index=False)