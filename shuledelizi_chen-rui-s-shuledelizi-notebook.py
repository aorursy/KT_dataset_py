# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# 读取数据

import numpy as np

import pandas as pd

import os



data  = pd.read_csv('/kaggle/input/bnu-esl-2020/train.csv')

test  = pd.read_csv('/kaggle/input/bnu-esl-2020/test.csv')
# 统一一下名称

data.columns = data.columns.str.replace('Unnamed: 0', 'Id')

test.columns = test.columns.str.replace('id', 'Id')
# 看一看数据长什么样

data.head()
# 看一看数据长什么样

data.shape
# 看一看数据长什么样

data.shape
# 看一看数据长什么样

data.info()
# 观察发现total_bedrooms存在缺失值

import matplotlib.pyplot as plt

import seaborn as sns



plt.figure(figsize = (16,6))

sns.heatmap(data.isnull(),cmap = 'viridis')



plt.figure(figsize = (16,6))

sns.heatmap(test.isnull(),cmap = 'viridis')
# 去除ID这一列

data.drop(['Id'], axis = 1,inplace = True)

data.shape

Id = test.Id

test.drop(['Id'], axis = 1,inplace = True)

test.shape
# 得到自变量与因变量，供以后处理

X = data.drop(['median_house_value'],axis = 1)

y = data.median_house_value
# 由于ocean_proximity是文本信息，因此把ocean_proximity转换为虚拟变量

X.ocean_proximity.value_counts()

X.ocean_proximity.replace({'<1H OCEAN':1.0,'INLAND':2.0,'ISLAND':3.0,'NEAR BAY':4.0,'NEAR OCEAN':5.0},inplace = True)
# total_bedrooms有缺失值，用平均数填充

data_inf=X.total_bedrooms.describe()

X.total_bedrooms = X.total_bedrooms.fillna(data_inf['mean'])
# 再次看看数据的样子，都是数值变量了

X.info()
# 数据可视化，可以发现经纬度还是有用的

housing = data.copy()

housing.plot(kind='scatter',x='longitude',y='latitude',alpha=0.4,

             s=housing['population'] / 100,label='population',c="median_house_value",

            cmap=plt.get_cmap("jet"),colorbar=True)

plt.legend()

plt.show()
# 看一看变量之间的相关，有一些变量存在高相关，应避免这种情况

corr = X.corr()

plt.figure(figsize=(16, 16))

sns.heatmap(corr, cmap='viridis')
# 因此，我们创建新特征。roomes_per_household、bedrooms_per_rooms、population_per_household相比原数据更有意义，更有解释性

X["roomes_per_household"]=X["total_rooms"]/X["households"]

X["bedrooms_per_rooms"]=X["total_bedrooms"]/X["total_rooms"]

X["population_per_household"]=X["population"]/X["households"]
# 看一看新变量与老变量的相关，发现变低了，这是好事

corr = X.corr()

plt.figure(figsize=(16, 16))

sns.heatmap(corr, cmap='viridis')
# 把一些相关较高的变量剔除

X.drop(['total_rooms'], axis = 1,inplace = True)

X.drop(['total_bedrooms'], axis = 1,inplace = True)

X.drop(['households'], axis = 1,inplace = True)
# 进行特征缩放

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X = sc_X.fit_transform(X)
# 准备工作完成，接下来进入模型部分，采用 Gradient Boosting

from sklearn import ensemble

from sklearn import datasets

from sklearn.utils import shuffle

from sklearn.metrics import mean_squared_error
# 选取90%作为训练集，10%作为验证集

offset = int(X.shape[0] * 0.9)

X_train, y_train = X[:offset], y[:offset]

X_test, y_test = X[offset:], y[offset:]
# 进行模型拟合，其中超参数是通过网格搜索来确定的，实际进行网格搜索时需要删除后面的一些分析（想试试可以把后面都删了），不然会报错

params = {'n_estimators': 1000, 'max_depth': 6, 'min_samples_split': 2,

          'learning_rate': 0.1, 'loss': 'ls'}

clf = ensemble.GradientBoostingRegressor(**params)



clf.fit(X_train, y_train)

mse = mean_squared_error(y_test, clf.predict(X_test))

rmse = np.sqrt(mse)

print("RMSE: %.4f" % rmse)



#网格搜索部分

#from sklearn import ensemble

#from sklearn import datasets

#from sklearn.utils import shuffle

#from sklearn.metrics import mean_squared_error

#from sklearn.model_selection import GridSearchCV



#offset = int(X.shape[0] * 0.9)

#X_train, y_train = X[:offset], y[:offset]

#X_test, y_test = X[offset:], y[offset:]



# Fit regression model

#param_grid = {

#        'n_estimators': [1000],

#        'max_depth': [4,6,8,10],

#        'learning_rate': [0.01,0.1],

#       'subsample': [1]

#    }



#gbr = ensemble.GradientBoostingRegressor(random_state=0)

#clf = GridSearchCV(estimator=gbr, param_grid=param_grid, n_jobs=1, cv=10)

#clf.fit(X_train, y_train)

#mse = mean_squared_error(y_test, clf.predict(X_test))

#rmse = np.sqrt(mse)

#print("RMSE: %.4f" % rmse)



#print('Gradient boosted tree regression...')

#print('Best Params:')

#print(clf.best_params_)

#print('Best CV Score:')

#print(-clf.best_score_)
# 画随着boosting的次数，模型的训练误差和测试误差的变化图

test_score = np.zeros((params['n_estimators'],), dtype=np.float64)



for i, y_pred in enumerate(clf.staged_predict(X_test)):

    test_score[i] = clf.loss_(y_test, y_pred)



plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)

plt.title('Deviance')

plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',

         label='Training Set Deviance')

plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',

         label='Test Set Deviance')

plt.legend(loc='upper right')

plt.xlabel('Boosting Iterations')

plt.ylabel('Deviance')
# 看看特征的重要性

feature_importance = clf.feature_importances_

feature_importance = 100.0 * (feature_importance / feature_importance.max())

sorted_idx = np.argsort(feature_importance)

pos = np.arange(sorted_idx.shape[0]) + .5

plt.subplot(1, 2, 2)

plt.barh(pos, feature_importance[sorted_idx], align='center')

feature_names = np.array(['longitude','latitude','housing_median_age','population','median_income','ocean_proximity','roomes_per_household','bedrooms_per_rooms','population_per_household'])

plt.yticks(pos, feature_names[sorted_idx])

plt.xlabel('Relative Importance')

plt.title('Variable Importance')

plt.show()
# 接下来做出预测，首先也要与训练一致，整理原数据

# 把ocean_proximity转换为虚拟变量

test.ocean_proximity.value_counts()

test.ocean_proximity.replace({'<1H OCEAN':1.0,'INLAND':2.0,'ISLAND':3.0,'NEAR BAY':4.0,'NEAR OCEAN':5.0},inplace = True)



# total_bedrooms有缺失值，用平均数填充

test_inf=test.total_bedrooms.describe()

test.total_bedrooms = test.total_bedrooms.fillna(test_inf['mean'])



# 创建新特征，去除某些特征

test["roomes_per_household"]=test["total_rooms"]/test["households"]

test["bedrooms_per_rooms"]=test["total_bedrooms"]/test["total_rooms"]

test["population_per_household"]=test["population"]/test["households"]



test.drop(['total_rooms'], axis = 1,inplace = True)

test.drop(['total_bedrooms'], axis = 1,inplace = True)

test.drop(['households'], axis = 1,inplace = True)



# 特征缩放

from sklearn.preprocessing import StandardScaler

sc_test = StandardScaler()

test = sc_test.fit_transform(test)
# 获得预测值

test_pred = clf.predict(test)
# 保存结果

output = pd.read_csv('/kaggle/input/bnu-esl-2020/test.csv')

output.insert(1,'predicted',test_pred)

output.drop(['longitude'], axis = 1,inplace = True)

output.drop(['latitude'], axis = 1,inplace = True)

output.drop(['housing_median_age'], axis = 1,inplace = True)

output.drop(['total_rooms'], axis = 1,inplace = True)

output.drop(['total_bedrooms'], axis = 1,inplace = True)

output.drop(['population'], axis = 1,inplace = True)

output.drop(['households'], axis = 1,inplace = True)

output.drop(['median_income'], axis = 1,inplace = True)

output.drop(['ocean_proximity'], axis = 1,inplace = True)

output.to_csv("predict.csv",index=False)