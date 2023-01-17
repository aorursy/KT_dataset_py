# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import xgboost as xgb

from sklearn.metrics import mean_squared_error

import pandas as pd

import numpy as np

import lightgbm

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
from sklearn.datasets import load_boston



#加载数据集

boston = load_boston()

boston.keys()
#一共有506个实例，13个特征

boston.data.shape
#查看特征

boston.feature_names
#对数据集的描述

print(boston.DESCR)
#把仅包含特征的数据转为DataFrameo类型

data = pd.DataFrame(boston.data)

data.columns = boston.feature_names

data.head()
#把价格添加到data中

data['PRICE'] = boston.target
#data的信息

data.info()
#各特征的平均数、标准差、最小值、1/4、1/2、3/4分位点，和最大值

data.describe()
#分出特征和目标变量

X, y = data.iloc[:,:-1], data.iloc[:,-1]
#生成XGBoost支持的DMatrix类型数据，用于交叉验证

data_dmatrix = xgb.DMatrix(data=X,label=y)
from sklearn.model_selection import train_test_split



#随机分割训诫集和测试集

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
#初始化模型

xg_reg = xgb.XGBRegressor(objective ='reg:linear', max_depth = 3, n_estimators = 3)
#模型训练

xg_reg.fit(X_train,y_train)



#模型预测

preds = xg_reg.predict(X_test)
#在测试集上的均方误差

rmse = np.sqrt(mean_squared_error(y_test, preds))

print("RMSE: %f" % (rmse))
for i in range(3):

    xgb.plot_tree(xg_reg,num_trees=i)

    plt.rcParams['figure.figsize'] = [19.2, 10.8]

    plt.show()
params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,

                'max_depth': 5, 'alpha': 10}



#XGBoost交叉验证

cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,

                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)
cv_results.head()
print((cv_results["test-rmse-mean"]).tail(1))
xg_reg = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=10)



xgb.plot_tree(xg_reg,num_trees=2)

plt.rcParams['figure.figsize'] = [10, 10]

plt.show()
#重要度

xgb.plot_importance(xg_reg)

plt.rcParams['figure.figsize'] = [5, 5]

plt.show()