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
from sklearn.datasets import load_boston

import pandas as pd

from pandas.plotting import scatter_matrix

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

from sklearn import preprocessing

import xgboost as xgb

from sklearn.metrics import mean_squared_error
#load data set

df = pd.read_csv('/kaggle/input/boston-housing-dataset/train.csv')

df.drop(columns = ['ID'], inplace = True)

df.head()
train_df, val_df = train_test_split(df)
plt.figure(figsize=(12,10))

cor = train_df.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.RdYlGn)

plt.show()
#prepare feature and target

train_y = train_df['MEDV']

train_X = train_df.drop(columns = 'MEDV' )

train_y.reset_index(drop=True, inplace=True)

train_X.reset_index(drop=True, inplace=True)
#try default XGBoostregression

xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,

                max_depth = 5, alpha = 10, n_estimators = 10)

xg_reg.fit(train_X,train_y)

preds = xg_reg.predict(val_df.iloc[:,:-1])

rmse = np.sqrt(mean_squared_error(val_df.iloc[:,-1], preds))

print("RMSE: %f" % (rmse))
# grid search on parameter of XGBoost

num_boost_round = 999

params = {

    # Parameters that we are going to tune.

    'max_depth':6,

    'min_child_weight': 1,

    'eta':.3,

    'subsample': 1,

    'colsample_bytree': 1,

    # Other parameters

    'objective':'reg:squarederror',

}

params['eval_metric'] = "rmse"
#train

dtrain = xgb.DMatrix(train_X, label=train_y)

dtest = xgb.DMatrix(val_df.iloc[:,:-1], label=val_df.iloc[:,-1])



model = xgb.train(

    params,

    dtrain,

    num_boost_round=num_boost_round,

    evals=[(dtest, "Test")],

    early_stopping_rounds=50

)

#finalize submission

exam_test = pd.read_csv('/kaggle/input/boston-housing-dataset/test.csv')

result = model.predict(xgb.DMatrix(exam_test.iloc[:, 1:]))



result_submit = pd.DataFrame({"ID": exam_test['ID'], "MEDV":result})

print(result_submit.head())



pd.DataFrame(result_submit).to_csv("submit_SY.csv", index=False)
xgb.plot_importance(model)

plt.rcParams['figure.figsize'] = [5, 5]

plt.show()