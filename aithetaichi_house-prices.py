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
df_train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

df_test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
df_train.columns
df_train["SalePrice"].describe()
df_train.select_dtypes(include=['float', 'int']).head()
df_train.select_dtypes(include='object').isnull().sum()[df_train.select_dtypes(include='object').isnull().sum()>0]
for col in ('Alley','Utilities','MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',

            'BsmtFinType2','Electrical','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond',

           'PoolQC','Fence','MiscFeature'):

    df_train[col] = df_train[col].fillna('None')

    df_test[col] = df_test[col].fillna('None')
for col in ('MSZoning','Exterior1st','Exterior2nd','KitchenQual','SaleType','Functional'):

    df_train[col] = df_train[col].fillna(df_train[col].mode()[0])

    df_test[col] = df_test[col].fillna(df_train[col].mode()[0])
import seaborn as sns

sns.distplot(df_train["SalePrice"])
var_list_all = df_train.columns

var_list_all = [var for var in var_list_all if var not in ['Id','SalePrice']]

len(var_list_all)
var_list_num = [var for var in var_list_all if df_train[var].dtypes!="object"]

var_list_num
#var_list = ["GrLivArea", "TotalBsmtSF", "OverallQual", "YearBuilt", "YearRemodAdd", "GarageArea", 'WoodDeckSF', 'OpenPorchSF']

#for var in var_list:

#for var in var_list_all:

for var in var_list_num:

    data = pd.concat([df_train["SalePrice"], df_train[var]], axis=1)

    #print(data)

    #print("var: {}\ttype:{}".format(var,type(df_train[var][0])))

    data.plot.scatter(x=var, y="SalePrice", ylim=(0,800000))
# not remove

# 0.16306



#var_list_num_fixed = var_list_num

# 0.16042

# 29 features will use

var_remove = ['LowQualFinSF', 'BsmtHalfBath', '3SsnPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']

var_list_num_fixed = [var for var in var_list_num if var not in var_remove]

# 25 fratures will use

# +'BsmtFullBath', 'KitchenAbvGr', 'Fireplaces','EnclosedPorch'

var_remove_add = ['BsmtFullBath', 'KitchenAbvGr', 'Fireplaces', 'EnclosedPorch']

var_list_num_fixed = [var for var in var_list_num_fixed if var not in var_remove_add]

#len(var_list_num),len(var_list_num_fixed)
import matplotlib.pyplot as plt

corrmat = df_train.corr()

f, ax = plt.subplots(figsize=(12,9))

sns.heatmap(corrmat, vmax=.8, square=True)
#saleprice correlation matrix

k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df_train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
from sklearn import preprocessing
df_train['HouseStyle'].head
df_train['HouseStyle'].describe()
hs_unique = pd.unique(df_train['HouseStyle'])

hs_unique
#le = preprocessing.LabelEncoder()

#le.fit(hs_unique)

#le.classes_
#hs_labeled = le.transform(df_train['HouseStyle'])

#hs_labeled
#df_hs_labeled = pd.Series(data=hs_labeled, name='HouseStyle_labeled')

#df_hs_labeled.head()
#data = pd.concat([df_train["SalePrice"], df_hs_labeled], axis=1)

#data.head
#data.plot.scatter(x='HouseStyle_labeled', y="SalePrice", ylim=(0,800000))

# 0: '1.5Fin', 1: '1.5Unf', 2: '1Story', 3: '2.5Fin', 

# 4: '2.5Unf', 5: '2Story', 6: 'SFoyer', 7: 'SLvl'
import xgboost as xgb

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
# 数値データの中から SalePrice との分布より独断と偏見で選んだ 29 の指標

# 欠損値処理なし。XGBoost の性能に委ねる

X = pd.concat([df_train[var] for var in var_list_num_fixed],

             axis=1,

             keys=var_list_num_fixed)

X.head(10)
y = df_train['SalePrice']

y.head(10)
X_train, X_test, y_train, y_test = train_test_split(X, y,

                                                    test_size=0.1,

                                                    shuffle=True,

                                                    random_state=42,

                                                    )

X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.model_selection import GridSearchCV
# ハイパーパラメータ探索



#"""

reg = xgb.XGBRegressor()

reg_cv = GridSearchCV(reg, {'max_depth': [2,4,6,8,10], 'n_estimators': [50,100,200]}, verbose=1)

reg_cv.fit(X_train, y_train)

print(reg_cv.best_params_, reg_cv.best_score_)



max_depth_best = reg_cv.best_params_['max_depth']

n_estimators_best = reg_cv.best_params_['n_estimators']

#"""
# ハイパーパラメータ探索

#"""

reg = xgb.XGBRegressor()

reg_cv = GridSearchCV(reg, {'max_depth': [max_depth_best],'n_estimators': [n_estimators_best], 'min_child_weight': [0,1,2,4,6,8,10], 'max_delta_step': [0,1,2,4,6,8,10]})

reg_cv.fit(X_train, y_train)

print(reg_cv.best_params_, reg_cv.best_score_)



min_child_weight_best = reg_cv.best_params_['min_child_weight']

max_delta_step_best = reg_cv.best_params_['max_delta_step']

#"""

dtrain = xgb.DMatrix(X_train, label=y_train)

dtest = xgb.DMatrix(X_test, label=y_test)



xgb_params = {

        # regression

        'objective': 'reg:linear',

        'eval_metric': 'rmse',

        # 29 features

        #'max_depth': 4,

        #'min_child_weight': 10,

        # 25 features

        'max_depth': max_depth_best,

        'min_child_weight': min_child_weight_best,

        'max_delta_step': max_delta_step_best,

        'n_estimators': n_estimators_best

}

evals = [(dtrain, 'train'), (dtest, 'eval')]

evals_result = {}

bst = xgb.train(xgb_params,

                dtrain,

                num_boost_round=1000,

                early_stopping_rounds=10,

                evals=evals,

                evals_result=evals_result,

)

y_pred = bst.predict(dtest)

mse = mean_squared_error(y_test, y_pred)

import math

print('RMSE:', math.sqrt(mse))



from matplotlib import pyplot as plt

train_metric = evals_result['train']['rmse']

plt.plot(train_metric, label='train rmse')

eval_metric = evals_result['eval']['rmse']

plt.plot(eval_metric, label='eval rmse')

plt.grid()

plt.legend()

plt.xlabel('rounds')

plt.ylabel('rmse')

plt.show()



bst.get_fscore()
# 数値データの中から SalePrice との分布より独断と偏見で選んだ 29 の指標

# 欠損値処理なし。XGBoost の性能に委ねる

submit = pd.concat([df_test[var] for var in var_list_num_fixed],

             axis=1,

             keys=var_list_num_fixed)

submit.head(10)
# predict

d_pred = xgb.DMatrix(submit)

pred = bst.predict(d_pred)
# testdata は 1459 あるのに対し、 pred は 1460 ある

pred = pred[:1459]

pred.shape
submission = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv", index_col=0)

submission.head
submission['SalePrice'] = pred

#submission.loc[df_test.index.values, "SalePrice"] =pred

submission.to_csv("submission.csv", index=True)