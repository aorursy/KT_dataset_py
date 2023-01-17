import pandas as pd

import numpy as np

import seaborn as sns

from scipy.stats import norm

%matplotlib inline

import matplotlib.pyplot as plt

import time

import warnings

warnings.simplefilter('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train=pd.read_csv('/kaggle/input/allstate-claims-severity/train.csv', index_col='id')

test=pd.read_csv('/kaggle/input/allstate-claims-severity/test.csv', index_col='id')

submission=pd.read_csv('/kaggle/input/allstate-claims-severity/sample_submission.csv', index_col='id')

print(train.shape, test.shape, submission.shape)
figure, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

figure.set_size_inches(14,6)

sns.distplot(train['loss'], fit=norm, ax=ax1)

sns.distplot(np.log(train['loss']+1), fit=norm, ax=ax2)
train=train.drop(train.loc[train['loss']>40000].index)
train['loss']=np.log(train['loss']+1)

Ytrain=train['loss']
data=train

train=train[list(test)]

all_data=pd.concat((train, test))

all_data.shape
cat_features=list(np.where(all_data.dtypes==np.object)[0])

print(cat_features)
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

non_numeric=list(all_data.select_dtypes(np.object))

for cols in non_numeric:

    le.fit(all_data[cols])

    all_data[cols]=le.transform(all_data[cols])
print(train.shape, test.shape)

Xtrain=all_data[:len(train)]

Xtest=all_data[len(train):]

print(Xtrain.shape, Ytrain.shape, Xtest.shape, submission.shape)
from catboost import CatBoostRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

#import optuna

from sklearn.model_selection import cross_val_score
model_xgb=XGBRegressor(tree_method='gpu_hist', seed=18, objective='reg:linear', n_jobs=-1, verbosity=0,

                       colsample_bylevel=0.764115402027029, colsample_bynode=0.29243734009596956, 

                       colsample_bytree= 0.7095719673041723, gamma= 4.127534050725986, learning_rate= 0.02387231810322894, 

                       max_depth=14, min_child_weight=135, n_estimators=828,reg_alpha=0.3170105723222332, 

                       reg_lambda= 0.3660379465131937, subsample=0.611471430211575)

model_xgb
model_LGB=LGBMRegressor(objective='regression_l1', random_state=18, subsample_freq=1,

                        colsample_bytree=0.3261853512759363, min_child_samples=221, n_estimators=2151, num_leaves= 45, 

                        reg_alpha=0.9113713668943361, reg_lambda=0.8220990333713991, subsample=0.49969995651550947, 

                        max_bin=202, learning_rate=0.02959820893211799) #,device='gpu')

model_LGB
model_Cat=CatBoostRegressor(loss_function='MAE', random_seed=18, task_type='GPU', cat_features=cat_features, verbose=False,

                            iterations=2681, learning_rate=0.2127106032536721, depth=7, l2_leaf_reg=5.266150673910493, 

                            random_strength=7.3001140226199315, bagging_temperature=0.26098669708900213)

model_Cat
model_Cat.fit(Xtrain, Ytrain)

model_LGB.fit(Xtrain, Ytrain)

model_xgb.fit(Xtrain, Ytrain)



lgb_predictions=model_LGB.predict(Xtest)

cat_predictions=model_Cat.predict(Xtest)

xgb_predictions=model_xgb.predict(Xtest)
predictions=(lgb_predictions + cat_predictions + xgb_predictions)/3



predictions=np.exp(predictions)-1

submission['loss']=predictions

submission.to_csv('Result.csv')

submission.head()