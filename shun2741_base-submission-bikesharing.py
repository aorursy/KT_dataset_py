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
import lightgbm as lgb

from sklearn.metrics import mean_squared_log_error

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv('/kaggle/input/bike-sharing-demand-for-education/train.csv')

test = pd.read_csv('/kaggle/input/bike-sharing-demand-for-education/test.csv')

sample_submission = pd.read_csv('/kaggle/input/bike-sharing-demand-for-education/sample_submission.csv')
#必要な列だけ残す

X = train.drop(['datetime','casual','registered','cnt'],axis = 1)

X_test = test.drop(['datetime'],axis = 1)

y = train[['year','cnt']]
%%time



columns = X.columns



y_preds = np.zeros(X_test.shape[0])



feature_importances = pd.DataFrame()

feature_importances['feature'] = columns



# 2011年を学習、2012年を検証データに

X_train = X.query("year == 2011")

X_valid = X.query("year == 2012")



y_train = y.query("year == 2011")

y_valid = y.query("year == 2012")



y_train.drop('year',axis=1, inplace=True)

y_valid.drop('year',axis=1, inplace=True)





dtrain = lgb.Dataset(X_train, label=y_train)



params = {

    'objective': 'mean_squared_error',

    'metric': 'rmse'

}

LGBM = lgb.train(params, dtrain, 1000, verbose_eval=100)



feature_importances['importance'] = LGBM.feature_importance(importance_type='gain')



y_pred_valid = LGBM.predict(X_valid)



y_pred_valid = np.where(y_pred_valid < 0, 0, y_pred_valid)



print("RMSLE:", mean_squared_log_error(y_valid, y_pred_valid))

print('******************************************************')



y_preds = LGBM.predict(X_test)

y_preds = np.where(y_preds < 0, 0, y_preds)
plt.figure(figsize=(16, 16))

sns.barplot(data=feature_importances.sort_values('importance', ascending=False).head(50), x='importance', y='feature');

plt.title('50 TOP feature importance')
submission = pd.concat([test['datetime'], pd.Series(y_preds)], axis=1) 

submission = submission.rename(columns={0:'cnt'})
submission.shape
submission.head()
submission.to_csv('base_submission.csv',header=True, index=False)