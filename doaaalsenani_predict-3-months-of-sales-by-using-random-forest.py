import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



#sklearn

from sklearn.model_selection import train_test_split, cross_val_score,KFold

from sklearn import datasets, linear_model

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LassoCV

from sklearn import preprocessing, svm





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df_train=pd.read_csv('../input/demand-forecasting-kernels-only/train.csv')
df_test= pd.read_csv('../input/demand-forecasting-kernels-only/test.csv', index_col='id')
df_smple=  pd.read_csv('../input/demand-forecasting-kernels-only/sample_submission.csv', index_col='id')
df_train.head()
df_train.info()
df_train.shape
df_test.head()
df_test.info()
df_test.shape
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (18, 6))

sns.heatmap(df_train.isnull(), yticklabels=False, ax = ax[0], cbar=False, cmap='viridis')

ax[0].set_title('Trian data')



sns.heatmap(df_test.isnull(), yticklabels=False, ax = ax[1], cbar=False, cmap='viridis')

ax[1].set_title('Test data');
# change date to DateTime

df_train['date'] = pd.to_datetime(df_train['date'])

df_train.dtypes
# change date to DateTime

df_test['date'] = pd.to_datetime(df_test['date'])

df_test.dtypes
#train data

df_train['year'] = df_train['date'].dt.year

df_train['month'] = df_train['date'].dt.month

df_train['day'] = df_train['date'].dt.day

df_train['week'] = df_train['date'].dt.week

df_train['weekofyear'] = df_train['date'].dt.weekofyear

df_train['dayofweek'] = df_train['date'].dt.dayofweek

df_train['weekday'] = df_train['date'].dt.weekday

df_train['dayofyear'] = df_train['date'].dt.dayofyear

df_train['quarter'] = df_train['date'].dt.quarter



df_train['is_month_start'] = df_train['date'].dt.is_month_start

df_train['is_month_end'] =df_train['date'].dt.is_month_end

df_train['is_quarter_start'] = df_train['date'].dt.is_quarter_start

df_train['is_quarter_end'] = df_train['date'].dt.is_quarter_end

df_train['is_year_start'] = df_train['date'].dt.is_year_start

df_train['is_year_end'] = df_train['date'].dt.is_year_end



# To convert data type from bool to int

# df_train['is_month_start'] = (df_train.date.dt.is_month_start).astype(int)
#Test data



df_test['year'] = df_test['date'].dt.year

df_test['month'] = df_test['date'].dt.month

df_test['day'] = df_test['date'].dt.day

df_test['week'] = df_test['date'].dt.week

df_test['weekofyear'] = df_test['date'].dt.weekofyear

df_test['dayofweek'] = df_test['date'].dt.dayofweek

df_test['weekday'] = df_test['date'].dt.weekday

df_test['dayofyear'] = df_test['date'].dt.dayofyear

df_test['quarter'] = df_test['date'].dt.quarter



df_test['is_month_start'] = df_test['date'].dt.is_month_start

df_test['is_month_end']= df_test['date'].dt.is_month_end

df_test['is_quarter_start'] = df_test['date'].dt.is_quarter_start

df_test['is_quarter_end'] = df_test['date'].dt.is_quarter_end

df_test['is_year_start'] = df_test['date'].dt.is_year_start

df_test['is_year_end'] = df_test['date'].dt.is_year_end

df_train.info()
df_test.info()
del df_train['date']

del df_test['date']
plt.figure(figsize=(33,30))

cor = df_train.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

plt.show()
y = df_train['sales']

X = df_train.drop('sales', axis=1)
ss = StandardScaler()

Xs =ss.fit_transform(X)

X_test_ss = ss.transform(df_test)
randomF = RandomForestRegressor()

randomF.fit(Xs, y)

print('Score :',randomF.score(Xs, y))
pred_f = randomF.predict(X_test_ss)
submission = df_smple.drop(['sales'],axis=1)
submission['sales'] = pred_f
submission.head()
submission.to_csv('submission.csv')