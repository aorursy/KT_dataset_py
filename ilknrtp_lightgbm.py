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
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
train = pd.read_csv('/kaggle/input/ml-challenge-tr-is-bankasi/train.csv')
test = pd.read_csv('/kaggle/input/ml-challenge-tr-is-bankasi/test.csv')
train_set = train.copy()
test_set = test.copy()
print(train.shape)
print(test.shape)
print(train.columns)
print(test.columns)
print(train.info())
print(test.info())
print('ISLEM_TUTARI:\n', train['ISLEM_TUTARI'].unique())
print('ISLEM_ADEDI:\n', train['ISLEM_ADEDI'].unique())
print('YIL_AY:\n', sorted(train['YIL_AY'].unique()))
print('SEKTOR:\n', sorted(train['SEKTOR'].unique()))
print('Record_Count:\n', sorted(train['Record_Count'].unique()))
print('CUSTOMER:\n', train['CUSTOMER'].unique())
print('NULL values:', train.isnull().any())
print('Min ISLEM_ADEDI:', train['ISLEM_ADEDI'].min())
print('Max ISLEM_ADEDI:', train['ISLEM_ADEDI'].max())

print('Min ISLEM_TUTARI:', train['ISLEM_TUTARI'].min())
print('Max ISLEM_TUTARI:', train['ISLEM_TUTARI'].max())
train = train.drop(['Record_Count'], axis=1)
test = test.drop(['Record_Count'], axis=1)
plt.figure(figsize=(10,10))
sns.heatmap(train.corr(), annot=True)
sektor_adet_index = train.groupby(train['SEKTOR'])['ISLEM_ADEDI'].mean().sort_values(ascending = False).index
sektor_adet_values = train.groupby(train['SEKTOR'])['ISLEM_ADEDI'].mean().sort_values(ascending = False).values
sektor_tutar_index = train.groupby(train['SEKTOR'])['ISLEM_TUTARI'].mean().sort_values(ascending = False).index
sektor_tutar_values = train.groupby(train['SEKTOR'])['ISLEM_TUTARI'].mean().sort_values(ascending = False).values

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(17,5))
sns.barplot(y=sektor_adet_index, x=sektor_adet_value, ax = ax1, palette='GnBu_d')
ax1.set_xlabel('İşlem Adedi')
sns.barplot(y=sektor_tutar_index, x=sektor_tutar_values, ax = ax2, palette='GnBu_d')
ax2.set_xlabel('İşlem Tutarı')
plt.subplots_adjust(wspace=30)
plt.tight_layout()
train['YIL_AY'] = train['YIL_AY'].apply(str).apply(lambda x: "-".join([x[:4],x[-2:]]))
train['YIL_AY'] = pd.DatetimeIndex(train['YIL_AY'])
train.groupby(train['YIL_AY'])['ISLEM_ADEDI'].mean().plot(linewidth=4, figsize=(15,6))
plt.show
train.groupby(train['YIL_AY'])['ISLEM_TUTARI'].mean().plot(linewidth=4, figsize=(15,6))
plt.show
#train = train.drop(['CUSTOMER'], axis=1)
#test = test.drop(['CUSTOMER'], axis=1)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

train_set['SEKTOR'] = le.fit_transform(train_set['SEKTOR'])
train_set['ISLEM_TURU'] = le.fit_transform(train_set['ISLEM_TURU'])

test_set['SEKTOR'] = le.fit_transform(test_set['SEKTOR'])
test_set['ISLEM_TURU'] = le.fit_transform(test_set['ISLEM_TURU'])
train.columns
y_train = train_set.ISLEM_TUTARI
X_train = train_set[['ISLEM_TUTARI', 'ISLEM_ADEDI', 'ISLEM_TURU', 'YIL_AY', 'SEKTOR',
       'Record_Count']]
X_test = test_set[['ISLEM_TUTARI', 'ISLEM_ADEDI', 'ISLEM_TURU', 'YIL_AY', 'SEKTOR',
       'Record_Count']]
lgb_train = lgb.Dataset(data=X_train, label=y_train,  free_raw_data=False)


params = {
    'task': 'train',
    'boosting_type': 'goss',
    'objective': 'regression_l2',
    'metric': 'rmse',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'verbose': -1
}


gbm = lgb.train(params,
                lgb_train)
y_pred = gbm.predict(X_test)
results = pd.DataFrame({'ID':test.ID, 'Predicted':y_pred})
results.to_csv('results2.csv', index=False)
