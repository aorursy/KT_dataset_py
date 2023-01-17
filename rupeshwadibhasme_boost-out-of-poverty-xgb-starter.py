# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os,operator
#import gc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import describe
%matplotlib inline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_squared_error
#import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')
test_df.shape
test_df = pd.read_csv('../input/test.csv')
test_df.shape
plt.figure(figsize=(12, 5))
plt.hist(train_df.Target.values, bins=4)
plt.title('Histogram target counts')
plt.xlabel('Count')
plt.ylabel('Target')
plt.show()
columns_to_use = train_df.columns[1:-1]
y = train_df['Target'].values
train_test_df = pd.concat([train_df[columns_to_use], test_df[columns_to_use]], axis=0)
cols = [f_ for f_ in train_test_df.columns if train_test_df[f_].dtype == 'object']
for col in cols:
    le = LabelEncoder()
    le.fit(train_test_df[col].astype(str))
    train_df[col] = le.transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))
del le
train_df.drop(['Id','Target'],axis=1,inplace=True)
test_df.drop(['Id'],axis=1,inplace=True)
# Check Final Shape of training and test data
scaler = MinMaxScaler()

df_train_scaled = pd.DataFrame(scaler.fit_transform(train_df), columns=train_df.columns)
df_test_scaled = pd.DataFrame(scaler.fit_transform(test_df), columns=test_df.columns)
print (df_train_scaled.shape)
print (df_test_scaled.shape)
params = {
    'min_child_weight': 10.0,
    'objective': 'multi:softmax',
    'max_depth': 7,
    'max_delta_step': 1.8,
    'colsample_bytree': 0.4,
    'subsample': 0.8,
    'eta': 0.025,
    'gamma': 0.65,
    'num_boost_round' : 700,
    'num_class' : 5
    }

# These can be modified and tuned further
X_train, X_valid, y_train, y_valid = train_test_split(
       df_train_scaled, y, test_size=0.33, random_state=42)
# Convert our data into XGBoost format
d_train = xgb.DMatrix(X_train, y_train)
d_valid = xgb.DMatrix(X_valid, y_valid)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    # Train the model! We pass in a max of 1,600 rounds (with early stopping after 70)
    # and the custom metric (maximize=True tells xgb that higher metric is better)
model = xgb.train(params, d_train, 1600, watchlist, early_stopping_rounds=70, verbose_eval=100)
d_test = xgb.DMatrix(df_test_scaled)
preds = model.predict(d_test)
preds
sample_submission = pd.read_csv("../input/sample_submission.csv")
sample_submission.head()
preds = [*map(int, preds)]
sample_submission['Target'] = preds
sample_submission.to_csv('sample_xgb_submission.csv', index=False)
sample_submission.head()
