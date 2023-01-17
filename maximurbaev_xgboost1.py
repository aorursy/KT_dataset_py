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
!pip install scikit-learn
!pip install xgboost
import xgboost as xgb
from sklearn.metrics import mean_squared_error
test_features = pd.read_csv('../input/lish-moa/test_features.csv')
train_features = pd.read_csv('../input/lish-moa/train_features.csv')
train_targets_nonscored = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')
train_targets_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

ss = pd.read_csv('../input/lish-moa/sample_submission.csv')
train_features.describe()
train_features.info()
def preprocess(df):
    df.loc[:, 'cp_type'] = df.loc[:, 'cp_type'].map({'trt_cp': 0, 'ctl_vehicle': 1})
    df.loc[:, 'cp_dose'] = df.loc[:, 'cp_dose'].map({'D1': 0, 'D2': 1})
    del df['sig_id']
    return df

train = preprocess(train_features)
test = preprocess(test_features)

del train_targets_scored['sig_id']
train_features.info()
X, y = train_features.iloc[:,:-1],train_features.iloc[:,-1]
train_dmatrix = xgb.DMatrix(train_features)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)
xg_reg.fit(X_train,y_train)

preds = xg_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))
y_test.shape
from sklearn.metrics import log_loss
log_loss(y_test, preds)
