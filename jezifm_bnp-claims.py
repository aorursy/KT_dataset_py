import numpy as np
import pandas as pd
import xgboost as xgb

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('../input/train.csv')
n_observation = train.shape[0]
n_features = train.shape[1]
n_missing_values = train.apply(lambda x: x.isnull().any()).sum()

print('%s Observations, %s Features, %s features with missing values' % (n_observation, n_features, n_missing_values))
print('114 numeric features')
print('19 string features')
print(train.dtypes.value_counts())
# target variable
pred_col = 'target'
str_cols = train.columns[train.dtypes=='object']
num_cols = train.columns[(train.dtypes=='int64') | (train.dtypes=='float64')]
sns.countplot(train.target, order=[0, 1])
sns.plt.title('Target Variable Distribution')
train[str_cols] = train[str_cols].apply(lambda x: x.factorize()[0])
x_train = train.drop(pred_col, axis=1)
y_train = train[pred_col]

bst = xgb.XGBClassifier()
bst.fit(x_train, y_train)
sample_submission = pd.read_csv('../input/sample_submission.csv')
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
data = train.append(test)
data[str_cols] = data[str_cols].apply(lambda x: x.factorize()[0])

train_mask = data[pred_col].notnull()
x_train = data[train_mask].drop(pred_col, axis=1)
y_train = data[train_mask][pred_col]
x_test = data[~train_mask].drop(pred_col, axis=1)

bst.fit(x_train, y_train)
pred = bst.predict_proba(x_test)
pred