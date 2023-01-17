# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn import *

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train = pd.read_csv('/kaggle/input/realestatepriceprediction/train.csv')
df_test = pd.read_csv('/kaggle/input/realestatepriceprediction/test.csv')
df_submission = pd.read_csv('/kaggle/input/realestatepriceprediction/sample_submission.csv')
df_train.head(5)
print(df_train.info())
print(df_train.shape)
df_train.describe().T
df_train[df_train['LifeSquare'].isnull()].describe().T
df_train['LifeSquare'] = df_train['LifeSquare'].fillna(0)
df_train[df_train['Healthcare_1'].isnull()].describe().T
df_train['Healthcare_1'] = df_train['Healthcare_1'].fillna(df_train['Healthcare_1'].median())
print(df_test.info())
print(df_test.shape)
df_test.describe().T
df_test[df_test['LifeSquare'].isnull()].describe().T
df_test['LifeSquare'] = df_test['LifeSquare'].fillna(df_test['LifeSquare'].median())
df_test[df_test['Healthcare_1'].isnull()].describe().T
df_test['Healthcare_1'] = df_test['Healthcare_1'].fillna(df_test['Healthcare_1'].median())
feat_numeric = list(df_train.select_dtypes(exclude='object').columns)
feat_object = list(df_train.select_dtypes(include='object').columns)
target = 'Price'
feat_numeric
feat_numeric.pop(0)
df_train[feat_numeric].hist(
    figsize=(16,16)
)
plt.show()
df_train['HouseYear'].describe()
df_train[df_train['HouseYear']>2020]
df_train.drop(
    index=df_train[df_train['HouseYear']>2020].index,
    axis=0,
    inplace=True
)
df_train[df_train['HouseYear']>2020]
df_test[df_test['HouseYear']>2020]
df_train[df_train['LifeSquare']>1000]
df_test[df_test['LifeSquare']>1000]
df_train.drop(
    index=df_train[df_train['LifeSquare']>1000].index,
    axis=0,
    inplace=True
)
df_train[df_train['LifeSquare']>1000]
df_train[df_train['KitchenSquare']>df_train['Square']]
df_train[df_train['KitchenSquare']>df_train['Square']]
df_test[df_test['KitchenSquare']>df_test['Square']]
_ = df_train[df_train['KitchenSquare']<=df_train['Square']][['Square', 'KitchenSquare']].median()
square, kitchen = _[0], _[1]
kitchen/square
df_train.loc[df_train['KitchenSquare']>df_train['Square'], 'KitchenSquare'] = \
    df_train.loc[df_train['KitchenSquare']>df_train['Square'], 'Square'] * kitchen/square
df_train[df_train['KitchenSquare']>df_train['Square']]
df_test[df_test['KitchenSquare']>df_test['Square']]
_ = df_test[df_test['KitchenSquare']<=df_test['Square']][['Square', 'KitchenSquare']].median()
square, kitchen = _[0], _[1]
df_test.loc[df_test['KitchenSquare']>df_test['Square'], 'KitchenSquare'] = \
    df_test.loc[df_test['KitchenSquare']>df_test['Square'], 'Square'] * kitchen/square
df_test[df_test['KitchenSquare']>df_test['Square']]
fig, ax = plt.subplots(
    nrows=1,
    ncols=3,
    figsize=(10, 10)
)

for n, f in enumerate(feat_object):
    
    sns.countplot(
        x=f,
        data=df_train,
        ax=ax[n]
    )
    
plt.show()
for col in feat_object:
    
    df_temp = pd.get_dummies(df_train[col])

    df_temp.columns = [col + '_' + c for c in df_temp]
    
    df_train.drop(
        labels=col,
        axis=1,
        inplace=True
    )
    
    df_train = pd.concat(
        objs=(df_train, df_temp),
        axis=1
    )
    
    del df_temp
df_train
for col in feat_object:
    
    df_temp = pd.get_dummies(df_test[col])

    df_temp.columns = [col + '_' + c for c in df_temp]
    
    df_test.drop(
        labels=col,
        axis=1,
        inplace=True
    )
    
    df_test = pd.concat(
        objs=(df_test, df_temp),
        axis=1
    )
    
    del df_temp
df_test
feat = list(df_train.columns)
feat.pop(0)
plt.figure(figsize = (20,15))

corr = df_train[feat].corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool))

sns.set(font_scale=1.4)
sns.heatmap(
    data=corr,
    mask=mask,
    annot=False
)

plt.title('Correlation matrix')
plt.show()
feat.pop(feat.index('Price'))
feat.pop(feat.index('Social_2'))

feat.pop(feat.index('Social_3'))
model = sklearn.ensemble.RandomForestRegressor(
    n_estimators=250,
    criterion='mse',
    max_depth=45,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features=.75,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    min_impurity_split=None,
    bootstrap=True,
    oob_score=True,
    n_jobs=-1,
    random_state=0,
    verbose=0,
    warm_start=False,
    ccp_alpha=0.0,
    max_samples=.75,
)

X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(
    df_train[feat],
    df_train[target]
)
model.fit(
    X=X_train,
    y=y_train
)
y_pred_train = model.predict(
    X=X_valid
)

sklearn.metrics.r2_score(
    y_true=y_valid,
    y_pred=y_pred_train
)
y_pred_test = model.predict(
    X=df_test[feat]
)
df_test[target] = y_pred_test
df_test[['Id', 'Price']].to_csv('submission.csv', index=None)