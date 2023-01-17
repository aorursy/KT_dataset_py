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
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
train.head()
test.head()
print('train shape:', train.shape, '\n','test shape:', test.shape)
missing_numeric = pd.concat([train.isnull().sum(), test.isnull().sum()], axis=1, keys=['train', 'test'])
missing_numeric = missing_numeric[(missing_numeric['train']>0) | (missing_numeric['test']>0)]
missing_numeric.sort_values(by=['train', 'test'], ascending=False)
train.isnull().sum()[train.isnull().sum()>0]
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm,skew
train = train.drop(['PoolQC','Fence','MiscFeature','Alley','FireplaceQu'], axis=1)
test = test.drop(['PoolQC','Fence','MiscFeature','Alley','FireplaceQu'], axis=1)
corr = train.corr()
plt.figure(figsize=(8,8))
sns.heatmap(corr)
plt.yticks(rotation=0, size=7)
plt.xticks(rotation=90, size=7)
plt.show()
rel_vars = corr.SalePrice[(corr.SalePrice > 0.5)]
rel_cols = list(rel_vars.index.values)

corr2 = train[rel_cols].corr()
plt.figure(figsize=(8,8))
hm = sns.heatmap(corr2, annot=True, annot_kws={'size':10})
plt.yticks(rotation=0, size=10)
plt.xticks(rotation=90, size=10)
plt.show()
X = train[rel_cols[:-1]].iloc[:,0:].values
y = train.iloc[:, -1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state = 0)
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300, random_state=0)
regressor.fit(X_train, y_train)

# Score model
regressor.score(X_train, y_train)
y_pred = regressor.predict(X_test)
# Plot y_test vs y_pred
plt.figure(figsize=(12,8))
plt.plot(y_test, color='red')
plt.plot(y_pred, color='blue')
plt.show()
from sklearn.preprocessing import LabelEncoder
from sklearn import ensemble
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn import model_selection
lb_make = LabelEncoder()

heads = train.columns
for i in range(len(train.columns)):
    if train[heads[i]].dtypes == 'O':
        train[heads[i]] = lb_make.fit_transform(train[heads[i]].astype(str))
models = []

models.append(('XGBoost',XGBRegressor()))
models.append(('GBR', ensemble.GradientBoostingRegressor(loss='quantile', alpha=0.1,
n_estimators=250, max_depth=3, learning_rate=.1, min_samples_leaf=9, min_samples_split=9)))

models.append(('RFR', RandomForestRegressor()))
results = []
names = []
seed = 7
scoring = 'r2'
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
sales=pd.DataFrame(y_pred,columns=['SalePrice'])
sample_submission['SalePrice']=sales['SalePrice']
sample_submission.head()
sample_submission.to_csv('xgb.csv',index=False)
dp = pd.read_csv('xgb.csv')
dp.head(10)