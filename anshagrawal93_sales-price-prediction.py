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
import pandas as pd

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

pd.set_option('display.max_columns', None)

train.head()
train.describe()
train.isnull().sum()
import seaborn as sns

sns.barplot(x = "BedroomAbvGr", y = "SalePrice", data = train)
import matplotlib.pyplot as plt

plt.style.use(style='ggplot')

plt.rcParams['figure.figsize'] = (10, 6)
train.SalePrice.describe()
plt.hist(train.SalePrice)
train.SalePrice.skew()
#target = np.log(train.SalePrice)

target = train.SalePrice

plt.hist(target)

target
target.skew()
features = train.select_dtypes(include = [np.number])

features.dtypes
corr = features.corr()

corr
print (corr['SalePrice'].sort_values(ascending=False)[:5], '\n')

print (corr['SalePrice'].sort_values(ascending=False)[-5:])
train.OverallQual.unique()
quality_pivot = train.pivot_table(index = 'OverallQual', values = 'SalePrice', aggfunc = np.median)

quality_pivot
quality_pivot.plot(kind = 'bar')
plt.scatter(train.GrLivArea, train.SalePrice)
plt.scatter(train.GarageArea, train.SalePrice)
train = train[train['GarageArea']<1200]
plt.scatter(train.GarageArea, train.SalePrice)
train.GarageCars.unique()
sns.barplot(train.GarageCars, train.SalePrice )
nulls = train.isnull().sum().sort_values(ascending = False)

nulls.columns = ['Null Count']

nulls.index.name = 'Feature'

nulls
train.MiscFeature.unique()
categorials = train.select_dtypes(exclude = [np.number])

categorials.describe()
target = train.SalePrice

train.drop('SalePrice', 1, inplace = True)
train
test.describe()
combined = train.append(test)

combined.describe()
combined.shape
combined1 = combined.select_dtypes(include = [np.number]).interpolate()

combined1.head()

combined1.shape
combined1.isnull().sum()
sns.barplot('SaleCondition', target, data = train)
def encode(x):

    return 1 if x=='Partial' else 0

combined.SaleCondition = combined.SaleCondition.apply(encode)
combined2 = pd.get_dummies(combined.select_dtypes(exclude = [np.number]))
combined2.shape
combined1.shape
combined2.isnull().sum()
result = pd.concat([combined1, combined2], axis=1)
result.head()
result = pd.concat([result, combined['SaleCondition']], axis = 1)
result.isnull().sum()
result.shape
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestRegressor



from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier

from sklearn.feature_selection import SelectKBest

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
def compute_score(clf, X, y, scoring='accuracy'):

    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)

    return np.mean(xval)
train = result.iloc[:1455]

test = result.iloc[1455:]
target = np.log(target)

target
clf = RandomForestRegressor(n_estimators=50, max_features='sqrt')

clf = clf.fit(train, target)
features = pd.DataFrame()

features['feature'] = train.columns

features['importance'] = clf.feature_importances_

features.sort_values(by=['importance'], ascending=False, inplace=True)

features.set_index('feature', inplace=True)
features.importance
features.plot(kind='barh', figsize=(25, 25))
model = SelectFromModel(clf, prefit=True)

train_reduced = model.transform(train)

print(train_reduced.shape)
test_reduced = model.transform(test)

print(test_reduced.shape)
X = train_reduced

y = target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

                          X, y, random_state=42, test_size=.33)
clf.fit(train_reduced, target)
clf.score(X_test, y_test)
predictions = clf.predict(X_test)

from sklearn.metrics import mean_squared_error

print ('RMSE is: \n', mean_squared_error(y_test, predictions))
plt.scatter(predictions, y_test)
pred = clf.predict(test_reduced)
submission = pd.DataFrame()

submission['Id'] = test.Id

pred = np.exp(pred)

submission['SalePrice'] = pred

submission.head()
submission.to_csv('submission1.csv', index=False)