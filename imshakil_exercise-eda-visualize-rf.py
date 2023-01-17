import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from IPython.display import display



# reading data input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# train data

train_data = pd.read_csv('/kaggle/input/train.csv')



# test data

test_data = pd.read_csv('/kaggle/input/test.csv')
# overview train data

print(train_data.shape)

display(train_data.head(3))
# overview test data

print(test_data.shape)

display(test_data.head(3))
# finiding data types

print(train_data.dtypes.unique())

#remove Id column

train_data.pop('Id')

y = train_data['SalePrice']
# numerical column

numerical_col = train_data.select_dtypes(exclude='object').columns.values.tolist()

print(len(numerical_col))

display(numerical_col)

# categorical column

categorical_col = train_data.select_dtypes(include='object').columns.values.tolist()

print(len(categorical_col))

display(categorical_col)
# plotting SalePrice

from scipy import stats

plt.figure(1); plt.title('Johnson SU')

sns.distplot(y, kde=False, fit=stats.johnsonsu)

plt.figure(2); plt.title('Normal')

sns.distplot(y, kde=False, fit=stats.norm)

plt.figure(3); plt.title('Log Normal')

sns.distplot(y, kde=False, fit=stats.lognorm)
# test normality for each numerical columns

test_normality = lambda x: stats.shapiro(x.fillna(0))[1] < 0.01

normal = pd.DataFrame(train_data[numerical_col])

normal = normal.apply(test_normality)

print(not normal.any())
# count missing values

missing = train_data.isnull().sum()

columns_with_missing_rows = missing[missing > 0]

columns_with_missing_rows.sort_values(inplace=True)

f, ax = plt.subplots(figsize=(12, 5))

columns_with_missing_rows.plot.bar(ax=ax)

plt.show()
pd.set_option('display.max_rows', 100)

print(len(columns_with_missing_rows))

display(columns_with_missing_rows)
# print columns with missing rows

display(train_data[numerical_col].isnull().sum())
display(train_data.LotFrontage.describe())

sns.kdeplot(train_data.LotFrontage.describe(), label='Lotfrontage', cbar=True, shade=True)
train_data['LotFrontage'].replace({np.nan:train_data.LotFrontage.median()}, inplace=True)
display(train_data.MasVnrArea.describe())

sns.kdeplot(train_data.MasVnrArea, label='MasVnrArea', cbar=True, shade=True)
train_data['MasVnrArea'].replace({np.nan:0}, inplace=True)
display(train_data.GarageYrBlt.describe())

sns.kdeplot(train_data.GarageYrBlt, label='GarageYrBlt', cbar=True, shade=True)
train_data['GarageYrBlt'].replace({np.nan:train_data.GarageYrBlt.mean()}, inplace=True)
"""

for col in categorical_col:

    train_data[col] = train_data[col].astype('category')

    if train_data[col].isnull().any():

        train_data[col] = train_data[col].cat.add_categories(['MISSING'])

        train_data[col] = train_data[col].fillna('MISSING')



def boxplot(x, y, **kwargs):

    sns.boxplot(x=x, y=y)

    x=plt.xticks(rotation=90)

f = pd.melt(train_data, id_vars=['SalePrice'], value_vars=categorical_col)

g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, height=5)

g = g.map(boxplot, "value", "SalePrice")

"""
display(train_data[categorical_col].isnull().sum())
def replacing_missing_category(column, value):

    train_data[column].replace({np.nan: value}, inplace=True)
display(train_data.Alley.describe())

sns.countplot(train_data['Alley'])
replacing_missing_category('Alley', 'NA')
display(train_data.MasVnrType.describe())

sns.countplot(train_data.MasVnrType)
replacing_missing_category('MasVnrType', 'None')
basement_cols = ['BsmtQual','BsmtCond','BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']



for col in basement_cols:

    sns.countplot(train_data[col])

    plt.show()
for col in basement_cols:

    replacing_missing_category(col, train_data[col].describe().top)
display(train_data.Electrical.describe())

sns.countplot(train_data['Electrical'])
# replacing with top category



replacing_missing_category('Electrical', train_data['Electrical'].describe().top)
garage_cols = ['GarageType','GarageFinish', 'GarageQual', 'GarageCond']

for col in garage_cols:

    sns.countplot(train_data[col])

    plt.show()
# replacing with top most category for each column

for col in garage_cols:

    replacing_missing_category(col, train_data[col].describe().top)
cols = ['FireplaceQu','PoolQC','Fence','MiscFeature']

for col in cols:

    sns.countplot(train_data[col])

    plt.show()
for col in cols:

    replacing_missing_category(col, 'NA')
# checking is there any missing value 

train_data.columns[train_data.isnull().any()]
#data splitting

X = train_data[numerical_col]

y = X.pop('SalePrice')

print(X.shape, y.shape)
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2



#choosing k=30 best features

best_fit = SelectKBest(chi2, k=30).fit(X, y)

dfScores = pd.DataFrame(best_fit.scores_)

dfColumns = pd.DataFrame(X.columns)



features_score = pd.concat([dfColumns, dfScores], axis=1)

features_score.head()
# naming columns of features_score

features_score.columns = ['Class', 'Score']

features_score.head()
#final features

features = list(features_score.Class[:30])

print(len(features))
# update X data frame with features

X = X[features]

X.shape
from sklearn import preprocessing

X = preprocessing.StandardScaler().fit(X).transform(X)
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor



train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=4)

forest_model = RandomForestRegressor(random_state=1)

forest_model.fit(X, y)

predictions = forest_model.predict(val_X)

print(mean_absolute_error(val_y, predictions))
from sklearn import linear_model

regr = linear_model.LinearRegression()

regr.fit(X, y)

regr_preds = regr.predict(val_X)

print(mean_absolute_error(val_y, regr_preds))
test_data.isnull().sum()
test_X = test_data.select_dtypes(exclude='object')

test_X.shape
test_X.isnull().sum()
test_X['GarageYrBlt'].replace({np.nan:test_X.GarageYrBlt.mean()}, inplace=True)
test_X['MasVnrArea'].replace({np.nan:0}, inplace=True)
test_X['LotFrontage'].replace({np.nan: test_X.LotFrontage.median()}, inplace=True)
cols = test_X.columns[test_X.isnull().any()].values.tolist()

cols
for col in cols:

    display(test_X[col].describe())
for col in cols:

    test_X[col].replace({np.nan:test_X[col].median()}, inplace=True)
test_X = test_X[features]

test_X.isnull().sum()
test_X = preprocessing.StandardScaler().fit(test_X).transform(test_X)
test_predictions = forest_model.predict(test_X)

# output for submission

output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': test_predictions})

output.head()
# output to csv

output.to_csv('submission.csv', index=False)