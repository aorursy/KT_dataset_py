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

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

description = pd.read_table("../input/house-prices-advanced-regression-techniques/data_description.txt", delim_whitespace=True, error_bad_lines=False)
train.columns
train['SalePrice'].describe()
sns.distplot(train['SalePrice']);
corrmat = train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
k = 10 

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.75)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
train_numeric = train.select_dtypes(include='number')

train_numeric
sns.distplot(train_numeric['SalePrice'], bins=50, kde=True, rug=True)
train_non_numeric = train.select_dtypes(exclude='number')





plt.figure(figsize=(25,7))

sns.countplot(x="SaleCondition", data=train_non_numeric)
train_non_numeric = train.select_dtypes(exclude='number')





plt.figure(figsize=(25,7))

sns.countplot(x="HouseStyle", data=train_non_numeric)
train_non_numeric = train.select_dtypes(exclude='number')





plt.figure(figsize=(25,7))

sns.countplot(x="Functional", data=train_non_numeric)
train_non_numeric = train.select_dtypes(exclude='number')





plt.figure(figsize=(25,7))

sns.countplot(x="LandContour", data=train_non_numeric)
import math

def plot_multiple_countplots(df, cols):

    num_plots = len(cols)

    num_cols = math.ceil(np.sqrt(num_plots))

    num_rows = math.ceil(num_plots/num_cols)

        

    fig, axs = plt.subplots(num_rows, num_cols)

    

    for ind, col in enumerate(cols):

        i = math.floor(ind/num_cols)

        j = ind - i*num_cols

        

        if num_rows == 1:

            if num_cols == 1:

                sns.countplot(x=df[col], ax=axs)

            else:

                sns.countplot(x=df[col], ax=axs[j])

        else:

            sns.countplot(x=df[col], ax=axs[i, j])

            

            

plot_multiple_countplots(train_non_numeric, ['LandContour', 'HouseStyle', 'Functional', 'SaleCondition'])
sns.relplot(x='YearBuilt', y='SalePrice', data=train, aspect=2.0)
train['SaleCondition'].value_counts()
train['HouseStyle'].value_counts()
train['RoofStyle'].value_counts()
train['Functional'].value_counts()
train['SaleType'].value_counts()
train_roof_year = train.groupby(['RoofStyle', 'SalePrice'])['YearBuilt'].count().reset_index()

train_roof_year_pivot = train_roof_year.pivot(index='RoofStyle', columns='SalePrice', values='YearBuilt').fillna(0)

sns.heatmap(train_roof_year_pivot, annot=True, fmt='.0f', cmap="YlGnBu")
cols = ['YearBuilt', 'RoofStyle', 'HouseStyle', 'SaleType', 'Functional', 'SaleCondition', 'SalePrice']

train_test = train[cols]

train_test.head()
numeric_columns = set(train_test.select_dtypes(include=['number']).columns)

non_numeric_columns = set(train_test.columns) - numeric_columns

print(numeric_columns)

print(non_numeric_columns)
for c in non_numeric_columns:

    cnt = train_test[c].value_counts()

    small_cnts = list(cnt[cnt < 5].index)

    

    s_replace = {}

    for sm in small_cnts:

        s_replace[sm] = 'other'

    

    train_test[c] = train_test[c].replace(s_replace)

    train_test[c] = train_test[c].fillna('other')
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

from sklearn.model_selection import cross_val_score



# we are going to look at feature importances so we like putting random features to act as a benchmark.

train_test['rand0'] = np.random.rand(train_test.shape[0])

train_test['rand1'] = np.random.rand(train_test.shape[0])

train_test['rand2'] = np.random.rand(train_test.shape[0])



# testing for relationships.

# for numeric targets.

reg = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, loss='ls', random_state=1)

# for categorical targets.

clf = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, loss='deviance', random_state=1)



train_test['YearBuilt'] = train_test['YearBuilt'].fillna(0) # only YearBuilt should have missing values.

        

# try to predict one feature using the rest of others to test collinearity, so it's easier to interpret the results

for c in cols:

    # c is the thing to predict.

    

    if c not in ['rand0', 'rand1', 'rand2']: 



        X = train_test.drop([c], axis=1) # drop the thing to predict.

        X = pd.get_dummies(X)

        y = train_test[c]



        print(c)



        if c in non_numeric_columns:

            scoring = 'accuracy'

            model = clf

            scores = cross_val_score(clf, X, y, cv=5, scoring=scoring)

            print(scoring + ": %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        elif c in numeric_columns:

            scoring = 'neg_root_mean_squared_error'

            model = reg

            scores = cross_val_score(reg, X, y, cv=5, scoring=scoring)

            print(scoring.replace('neg_', '') + ": %0.2f (+/- %0.2f)" % (-scores.mean(), scores.std() * 2))

        else:

            print('what is this?')



        model.fit(X, y)

        train_importances = pd.DataFrame(data={'feature_name': X.columns, 'importance': model.feature_importances_}).sort_values(by='importance', ascending=False)

        top5_features = train_importances.iloc[:5]

        print('top 5 features:')

        print(top5_features)



        print()
# SaleType, SaleCondition



sns.relplot(x='SaleCondition', y='SaleType', size='SalePrice', sizes=(10, 1000), data=train, aspect=3.0)
missing = train.isnull().sum()

missing = missing[missing > 0]

missing.sort_values(inplace=True)

missing.plot.bar()
train.shape
test.shape
pd.set_option("display.max_columns", 81)
train.isna().sum().sum()
dataframe_train = train.drop('Id',1)

Y_train = dataframe_train['SalePrice']

df_features_train = dataframe_train.drop('SalePrice',1)



print(Y_train.shape, df_features_train.shape)
df_features_test = test.drop('Id',1)



print(df_features_test.shape)
na_total = df_features_train.isnull().sum().sort_values(ascending=False) ## FInding total null values

df_features_train.fillna(0,inplace=True)
na_total = df_features_test.isnull().sum().sort_values(ascending=False) ## FInding total null values

df_features_test.fillna(0,inplace=True)
numeric_cols = [x for x in df_features_train.columns if ('Area' in x) | ('SF' in x)]+['LotFrontage','MiscVal','EnclosedPorch','3SsnPorch','ScreenPorch','OverallQual','OverallCond','YearBuilt']
categorical_cols = [x for x in df_features_train.columns if x not in numeric_cols]
numeric_cols_test = [x for x in df_features_test.columns if ('Area' in x) | ('SF' in x)]+['LotFrontage','MiscVal','EnclosedPorch','3SsnPorch','ScreenPorch','OverallQual','OverallCond','YearBuilt']
categorical_cols_test = [x for x in df_features_test.columns if x not in numeric_cols_test]
train_num = df_features_train[numeric_cols]

train_cat = df_features_train[categorical_cols]
test_num = df_features_test[numeric_cols_test]

test_cat = df_features_test[categorical_cols_test]
Y_train = np.log(Y_train)

Y_train.hist()
train = pd.concat([train_cat,train_num],axis=1)

train.shape
test = pd.concat([test_cat,test_num],axis=1)

test.shape
train_objs_num = len(train)

print(train_objs_num)
dataset = pd.concat(objs=[train, test], axis=0)

dataset = pd.get_dummies(dataset)



train = dataset[:train_objs_num]

test = dataset[train_objs_num:]

print(train.shape,test.shape)
from sklearn import datasets, linear_model

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.impute import SimpleImputer
model = linear_model.LinearRegression()

model.fit(train,Y_train)
prediction =  model.predict(test)

final_prediction = np.exp(prediction)
print(final_prediction)
main_file_path = '../input/house-prices-advanced-regression-techniques/test.csv'

dataframe = pd.read_csv(main_file_path)

dataframe.head()

dataframe.info()
dataframe['Id']
Submission = pd.DataFrame()

Submission['Id'] = dataframe.Id

Submission.info()
Submission['SalePrice'] = final_prediction
print(Submission.head())
Submission.to_csv('submission.csv', index=False)