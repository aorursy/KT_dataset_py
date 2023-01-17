import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
train_initial = pd.read_csv('../input/train.csv')

test_initial = pd.read_csv('../input/test.csv')
train = train_initial.drop(columns=['LotFrontage', 'Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'])

test = test_initial.drop(columns=['LotFrontage', 'Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'])
train.head()
train.shape
train = train.fillna(method="bfill")

test = test.fillna(method="bfill")
p = sns.countplot(data=train, x="MSSubClass") 

# Type of dwelling involved in the sale

 #   20  1-STORY 1946 & NEWER ALL STYLES

 #   30  1-STORY 1945 & OLDER

 #   40  1-STORY W/FINISHED ATTIC ALL AGES

 #   45  1-1/2 STORY - UNFINISHED ALL AGES

 #   50  1-1/2 STORY FINISHED ALL AGES

 #   60  2-STORY 1946 & NEWER

 #   70  2-STORY 1945 & OLDER

 #   75  2-1/2 STORY ALL AGES

 #   80  SPLIT OR MULTI-LEVEL

 #   85  SPLIT FOYER

 #   90  DUPLEX - ALL STYLES AND AGES

 #  120  1-STORY PUD (Planned Unit Development) - 1946 & NEWER

 #  150  1-1/2 STORY PUD - ALL AGES

 #  160  2-STORY PUD - 1946 & NEWER

 #  180  PUD - MULTILEVEL - INCL SPLIT LEV/FOYER

 #  190  2 FAMILY CONVERSION - ALL STYLES AND AGES
p = sns.countplot(data=train, x="MSZoning")

# General Zoning Classification of the sale

 #  A    Agriculture

 #  C    Commercial

 #  FV   Floating Village Residential

 #  I    Industrial

 #  RH   Residential High Density

 #  RL   Residential Low Density

 #  RP   Residential Low Density Park 

 #  RM   Residential Medium Density
plt.figure(figsize=(35,25))

p = sns.heatmap(train.corr(), annot=True)
p = sns.countplot(data=train, x="OverallQual")

# Overall Quality of the house

#   10   Very Excellent

#   9    Excellent

#   8    Very Good

#   7    Good

#   6    Above Average

#   5    Average

#   4    Below Average

#   3    Fair

#   2    Poor

#   1    Very Poor
# Years in which houses were built

plt.figure(figsize=(20, 10))

p = sns.countplot(data=train, x="YearBuilt")

_ = plt.setp(p.get_xticklabels(), rotation=90)
plt.figure(figsize=(10, 5))

plt.title("Sales Price Distribution")

p = sns.distplot(train["SalePrice"], color='g')
attributes_train = ['SalePrice', 'MSSubClass', 'MSZoning', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF','1stFlrSF', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'MasVnrType', 'LotArea', 'GarageCars', 'GarageArea', 'EnclosedPorch']

attributes_test =  ['MSSubClass', 'MSZoning', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'MasVnrType', 'LotArea', 'GarageCars', 'GarageArea', 'EnclosedPorch']

train_clean = train[attributes_train]

test_clean =test[attributes_test]
train_clean.loc[:, 'MSZoning'] = train_clean.loc[:, 'MSZoning'].apply(lambda x: {'RL': 0, 'RM': 1, 'C (all)': 2, 'FV': 3, 'RH': 4}[x])

train_clean.loc[:, 'MasVnrType'] = train_clean.loc[:, 'MasVnrType'].apply(lambda x: {'BrkFace': 0, 'None': 1, 'Stone': 2, 'BrkCmn': 3, 'nan': 4}[x])

test_clean.loc[:, 'MSZoning'] = test_clean.loc[:, 'MSZoning'].apply(lambda x: {'RL': 0, 'RM': 1, 'C (all)': 2, 'FV': 3, 'RH': 4}[x])

test_clean.loc[:, 'MasVnrType'] = test_clean.loc[:, 'MasVnrType'].apply(lambda x: {'BrkFace': 0, 'None': 1, 'Stone': 2, 'BrkCmn': 3, 'nan': 4}[x])
from sklearn.model_selection import train_test_split

train_df = train_clean.drop(columns=['SalePrice'])
X_train, X_test, Y_train, Y_test = train_test_split(train_df, train['SalePrice'], test_size=0.1, random_state=42)
from sklearn import ensemble
model = ensemble.GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=3, max_features='sqrt',

                                               min_samples_leaf=15, min_samples_split=10, loss='huber')
model.fit(X_train, Y_train)
from sklearn import tree
model_tree = tree.DecisionTreeRegressor()
model_tree.fit(X_train, Y_train)
model.score(X_test, Y_test)
model_tree.score(X_test, Y_test)
model.fit(X_train, Y_train)

model_tree.fit(X_train, Y_train)
predictions = model.predict(test_clean)
submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predictions})

submission.to_csv('submission.csv', index=False)