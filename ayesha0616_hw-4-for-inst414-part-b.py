import numpy as np

import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score

from sklearn.metrics import precision_score, recall_score

from sklearn import preprocessing

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import skew 

%matplotlib inline



import csv

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

train.head()
test.head()
test.describe()
train.describe()
test.dtypes

train.dtypes
#summary of SalesPrice

train['SalePrice'].describe()
sns.distplot(train['SalePrice']);
corr = train.corr()

corr.sort_values(["SalePrice"], ascending = False, inplace = True)

print(corr.SalePrice)
# We will see if there's a relationship between SalesPrice and OverallQual: overall material and finish of the house.

# 1-10 where 1=Very Poor and 10=Very Excellent

sns.barplot(train.OverallQual,train.SalePrice)

#scatter plot 

data = pd.concat([train['SalePrice'], train['GrLivArea']], axis=1)

data.plot.scatter(x='GrLivArea', y='SalePrice');
#scatter plot 

data = pd.concat([train['SalePrice'], train['TotalBsmtSF']], axis=1)

data.plot.scatter(x='TotalBsmtSF', y='SalePrice');


#scatter plot 

data = pd.concat([train['SalePrice'], train['YearBuilt']], axis=1)

data.plot.scatter(x='YearBuilt', y='SalePrice');
sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(train[cols], size = 2.5)

plt.show();
# For train

train_null = train.isnull().sum()

train_null = train_null[train_null>0]

train_null.sort_values(ascending=False)
# For test

test_null = test.isnull().sum()

test_null = test_null[test_null>0]

test_null.sort_values(ascending=False)
categorical_features = train.select_dtypes(include=['object']).columns

categorical_features
numerical_features = train.select_dtypes(exclude = ["object"]).columns

numerical_features
# Differentiate numerical features (minus the target) and categorical features

categorical_features = train.select_dtypes(include = ["object"]).columns

numerical_features = train.select_dtypes(exclude = ["object"]).columns

numerical_features = numerical_features.drop("SalePrice")

print("Numerical features : " + str(len(numerical_features)))

print("Categorical features : " + str(len(categorical_features)))

train_num = train[numerical_features]

train_cat = train[categorical_features]
# Now we will do the remaining missing values for numerical features by using median as replacement

print("NAs for numerical features in train : " + str(train_num.isnull().values.sum()))

train_num = train_num.fillna(train_num.median())

print("Remaining NAs for numerical features in train : " + str(train_num.isnull().values.sum()))
skewness = train_num.apply(lambda x: skew(x))

skewness.sort_values(ascending=False)
skewness = skewness[abs(skewness)>0.5]

skewness.index
skew_features = train[skewness.index]

skew_features.columns
skew_features = np.log1p(skew_features)

train_cat.shape 
train_cat = pd.get_dummies(train_cat)

train_cat.shape
train_cat.head()
str(train_cat.isnull().values.sum())

# now we can see there is no null values. 
# pulling data into  the target (y) which is the SalePrice and predictors (X)

train_y = train.SalePrice

pred_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']
#prediction data

train_x = train[pred_cols]



model =  LogisticRegression()



model.fit(train_x, train_y)
# pulling same columns from the test data

test_x = test[pred_cols]

pred_prices = model.predict(test_x)

print(pred_prices)
#save file

ayesha_submission2 = pd.DataFrame({'Id': test.Id, 'SalePrice' : pred_prices})

ayesha_submission2.to_csv('submission.csv', index=False)