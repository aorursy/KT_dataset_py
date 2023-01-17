# Data Processing

import numpy as np 

import pandas as pd

pd.set_option('display.max_columns', None)

pd.set_option("max_rows", None)



# Data Visualization

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set(style='whitegrid')



# Modeling

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold



from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor



from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error

from math import sqrt



from sklearn.model_selection import RandomizedSearchCV
train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

sample_submission = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")
train.head()
train.shape
test.head()
test.shape
plt.figure(figsize=(20,10))

b = sns.distplot(train['SalePrice'])

b.set_title("SalePrice Distribution");
plt.figure(figsize=(20,10))

b = sns.boxplot(y = 'SalePrice', data = train)

b.set_title("SalePrice Distribution");
len(train[train['SalePrice'] > 700000])
train.shape
train = train[train['SalePrice'] <= 700000]
train.shape
train.columns[train.isna().any()].tolist()
test.columns[test.isna().any()].tolist()
#missing data

total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
train = train.drop(['PoolQC'], axis=1)

test = test.drop(['PoolQC'], axis=1)



train = train.drop(['MiscFeature'], axis=1)

test = test.drop(['MiscFeature'], axis=1)



train = train.drop(['Alley'], axis=1)

test = test.drop(['Alley'], axis=1)



train = train.drop(['Fence'], axis=1)

test = test.drop(['Fence'], axis=1)



train = train.drop(['FireplaceQu'], axis=1)

test = test.drop(['FireplaceQu'], axis=1)



train = train.drop(['LotFrontage'], axis=1)

test = test.drop(['LotFrontage'], axis=1)
train = train.fillna(train.median())

test = test.fillna(test.median())
test['MSZoning'] = test['MSZoning'].fillna('None')
train = train.drop(['Utilities'], axis=1)

test = test.drop(['Utilities'], axis=1)
test['Exterior1st'] = test['Exterior1st'].fillna('None')



train.loc[train['Exterior1st'].value_counts()[train['Exterior1st']].values < 18,'Exterior1st'] = 'Rare'

test.loc[test['Exterior1st'].value_counts()[test['Exterior1st']].values < 18,'Exterior1st'] = 'Rare'
test['Exterior2nd'] = test['Exterior2nd'].fillna('None')



train.loc[train['Exterior2nd'].value_counts()[train['Exterior2nd']].values < 10,'Exterior2nd'] = 'Rare'

test.loc[test['Exterior2nd'].value_counts()[test['Exterior2nd']].values < 10,'Exterior2nd'] = 'Rare'
train['MasVnrType'] = train['MasVnrType'].fillna('Missing')

test['MasVnrType'] = test['MasVnrType'].fillna('Missing')
train['BsmtQual'] = train['BsmtQual'].fillna('None')

test['BsmtQual'] = test['BsmtQual'].fillna('None')
train['BsmtCond'] = train['BsmtCond'].fillna('None')

test['BsmtCond'] = test['BsmtCond'].fillna('None')
train['BsmtExposure'] = train['BsmtExposure'].fillna('None')

test['BsmtExposure'] = test['BsmtExposure'].fillna('None')
train['BsmtFinType1'] = train['BsmtFinType1'].fillna('None')

test['BsmtFinType1'] = test['BsmtFinType1'].fillna('None')
train['BsmtFinType2'] = train['BsmtFinType2'].fillna('None')

test['BsmtFinType2'] = test['BsmtFinType2'].fillna('None')
train['Electrical'] = train['Electrical'].fillna('None')

test['Electrical'] = test['Electrical'].fillna('None')
test['KitchenQual'] = test['KitchenQual'].fillna('None')
test['Functional'] = test['Functional'].fillna('None')
train['GarageType'] = train['GarageType'].fillna('None')

test['GarageType'] = test['GarageType'].fillna('None')
train['GarageFinish'] = train['GarageFinish'].fillna('None')

test['GarageFinish'] = test['GarageFinish'].fillna('None')
train['GarageQual'] = train['GarageQual'].fillna('None')

test['GarageQual'] = test['GarageQual'].fillna('None')
train['GarageCond'] = train['GarageCond'].fillna('None')

test['GarageCond'] = test['GarageCond'].fillna('None')
train['SaleType'] = train['SaleType'].fillna('None')

test['SaleType'] = test['SaleType'].fillna('None')
train.isna().all().sum()
test.isna().all().sum()
y_train = train['SalePrice'].values

df = pd.concat((train, test)).reset_index(drop=True)

df.drop(['SalePrice'], axis=1, inplace=True)
from sklearn.preprocessing import LabelEncoder

cols = ('BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')

# process columns, apply LabelEncoder to categorical features

for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(df[c].values)) 

    df[c] = lbl.transform(list(df[c].values))

corrmat = train.corr()

plt.subplots(figsize=(12,9))

sns.heatmap(corrmat, vmax=0.9, square=True);
df = pd.get_dummies(df)

print(df.shape)
train = df[df['Id'] < 1461]

test = df[df['Id'] >= 1461]
# Everything except target variable

X = train



# Target variable

y = y_train
# Random seed for reproducibility

np.random.seed(42)



# Split into train & test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
# Put models in a dictionary

models = {"ElasticNet": ElasticNet(tol=0.1),

          "Lasso": Lasso(tol=0.1), 

          "BayesianRidge": BayesianRidge(n_iter=1000),

          "LassoLarsIC" : LassoLarsIC(max_iter=45),

          "RandomForestRegressor" : RandomForestRegressor(),

          "GradientBoostingRegressor" : GradientBoostingRegressor(),

          "XGBRegressor": XGBRegressor(),

          "LGBMRegressor": LGBMRegressor()

}



# Create function to fit and score models

def fit_and_score(models, X_train, X_test, y_train, y_test):

    """

    Fits and evaluates given machine learning models.

    models : a dict of different Scikit-Learn machine learning models

    X_train : training data

    X_test : testing data

    y_train : labels assosciated with training data

    y_test : labels assosciated with test data

    """

    # Random seed for reproducible results

    np.random.seed(42)

    # Make a list to keep model scores

    model_scores = {}

    # Loop through models

    for name, model in models.items():

        # Fit the model to the data

        model.fit(X_train, y_train)

        # Predicting target values

        y_pred = model.predict(X_test)

        # Evaluate the model and append its score to model_scores

        model_scores[name] = np.sqrt(mean_squared_error(y_test, y_pred))

    return model_scores
model_scores = fit_and_score(models=models,

                             X_train=X_train,

                             X_test=X_test,

                             y_train=y_train,

                             y_test=y_test)

model_scores
np.random.seed(42)



gbr = GradientBoostingRegressor(n_estimators=5000)

gbr.fit(X_train, y_train)

y_pred = gbr.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_pred))
# Plotting the Train and Test scores

plt.figure(figsize=(20,10))

plt.plot(y_test, label="True values")

plt.plot(y_pred, label="Predicted values")

plt.xticks(np.arange(1, 51, 1))

plt.xlabel("Object")

plt.ylabel("Values")

plt.legend();
y_pred = gbr.predict(test)
sample_submission.head()
sample_submission['SalePrice'] = y_pred

sample_submission.to_csv("submission.csv", index=False)

sample_submission.head()