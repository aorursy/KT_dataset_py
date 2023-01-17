import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

train
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

test
train_number_columns = train.select_dtypes(include=np.number).columns

train_category_columns = train.select_dtypes(include='category').columns

train_object_columns = train.select_dtypes(include='object').columns



test_number_columns = test.select_dtypes(include=np.number).columns

test_category_columns = test.select_dtypes(include='category').columns

test_object_columns = test.select_dtypes(include='object').columns

print("Number columns: ",train_number_columns)

print("\n----------------------------------------\n")

print("Categorical Columns:", train_category_columns)

print("\n----------------------------------------\n")

print("Object Columns:", train_object_columns)

print("\n----------------------------------------\n")

print("Length of Train and Test are: {0}, {1}".format(train.shape[0], test.shape[0]))
fig, axs = plt.subplots(11,4, figsize=(40,80))



for ax, col in zip(axs.flatten(), train_object_columns[1:]):

    obj_summary_df = pd.DataFrame()

    obj_summary_df = pd.DataFrame(test[col].value_counts()).reset_index()

    obj_summary_df['From'] = 'Train'

    obj_summary_df = obj_summary_df.append(train[col].value_counts().reset_index(), sort=True).fillna('Test')

    sns.barplot(x=obj_summary_df['index'], y=obj_summary_df[col], hue=obj_summary_df['From'], ax=ax)

    ax.set_xlabel(col, fontsize=20)

    

fig.show()
fig, axs = plt.subplots(11,4, figsize=(40,80))



for ax, col in zip(axs.flatten(), train_object_columns[1:]):

    sns.boxplot(x=train[col], y=train['SalePrice'], ax=ax)

    ax.set_xlabel(col, fontsize=20)

    

fig.show()
fig, axs = plt.subplots(6,6, figsize=(40,30))



for ax, col in zip(axs.flatten(), train_number_columns[1:-1]):

    sns.distplot(train[col].dropna(), hist=False, ax=ax)

    sns.distplot(test[col].dropna(), hist=False, ax=ax,  kde_kws = {'linewidth': 3},

                 label = "Test {}".format(col))

    ax.set_xlabel(col,fontsize=18)

    

fig.show()
unsimilar_dist_cols = ['BsmtFinSF2', 'LowQualFinSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea']

sns.heatmap(train[unsimilar_dist_cols].corr(), annot=True, fmt='.2f')

plt.show()
plt.figure(figsize=(30,20))

sns.set(font_scale=0.75)

sns.heatmap(train.corr(), annot=True, cmap="YlGnBu", fmt='.2f')

plt.show()
train.corr()[np.abs(train.corr()['SalePrice']) > 0.2]['SalePrice'].index

low_corr_cols = train.corr()[np.abs(train.corr()['SalePrice']) < 0.2]['SalePrice'].index
train.isna().sum()[train.isna().sum() > 0]
many_na_cols = ['Alley','FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']
obj_col = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']



def feature_engineer(obj_col, df):

    for col in obj_col:

        df[col].fillna('NA', inplace=True)

    

    df['MasVnrType'].fillna('None', inplace=True)

    df.loc[train['MasVnrType'] == 'None', ['MasVnrArea']] = 0 

    df['MasVnrArea'].fillna(df['MasVnrArea'].median(), inplace=True)

    df['Electrical'].fillna(df['Electrical'].mode()[0], inplace=True)

    df['GarageYrBlt'].fillna(-1, inplace=True)

    df['LotFrontage'].fillna(df['LotFrontage'].median(), inplace=True)

    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

    

    return df

    

train = feature_engineer(obj_col, train)

test = feature_engineer(obj_col, test)



drop_cols = list(many_na_cols) + unsimilar_dist_cols + list(low_corr_cols[1:])

drop_cols = list(set(drop_cols))



for cols in drop_cols:

    train.drop(columns=cols, inplace=True)

    test.drop(columns=cols, inplace=True)

    

col_category = train.select_dtypes(include='object').columns



for col in col_category:

    train[col] = train[col].astype('category').cat.codes

    test[col] = test[col].astype('category').cat.codes
train['LogSalePrice'] = np.log(train['SalePrice'])
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score

from catboost import CatBoostRegressor

from sklearn.ensemble import VotingRegressor



X = train.drop(columns=['SalePrice', 'LogSalePrice'])

y = train['LogSalePrice']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)



xgb = XGBRegressor()

#xgb.fit(X_train, y_train)

#xgb_y_pred = xgb.predict(X_test)



rfr = RandomForestRegressor()

#rfr.fit(X_train, y_train)

#rfr_y_pred = rfr.predict(X_test)



cat = CatBoostRegressor(verbose=False)

#cat.fit(X_train, y_train)

#cat_y_pred = cat.predict(X_test)



vr = VotingRegressor([('xgb', xgb), ('rfr', rfr), ('cat', cat)])

vr = vr.fit(X_train, y_train)

y_pred = vr.predict(X_test)



def get_prediction_scores(y_test, y_pred):

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    r2 = r2_score(y_test, y_pred)

    print("The rmse is: {}".format(rmse))

    print("The r2 score is: {}".format(r2))

    



get_prediction_scores(y_test, y_pred)
pred = vr.predict(test.fillna(test.median()))

submission = pd.DataFrame()

submission['Id'] = test['Id']

submission['SalePrice'] = np.exp(pred)

submission.to_csv('../working/submission.csv', index=False)