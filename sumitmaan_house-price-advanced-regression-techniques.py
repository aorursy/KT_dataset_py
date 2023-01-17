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
import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns 
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

# data_des = open('/kaggle/input/house-prices-advanced-regression-techniques/data_description.txt', 'r')

print('Training Data Shape:{} \n Test Data Shape: {}'.format(train.shape, test.shape))

train.head()
def get_unwanted_data(df, null = False, zero = False):

    if null == True:

        df_missing = df.isnull().sum()[df.isnull().sum() >0]

        df_missing_percent = df_missing.sort_values(ascending=False)/len(train)*100

        

        ## Find out columns with more than 50% missing data , we are going to drop these columns

    

        more_than_50_missing = df_missing_percent[df_missing_percent>=50].index.tolist()

        

    if zero == True:

        df_zero = df.isin([0]).sum()[df.isin([0]).sum() >0]

        df_zero_percent = df_zero.sort_values(ascending=False)/len(train)*100

        

        ## Find out columns with more than 90% zero values data , we are going to drop these columns

    

        more_than_90_zero = df_zero_percent[df_zero_percent>=90].index.tolist()

        

    return df_missing_percent, more_than_50_missing, more_than_90_zero



train_missing_percent, train_cols_50, train_zero_90 = get_unwanted_data(train, null = True, zero = True )

test_missing_percent, test_cols_50, test_zero_90  = get_unwanted_data(test, null = True, zero = True)



## List down the columns with more than 50% Missing data or more than 90% Zero valued data and drop these columns



cols_to_drop = list(set(train_cols_50 + test_cols_50 + train_zero_90 + test_zero_90))

train.drop(cols_to_drop, axis=1, inplace=True)

test.drop(cols_to_drop,axis=1, inplace=True)



## Look for the remaining columns with missing data less than 50% 



train_missing_percent = train_missing_percent[~train_missing_percent.index.isin(cols_to_drop)]

test_missing_percent = test_missing_percent[~test_missing_percent.index.isin(cols_to_drop)]

print(train_missing_percent.head(6))



## Create a seprate dataframe of the columns with missing data/ Just to deal with only these columns 

train_missing_df = train[train_missing_percent.index.tolist()]

test_missing_df = test[test_missing_percent.index.tolist()]

missing_data_cols = train_missing_df.columns.tolist()

print('\nMissing Data Columns : {}\n{}'.format(len(missing_data_cols), missing_data_cols))

train_missing_df.head()
train_des = train_missing_df.describe(include='all')

train_des
## Fill the Data/Columns with greater than 80% frequency (most accurance element) values with the mode values



fill_with_mode = train_des.loc['freq'][train_des.loc['freq']>len(train)*.8].index.tolist()

train[fill_with_mode] = train[fill_with_mode].fillna(train.mode().iloc[0])

test[fill_with_mode] = test[fill_with_mode].fillna(test.mode().iloc[0])

missing_data_cols = list(set(missing_data_cols) - set(fill_with_mode))



print('\nMissing Data Columns : {}\n{}'.format(len(missing_data_cols), missing_data_cols))
corr = train.corr()

plt.figure(figsize=(15, 15))

sns.heatmap(corr.abs()>0.7, annot=True)
train[missing_data_cols].nunique()
fig, ax = plt.subplots(2, 2, figsize=(12,12))

sns.violinplot(x = 'LotFrontage', data = train, ax = ax[0][0])

sns.violinplot(x = 'LotFrontage', data = train[train['LotFrontage']<110], color='r',ax = ax[0][0] )



sns.violinplot(x = 'MasVnrArea',data = train, ax = ax[0][1])



sns.boxplot(x = 'GarageYrBlt', data = train, ax = ax[1][0])

sns.boxplot(x = 'YearBuilt', data = train, ax = ax[1][0], color='r')



sns.regplot(x = 'YearBuilt', y = 'GarageYrBlt', data=train, ax = ax[1][1])
## It shows that we can fill the two columnns with mean, and Fill GarageYrBlt Column shows a linear relationship with YearBuilt Col



train[['LotFrontage', 'MasVnrArea']] = train[['LotFrontage', 'MasVnrArea']].fillna(train.mean())

test[['LotFrontage', 'MasVnrArea']] = test[['LotFrontage', 'MasVnrArea']].fillna(train.mean())



from scipy.stats import linregress

slope, intercept, _, _, _ = linregress(x = train['YearBuilt'], y = train['GarageYrBlt'].fillna(train['GarageYrBlt'].mean()))

print('Slope and Intersept for linear relatioship b/w Year Built and Garage Year Buit\nSlope: {},\nIntersept: {}'.format(slope, intercept))

train['GarageYrBlt'] = train['GarageYrBlt'].fillna((train['YearBuilt']*slope + intercept).astype(int))

test['GarageYrBlt'] = test['GarageYrBlt'].fillna((test['YearBuilt']*slope + intercept).astype(int))
missing_data_cols = list(set(missing_data_cols) - set(['LotFrontage', 'MasVnrArea', 'GarageYrBlt']))

print('\nMissing Data Columns : {}\n{}'.format(len(missing_data_cols), missing_data_cols))



train.fillna(train.mode().iloc[0], inplace=True)

test.fillna(test.mode().iloc[0], inplace=True)



### Check if we have any null data 

train.isnull().sum()[train.isnull().sum()>0]
print('Training Data Shape: {}\nTest Data Shape: {}'.format(train.shape, test.shape))



print('\nTraining Data Datatypes: {}'.format(train.dtypes.unique()))

numerical_cols = train.select_dtypes(include=['int64','float64' ]).columns.tolist()

categricol_cols = train.select_dtypes(include='object').columns.tolist()

train.describe(include='all')



print('\nTotal Numerical Valued Columns: {}\nTotal Categorical Valued Columns: {}'.

      format(len(numerical_cols), len(categricol_cols)))

train.head()
X = train.drop(['SalePrice'], axis=1)

y = np.log1p(train['SalePrice'])

def inv_y(y_transformed):

    return np.exp(y_transformed)

X_test = test
from sklearn import preprocessing

def ohe(df, cat_cols):

    # creating instance of one-hot-encoder

    ohe = preprocessing.OneHotEncoder(handle_unknown='error', drop='first')

    # passing bridge-types-cat column (label encoded values of bridge_types)

    df_ohe = pd.DataFrame(ohe.fit_transform(df[cat_cols]).toarray())

    # merge with main df bridge_df on key values

    df_final = df.join(df_ohe)

    df_final.drop(cat_cols + ['Id'], axis=1, inplace=True)

    return df_final
X = ohe(X, categricol_cols)

X_test = ohe(X_test, categricol_cols)

X, X_test = X.align(X_test, join='inner', axis=1)  # inner join

print('Training Data Shape : {}\nTest Data Shape: {}'.format(X.shape, X_test.shape))
def normalize_cols(df):

    x = df.values #returns a numpy array

    min_max_scaler = preprocessing.MinMaxScaler()

    x_scaled = min_max_scaler.fit_transform(x)

    return x_scaled
X = normalize_cols(X)

x_num = normalize_cols(train[list(set(numerical_cols) - set(['SalePrice']))])

x_num_test = normalize_cols(test[list(set(numerical_cols) - set(['SalePrice']))])

X_test = normalize_cols(X_test)
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor



def get_mae(x, y):

    # multiple by -1 to make positive MAE score instead of neg value returned as sklearn convention

    return -1 * cross_val_score(XGBRegressor(), 

                                x, y, 

                                scoring = 'neg_mean_absolute_error').mean()



mae_without_categoricals = get_mae(x_num, train['SalePrice'])



mae_one_hot_encoded = get_mae(X, train['SalePrice'])



print('Mean Absolute Error when Dropping Categoricals: ' + str(int(mae_without_categoricals)))

print('Mean Abslute Error with One-Hot Encoding: ' + str(int(mae_one_hot_encoded)))
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

X_train, X_valid, y_train,  y_valid = train_test_split(X,y, test_size=0.2, random_state=42 )
# learning_rate=0.01, n_estimators=3460, max_depth=3, min_child_weight=0,

#                      gamma=0, subsample=0.7,colsample_bytree=0.7,objective='reg:squarederror', 

#                      nthread=-1,scale_pos_weight=1, seed=27, reg_alpha=0.00006)
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import LassoCV

from sklearn.ensemble import GradientBoostingRegressor



n_folds = 10



# XGBoost

# 

model = XGBRegressor(learning_rate=0.01,subsample=0.7,

                       n_estimators=3460,colsample_bytree=0.7,min_child_weight=0,

                       max_depth=3, n_jobs=4)



model.fit(X_train, y_train)

predict = model.predict(X_valid)

print('XGBoost: ' + str(mean_absolute_error(inv_y(predict), inv_y(y_valid))))



      

# Lasso   

model = LassoCV(max_iter=1e7,  random_state=14, cv=n_folds)

model.fit(X_train, y_train)

predict = model.predict(X_valid)

print('Lasso: ' + str(mean_absolute_error(inv_y(predict), inv_y(y_valid))))



      

# GradientBoosting   

model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=5)

model.fit(X_train, y_train)

predict = model.predict(X_valid)

print('Gradient: ' + str(mean_absolute_error(inv_y(predict), inv_y(y_valid))))
model = XGBRegressor(learning_rate=0.01,subsample=0.7,

                       n_estimators=3460,colsample_bytree=0.7,min_child_weight=0,

                       max_depth=3, n_jobs=4)



model.fit(X, y)

predict = model.predict(X_test)

y_test = inv_y(predict)
df_sub = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')

df_sub['SalePrice'] = y_test

df_sub.to_csv('submission.csv', index=False)

df_sub.head()