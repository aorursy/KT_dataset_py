# Load all necessary modules.

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



%matplotlib inline

import matplotlib.pyplot as plt  # Matlab-style plotting

import seaborn as sns # data visualization



from scipy import stats

from scipy.stats import norm, skew #for some statistics



pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points



# ignore warnings (NB: it's not a good practise to ignore warnings)

import warnings

warnings.filterwarnings("ignore")



# check if the dataset has been loaded

import os

assert 'house-prices-advanced-regression-techniques' in os.listdir('../input/'), 'House Prices dataset not loaded!'
# Base folder

HOME_FOLDER = '../input/house-prices-advanced-regression-techniques/'



# Read the data

df_train = pd.read_csv(HOME_FOLDER + 'train.csv', index_col='Id')

df_test = pd.read_csv(HOME_FOLDER + 'test.csv', index_col='Id')
# Print some useful info for the train set

print(f'Train test size: {df_train.shape}')

print(f'Test test size: {df_test.shape}')
# Descriptive statistics summary

print(df_train['SalePrice'].describe())



# We'll need again to plot the distribution and the qqplot, so let's make a function

def plot_distribution_and_qqplot(data):

    # Get the fitted parameters used by the function

    (mu, sigma) = norm.fit(data)

    print('mu = {:.2f} and sigma = {:.2f}'.format(mu, sigma))



    # Plot the distribution

    sns.distplot(data, fit=norm)

    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')

    plt.ylabel('Frequency')

    plt.title(f'{data.name} distribution')



    # Get also the QQ-plot

    fig = plt.figure()

    res = stats.probplot(data, plot=plt)

    plt.show()

    

plot_distribution_and_qqplot(df_train['SalePrice'])
# We use the numpy fuction log1p which  applies log(1+x) to all elements of the column

df_train["SalePrice"] = np.log1p(df_train["SalePrice"])



# Check the new distribution

plot_distribution_and_qqplot(df_train['SalePrice'])
# As before, before plotting let's create a useful function to use it again later

def plot_scatter(x, y):

    fig, ax = plt.subplots()

    ax.scatter(x=x, y=y)

    plt.ylabel(y.name, fontsize=13)

    plt.xlabel(x.name, fontsize=13)

    plt.show()

    

plot_scatter(df_train['GrLivArea'], df_train['SalePrice'])
# Deleting outliers

outliers_idx = df_train['GrLivArea'].sort_values(ascending=False)[:2].index

df_train.drop(outliers_idx, inplace=True)



# Check the graphic again

plot_scatter(df_train['GrLivArea'], df_train['SalePrice'])
# We are gonna need this later for split back the fill df into train and test

n_train, n_test = df_train.shape[0], df_test.shape[0]



# Create one, whole dataframe

df_full = pd.concat((df_train, df_test), sort=False).reset_index(drop=True)



# Store the target variable for later use

y_train = df_train['SalePrice']



# There shouldn't be missing data in the target variable

df_full.drop(['SalePrice'], axis=1, inplace=True)
# Compute the percentage of missing values for each column and sort the result

missing_ratio = (df_full.isnull().sum() / df_full.shape[0] * 100).sort_values(ascending=False)



# Drop all columns which have no missing data and sort them

missing_ratio.drop(missing_ratio[missing_ratio == 0].index, inplace=True)



# Create a simple table for better visualization (and understanding)

missing_values = pd.DataFrame(missing_ratio, columns=['Missing Ratio'])



print(missing_values)
df_full['PoolQC'].fillna('None', inplace=True)
df_full['MiscFeature'].fillna('None', inplace=True)
cols = ['Alley', 'Fence', 'FireplaceQu']

df_full[cols] = df_full[cols].fillna('None', inplace=False)
# Group by neighborhood

df_full['LotFrontage'] = df_full.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
cols = ['GarageCond', 'GarageQual', 'GarageFinish', 'GarageType']

df_full[cols] = df_full[cols].fillna('None')
cols = ['GarageYrBlt', 'GarageArea', 'GarageCars']

df_full[cols] = df_full[cols].fillna(0)
cols = ['BsmtCond', 'BsmtExposure', 'BsmtQual', 'BsmtFinType2', 'BsmtFinType1']

df_full[cols] = df_full[cols].fillna('None')
cols = ['BsmtHalfBath', 'BsmtFullBath', 'BsmtFinSF2', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF']

df_full[cols] = df_full[cols].fillna(0)
# House with a value for 'MasVnrArea' but a missing value in 'MasVnrType'

print(df_full[['MasVnrType', 'MasVnrArea']].loc[df_full['MasVnrType'].isnull() & df_full['MasVnrArea'].notnull()])



# In respect to the house of observation above, what 'MasVnrType' do the similar houses have?

print('\nHouses with price between 196.000 and 200.000:')

print(df_full[['MasVnrType']].loc[(df_full['MasVnrArea'] <= 200.000) & (df_full['MasVnrArea'] >= 196.000)].squeeze().value_counts())
# Replace the single value

mask = (df_full['MasVnrType'].isnull()) & (df_full['MasVnrArea'].notnull())

df_full.loc[mask, 'MasVnrType'] = df_full.loc[mask, 'MasVnrType'].fillna('None')



# The others

df_full['MasVnrType'].fillna('None', inplace=True)

df_full['MasVnrArea'].fillna(0, inplace=True)
df_full['MSZoning'].fillna(df_full['MSZoning'].mode()[0], inplace=True)
df_full.drop(['Utilities'], axis=1, inplace=True)
df_full['Functional'].fillna('Typ', inplace=True)
df_full['Exterior2nd'].fillna(df_full['Exterior2nd'].mode()[0], inplace=True)

df_full['Exterior1st'].fillna(df_full['Exterior1st'].mode()[0], inplace=True)
df_full['SaleType'].fillna(df_full['SaleType'].mode()[0], inplace=True)

df_full['Electrical'].fillna(df_full['Electrical'].mode()[0], inplace=True)

df_full['KitchenQual'].fillna(df_full['KitchenQual'].mode()[0], inplace=True)
assert df_full.isnull().sum().sort_values(ascending=False)[0] == 0, 'Mmm, there are still missing values! Check again.'
# Change column tpye to set the feature as categorical

df_full['MSSubClass'] = df_full['MSSubClass'].apply(str)

df_full['OverallCond'] = df_full['OverallCond'].apply(str)



import datetime

def mapper(month):

    date = datetime.datetime(2000, month, 1)  # You need a dateobject with the proper month

    return date.strftime('%b')  # %b returns the months abbreviation, other options [here][1]



df_full['MoSold'] = df_full['MoSold'].apply(mapper)
from sklearn.preprocessing import LabelEncoder



# Write down explicitly all the feature names

cols = ['MSSubClass', 'Street', 'Alley','LandSlope', 'OverallQual', 'OverallCond', 

        'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',

        'BsmtFinType2', 'HeatingQC', 'CentralAir', 'KitchenQual', 'Functional',

        'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive',

        'PoolQC', 'Fence',  'YrSold', 'MoSold']



# For each column create and apply the encoder

for c in cols:   

    le = LabelEncoder()

    

    le.fit(list(df_full[c].values))

    

    df_full[c] = le.transform(list(df_full[c].values))
# Get the non categorical features

numerical_feats = df_full.dtypes[df_full.dtypes != "object"].index



# Check the skew of all numerical features

skewed_feats = df_full[numerical_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print('Skew in numerical features:')

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness.head(10)
skewness = skewness[abs(skewness['Skew']) > 0.75]

print(f'There are {skewness.shape[0]} skewed numerical features to Box Cox transform.')



from scipy.special import boxcox1p



lambda_param = 0.15

for feature in skewness.index:

    df_full[feature] = boxcox1p(df_full[feature], lambda_param)
df_full = pd.get_dummies(df_full)

print(f'Now there are {df_full.shape[1]} features.')
X_train = df_full[:n_train]

X_test = df_full[n_train:]
from sklearn.model_selection import KFold, cross_val_score

#from sklearn.preprocessing import RobustScaler

from xgboost import XGBRegressor
# Define the validation function

def model_validation(model, X, y, n_folds=5):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X.values)

    

    score = -cross_val_score(model, X.values, y, scoring='neg_mean_squared_error', cv=kf)

    

    return np.sqrt(score)
# Create the model

model_xgb = XGBRegressor(colsample_bytree=0.8, subsample=0.5,

                         learning_rate=0.05, max_depth=3,

                         min_child_weight=1.8, n_estimators=2000,

                         reg_alpha=0.1, reg_lambda=0.8, gamma=0.01,

                         silent=1, random_state=7, n_jobs=-1,

                         early_stopping_rounds=10)



# Check the performance using the defined strategy

score = model_validation(model_xgb, X_train, y_train)

print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# Fit using the whole train data

model_xgb.fit(X_train, y_train)



# Predict using the test set

# NOTE: remember that we log-transformed the target variable, so now we need to apply the inverse process to get the actual predictions

predictions = np.expm1(model_xgb.predict(X_test))
# Save predictions to file

output = pd.DataFrame({'Id': df_test.index,

                       'SalePrice': predictions})



output.to_csv('submission.csv', index=False)