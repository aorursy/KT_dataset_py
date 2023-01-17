# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.model_selection import train_test_split, RandomizedSearchCV

from sklearn.linear_model import LinearRegression, Ridge, RidgeCV

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# import data

PATH = ('../input/Melbourne_housing_FULL.csv')

raw_data = pd.read_csv(PATH, index_col=None)

df_raw = pd.DataFrame(raw_data)

df_raw.head()



#PATH_clean = ('assets\Melbourne_housing_FULL_clean.csv')

#clean_data = pd.read_csv(PATH_clean, index_col=0)

#df_clean = pd.DataFrame(clean_data)

#df_clean.head()
df_raw.info()
df_raw.describe()
# view price data

display(df_raw.Price.head())

df_raw.Price.dropna().hist()

plt.title('Price');
# log transform price data

df_raw['log_Price'] = np.log1p(df_raw.Price.dropna())



log_price_mean = df_raw['log_Price'].mean()

log_price_std = df_raw['log_Price'].std()



# view log(price) data

df_raw.log_Price.hist(bins=20)

plt.axvline((log_price_mean+log_price_std), color='k', linestyle='--')

plt.axvline((log_price_mean-log_price_std), color='k', linestyle='--')

plt.axvline(log_price_mean, color='k', linestyle='-')

plt.title('log(Price)');
# view missing data

display(df_raw.shape)

display(df_raw.dropna().shape)
# view features

df_raw.columns.values



# view sum of NaN values if

# all missing Price values are dropped

#df_raw.dropna(subset=['Price']).isna().sum()



# view sum of NaN values

df_raw.isna().sum()
# define 'high price' and 'low price' binary features

df_raw['high_price'] = np.where(

    df_raw['log_Price'] > (log_price_mean+log_price_std), 1, 0

)



df_raw['low_price'] = np.where(

    df_raw['log_Price'] < (log_price_mean-log_price_std), 1, 0

)



display(df_raw['high_price'].value_counts())

df_raw['low_price'].value_counts()
# drop features

drop_list = ['Suburb', 'Address', 'SellerG', 'CouncilArea']

df_raw = df_raw.drop(drop_list, axis=1)



df_raw.head()
df_raw[['Type', 'Method', 'Regionname']].isna().sum()
# Regionname has NaN values,

# so we'll have to deal with those

df_raw = df_raw.dropna(subset=['Type', 'Method', 'Regionname'], axis=0)
# one-hot encoding

# set sparse=False to return an array

cat_encoder = OneHotEncoder(sparse=False)

df_raw_type_reshaped = df_raw['Type'].values.reshape(-1,1)

df_raw_type_1hot = cat_encoder.fit_transform(df_raw_type_reshaped)

categories = cat_encoder.categories_

df_raw_type_1hot = pd.DataFrame(df_raw_type_1hot, columns=categories)
# concat 1hot DataFrame w/ df_na

# reset index of df_na and concat

df_raw = df_raw.reset_index().drop('index', axis=1)

df_raw = pd.concat([df_raw, df_raw_type_1hot], axis=1)

df_raw.head()
# one-hot encode 'Method' feature

df_raw_meth_reshaped = df_raw['Method'].values.reshape(-1,1)

df_raw_meth_1hot = cat_encoder.fit_transform(df_raw_meth_reshaped)

categories = cat_encoder.categories_

df_raw_meth_1hot = pd.DataFrame(df_raw_meth_1hot, columns=categories)



# concat 1hot DataFrame w/ df_na

df_raw = pd.concat([df_raw, df_raw_meth_1hot], axis=1)

df_raw.head()
# one-hot encode 'Regionname' feature

df_raw_reg_reshaped = df_raw['Regionname'].values.reshape(-1,1)

df_raw_reg_1hot = cat_encoder.fit_transform(df_raw_reg_reshaped)

categories = cat_encoder.categories_

df_raw_reg_1hot = pd.DataFrame(df_raw_reg_1hot, columns=categories)



# concat 1hot DataFrame w/ df_na

df_raw = pd.concat([df_raw, df_raw_reg_1hot], axis=1)

df_raw.head()
df_raw.isna().sum()
# drop NaNs from target column(s)

df_na = df_raw.dropna(subset=['Price', 'log_Price'], axis=0)

df_na.isna().sum()
# find our features with dtype == object

objects = []



for i in df_raw.columns.values:

    if df_raw[i].dtype == 'O':

        objects.append(str(i))



df_na = df_na.drop(objects, axis=1)



df_na.isna().sum()
# drop regional data

cols = ['Bedroom2', 'Bathroom', 'Car',

        'Landsize','BuildingArea', 'YearBuilt',

        'high_price', 'low_price']

df_num = df_na[cols]



# check nulls

df_num.isna().sum()
# fill NaN values by median of price category

df_num.loc[df_num['high_price']==1] = df_num.loc[

    df_num['high_price']==1].apply(lambda x: x.fillna(x.median()),axis=0)

df_num.loc[df_num['low_price']==1] = df_num.loc[

    df_num['low_price']==1].apply(lambda x: x.fillna(x.median()),axis=0)

df_num.loc[df_num['high_price' and 'low_price']==0] = df_num.loc[

    df_num['high_price' and 'low_price']==0].apply(lambda x: x.fillna(x.median()),axis=0)



df_num.isna().sum()
# concat filled NaNs w/ the rest of our data

df_na = pd.concat([df_na.drop(cols, axis=1), df_num], axis=1)

df_na.head()

df_na.isna().sum()
# clean up columns from 1hot encoding

col_list = ['Rooms', 'Price', 'Distance', 'Postcode', 'Lattitude',

            'Longtitude', 'Propertycount', 'log_Price', 'h', 't', 'u',

            'PI', 'PN', 'S', 'SA', 'SN', 'SP', 'SS',

            'VB', 'W', 'Eastern_Metropolitan', 'Eastern_Victoria',

            'Northern_Metropolitan', 'Northern_Victoria',

            'South_Eastern_Metropolitan', 'Southern_Metropolitan',

            'Western_Metropolitan', 'Western_Victoria', 'Bedroom2',

            'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'YearBuilt',

            'high_price', 'low_price']

df_na.columns = col_list

reg_cols = ['Eastern_Metropolitan', 'Eastern_Victoria',

            'Northern_Metropolitan', 'Northern_Victoria',

            'South_Eastern_Metropolitan', 'Southern_Metropolitan',

            'Western_Metropolitan', 'Western_Victoria']



for i in reg_cols:

    df_na.loc[df_na[i]==1] = df_na.loc[

        df_na[i]==1].apply(lambda x: x.fillna(x.median()),axis=0)

    

df_na.isna().sum()
# convert Date feature to datetime

df_na['Date'] = df_raw['Date']

df_na['Date'].head()

df_na['Date'] = pd.to_datetime(df_na['Date'], errors='raise', dayfirst=1)

df_na['Date'].head()
# store clean data

df_clean = df_na

#df_clean.to_csv('assets\Melbourne_housing_FULL_clean.csv')
# check data

df_clean.info()
# define data and target

# drop our target value and derived values

# drop datetime as it likely won't help our model

drop_list = ['Price', 'log_Price', 'high_price', 'low_price', 'Date']

data = df_clean.drop(drop_list, axis=1)

target = df_clean['log_Price']



# plot a heatmap

sns.heatmap(data.corr());
# Create correlation matrix

corr_matrix = data.corr().abs()



# Select upper triangle of correlation matrix

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=4).astype(np.bool))



# Find index of feature columns with correlation greater than 0.90

to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]



display(data.shape)



# Drop correlated features 

for i in to_drop:

    data = data.drop(i, axis=1)



data.shape
# define training and test set

X_train, X_test, y_train, y_test = train_test_split(

    data, target, test_size=0.2, random_state=42)



# scale X_train values

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))

X_test_scaled = scaler.transform(X_test.astype(np.float64))
# determine which model to use

OLS = LinearRegression()

OLS.fit(X_train, y_train)

y_pred = OLS.predict(X_test)



# Display.

print('Linear Regression')

print('\nR-squared training set:')

print(OLS.score(X_train, y_train))



print('\nR-squared test set:')

print(OLS.score(X_test, y_test))
# determine which model to use

OLS_scaled = LinearRegression()

OLS_scaled.fit(X_train_scaled, y_train)

y_pred = OLS_scaled.predict(X_test_scaled)



# Display.

print('Scaled Linear Regression')

print('\nR-squared training set:')

print(OLS_scaled.score(X_train_scaled, y_train))



print('\nR-squared test set:')

print(OLS_scaled.score(X_test_scaled, y_test))
# determine which model to use

RF = RandomForestRegressor(n_estimators=10)

RF.fit(X_train, y_train)

y_pred = RF.predict(X_test)



# Display.

print('Random Forest Regressor')

print('\nR-squared training set:')

print(RF.score(X_train, y_train))



print('\nR-squared test set:')

print(RF.score(X_test, y_test))
# determine which model to use

RF_scaled = RandomForestRegressor(n_estimators=10)

RF_scaled.fit(X_train_scaled, y_train)

y_pred = RF_scaled.predict(X_test_scaled)



# Display.

print('Scaled Random Forest Regressor')

print('\nR-squared training set:')

print(RF_scaled.score(X_train_scaled, y_train))



print('\nR-squared test set:')

print(RF_scaled.score(X_test_scaled, y_test))
# define empty list

alphas = []

train_scores = []

test_scores = []



#Run the model for many alphas.

for lambd in range(1, 50, 2):

    ridge = Ridge(alpha=lambd, fit_intercept=False)

    ridge.fit(X_train, y_train)

    alphas.append(lambd)

    train_scores.append(ridge.score(X_train, y_train))

    test_scores.append(ridge.score(X_test, y_test))



plt.plot(alphas, train_scores, label='Training Data')

plt.plot(alphas, test_scores, label='Test Data')

plt.title('Ridge Regression')

plt.xlabel('Lambda')

plt.ylabel('R-Squared')

plt.legend()

plt.show();
# instantiate ridgeCV regression

alpha_list = range(1, 50, 2)

ridge = RidgeCV(alphas=alpha_list, cv=5)

ridge.fit(X_train, y_train)



# Display

print('Ridge Regression')

print('\nR-squared training set:')

print(ridge.score(X_train, y_train))



print('\nR-squared test set:')

print(ridge.score(X_test, y_test))



print('\nRidge regression alpha:')

print(ridge.alpha_)
# instantiate ridgeCV regression

alpha_list = range(1, 50, 2)

ridge_scaled = RidgeCV(alphas=alpha_list, cv=5)

ridge_scaled.fit(X_train_scaled, y_train)



# Display

print('Scaled Ridge Regression')

print('\nR-squared training set:')

print(ridge_scaled.score(X_train_scaled, y_train))



print('\nR-squared test set:')

print(ridge_scaled.score(X_test_scaled, y_test))



print('\nRidge regression alpha:')

print(ridge_scaled.alpha_)
# determine which model to use

GBRT = GradientBoostingRegressor(max_depth=2, n_estimators=120)

GBRT.fit(X_train, y_train)



errors = [mean_squared_error(y_test, y_pred)

         for y_pred in GBRT.staged_predict(X_test)]

best_n_estimators = np.argmin(errors)



GBRT_best = GradientBoostingRegressor(max_depth=2, n_estimators=best_n_estimators)

GBRT_best.fit(X_train, y_train)

y_pred = GBRT_best.predict(X_test)



# Display

print('Gradient Boosting Regressor')

print('\nR-squared training set:')

print(GBRT_best.score(X_train, y_train))



print('\nR-squared test set:')

print(GBRT_best.score(X_test, y_test))
# determine which model to use

GBRT_scaled = GradientBoostingRegressor(max_depth=2, n_estimators=120)

GBRT_scaled.fit(X_train_scaled, y_train)



errors = [mean_squared_error(y_test, y_pred)

         for y_pred in GBRT_scaled.staged_predict(X_test_scaled)]

best_n_estimators = np.argmin(errors)



GBRT_scaled_best = GradientBoostingRegressor(max_depth=2, n_estimators=best_n_estimators)

GBRT_scaled_best.fit(X_train_scaled, y_train)

y_pred = GBRT_scaled_best.predict(X_test_scaled)



# Display

print('Scaled Gradient Boosting Regressor')

print('\nR-squared training set:')

print(GBRT_scaled_best.score(X_train_scaled, y_train))



print('\nR-squared test set:')

print(GBRT_scaled_best.score(X_test_scaled, y_test))
# sample training set to speed up parameter optimization

X_train_sample, X_test_sample, y_train_sample, y_test_sample = train_test_split(

    X_train, y_train, test_size=0.5, random_state=42)
# define our parameter ranges

learning_rate=[0.01]

alpha=[0.01,0.03,0.05,0.1,0.3, 0.9]

n_estimators=[int(x) for x in np.linspace(start = 10, stop = 500, num = 4)]

max_depth=[int(x) for x in np.linspace(start = 3, stop = 15, num = 4)]

max_depth.append(None)

min_samples_split=[int(x) for x in np.linspace(start = 2, stop = 5, num = 4)]

min_samples_leaf=[int(x) for x in np.linspace(start = 1, stop = 4, num = 4)]

max_features=['auto', 'sqrt']



# Create the random grid

param_grid = {'learning_rate':learning_rate,

              'alpha':alpha,

              'n_estimators': n_estimators,

              'max_features': max_features,

              'max_depth': max_depth,

              'min_samples_split': min_samples_split,

              'min_samples_leaf': min_samples_leaf,

             }



print(param_grid)



# Initialize and fit the model.

model = GradientBoostingRegressor()

model = RandomizedSearchCV(model, param_grid, cv=3)

model.fit(X_train_sample, y_train_sample)



# get the best parameters

best_params = model.best_params_

print(best_params)
# refit model with best parameters

model_best = GradientBoostingRegressor(**best_params)

model_best.fit(X_train, y_train)

y_pred = model_best.predict(X_test)
feature_importance = model_best.feature_importances_



# Make importances relative to max importance.

feature_importance = 100.0 * (feature_importance / feature_importance.max())

sorted_idx = np.argsort(feature_importance)

pos = np.arange(sorted_idx.shape[0]) + 0.5



plt.subplot(1,2,2)

plt.barh(pos, feature_importance[sorted_idx], align='center')



plt.yticks(pos, data.columns.values[sorted_idx])

plt.xlabel('Relative Importance')

plt.title('Variable Importance')

plt.show()
# sort top features

top_features = np.where(feature_importance > 20)

top_features = data.columns[top_features].ravel()

print(top_features)
# Display.

print('Optimized Gradient Boosting Regressor')

print('\nR-squared training set:')

print(model_best.score(X_train, y_train))

print('\nMean absolute error training set: ')

print(mean_absolute_error(y_train, model_best.predict(X_train)))

print('\nMean squared error training set: ')

print(mean_squared_error(y_train, model_best.predict(X_train)))



print('\n\nR-squared test set:')

print(model_best.score(X_test, y_test))

print('\nMean absolute error test set: ')

print(mean_absolute_error(y_test, y_pred))

print('\nMean squared error test set: ')

print(mean_squared_error(y_test, y_pred))



# top features

print('\nTop indicators:')

print(top_features)