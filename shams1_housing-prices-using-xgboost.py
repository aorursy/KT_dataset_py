import numpy as np

import pandas as pd

import pickle, gzip, urllib.request, json



%matplotlib inline

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split





import os

import re

import copy

import time

import io

import struct

from time import gmtime, strftime

import seaborn as sns



from numpy import loadtxt

import xgboost as xgb

from xgboost import XGBClassifier

from xgboost import plot_tree

from xgboost import plot_importance



from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV

from sklearn.feature_selection import SelectFromModel
%%time 



data_dir = "/kaggle/input/house-prices-advanced-regression-techniques/"



# load the train dataset

train = 'train.csv'

train_df = pd.read_csv(data_dir + train, sep=",")



# load the test dataset

test = 'test.csv'

test_df = pd.read_csv(data_dir + test, sep=",")

# lets peek at the train dataset

train_df.head()
# get the shape of the train dataset

train_df.shape
# lets peek at the test dataset

test_df.head()
# test dimension

test_df.shape
# lets look at the data types of train df columns

train_df.info()
# define helper functions to get categorical and numerical columns

def get_object_cols(df):

    return list(df.select_dtypes(include='object').columns)



def get_numerical_cols(df):

    return list(df.select_dtypes(exclude='object').columns)
corr = train_df[get_numerical_cols(train_df)].corr()



print (corr['SalePrice'].sort_values(ascending=False)[:10], '\n') # print top ten features with high correlation

print (corr['SalePrice'].sort_values(ascending=False)[-10:])
# plot correlation values

corr_df = pd.DataFrame(corr['SalePrice'].sort_values(ascending=False))



corr_df = corr_df.reset_index()



corr_df.columns = ['cols', 'values']



plt.rcdefaults()

plt.figure(figsize=(10,5))

ax = sns.barplot(x="cols", y="values", data=corr_df)

ax.set_ylim(-1.1, 1.1)

ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

ax.set_title('Correlations between the SalePrice and other features')

plt.show()
pd.unique(train_df['OverallQual'])
# lets plot the unique overall quality values against median price

groupedby_quality_df = train_df[['OverallQual', 'SalePrice']].groupby(by='OverallQual').median().reset_index()



groupedby_quality_df.columns = ['Overall Quality', 'Median Sale Price']

plt.rcdefaults()

plt.figure(figsize=(10,5))

ax = sns.barplot(x="Overall Quality", y="Median Sale Price", data=groupedby_quality_df)

ax.set_ylim(0, max(groupedby_quality_df['Median Sale Price'])+10000)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

ax.set_title('Overall Quality vs. Median Sale Prices')

plt.show()
plt.scatter(x=train_df['GrLivArea'], y=train_df['SalePrice'])

plt.ylabel('Sale Price')

plt.xlabel('GrLivArea: Above grade (ground) living area square feet')

plt.show()
groupedby_garagecars_df = train_df[['GarageCars', 'SalePrice']].groupby(by='GarageCars').median().reset_index()



groupedby_quality_df.columns = ['Garage Size', 'Median Sale Price']

plt.rcdefaults()

plt.figure(figsize=(10,5))

ax = sns.barplot(x="Garage Size", y="Median Sale Price", data=groupedby_quality_df)

ax.set_ylim(0, max(groupedby_quality_df['Median Sale Price'])+10000)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

ax.set_title('Size of garage in car capacity vs. Median Sale Prices')

plt.show()
plt.scatter(x=train_df['GarageArea'], y=train_df['SalePrice'])

plt.ylabel('Sale Price')

plt.xlabel('GarageArea: Size of garage in square feet')

plt.show()
def visualize_missing_values(df):

    total_nans_df = pd.DataFrame(df.isnull().sum(), columns=['values'])

    total_nans_df = total_nans_df.reset_index()

    total_nans_df.columns = ['cols', 'values']

    # calculate % missing values

    total_nans_df['% missing values'] = 100*total_nans_df['values']/df.shape[0]

    total_nans_df = total_nans_df[total_nans_df['% missing values'] > 0 ]

    total_nans_df = total_nans_df.sort_values(by=['% missing values'])



    plt.rcdefaults()

    plt.figure(figsize=(10,5))

    ax = sns.barplot(x="cols", y="% missing values", data=total_nans_df)

    ax.set_ylim(0, 100)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    plt.show()
# visualise missing values in train data set

visualize_missing_values(train_df)
# visualize columns with missing values in the test dataset

visualize_missing_values(test_df)
# function to drop columns with missing that values based on a predefine cutoff criteria

def drop_columns_with_missing_values(df, cutoff):

    """Drop columns with missing values greater than the specified cut-off %

    

    Parameters:

    -----------

    df     : pandas dataframe

    cutoff : % missing values

    

    Returns:

    ---------

    Returns clean dataframe

    """

    # create a dataframe for missing values by column

    total_nans_df = pd.DataFrame(df.isnull().sum(), columns=['values'])

    total_nans_df = total_nans_df.reset_index()

    total_nans_df.columns = ['cols', 'values']

    

    # calculate % missing values

    total_nans_df['% missing values'] = 100*total_nans_df['values']/df.shape[0]

    

    total_nans_df = total_nans_df[total_nans_df['% missing values'] >= cutoff ]

    

    # get columns to drop

    cols = list(total_nans_df['cols'])

    print('Features with missing values greater than specified cutoff : ', cols)

    print('Shape before dropping: ', df.shape)

    new_df = df.drop(labels=cols, axis=1)

    print('Shape after dropping: ',new_df.shape)

    

    return new_df
# drop columns with over 80% missing values from the train dataset

new_train_df = drop_columns_with_missing_values(train_df, 80)
# drop columns with over 80% missing values from the test dataset

new_test_df = drop_columns_with_missing_values(test_df, 80)
# Separate features into object and numerical variables



# train object cols

object_cols_train = get_object_cols(new_train_df)

# train numerical cols

numerical_cols_train = get_numerical_cols(new_train_df)
# test object cols

object_cols_test = get_object_cols(new_test_df)

# test numerical cols

numerical_cols_test = get_numerical_cols(new_test_df)
# columns whose missing values are to be filled with 'None'

cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 

        'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']



# Now for these columns in train and test dataframes, replace missing values with 'None'

values = {col: 'None' for col in cols}



new_train_df = new_train_df.fillna(value=values)

new_test_df = new_test_df.fillna(value=values)
# for other categorical variables, replace NaNs with the most frequent variable

other_cols = [col for col in object_cols_train if col not in cols]

print(other_cols)



# create a dictionary to use to replace missing values in train dataframe

values = {col:  new_train_df[col].mode()[0] for col in other_cols}



# replace missing values

new_train_df = new_train_df.fillna(value=values)



# create a dictionary to use to replace missing values in test dataframe

values = {col:  new_test_df[col].mode()[0] for col in other_cols}



# replace missing values

new_test_df = new_test_df.fillna(value=values)
# check missing values in train df

visualize_missing_values(new_train_df)
# check missing values in test df

visualize_missing_values(new_test_df)
# create a dictionary to replace missing numerical values in train dataset

values = {'LotFrontage': new_train_df['LotFrontage'].median(),

          'GarageYrBlt': new_train_df['GarageYrBlt'].median(),

          'MasVnrArea': 0

         }



# replace missing values

new_train_df = new_train_df.fillna(value=values)
# get columns with missing values in test dataframe

cols = list(new_test_df.columns[new_test_df.isnull().any()])



values = {}

for col in cols:

    if col in ['LotFrontage', 'GarageYrBlt']:

        values[col] = new_test_df[col].median()

    else:

        values[col] = 0

# now replace missing values in test dataframe

new_test_df = new_test_df.fillna(value=values)
# check missing values in train df

np.sum(new_train_df.isnull().sum())
# check missing values in test df

np.sum(new_test_df.isnull().sum())
# from our correlation graph, we are going to drop columns that do not positively influence the SalePrice

cols_to_drop = list(corr_df[corr_df['values'] < 0]['cols'])

cols_to_drop
# drop these columns from train and test dfs

new_train_df = new_train_df.drop(labels=cols_to_drop, axis=1)

new_test_df = new_test_df.drop(labels=cols_to_drop, axis=1)
new_train_df.shape
new_test_df.shape
# lets combine the train and test dataframes before performing label and one-hot encoding schemes

new_train_df['train']  = 1 # the train column will later be used to split the combined dataframes

new_test_df['train']  = 0



df = pd.concat([new_train_df, new_test_df], axis=0,sort=False)
df.head()
# Using the data description file, lets create a list for columns marked for label encoding

label_enc_cols = [

    'ExterQual',

    'ExterCond',

    'GarageCond',

    'GarageQual',

    'FireplaceQu',

    'KitchenQual',

    'CentralAir',

    'HeatingQC',

    'BsmtFinType2',

    'BsmtFinType1',

    'BsmtExposure',

    'BsmtCond',

    'BsmtQual'

]
# Using the data description file, lets create a dictionary to use for substituting each value in a column Series with the key values

map_dict = {

    'Ex': 5,

    'Gd': 4,

    'TA': 3,

    'Fa': 2,

    'Po': 1,

    'None': 0,

    'Av':   3,

    'Mn':   2,

    'No':   1,

    'GLQ':  6,

    'ALQ':  5,

    'BLQ':  4,

    'Rec':  3,

    'LwQ':  2,

    'Unf':  1,

    'N':    0,

    'Y':    1

}
# now perform label encoding in combined df

for col in label_enc_cols:

    df[col] = df[col].map(map_dict)
# check label encoding

for col in label_enc_cols:

    print(pd.unique(df[col]))
# create a list of columns to perfom one-hot encoding on

one_hot_enc_cols = [col for col in object_cols_train if col not in label_enc_cols ]
# perform one hot encoding using pd.get_dummies 

one_hot_df = pd.get_dummies(df[one_hot_enc_cols], drop_first=True)
one_hot_df.head()
# now combine dummies dataframe with the combined train and test dfs

df_final = pd.concat([df, one_hot_df], axis=1, sort=False)
# drop columns used for one hot encoding

df_final = df_final.drop(labels=one_hot_enc_cols, axis=1)
df_final.head()
# Now that we are done with preparing our data for modelling, separate df_final into train and test



# use the 'train' column to separate the combined df

train_df_final =  df_final[df_final['train'] == 1]

train_df_final = train_df_final.drop(labels=['train'], axis=1)





test_df_final = df_final[df_final['train'] == 0]

test_df_final = test_df_final.drop(labels=['SalePrice'], axis=1)

test_df_final = test_df_final.drop(labels=['train'], axis=1)
# check dimensions

print(train_df_final.shape)

print(test_df_final.shape)
y= train_df_final['SalePrice']

X = train_df_final.drop(labels=['SalePrice'], axis=1)



# split data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=50)
# specify parameters via map, definition are same as c++ version

param = {

    'max_depth': 3,

    'eta' : .1,

    'gamma' : 0,

    'min samples split' : 2,

    'min samples leaf': 1,

    'objective': "reg:squarederror",

    'subsample': 1

}

kfold = KFold(n_splits=10, shuffle=True, random_state=7)

params = {

    'max_depth': [2, 4, 6],

    'n_estimators': [50, 100, 200, 500, 1000],

    'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]

}

print("Parameter optimization")

xgb_model = xgb.XGBRegressor()

clf = GridSearchCV(xgb_model,param_grid=params, verbose=0, cv=kfold, n_jobs=-1)

clf.fit(X, y)

print(clf.best_score_)

print(clf.best_params_)
# lets take a peek at the best model

clf.best_estimator_
# plot tree

plot_tree(clf.best_estimator_)

plt.show()
# plot feature importance

ax = plot_importance(clf.best_estimator_, height=1)

fig = ax.figure

fig.set_size_inches(10, 30)

plt.show()
# fit model on all training data

model = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,

             importance_type='gain', interaction_constraints='',

             learning_rate=0.1, max_delta_step=0, max_depth=2,

             min_child_weight=1, monotone_constraints='()',

             n_estimators=1000, n_jobs=0, num_parallel_tree=1, random_state=0,

             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,

             tree_method='exact', validate_parameters=1, verbosity=None)



model.fit(X_train, y_train)

# make predictions for test data and evaluate

predictions = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test.values, predictions))

print("RMSE:", (rmse))

# Fit model using each importance as a threshold

thresholds = np.sort(model.feature_importances_)

best_threshold = X_train.shape[1]

best_score = rmse

for thresh in thresholds:

    # select features using threshold

    selection = SelectFromModel(model, threshold=thresh, prefit=True)

    select_X_train = selection.transform(X_train)

    

    # train model

    selection_model = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,

             importance_type='gain', interaction_constraints='',

             learning_rate=0.1, max_delta_step=0, max_depth=2,

             min_child_weight=1, monotone_constraints='()',

             n_estimators=1000, n_jobs=0, num_parallel_tree=1, random_state=0,

             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,

             tree_method='exact', validate_parameters=1, verbosity=None)

    

    selection_model.fit(select_X_train, y_train)

    # eval model

    select_X_test = selection.transform(X_test)

    predictions = selection_model.predict(select_X_test)

    score = np.sqrt(mean_squared_error(y_test, predictions))

    if score < best_score:

        best_score = score

        best_threshold = select_X_train.shape[1]

    

    print("Thresh={}, n={}, RMSE: {}".format(thresh, select_X_train.shape[1], score))

print('Best RMSE: {}, n={}'.format(best_score, best_threshold))
feature_importance = pd.DataFrame(pd.Series(model.feature_importances_, index=X_train.columns, 

                               name='Feature_Importance').sort_values(ascending=False)).reset_index()

selected_features = feature_importance.iloc[0:best_threshold]['index']

selected_features = list(selected_features)
# now use the selected features  and fit the model on X and Y

new_X = X[selected_features]

new_test = test_df_final[selected_features]



model.fit(new_X, y)
# Now lets make predictions on the test dataset for submission

submission_predictions = model.predict(new_test)
# prepare a csv file for submission

sub_df = pd.DataFrame(submission_predictions)

sub_df['Id'] = test_df['Id']

sub_df.columns = ['SalePrice', 'Id']

sub_df = sub_df[['Id', 'SalePrice']]



sub_df.to_csv('submission.csv', index=False)