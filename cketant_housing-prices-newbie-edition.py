import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas_profiling



# Data Viz

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set(rc={'figure.figsize':(16,16)})



# import the data

df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

df_joined = pd.concat([df_train, df_test], sort=False)
df_joined.head()
print("Total Columns: %s\nTotal Rows: %s" % (df_joined.shape[1], df_joined.shape[0]))
types = {}

for d in df_joined.dtypes.tolist():

    if d not in types.keys():

        types[d] = 1

    else:

        types[d] += 1

print("Total count of column types\n-------------------")

types
print("Total columns with null values: %s" % (len(df_joined.columns[df_joined.isna().any()]),))
print("Columns with null values\n-------------------")

null_series = pd.isnull(df_joined).sum()

null_series[null_series > 0]
types = {}

indices = null_series[null_series > 0].index.tolist()

for d in df_joined[indices].dtypes.tolist():

    if d not in types.keys():

        types[d] = 1

    else:

        types[d] += 1

print("Total count of column types for columns with null values\n-------------------")

types
# df_joined.profile_report(style={'full_width': True})
sns.barplot(x='SaleType', y='SalePrice', data=df_train)
sns.pointplot(x='SaleCondition', y='SalePrice', data=df_train)
sns.pointplot(x='MSSubClass', y='SalePrice', data=df_train)
sns.pointplot(x='GarageQual', y='SalePrice', data=df_train, order=['Po', 'Fa', 'TA', 'Gd', 'Ex'], color='blue', estimator=np.median)
sns.pointplot(x='GarageCond', y='SalePrice', data=df_train, order=['NA','Po', 'Fa', 'TA', 'Gd', 'Ex'])
sns.pointplot(x='HeatingQC', y='SalePrice', data=df_train, order=['Po', 'Fa', 'TA', 'Gd', 'Ex'], color='green', estimator=np.median)
sns.pointplot(x='ExterQual', y='SalePrice', data=df_train, order=['Po', 'Fa', 'TA', 'Gd', 'Ex'], color='orange', estimator=np.median)
sns.pointplot(x='KitchenQual', y='SalePrice', data=df_train, order=['Po', 'Fa', 'TA', 'Gd', 'Ex'], color='purple', estimator=np.median)
sns.pointplot(x='BsmtCond', y='SalePrice', data=df_train, order=['NA','Po', 'Fa', 'TA', 'Gd', 'Ex'])
sns.pointplot(x='ExterCond', y='SalePrice', data=df_train, order=['NA','Po', 'Fa', 'TA', 'Gd', 'Ex'])
sns.pointplot(x='OverallCond', y='SalePrice', data=df_train)
sns.pointplot(x='Functional', y='SalePrice', data=df_train, order=['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'])
sns.regplot(x='LotArea', y='SalePrice', data=df_train)
sns.regplot(x='1stFlrSF', y='SalePrice', data=df_train)
sns.regplot(x='2ndFlrSF', y='SalePrice', data=df_train)
sns.regplot(x='MasVnrArea', y='SalePrice', data=df_train)
sns.regplot(x='TotalBsmtSF', y='SalePrice', data=df_train)
sns.pointplot(x='BldgType', y='SalePrice', data=df_train)
sns.pointplot(x='HouseStyle', y='SalePrice', data=df_train, order=['1Story', '1.5Fin', '1.5Unf', '2Story', '2.5Fin', '2.5Unf', 'SFoyer', 'SLvl'], estimator=np.mean)
sns.barplot(x='MSZoning', y='SalePrice', data=df_train)
chart = sns.barplot(x='Neighborhood', y="SalePrice", data=df_train)

chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
sns.pointplot(x='YrSold', y='SalePrice', data=df_train)
def num_clean(df_train, df_test):

    """ Clean the data before we encode"""

    #1) df_train

    df_train_num = df_train.select_dtypes(include='number') # fetch num columns

    train_missing_cols = df_train_num.columns[df_train_num.isnull().any()].tolist() # fetch num columns with missing

    df_train = _fill_num_df(df_train, train_missing_cols)

    

    # 2) df_test

    df_test_num = df_test.select_dtypes(include='number') # fetch num columns

    test_missing_cols = df_test_num.columns[df_test_num.isnull().any()].tolist() # fetch num columns with missing

    df_test = _fill_num_df(df_test, test_missing_cols)

    

    return df_train, df_test

    

def _fill_num_df(df, cols):

    """ Fill in the missing values for the dataframe """

    for col in cols:

        df[col] = df[col].fillna(df[col].mean())

    return df

    
def cat_clean(df_train, df_test):

    """ Clean the data data before we encode """

    #1) df_train

    df_train_cat = df_train.select_dtypes(include='object') # fetch cat columns

    train_missing_cols = df_train_cat.columns[df_train_cat.isnull().any()].tolist() # fetch cat columns with missing

    df_train = _fill_cat_df(df_train, train_missing_cols)

    

    # 2) df_test

    df_test_cat = df_test.select_dtypes(include='object') # fetch cat columns

    test_missing_cols = df_test_cat.columns[df_test_cat.isnull().any()].tolist() # fetch cat columns with missing

    df_test = _fill_cat_df(df_test, test_missing_cols)

    

    return df_train, df_test

    

def _fill_cat_df(df, cols):

    """ Fill in the missing values for the dataframe """

    for col in cols:

        df[col] = df[col].fillna(df[col].mode().values[0])

    return df
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
def nominal_encode(df_train, df_test, cols):

    """Encode the nominal features"""

    # create encoder

    encoder = OneHotEncoder(dtype='int', sparse=False) # sparse returns array not matrix

    # transform columns

    encoded_df_train = pd.DataFrame(encoder.fit_transform(df_train[cols])) # remove target and put back after transform

    encoded_df_test = pd.DataFrame(encoder.transform(df_test[cols]))

    # Add index back to the transformed dfs

    encoded_df_train.index = df_train.index

    encoded_df_test.index = df_test.index

    # remove the original cols b/c we're about add the encoded

    df_train = df_train.drop(cols, axis=1)

    df_test = df_test.drop(cols, axis=1)

    # create the new dfs

    df_train = pd.concat([df_train, encoded_df_train], axis=1)

    df_test = pd.concat([df_test, encoded_df_test], axis=1)

    

    return df_train, df_test

    

    

def ordinal_encode(df_train, df_test, cols):

    """Encode the ordinal features"""

    # Encoder

    encoder = OrdinalEncoder(dtype='int')

    # transform

    encoded_df_train = pd.DataFrame(encoder.fit_transform(df_train[cols]))

    encoded_df_test = pd.DataFrame(encoder.transform(df_test[cols]))

    # add index 

    encoded_df_train.index = df_train.index

    encoded_df_test.index = df_test.index

    # remove original columsn b/c we transformed them

    df_train = df_train.drop(cols, axis=1)

    df_test = df_test.drop(cols, axis=1)

    # concat

    df_train = pd.concat([df_train, encoded_df_train], axis=1)

    df_test = pd.concat([df_test, encoded_df_test], axis=1)

    

    return df_train, df_test
# 1) Clean

df_train, df_test = num_clean(df_train, df_test)

df_train, df_test = cat_clean(df_train, df_test)



# 2) Encode

df_train, df_test = nominal_encode(df_train, df_test, ["MSZoning", "Street", "Alley", "Utilities", "Exterior1st", "Exterior2nd", "MasVnrType", "MiscFeature", "SaleType", "Electrical", "GarageType", "LotShape", "LandContour", "LotConfig", "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Foundation", "Heating", "PavedDrive", "CentralAir", "SaleCondition"])

df_train, df_test = ordinal_encode(df_train, df_test, ["BsmtQual", "BsmtCond", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC", "Functional", "BsmtExposure", "GarageFinish", "Fence", "LandSlope", "ExterQual", "ExterCond", "BsmtFinType1", "BsmtFinType2", "HeatingQC"])
from sklearn.model_selection import train_test_split
TEST_SIZE = 0.25

X_all = df_train.drop(['SalePrice', 'Id'], axis=1)

y_all = df_train[['SalePrice']]



X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=TEST_SIZE, random_state=3)
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.svm import LinearSVR

from sklearn.linear_model import LinearRegression



from sklearn.metrics import max_error,mean_absolute_error,mean_squared_error,median_absolute_error,r2_score



import math
models = ['LinearRegression',

            'LinearSVR',

            'DecisionTreeRegressor',

            'GradientBoostingRegressor',

            'RandomForestRegressor']



results = {

    "models": [],

    "mean_absolute_error": [],

    "root_mean_squared_error": [],

    "median_absolute_error": [],

    "max_error": [],

    "r2_score": [],

}



for model in models:

    m = eval(model)()

    m.fit(X_train, y_train)

    y_pred = m.predict(X_test)

    

    results['models'].append(model)

    results['mean_absolute_error'].append(mean_absolute_error(y_test, y_pred))

    results['root_mean_squared_error'].append(math.sqrt(mean_squared_error(y_test, y_pred)))

    results['median_absolute_error'].append(median_absolute_error(y_test, y_pred))

    results['max_error'].append(max_error(y_test, y_pred))

    results['r2_score'].append(r2_score(y_test, y_pred))

                                       

results_df = pd.DataFrame(results)

results_df.sort_values(by=['root_mean_squared_error', 'r2_score'], ascending=True, inplace=True)

results_df
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer
# Define scorer

def rmse_metric(y_test, y_pred):

    score = math.sqrt(mean_squared_error(y_test, y_pred))

    return score



# Scorer function would try to maximize calculated metric

rmse_scorer = make_scorer(rmse_metric, greater_is_better=False)



def tune_model(model, X, y, param_grid, cv):

    reg = GridSearchCV(estimator=eval(model)(), param_grid=param_grid, cv=cv, scoring=rmse_scorer, n_jobs=-1, verbose=False)

    reg.fit(X, y)

    return (model, reg.best_score_, reg.best_estimator_)
tuning_models = ['GradientBoostingRegressor',

                'RandomForestRegressor']



param_grid = {

    'RandomForestRegressor': {

        'n_estimators': [10, 25, 50],

        'max_depth': [10, 25, 50],

        'max_features': ['auto', 'sqrt', 'log2']

    },

    'GradientBoostingRegressor': {

        'learning_rate': [0.01, 0.001, 0.0001],

        'n_estimators': [100, 150, 200],

        'max_depth': [3, 5, 10],

        'max_features': ['auto', 'sqrt', 'log2'],

    }

}



tuned_results = {

    'model': [],

    'best_score': [],

    'best_estimator': []

}



for model in tuning_models:

    m, score, est = tune_model(model, X_train, np.ravel(y_train), param_grid[model], 5)

    tuned_results['model'].append(m)

    tuned_results['best_score'].append(score)

    tuned_results['best_estimator'].append(est)

    

tuned_results_df = pd.DataFrame(tuned_results)

tuned_results_df.sort_values(by=['best_score'], ascending=True, inplace=True)

tuned_results_df
best_estimator = tuned_results_df.iloc[0,2]

y_predict = best_estimator.predict(df_test.drop(['Id'], axis=1))

y_predict_df = pd.DataFrame({'SalePrice': y_predict})



submission_df = pd.concat([df_test[['Id']], y_predict_df], axis=1)



submission_df.head()
submission_df.to_csv('Housing_Prices_Prediction_1.csv', index=False)