# import data manipulation library

import numpy as np

import pandas as pd



# import data visualization library

import matplotlib.pyplot as plt

import seaborn as sns



# import scientific computing library

import scipy



# import sklearn data preprocessing

from sklearn.preprocessing import RobustScaler



# import sklearn model class

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge

from sklearn.linear_model import ElasticNet

from sklearn.kernel_ridge import KernelRidge

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor



# import xgboost model class

import xgboost as xgb



# import sklearn model selection

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split



# import sklearn model evaluation regression metrics

from sklearn.metrics import mean_squared_error
# acquiring training and testing data

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
# visualize head of the training data

df_train.head(n=5)
# visualize tail of the testing data

df_test.tail(n=5)
# combine training and testing dataframe

df_train['DataType'], df_test['DataType'] = 'training', 'testing'

df_test.insert(df_test.shape[1] - 1, 'SalePrice', np.nan)

df_data = pd.concat([df_train, df_test], ignore_index=True)
def boxplot(categorical_x: list or str, numerical_y: list or str, data: pd.DataFrame, figsize: tuple = (4, 3), ncols: int = 5, nrows: int = None) -> plt.figure:

    """ Return a box plot applied for categorical variable in x-axis vs numerical variable in y-axis.

    

    Args:

        categorical_x (list or str): The categorical variable in x-axis.

        numerical_y (list or str): The numerical variable in y-axis.

        data (pd.DataFrame): The data to plot.

        figsize (tuple): The matplotlib figure size width and height in inches. Default to (4, 3).

        ncols (int): The number of columns for axis in the figure. Default to 5.

        nrows (int): The number of rows for axis in the figure. Default to None.

    

    Returns:

        plt.figure: The plot figure.

    """

    

    categorical_x, numerical_y = [categorical_x] if type(categorical_x) == str else categorical_x, [numerical_y] if type(numerical_y) == str else numerical_y

    if nrows is None: nrows = (len(categorical_x)*len(numerical_y) - 1) // ncols + 1

    

    fig, axes = plt.subplots(figsize=(figsize[0]*ncols , figsize[1]*nrows), ncols=ncols, nrows=nrows)

    axes = axes.flatten()

    _ = [sns.boxplot(x=vj, y=vi, data=data, ax=axes[i*len(categorical_x) + j]) for i, vi in enumerate(numerical_y) for j, vj in enumerate(categorical_x)]

    return fig
def scatterplot(numerical_x: list or str, numerical_y: list or str, data: pd.DataFrame, figsize: tuple = (4, 3), ncols: int = 5, nrows: int = None) -> plt.figure:

    """ Return a scatter plot applied for numerical variable in x-axis vs numerical variable in y-axis.

    

    Args:

        numerical_x (list or str): The numerical variable in x-axis.

        numerical_y (list or str): The numerical variable in y-axis.

        data (pd.DataFrame): The data to plot.

        figsize (tuple): The matplotlib figure size width and height in inches. Default to (4, 3).

        ncols (int): The number of columns for axis in the figure. Default to 5.

        nrows (int): The number of rows for axis in the figure. Default to None.

    

    Returns:

        plt.figure: The plot figure.

    """

    

    numerical_x, numerical_y = [numerical_x] if type(numerical_x) == str else numerical_x, [numerical_y] if type(numerical_y) == str else numerical_y

    if nrows is None: nrows = (len(numerical_x)*len(numerical_y) - 1) // ncols + 1

    

    fig, axes = plt.subplots(figsize=(figsize[0]*ncols , figsize[1]*nrows), ncols=ncols, nrows=nrows)

    axes = axes.flatten()

    _ = [sns.scatterplot(x=vj, y=vi, data=data, ax=axes[i*len(numerical_x) + j], rasterized=True) for i, vi in enumerate(numerical_y) for j, vj in enumerate(numerical_x)]

    return fig
# describe training and testing data

df_data.describe(include='all')
# convert dtypes numeric to object

col_convert = ['MSSubClass']

df_data[col_convert] = df_data[col_convert].astype('object')
# list all features type number

col_number = df_data.select_dtypes(include=['number']).columns.tolist()

print('features type number:\n items %s\n length %d' %(col_number, len(col_number)))



# list all features type object

col_object = df_data.select_dtypes(include=['object']).columns.tolist()

print('features type object:\n items %s\n length %d' %(col_object, len(col_object)))
# feature exploration: histogram of all numeric features

_ = df_data.hist(bins=20, figsize=(20, 15))
# feature extraction: sale price

df_data['SalePrice'] = np.log1p(df_data['SalePrice'])
# feature extraction: value of miscellaneous feature

df_data['MiscVal'] = np.log1p(df_data['MiscVal'])
# feature exploration: sale price

col_number = df_data.select_dtypes(include=['number']).columns.drop(['Id']).tolist()

col_object = df_data.select_dtypes(include=['object']).columns.tolist()

_ = scatterplot(col_number, 'SalePrice', df_data[df_data['DataType'] == 'training'])

_ = boxplot(col_object, 'SalePrice', df_data[df_data['DataType'] == 'training'])
# feature exploration: lot frontage

col_number = df_data.select_dtypes(include=['number']).columns.drop(['Id']).tolist()

col_object = df_data.select_dtypes(include=['object']).columns.tolist()

_ = scatterplot(col_number, 'LotFrontage', df_data)

_ = boxplot(col_object, 'LotFrontage', df_data)
# feature extraction: lot frontage

df_data['LotFrontage'] = df_data['LotFrontage'].fillna(df_data.groupby(['Neighborhood'])['LotFrontage'].transform('mean'))

df_data.loc[(df_data['LotFrontage'] > 200) & (df_data['DataType'] == 'trainnig'), 'DataType'] = 'excluded'
# feature extraction: lot area

df_data.loc[(df_data['LotArea'] > 100000) & (df_data['DataType'] == 'training'), 'DataType'] = 'excluded'
# feature extraction: basement type 1 finished area square feet

df_data.loc[(df_data['BsmtFinSF1'] > 4000) & (df_data['DataType'] == 'training'), 'DataType'] = 'excluded'
# feature extraction: basement total area square feet

df_data.loc[(df_data['TotalBsmtSF'] > 5000) & (df_data['DataType'] == 'training'), 'DataType'] = 'excluded'
# feature extraction: first floor area square feet

df_data.loc[(df_data['1stFlrSF'] > 4000) & (df_data['DataType'] == 'training'), 'DataType'] = 'excluded'
# feature extraction: above grade (ground) living area square feet

df_data.loc[(df_data['GrLivArea'] > 4500) & (df_data['DataType'] == 'training'), 'DataType'] = 'excluded'
# feature extraction: open porch area square feet

df_data.loc[(df_data['OpenPorchSF'] > 500) & (df_data['SalePrice'] < 11) & (df_data['DataType'] == 'training'), 'DataType'] = 'excluded'
# feature extraction: all features related to area

col_convert = ['LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',

               '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea',

               'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea']

df_data[col_convert] = df_data[col_convert].fillna(0)

df_data['TotalSF'] = df_data['TotalBsmtSF'] + df_data['GrLivArea']

df_data['TotalPorch'] = df_data['OpenPorchSF'] + df_data['EnclosedPorch'] + df_data['3SsnPorch'] + df_data['ScreenPorch']

df_data['TotalArea'] = df_data['TotalSF'] + df_data['TotalPorch'] + df_data['GarageArea'] + df_data['WoodDeckSF']
# feature extraction: all features related to room

col_convert = ['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',

               'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd']

df_data[col_convert] = df_data[col_convert].fillna(0)

df_data['TotalBathBsmt'] = df_data['BsmtFullBath'] + 0.5 * df_data['BsmtHalfBath']

df_data['TotalBathAbvGrd'] = df_data['FullBath'] + 0.5 * df_data['HalfBath']

df_data['TotalRmsAbvGrdIncBath'] = df_data['TotRmsAbvGrd'] + df_data['TotalBathAbvGrd']

df_data['TotalRms'] = df_data['TotalRmsAbvGrdIncBath'] + df_data['TotalBathBsmt']
# feature extraction: total area per rooms

df_data['AreaPerRmsBsmt'] = df_data['TotalBsmtSF'] / (df_data['TotalBathBsmt'] + 1)

df_data['AreaPerRmsGrLivAbvGrd'] = df_data['GrLivArea'] / (df_data['TotalRmsAbvGrdIncBath'] + 1)

df_data['AreaPerRmsTotal'] = df_data['TotalSF'] / (df_data['TotalRms'] + 1)
# feature extraction: all features related to quality and condition

col_convert = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',

               'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']

df_data[col_convert] = df_data[col_convert].replace('Ex', 5).replace('Gd', 4).replace('TA', 3).replace('Fa', 2).replace('Po', 1).replace('NA', 0)

df_data[col_convert] = df_data[col_convert].fillna(0).astype(int)

df_data['ExterQualCond'] = df_data['ExterQual'] * df_data['ExterCond']

df_data['BsmtQualCond'] = df_data['BsmtQual'] * df_data['BsmtCond']

df_data['GarageQualCond'] = df_data['GarageQual'] * df_data['GarageCond']

df_data['OverallQualCond'] = df_data['OverallQual'] * df_data['OverallCond']
# feature extraction: all features related to exposure

col_convert = ['BsmtExposure']

df_data[col_convert] = df_data[col_convert].replace('Gd', 4).replace('Av', 3).replace('Mn', 2).replace('No', 1).replace('NA', 0)

df_data[col_convert] = df_data[col_convert].fillna(0).astype(int)
# feature extraction: all features related to basement finished

col_convert = ['BsmtFinType1', 'BsmtFinType2']

df_data[col_convert] = df_data[col_convert].replace('GLQ', 6).replace('ALQ', 5).replace('BLQ', 4).replace('Rec', 3).replace('LwQ', 2).replace('Unf', 1).replace('NA', 0)

df_data[col_convert] = df_data[col_convert].fillna(0).astype(int)
# feature extraction: all features related to garage finished

col_convert = ['GarageFinish']

df_data[col_convert] = df_data[col_convert].replace('Fin', 3).replace('RFn', 2).replace('Unf', 1).replace('NA', 0)

df_data[col_convert] = df_data[col_convert].fillna(0).astype(int)
# feature extraction: all features related to fence

col_convert = ['Fence']

df_data[col_convert] = df_data[col_convert].replace('GdPrv', 4).replace('MnPrv', 3).replace('GdWo', 2).replace('MnWw', 1).replace('NA', 0)

df_data[col_convert] = df_data[col_convert].fillna(0).astype(int)
# feature extraction: all features related to year

df_data['GarageYrBlt'] = df_data['GarageYrBlt'].fillna(df_data['YearBuilt'])

df_data['YearBuiltRemod'] = df_data['YearRemodAdd'] - df_data['YearBuilt']

df_data['YearBuiltSold'] = df_data['YrSold'] - df_data['YearBuilt']

df_data['YearRemodSold'] = df_data['YrSold'] - df_data['YearRemodAdd']

df_data['YearGarageSold'] = df_data['YrSold'] - df_data['GarageYrBlt']
# feature exploration: season dataframe

df_season = df_data.loc[df_data['DataType'] == 'training'].groupby(['YrSold', 'MoSold'], as_index=False).agg({

    'SalePrice': 'mean'

})

fig, axes = plt.subplots(figsize=(20, 3))

_ = sns.pointplot(x='MoSold', y='SalePrice', data=df_season, join=True, hue='YrSold')
# feature extraction: fillna on type of utilities available

df_data['Utilities'] = df_data['Utilities'].fillna('ELO')
# feature extraction: fillna on type of sale

df_data['SaleType'] = df_data['SaleType'].fillna('Oth')
# feature extraction: fillna with repetitive

col_fillnas = ['MSZoning', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Electrical', 'Functional']

for col_fillna in col_fillnas: df_data[col_fillna] = df_data[col_fillna].fillna(df_data[col_fillna].value_counts().idxmax())
# feature extraction: fillna with na

col_fillnas = ['Alley', 'GarageType']

for col_fillna in col_fillnas: df_data[col_fillna] = df_data[col_fillna].fillna('NA')
# feature extraction: fillna with 0

col_fillnas = ['GarageCars', 'MiscFeature']

df_data[col_fillnas] = df_data[col_fillnas].fillna(0)
# verify and print columns contain nan

if df_data.isna().any().any(): print(df_data.loc[:, df_data.columns[df_data.isna().any()].tolist()].describe(include='all'))

else: print('no null entry')
# feature extraction: apply log1p transform for all high skewness numeric features

col_number = df_data.select_dtypes(include=['number']).columns.drop(['MiscVal', 'SalePrice', 'YearBuiltRemod', 'YearBuiltSold', 'YearRemodSold', 'YearGarageSold']).tolist()

for col_transform in col_number:

    skewness = scipy.stats.skew(df_data[col_transform].dropna())

    if skewness > 0.75: df_data[col_transform] = np.log1p(df_data[col_transform])
# feature exploration: sale price

col_number = df_data.select_dtypes(include=['number']).columns.drop(['Id']).tolist()

col_object = df_data.select_dtypes(include=['object']).columns.tolist()

_ = scatterplot(col_number, 'SalePrice', df_data[df_data['DataType'] == 'training'])

_ = boxplot(col_object, 'SalePrice', df_data[df_data['DataType'] == 'training'])
# feature extraction: sale price

df_data['SalePrice'] = df_data['SalePrice'].fillna(0)
# convert category codes for data dataframe

df_data = pd.get_dummies(df_data, columns=None, drop_first=True)
# describe data dataframe

df_data.describe(include='all')
# verify dtypes object

df_data.info()
# compute pairwise correlation of columns, excluding NA/null values and present through heat map

corr = df_data[df_data['DataType_training'] == 1].corr()

fig, axes = plt.subplots(figsize=(200, 150))

heatmap = sns.heatmap(corr, annot=True, cmap=plt.cm.RdBu, fmt='.1f', square=True, vmin=-0.8, vmax=0.8)
# select all features to evaluate the feature importances

x = df_data[df_data['DataType_training'] == 1].drop(['Id', 'SalePrice', 'DataType_training', 'DataType_testing'], axis=1)

y = df_data.loc[df_data['DataType_training'] == 1, 'SalePrice']
# set up lasso regression to find the feature importances

lassoreg = Lasso(alpha=1e-5).fit(x, y)

feat = pd.DataFrame(data=lassoreg.coef_, index=x.columns, columns=['FeatureImportances']).sort_values(['FeatureImportances'], ascending=False)
# plot the feature importances

feat[(feat['FeatureImportances'] < -1e-3) | (feat['FeatureImportances'] > 1e-3)].dropna().plot(y='FeatureImportances', figsize=(20, 5), kind='bar')

plt.axhline(-0.005, color="grey")

plt.axhline(0.005, color="grey")
# list feature importances

model_feat = feat[(feat['FeatureImportances'] < -0.005) | (feat['FeatureImportances'] > 0.005)].index
# select the important features

x = df_data.loc[df_data['DataType_training'] == 1, model_feat]

y = df_data.loc[df_data['DataType_training'] == 1, 'SalePrice']
# create scaler to the features

scaler = RobustScaler()

x = scaler.fit_transform(x)
# perform train-test (validate) split

x_train, x_validate, y_train, y_validate = train_test_split(x, y, test_size=0.25, random_state=58)
# linear regression model setup

model_linreg = LinearRegression()



# linear regression model fit

model_linreg.fit(x_train, y_train)



# linear regression model prediction

model_linreg_ypredict = model_linreg.predict(x_validate)



# linear regression model metrics

model_linreg_rmse = mean_squared_error(y_validate, model_linreg_ypredict) ** 0.5

model_linreg_cvscores = np.sqrt(np.abs(cross_val_score(model_linreg, x, y, cv=5, scoring='neg_mean_squared_error')))

print('linear regression\n  root mean squared error: %0.4f, cross validation score: %0.4f (+/- %0.4f)' %(model_linreg_rmse, model_linreg_cvscores.mean(), 2 * model_linreg_cvscores.std()))
# lasso regression model setup

model_lassoreg = Lasso(alpha=0.001, max_iter=1024)



# lasso regression model fit

model_lassoreg.fit(x_train, y_train)



# lasso regression model prediction

model_lassoreg_ypredict = model_lassoreg.predict(x_validate)



# lasso regression model metrics

model_lassoreg_rmse = mean_squared_error(y_validate, model_lassoreg_ypredict) ** 0.5

model_lassoreg_cvscores = np.sqrt(np.abs(cross_val_score(model_lassoreg, x, y, cv=5, scoring='neg_mean_squared_error')))

print('lasso regression\n  root mean squared error: %0.4f, cross validation score: %0.4f (+/- %0.4f)' %(model_lassoreg_rmse, model_lassoreg_cvscores.mean(), 2 * model_lassoreg_cvscores.std()))
# specify the hyperparameter space

params = {'alpha': np.logspace(-4, 4, base=10, num=9),

          'max_iter': [1024],

}



# lasso regression grid search model setup

model_lassoreg_cv = GridSearchCV(model_lassoreg, params, iid=False, cv=5)



# lasso regression grid search model fit

model_lassoreg_cv.fit(x_train, y_train)



# lasso regression grid search model prediction

model_lassoreg_cv_ypredict = model_lassoreg_cv.predict(x_validate)



# lasso regression grid search model metrics

model_lassoreg_cv_rmse = mean_squared_error(y_validate, model_lassoreg_cv_ypredict) ** 0.5

model_lassoreg_cv_cvscores = np.sqrt(np.abs(cross_val_score(model_lassoreg_cv, x, y, cv=5, scoring='neg_mean_squared_error')))

print('lasso regression grid search\n  root mean squared error: %0.4f, cross validation score: %0.4f (+/- %0.4f)' %(model_lassoreg_cv_rmse, model_lassoreg_cv_cvscores.mean(), 2 * model_lassoreg_cv_cvscores.std()))

print('  best parameters: %s' %model_lassoreg_cv.best_params_)
# ridge regression model setup

model_ridgereg = Ridge(alpha=10)



# ridge regression model fit

model_ridgereg.fit(x_train, y_train)



# ridge regression model prediction

model_ridgereg_ypredict = model_ridgereg.predict(x_validate)



# ridge regression model metrics

model_ridgereg_rmse = mean_squared_error(y_validate, model_ridgereg_ypredict) ** 0.5

model_ridgereg_cvscores = np.sqrt(np.abs(cross_val_score(model_ridgereg, x, y, cv=5, scoring='neg_mean_squared_error')))

print('ridge regression\n  root mean squared error: %0.4f, cross validation score: %0.4f (+/- %0.4f)' %(model_ridgereg_rmse, model_ridgereg_cvscores.mean(), 2 * model_ridgereg_cvscores.std()))
# specify the hyperparameter space

params = {'alpha': np.logspace(-4, 4, base=10, num=9)}



# ridge regression grid search model setup

model_ridgereg_cv = GridSearchCV(model_ridgereg, params, iid=False, cv=5)



# ridge regression grid search model fit

model_ridgereg_cv.fit(x_train, y_train)



# ridge regression grid search model prediction

model_ridgereg_cv_ypredict = model_ridgereg_cv.predict(x_validate)



# ridge regression grid search model metrics

model_ridgereg_cv_rmse = mean_squared_error(y_validate, model_ridgereg_cv_ypredict) ** 0.5

model_ridgereg_cv_cvscores = np.sqrt(np.abs(cross_val_score(model_ridgereg_cv, x, y, cv=5, scoring='neg_mean_squared_error')))

print('ridge regression grid search\n  root mean squared error: %0.4f, cross validation score: %0.4f (+/- %0.4f)' %(model_ridgereg_cv_rmse, model_ridgereg_cv_cvscores.mean(), 2 * model_ridgereg_cv_cvscores.std()))

print('  best parameters: %s' %model_ridgereg_cv.best_params_)
# elastic net regression model setup

model_elasticnetreg = ElasticNet(alpha=0.1, l1_ratio=0.1, max_iter=1024)



# elastic net regression model fit

model_elasticnetreg.fit(x_train, y_train)



# elastic net regression model prediction

model_elasticnetreg_ypredict = model_elasticnetreg.predict(x_validate)



# elastic net regression model metrics

model_elasticnetreg_rmse = mean_squared_error(y_validate, model_elasticnetreg_ypredict) ** 0.5

model_elasticnetreg_cvscores = np.sqrt(np.abs(cross_val_score(model_elasticnetreg, x, y, cv=5, scoring='neg_mean_squared_error')))

print('elastic net regression\n  root mean squared error: %0.4f, cross validation score: %0.4f (+/- %0.4f)' %(model_elasticnetreg_rmse, model_elasticnetreg_cvscores.mean(), 2 * model_elasticnetreg_cvscores.std()))
# specify the hyperparameter space

params = {'alpha': np.logspace(-4, 4, base=10, num=9),

          'l1_ratio': np.linspace(0.1, 0.9, num=5),

          'max_iter': [1024],

}



# elastic net regression grid search model setup

model_elasticnetreg_cv = GridSearchCV(model_elasticnetreg, params, iid=False, cv=5)



# elastic net regression grid search model fit

model_elasticnetreg_cv.fit(x_train, y_train)



# elastic net regression grid search model prediction

model_elasticnetreg_cv_ypredict = model_elasticnetreg_cv.predict(x_validate)



# elastic net regression grid search model metrics

model_elasticnetreg_cv_rmse = mean_squared_error(y_validate, model_elasticnetreg_cv_ypredict) ** 0.5

model_elasticnetreg_cv_cvscores = np.sqrt(np.abs(cross_val_score(model_elasticnetreg_cv, x, y, cv=5, scoring='neg_mean_squared_error')))

print('elastic net regression grid search\n  root mean squared error: %0.4f, cross validation score: %0.4f (+/- %0.4f)' %(model_elasticnetreg_cv_rmse, model_elasticnetreg_cv_cvscores.mean(), 2 * model_elasticnetreg_cv_cvscores.std()))

print('  best parameters: %s' %model_elasticnetreg_cv.best_params_)
# kernel ridge regression model setup

model_kernelridgereg = KernelRidge(alpha=0.1, kernel='polynomial', degree=2)



# kernel ridge regression model fit

model_kernelridgereg.fit(x_train, y_train)



# kernel ridge regression model prediction

model_kernelridgereg_ypredict = model_kernelridgereg.predict(x_validate)



# kernel ridge regression model metrics

model_kernelridgereg_rmse = mean_squared_error(y_validate, model_kernelridgereg_ypredict) ** 0.5

model_kernelridgereg_cvscores = np.sqrt(np.abs(cross_val_score(model_kernelridgereg, x, y, cv=5, scoring='neg_mean_squared_error')))

print('kernel ridge regression\n  root mean squared error: %0.4f, cross validation score: %0.4f (+/- %0.4f)' %(model_kernelridgereg_rmse, model_kernelridgereg_cvscores.mean(), 2 * model_kernelridgereg_cvscores.std()))
# specify the hyperparameter space

params = {'alpha': np.logspace(-4, 4, base=10, num=9),

          'degree': [1, 2, 3, 4, 5],

}



# kernel ridge regression grid search model setup

model_kernelridgereg_cv = GridSearchCV(model_kernelridgereg, params, iid=False, cv=5)



# kernel ridge regression grid search model fit

model_kernelridgereg_cv.fit(x_train, y_train)



# kernel ridge regression grid search model prediction

model_kernelridgereg_cv_ypredict = model_kernelridgereg_cv.predict(x_validate)



# kernel ridge regression grid search model metrics

model_kernelridgereg_cv_rmse = mean_squared_error(y_validate, model_kernelridgereg_cv_ypredict) ** 0.5

model_kernelridgereg_cv_cvscores = np.sqrt(np.abs(cross_val_score(model_kernelridgereg_cv, x, y, cv=5, scoring='neg_mean_squared_error')))

print('kernel ridge regression grid search\n  root mean squared error: %0.4f, cross validation score: %0.4f (+/- %0.4f)' %(model_kernelridgereg_cv_rmse, model_kernelridgereg_cv_cvscores.mean(), 2 * model_kernelridgereg_cv_cvscores.std()))

print('  best parameters: %s' %model_kernelridgereg_cv.best_params_)
# decision tree regression model setup

model_treereg = DecisionTreeRegressor(splitter='best', min_samples_split=5)



# decision tree regression model fit

model_treereg.fit(x_train, y_train)



# decision tree regression model prediction

model_treereg_ypredict = model_treereg.predict(x_validate)



# decision tree regression model metrics

model_treereg_rmse = mean_squared_error(y_validate, model_treereg_ypredict) ** 0.5

model_treereg_cvscores = np.sqrt(np.abs(cross_val_score(model_treereg, x, y, cv=5, scoring='neg_mean_squared_error')))

print('decision tree regression\n  root mean squared error: %0.4f, cross validation score: %0.4f (+/- %0.4f)' %(model_treereg_rmse, model_treereg_cvscores.mean(), 2 * model_treereg_cvscores.std()))
# random forest regression model setup

model_forestreg = RandomForestRegressor(n_estimators=100, min_samples_split=3, random_state=58)



# random forest regression model fit

model_forestreg.fit(x_train, y_train)



# random forest regression model prediction

model_forestreg_ypredict = model_forestreg.predict(x_validate)



# random forest regression model metrics

model_forestreg_rmse = mean_squared_error(y_validate, model_forestreg_ypredict) ** 0.5

model_forestreg_cvscores = np.sqrt(np.abs(cross_val_score(model_forestreg, x, y, cv=5, scoring='neg_mean_squared_error')))

print('random forest regression\n  root mean squared error: %0.4f, cross validation score: %0.4f (+/- %0.4f)' %(model_forestreg_rmse, model_forestreg_cvscores.mean(), 2 * model_forestreg_cvscores.std()))
# specify the hyperparameter space

params = {'n_estimators': [100],

          'max_depth': [10, 20, None],

          'min_samples_split': [3, 5, 7, 9],

          'random_state': [58],

}



# random forest regression grid search model setup

model_forestreg_cv = GridSearchCV(model_forestreg, params, iid=False, cv=5)



# random forest regression grid search model fit

model_forestreg_cv.fit(x_train, y_train)



# random forest regression grid search model prediction

model_forestreg_cv_ypredict = model_forestreg_cv.predict(x_validate)



# random forest regression grid search model metrics

model_forestreg_cv_rmse = mean_squared_error(y_validate, model_forestreg_cv_ypredict) ** 0.5

model_forestreg_cv_cvscores = np.sqrt(np.abs(cross_val_score(model_forestreg_cv, x, y, cv=5, scoring='neg_mean_squared_error')))

print('random forest regression grid search\n  root mean squared error: %0.4f, cross validation score: %0.4f (+/- %0.4f)' %(model_forestreg_cv_rmse, model_forestreg_cv_cvscores.mean(), 2 * model_forestreg_cv_cvscores.std()))

print('  best parameters: %s' %model_forestreg_cv.best_params_)
# xgboost regression model setup

model_xgbreg = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=1000, objective='reg:linear', booster='gbtree',

                                gamma=0, subsample=0.9, colsample_bytree=0.9, reg_alpha=0.1, reg_lambda=0.9, random_state=58)



# xgboost regression model fit

model_xgbreg.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_validate, y_validate)], early_stopping_rounds=50, verbose=False, callbacks=[xgb.callback.print_evaluation(period=50)])



# xgboost regression model prediction

model_xgbreg_ypredict = model_xgbreg.predict(x_validate)



# xgboost regression model metrics

model_xgbreg_rmse = mean_squared_error(y_validate, model_xgbreg_ypredict) ** 0.5

model_xgbreg_cvscores = np.sqrt(np.abs(cross_val_score(model_xgbreg, x, y, cv=5, scoring='neg_mean_squared_error')))

print('xgboost regression\n  root mean squared error: %0.4f, cross validation score: %0.4f (+/- %0.4f)' %(model_xgbreg_rmse, model_xgbreg_cvscores.mean(), 2 * model_xgbreg_cvscores.std()))
# model selection

final_model = model_kernelridgereg



# prepare testing data and compute the observed value

x_test = df_data.loc[df_data['DataType_testing'] == 1, model_feat]

x_test = scaler.transform(x_test)

y_test = pd.DataFrame(final_model.predict(x_test), columns=['SalePrice'], index=df_data.loc[df_data['DataType_testing'] == 1, 'Id'])

y_test['SalePrice'] = np.expm1(y_test['SalePrice'])
# submit the results

out = pd.DataFrame({'Id': y_test.index, 'SalePrice': y_test['SalePrice']})

out.to_csv('submission.csv', index=False)