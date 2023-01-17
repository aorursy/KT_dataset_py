# data analysis and wrangling

import math

import numpy 

import pandas 

import random

import scipy.stats



# visualization

import seaborn ; seaborn.set(style='whitegrid') # nicer plotting style

import matplotlib.pyplot as pyplot

%matplotlib inline



from IPython.core.display import display, HTML

display(HTML("<style>.container { width:80% !important; }</style>"))

pandas.options.display.max_columns = None  # to display all the columns
import matplotlib.gridspec as gridspec



# Boxplots with histograms above

def plot_boxplot_with_hist(df_input, target='SalePrice', ncols=5, figsize=(20, 32)):

    

    unique_count = df_input.nunique()

    df = df_input[list(unique_count[unique_count <= 25].index) + ['SalePrice']].copy()

    nrows = math.ceil((df.shape[1] - 1) / ncols) 

    

    row_width_increment = 1 / ncols

    col_width_decrement = 1 / nrows



    fig = pyplot.figure(figsize=figsize)



    for count, col in enumerate(df.columns.drop(target)):

        col_num, row_num = count % ncols, count // ncols

        left = row_width_increment * col_num

        right = row_width_increment * (col_num + 1) - 0.05

        top = 1 - row_num * col_width_decrement

        bottom = 1.03 - (row_num + 1) * col_width_decrement 



        gs1 = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=(.3, .7), left=left, bottom=bottom, top=top, right=right, hspace=0.05)

        hist_ax = fig.add_subplot(gs1[0]) ; hist_ax.tick_params(axis='x', which='major', pad=.1)

        seaborn.countplot(x=col, data=df, ax=hist_ax)

        box_ax = fig.add_subplot(gs1[1]) ; box_ax.tick_params(axis='x', which='major', pad=.1)

        seaborn.boxplot(x=col, y=target, data=df, ax=box_ax)

        

        if df[col].dtype == 'O':

            hist_ax.set_xticklabels(hist_ax.get_xticklabels(), rotation=45) 

            box_ax.set_xticklabels(box_ax.get_xticklabels(), rotation=45) 

        

        if col_num != 0:

            hist_ax.set_ylabel('')

            box_ax.set_ylabel('')



# Scatter plots between the features and the target 

def plot_pairwise_with_target(df_input, target='SalePrice', ncols=6):

    

    unique_count = df_input.nunique()

    df = df_input[list(unique_count[unique_count > 25].index)].drop('Id', axis=1)

    nrows = math.ceil((df.shape[1] - 1) / ncols) 

    

    fig, ax = pyplot.subplots(nrows=nrows, ncols=ncols, figsize=(20, 11), sharey='row')

    

    for count, col in enumerate(df.columns.drop(target)):

        this_ax = ax[count // ncols][count % ncols]

        seaborn.regplot(x=col, y=target, data=df, color='b', fit_reg=False, scatter_kws={'s': 10, 'alpha': 0.5}, ax=this_ax)  

        #this_ax.set_xticklabels(this_ax.get_xticklabels(), rotation=45) 

        if count % ncols != 0:

            ax[count // ncols][count % ncols].set_ylabel('')

            

# Correlation matrix (upper triangle)          

def plot_correlation_map(df, figsize=(18,18)):

    

    corr = df.corr()

    

    # Generate a mask for the upper triangle

    mask = numpy.zeros_like(corr, dtype=numpy.bool)

    mask[numpy.triu_indices_from(mask)] = True

    

    _ , ax = pyplot.subplots(figsize=figsize)

    _ = seaborn.heatmap(

        corr, 

        mask=mask,

        cmap="RdBu_r",

        square=True, 

        cbar_kws={'shrink': .7}, 

        ax=ax, 

        annot=False, 

        annot_kws={'fontsize': 9}

    )
# Load both data sets

df_train = pandas.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

df_test = pandas.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')



# Store both data sets in one dictionary in order to make each transformation step easier and clearer

data_sets = {'df_train': df_train, 'df_test': df_test}
print(data_sets['df_train'].shape, data_sets['df_test'].shape)

data_sets['df_train'].head(2)
data_sets['df_test'].head(2)
# Differences in data types

print(pandas.DataFrame({'train': data_sets['df_train'].dtypes.value_counts(), 'test': data_sets['df_test'].dtypes.value_counts()}))

pandas.DataFrame({'train': data_sets['df_train'].dtypes, 'test': data_sets['df_test'].dtypes}).query('train != test')
# Valid values

pandas.DataFrame({'train': data_sets['df_train'].count(), 'test': data_sets['df_test'].count()}).sort_values('train')
# Transform ordinal categorical variables into numerical ones

# For these variables the NaN's correspond to houses that don't have the feature. 

# So we'll map those to 0 (lowest rank value).

for key, df in data_sets.items():

    

    print(f'Data set being processed: {key}')



    mapping = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

    cols_to_replace = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 

                       'GarageCond', 'PoolQC']

    df.loc[:, cols_to_replace] = df[cols_to_replace].replace(mapping).fillna(0)



    mapping = {'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}

    cols_to_replace = ['BsmtExposure']

    df.loc[:, cols_to_replace] = df[cols_to_replace].replace(mapping).fillna(0)



    mapping = {'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}

    cols_to_replace = ['BsmtFinType1', 'BsmtFinType2']

    df.loc[:, cols_to_replace] = df[cols_to_replace].replace(mapping).fillna(0)

    

    mapping = {'Unf': 1, 'RFn': 2, 'Fin': 3}

    cols_to_replace = ['GarageFinish']

    df.loc[:, cols_to_replace] = df[cols_to_replace].replace(mapping).fillna(0)

    

    mapping = {'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4}

    cols_to_replace = ['Fence']

    df.loc[:, cols_to_replace] = df[cols_to_replace].replace(mapping).fillna(0)

    

    mapping = {'Reg': 0, 'IR1': 1, 'IR2': 2, 'IR3': 3}

    cols_to_replace = ['LotShape']

    df.loc[:, cols_to_replace] = df[cols_to_replace].replace(mapping)

    

    mapping = {'ELO': 0, 'NoSeWa': 1, 'NoSewr': 2, 'AllPub': 3}

    cols_to_replace = ['Utilities']

    df.loc[:, cols_to_replace] = df[cols_to_replace].replace(mapping)

    

    mapping = {'Gtl': 0, 'Mod': 1, 'Sev': 2}

    cols_to_replace = ['LandSlope']

    df.loc[:, cols_to_replace] = df[cols_to_replace].replace(mapping)

    

    mapping = {'Sal': 0, 'Sev': 1, 'Maj2': 2, 'Maj1': 3, 'Mod': 4, 'Min2': 5, 'Min1': 6, 'Typ': 7}

    cols_to_replace = ['Functional']

    df.loc[:, cols_to_replace] = df[cols_to_replace].replace(mapping)

    

    mapping = {'N': 0, 'P': 1, 'Y': 2}

    cols_to_replace = ['PavedDrive']

    df.loc[:, cols_to_replace] = df[cols_to_replace].replace(mapping)

    

    data_sets[key] = df

    

    print(f'Data set processed successfully: {key}')
# Transform numerical features into string (for the ones that are categories)

for key, df in data_sets.items():

    

    print(f'Data set being processed: {key}')

    

    mapping = {num: f'MS{num}' for num in data_sets[key]['MSSubClass'].unique()}

    cols_to_replace = ['MSSubClass']

    df.loc[:, cols_to_replace] = df[cols_to_replace].replace(mapping)

    

    print(f'Data set processed successfully: {key}')
# We fill missing values with 0 for the numerical columns (expect GarageYrBlt)

# and with a new category ('no feature') for the remaining nominal columns

for key, df in data_sets.items():

    

    print(f'Data set being processed: {key}')

    

    df = df.apply(lambda x: x.fillna(0) if (x.dtype in ['float64', 'int64'] and x.name != 'GarageYrBlt') else x)

                                          

    df.loc[:, ['Alley', 'GarageType', 'MasVnrType', 'MiscFeature']] = df[['Alley', 'GarageType', 'MasVnrType', 'MiscFeature']].fillna('No feature')

    

    data_sets[key] = df

                                                                                    

    print(f'Data set processed successfully: {key}')
# Distribution of SalePrice

seaborn.distplot(data_sets['df_train']['SalePrice'])



print("Skewness: {}".format(df_train['SalePrice'].skew()))

data_sets['df_train'][['SalePrice']].describe()
# We first take a look at the categorical (and ordinal) variables (up to 12 unique values)

data_sets['df_train'].nunique().sort_values().iloc[:20]
# We observe the relation between SalePrice and all the categorical variables, while checking their distribution

plot_boxplot_with_hist(data_sets['df_train'])
# Pairwise relations between SalePrice and continuous features

plot_pairwise_with_target(data_sets['df_train'])
# Linear correlations between numerical features

plot_correlation_map(data_sets['df_train'])
data_sets['df_train'].corr()['SalePrice'].sort_values(ascending=False).iloc[1:].plot(kind='bar', figsize=(12, 4), color='b')
for key, df in data_sets.items():

    

    print(f'Data set being processed: {key}')

    

    # 'YearSold' and 'MoSold' into 'YrsFromSale' (we take latest year 2010 as a reference).

    # Same for 'YearBuilt' which we transform into 'YrsFromConst' (better interpretable).

    df = df.assign(

        TimeFromSale=2010 - df['YrSold'] + (12 - df['MoSold']) / 12,

        YrsFromConst=2010 - df['YearBuilt'],

        YrsFromRemod=2010 - df['YearRemodAdd'], 

    ).drop(['YrSold', 'MoSold', 'YearBuilt', 'YearRemodAdd'], axis=1)

    

    # Combine the freshly created 'YrsFromConst' and 'YrsFromRemod' into a single feature (take the mean)

    df = df.assign(

        YrsFromCombined=0.5 * (df['YrsFromConst'] + df['YrsFromRemod']),

    ).drop(['YrsFromConst', 'YrsFromRemod'], axis=1)

        

    # Combine 'FullBath' and 'HalfBath' into BathTot (same for 'BsmtFullBath' and 'BsmtHalfBath').

    df = df.assign(

        BathTot=lambda x: x['FullBath'] + 0.5 * x['HalfBath'],

        BsmtBathTot=lambda x: x['BsmtFullBath'] + 0.5 * x['BsmtHalfBath'],

    ).drop(['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath'], axis=1)

    

    # Combine highly correlated features with underlying logic ('GarageQual' and 'GarageCond').

    df = df.assign(

        FirePlaceScore=lambda x: x['Fireplaces'] * x['FireplaceQu'],

        GarageScore=lambda x: x['GarageQual'] * x['GarageCond'],

    ).drop(['Fireplaces', 'FireplaceQu', 'GarageQual', 'GarageCond'], axis=1)

    

    # MasVnrType

    mapping = {'None': 0, 'No feature': 0, 'BrkCmn': 1, 'BrkFace': 1, 'CBlock': 1, 'Stone': 1}

    df = df.assign(

        MasVnrType=df['MasVnrType'].replace(mapping),

        MasVnrTypeScore=lambda x: x['MasVnrType'] * x['MasVnrArea']

    ).drop(['MasVnrArea', 'MasVnrType'], axis=1)

    

    # BsmtScore

    df = df.assign(

        BsmtFinScore1=df['BsmtFinSF1'] * df['BsmtFinType1'],

        BsmtFinScore2=df['BsmtFinSF2'] * df['BsmtFinType2'],

    ).drop(['BsmtFinSF1', 'BsmtFinType1', 'BsmtFinSF2', 'BsmtFinType2'], axis=1)

    

    # 'GarageCars' and 'GarageArea' highly correlated (almost 0.9 on df_train!).

    # We choose to drop 'GarageArea' which is slightly less correlated with the target.

    # 'GarageYrBlt' is correlated with 'YearBuilt' and there is no obvious way to deal with the missing values.

    # We choose to drop it.

    df = df.drop(['GarageArea', 'GarageYrBlt'], axis=1)

    

    # Drop 'PoolArea' as the feature is too correlated to 'PoolQC'

    # and according to box plots the relation between 'SalePrice' and 'PoolQC' seems more robust

    df = df.drop(['PoolArea'], axis=1)

    

    # Drop 'BsmtFinType2' as too correlated with 'BsmtFinSF2'

    # Same thing for 'TotalBsmtSF' too correlated with '1stFlrSF' (which has more non zero values)

    df = df.drop(['TotalBsmtSF'], axis=1)

    

    # Drop 'TotRmsAbvGrd' as too correlated with 'GrLivArea'

    df = df.drop(['TotRmsAbvGrd'], axis=1)

    

    data_sets[key] = df

    

    print(f'Data set processed successfully: {key}')
plot_correlation_map(data_sets['df_train'], figsize=(13, 13))
modes_train = data_sets['df_train'].select_dtypes('object').mode().iloc[0]



for key, df in data_sets.items():

    

    print(f'Data set being processed: {key}')

    

    temp = df.select_dtypes('object').isnull().any(axis=0)

    print(temp[temp].index)

    data_sets[key] = df.select_dtypes('object').fillna(modes_train).join(df.select_dtypes(exclude=['object']))

                                                                                     

    print(f'Data set processed successfully: {key} \n')

    

    

del modes_train, temp
data_sets['df_train'].select_dtypes(['int64', 'float64']).info()
# compute skewness on df_train

skewed_feats = data_sets['df_train'].select_dtypes(['int64', 'float64']).drop('SalePrice', axis=1).apply(lambda x: scipy.stats.skew(x.dropna()))

skewed_feats = skewed_feats[skewed_feats > 1]

skewed_feats = list(skewed_feats.index)

print(skewed_feats)



# Remove the categorical variables from the list (it's definitely not a nice way to do it)

for feature in ['LotShape', 'LandSlope', 'ExterCond', 'BsmtExposure', 'KitchenAbvGr', 'PoolQC', 'Fence', 'MasVnrTypeScore',

                'BsmtFinScore1', 'BsmtFinScore2']:

    skewed_feats.remove(feature)

    



# Remove the skewed (and continuous) features from both data sets

for key, df in data_sets.items():

    print(f'Data set being processed: {key}')

    df[list(skewed_feats)] = numpy.log1p(df[list(skewed_feats)])

    print(f'Data set processed successfully: {key} \n')
# Encode categorical columns into dummy variables

for key, df in data_sets.items():

    

    print(f'Data set being processed: {key}')

    

    columns_to_process = df.select_dtypes(include='object').columns

    other_columns = df.select_dtypes(exclude='object').columns

    

    data_sets[key] = df.loc[:, other_columns].join(pandas.get_dummies(df.loc[:, columns_to_process], drop_first=True))

                                                                                                                                  

    print(f'Data set processed successfully: {key}')

    

    

# There are categories in df_train that don't appear in df_test.

# We therefore have to add the corresponding columns in df_test and fill the column with 0.

# Besides there is an additional column in df_test ('MSSubClass_MS150') which we drop.

def complete_df_test():

    cols_dict = {}

    for col in data_sets['df_train'].columns.drop('SalePrice'):

        if not(col in data_sets['df_test'].columns):

            cols_dict[col] = lambda x: 0

    data_sets['df_test'] = data_sets['df_test'].assign(**cols_dict).drop(['MSSubClass_MS150'], axis=1)

    

complete_df_test()
data_sets['df_train'].shape, data_sets['df_test'].shape
# Regressors

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import Ridge



# Scalers

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import RobustScaler



# Pipeline

from sklearn.pipeline import Pipeline



# Model Selection 

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV

from sklearn.feature_selection import SelectFromModel



# Metrics

from sklearn.metrics import mean_squared_error

from sklearn.metrics import make_scorer
def generate_train_test_with_preselection(threshold_mult=2, plot=True):

    

    # We use log(SalePrice) as a target since the evaluation metric is the RMSE on log(SalePrice)

    y_train = numpy.log(data_sets['df_train']['SalePrice'])

    X_train = data_sets['df_train'].drop(['SalePrice', 'Id'], axis=1)

    X_test = data_sets['df_test'].loc[:, list(X_train.columns)].copy()

    assert (X_train.columns == X_test.columns).all(), 'At least one mismatch in the columns ordering'

    

    # Fit randome forest regressor and plot feature importance

    reg = RandomForestRegressor(n_estimators=100, criterion='mse', max_features='sqrt', random_state=42)

    reg = reg.fit(X_train, y_train)



    df = pandas.DataFrame({

        'features': X_train.columns,

        'importance': reg.feature_importances_,

    }).sort_values('importance', ascending=True)

    

    threshold = threshold_mult * df['importance'].mean()

    if plot:

        fig, ax = pyplot.subplots(figsize=(15, 34))

        df.plot(y='importance', x='features', kind='barh', color='b', ax=ax)

        ax.axvline(threshold, linestyle='--', color='black')

        pyplot.show()

        

    # Pre feature selection

    features_selector = SelectFromModel(reg, prefit=True, threshold=threshold)

    reduced_X_train = X_train[X_train.columns[features_selector.get_support(indices=True)]]

    reduced_X_test = X_test[X_test.columns[features_selector.get_support(indices=True)]]

            

    return reduced_X_train, y_train, reduced_X_test





# We also define a score function that returns directly the rmse (SalePrice is already log-transformed)

def rmse(y_true, y_pred):

    return math.sqrt(mean_squared_error(y_true, y_pred))



rmse_score = make_scorer(rmse, greater_is_better=False)
_ = generate_train_test_with_preselection()
reduced_X_train, y_train, reduced_X_test = generate_train_test_with_preselection(threshold_mult=.2, plot=False)



reg = RandomForestRegressor()

parameter_grid = {

                 'n_estimators': [100],

                 'max_depth' : [4, 6, 8, 10, 14],

                 'max_features': ['sqrt'],

                 'min_samples_split': [3, 5, 10],

                 'min_samples_leaf': [3, 5, 10],

                 }

cross_validation = KFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(estimator=reg, param_grid=parameter_grid, cv=cross_validation, scoring=rmse_score, refit=True, return_train_score=True)

grid_search.fit(reduced_X_train, y_train) 



print('Best score: {}'.format(grid_search.best_score_))

print('Best parameters: {}'.format(grid_search.best_params_))

print('Best estimator: {}'.format(grid_search.best_estimator_))
# Validation score and comparions with CV scores

pred_validation = grid_search.predict(reduced_X_train) 

print('validation_rmse: {}'.format(math.sqrt(mean_squared_error(pred_validation, y_train))))
output = grid_search.predict(reduced_X_test)

output = numpy.exp(output)



pandas.DataFrame(index=data_sets['df_test']['Id'], data=output, columns=['SalePrice']).to_csv('rf_submission.csv')
reduced_X_train, y_train, reduced_X_test = generate_train_test_with_preselection(plot=False, threshold_mult=.1)



pipe = Pipeline([

    ('scaler', RobustScaler(with_centering=False)),

    ('estimator', Ridge(fit_intercept=True, normalize=False)),

])



alphas = [0.01, 0.05, 0.1, 0.3, 0.5, 1, 2, 3, 5, 8, 10, 12, 14, 16, 18, 30, 40, 50]

parameter_grid = {'estimator__alpha': alphas}

cross_validation = KFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(estimator=pipe, param_grid=parameter_grid, cv=cross_validation, scoring=rmse_score, refit=True, return_train_score=True)

grid_search.fit(reduced_X_train, y_train) 



print('Best score: {}'.format(grid_search.best_score_))

print('Best parameters: {}'.format(grid_search.best_params_))

print('Best estimator: {}'.format(grid_search.best_estimator_))
# Validation score and comparions with CV scores

pred_validation = grid_search.predict(reduced_X_train) 

entire_train_score = math.sqrt(mean_squared_error(pred_validation, y_train))

print('validation_rmse: {}'.format(entire_train_score))



fig, ax = pyplot.subplots(figsize=(12, 4))

pyplot.rcParams.update({'lines.markeredgewidth': 1})

ax.errorbar(alphas, -grid_search.cv_results_['mean_train_score'], yerr=0.5*grid_search.cv_results_['std_train_score'], 

                fmt='-o', capsize=5, label='cv_mean_train_score')

ax.errorbar(alphas, -grid_search.cv_results_['mean_test_score'], yerr=0.5*grid_search.cv_results_['std_test_score'], 

                fmt='-o', capsize=5, label='cv_mean_test_score')

ax.axhline(entire_train_score, linestyle='--', color='black')

pyplot.legend()

pyplot.show()
output = grid_search.predict(reduced_X_test)

output = numpy.exp(output)



pandas.DataFrame(index=data_sets['df_test']['Id'], data=output, columns=['SalePrice']).to_csv('submission_ridge.csv')