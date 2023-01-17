%pip install --upgrade scikit-learn



# Did this to use latest regressors from sklearn...
# Loading neccesary packages.





import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from datetime import datetime



#



from scipy import stats

from scipy.stats import skew, boxcox_normmax, norm

from scipy.special import boxcox1p



#



import matplotlib.gridspec as gridspec

from matplotlib.ticker import MaxNLocator



#



import warnings

pd.options.display.max_columns = 250

pd.options.display.max_rows = 250

warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')
# Loading datasets.



train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
# Shape of the train data



display(train.shape)

# Shape of the test data



display(test.shape)

# First 5 entries of train data



train.head()
# First 5 entries of test data



test.head()
# General statistics of the train data



train.describe()
# General statistics of the test data



test.describe()
# Dropping unnecessary Id columns.





train.drop('Id', axis=1, inplace=True)

test.drop('Id', axis=1, inplace=True)
# Backing up target variables and dropping them from train data.





y = train['SalePrice'].reset_index(drop=True)

train_features = train.drop(['SalePrice'], axis=1)

test_features = test
# Display numerical correlations between features on heatmap.



sns.set(font_scale=1.1)

correlation_train = train.corr()

mask = np.triu(correlation_train.corr())

plt.figure(figsize=(20, 20))

sns.heatmap(correlation_train,

            annot=True,

            fmt='.1f',

            cmap='coolwarm',

            square=True,

            mask=mask,

            linewidths=1,

            cbar=False)



plt.show()
# Merging features



features = pd.concat([train_features, test_features]).reset_index(drop=True)

print(features.shape)
def missing_percentage(df):

    

    '''A function for showing missing data values'''

    

    total = df.isnull().sum().sort_values(

        ascending=False)[df.isnull().sum().sort_values(ascending=False) != 0]

    percent = (df.isnull().sum().sort_values(ascending=False) / len(df) *

               100)[(df.isnull().sum().sort_values(ascending=False) / len(df) *

                     100) != 0]

    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# Checking NaN values.



missing = missing_percentage(features)



fig, ax = plt.subplots(figsize=(20, 5))

sns.barplot(x=missing.index, y='Percent', data=missing, palette='Reds_r')

plt.xticks(rotation=90)



display(missing.T.style.background_gradient(cmap='Reds', axis=1))
# List of NaN's including columns where NaN's mean none.



none_cols = [

    'Alley', 'PoolQC', 'MiscFeature', 'Fence', 'FireplaceQu', 'GarageType',

    'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond',

    'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType'

]

# List of NaN's including columns where NaN's mean 0.



zero_cols = [

    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath',

    'BsmtHalfBath', 'GarageYrBlt', 'GarageArea', 'GarageCars', 'MasVnrArea'

]



# List of NaN's including columns where NaN's actually missing gonna replaced with mode.



freq_cols = [

    'Electrical', 'Exterior1st', 'Exterior2nd', 'Functional', 'KitchenQual',

    'SaleType', 'Utilities'

]



# Filling the list of columns above:



for col in zero_cols:

    features[col].replace(np.nan, 0, inplace=True)



for col in none_cols:

    features[col].replace(np.nan, 'None', inplace=True)



for col in freq_cols:

    features[col].replace(np.nan, features[col].mode()[0], inplace=True)
# Filling MSZoning according to MSSubClass.



features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].apply(

    lambda x: x.fillna(x.mode()[0]))
# Filling MSZoning according to Neighborhood.



features['LotFrontage'] = features.groupby(

    ['Neighborhood'])['LotFrontage'].apply(lambda x: x.fillna(x.median()))
# Features which numerical on data but should be treated as category.



features['MSSubClass'] = features['MSSubClass'].astype(str)



features['YrSold'] = features['YrSold'].astype(str)



features['MoSold'] = features['MoSold'].astype(str)
# Transforming rare values(less than 10) into one group.



others = [

    'Condition1', 'Condition2', 'RoofMatl', 'Exterior1st', 'Exterior2nd',

    'Heating', 'Electrical', 'Functional', 'SaleType'

]



for col in others:

    mask = features[col].isin(

        features[col].value_counts()[features[col].value_counts() < 10].index)

    features[col][mask] = 'Other'
def srt_box(y, df):

    fig, axes = plt.subplots(14, 3, figsize=(25, 80))

    axes = axes.flatten()



    for i, j in zip(df.select_dtypes(include=['object']).columns, axes):



        sortd = df.groupby([i])[y].median().sort_values(ascending=False)

        sns.boxplot(x=i,

                    y=y,

                    data=df,

                    palette='plasma',

                    order=sortd.index,

                    ax=j)

        j.tick_params(labelrotation=45)

        j.yaxis.set_major_locator(MaxNLocator(nbins=18))



        plt.tight_layout()
# Displaying sale prices vs. categorical values.



srt_box('SalePrice', train)
# Converting some of the categorical values to numeric ones.



neigh_map = {

    'MeadowV': 1,

    'IDOTRR': 1,

    'BrDale': 1,

    'BrkSide': 2,

    'OldTown': 2,

    'Edwards': 2,

    'Sawyer': 3,

    'Blueste': 3,

    'SWISU': 3,

    'NPkVill': 3,

    'NAmes': 3,

    'Mitchel': 4,

    'SawyerW': 5,

    'NWAmes': 5,

    'Gilbert': 5,

    'Blmngtn': 5,

    'CollgCr': 5,

    'ClearCr': 6,

    'Crawfor': 6,

    'Veenker': 7,

    'Somerst': 7,

    'Timber': 8,

    'StoneBr': 9,

    'NridgHt': 10,

    'NoRidge': 10

}



features['Neighborhood'] = features['Neighborhood'].map(neigh_map).astype(

    'int')

ext_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

features['ExterQual'] = features['ExterQual'].map(ext_map).astype('int')

features['ExterCond'] = features['ExterCond'].map(ext_map).astype('int')

bsm_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

features['BsmtQual'] = features['BsmtQual'].map(bsm_map).astype('int')

features['BsmtCond'] = features['BsmtCond'].map(bsm_map).astype('int')

bsmf_map = {

    'None': 0,

    'Unf': 1,

    'LwQ': 2,

    'Rec': 3,

    'BLQ': 4,

    'ALQ': 5,

    'GLQ': 6

}



features['BsmtFinType1'] = features['BsmtFinType1'].map(bsmf_map).astype('int')

features['BsmtFinType2'] = features['BsmtFinType2'].map(bsmf_map).astype('int')

heat_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

features['HeatingQC'] = features['HeatingQC'].map(heat_map).astype('int')

features['KitchenQual'] = features['KitchenQual'].map(heat_map).astype('int')

features['FireplaceQu'] = features['FireplaceQu'].map(bsm_map).astype('int')

features['GarageCond'] = features['GarageCond'].map(bsm_map).astype('int')

features['GarageQual'] = features['GarageQual'].map(bsm_map).astype('int')
# Plotting numerical features with polynomial order to detect outliers.



def srt_reg(y, df):

    fig, axes = plt.subplots(12, 3, figsize=(25, 80))

    axes = axes.flatten()



    for i, j in zip(df.select_dtypes(include=['number']).columns, axes):



        sns.regplot(x=i,

                    y=y,

                    data=df,

                    ax=j,

                    order=3,

                    ci=None,

                    color='#e74c3c',

                    line_kws={'color': 'black'},

                    scatter_kws={'alpha':0.4})

        j.tick_params(labelrotation=45)

        j.yaxis.set_major_locator(MaxNLocator(nbins=10))



        plt.tight_layout()
srt_reg('SalePrice', train)
# Dropping outliers after detecting them by eye.



features = features.join(y)

features = features.drop(features[(features['OverallQual'] < 5)

                                  & (features['SalePrice'] > 200000)].index)

features = features.drop(features[(features['GrLivArea'] > 4000)

                                  & (features['SalePrice'] < 200000)].index)

features = features.drop(features[(features['GarageArea'] > 1200)

                                  & (features['SalePrice'] < 200000)].index)

features = features.drop(features[(features['TotalBsmtSF'] > 3000)

                                  & (features['SalePrice'] > 320000)].index)

features = features.drop(features[(features['1stFlrSF'] < 3000)

                                  & (features['SalePrice'] > 600000)].index)

features = features.drop(features[(features['1stFlrSF'] > 3000)

                                  & (features['SalePrice'] < 200000)].index)



y = features['SalePrice']

y.dropna(inplace=True)

features.drop(columns='SalePrice', inplace=True)
# Creating new features based on previous observations.



features['TotalSF'] = (features['BsmtFinSF1'] + features['BsmtFinSF2'] +

                       features['1stFlrSF'] + features['2ndFlrSF'])

features['TotalBathrooms'] = (features['FullBath'] +

                              (0.5 * features['HalfBath']) +

                              features['BsmtFullBath'] +

                              (0.5 * features['BsmtHalfBath']))



features['TotalPorchSF'] = (features['OpenPorchSF'] + features['3SsnPorch'] +

                            features['EnclosedPorch'] +

                            features['ScreenPorch'] + features['WoodDeckSF'])



features['YearBlRm'] = (features['YearBuilt'] + features['YearRemodAdd'])



# Merging quality and conditions.



features['TotalExtQual'] = (features['ExterQual'] + features['ExterCond'])

features['TotalBsmQual'] = (features['BsmtQual'] + features['BsmtCond'] +

                            features['BsmtFinType1'] +

                            features['BsmtFinType2'])

features['TotalGrgQual'] = (features['GarageQual'] + features['GarageCond'])

features['TotalQual'] = features['OverallQual'] + features[

    'TotalExtQual'] + features['TotalBsmQual'] + features[

        'TotalGrgQual'] + features['KitchenQual'] + features['HeatingQC']



# Creating new features by using new quality indicators.



features['QualGr'] = features['TotalQual'] * features['GrLivArea']

features['QualBsm'] = features['TotalBsmQual'] * (features['BsmtFinSF1'] +

                                                  features['BsmtFinSF2'])

features['QualPorch'] = features['TotalExtQual'] * features['TotalPorchSF']

features['QualExt'] = features['TotalExtQual'] * features['MasVnrArea']

features['QualGrg'] = features['TotalGrgQual'] * features['GarageArea']

features['QlLivArea'] = (features['GrLivArea'] -

                         features['LowQualFinSF']) * (features['TotalQual'])

features['QualSFNg'] = features['QualGr'] * features['Neighborhood']
# Observing the effects of newly created features on sale price.



def srt_reg(feature):

    merged = features.join(y)

    fig, axes = plt.subplots(5, 3, figsize=(25, 40))

    axes = axes.flatten()



    new_features = [

        'TotalSF', 'TotalBathrooms', 'TotalPorchSF', 'YearBlRm',

        'TotalExtQual', 'TotalBsmQual', 'TotalGrgQual', 'TotalQual', 'QualGr',

        'QualBsm', 'QualPorch', 'QualExt', 'QualGrg', 'QlLivArea', 'QualSFNg'

    ]



    for i, j in zip(new_features, axes):



        sns.regplot(x=i,

                    y=feature,

                    data=merged,

                    ax=j,

                    order=3,

                    ci=None,

                    color='#e74c3c',

                    line_kws={'color': 'black'},

                    scatter_kws={'alpha':0.4})

        j.tick_params(labelrotation=45)

        j.yaxis.set_major_locator(MaxNLocator(nbins=10))



        plt.tight_layout()





srt_reg('SalePrice')
# Creating some simple features.



features['HasPool'] = features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)



features['Has2ndFloor'] = features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)



features['HasGarage'] = features['QualGrg'].apply(lambda x: 1 if x > 0 else 0)



features['HasBsmt'] = features['QualBsm'].apply(lambda x: 1 if x > 0 else 0)



features['HasFireplace'] = features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)



features['HasPorch'] = features['QualPorch'].apply(lambda x: 1 if x > 0 else 0)
possible_skewed = [

    'LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',

    'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea',

    'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',

    'ScreenPorch', 'PoolArea', 'LowQualFinSF', 'MiscVal'

]
# Finding skewness of the numerical features.



skew_features = np.abs(features[possible_skewed].apply(lambda x: skew(x)).sort_values(

    ascending=False))



# Filtering skewed features.



high_skew = skew_features[skew_features > 0.3]



# Taking indexes of high skew.



skew_index = high_skew.index



# Applying boxcox transformation to fix skewness.



for i in skew_index:

    features[i] = boxcox1p(features[i], boxcox_normmax(features[i] + 1))
# Features to drop.



to_drop = [

    'Utilities',

    'PoolQC',

    'YrSold',

    'MoSold',

    'ExterQual',

    'BsmtQual',

    'GarageQual',

    'KitchenQual',

    'HeatingQC',

]



# Dropping features



features.drop(columns=to_drop, inplace=True)
# Getting dummy variables for ategorical data.



features = pd.get_dummies(data=features)
print(f'Number of missing values: {features.isna().sum().sum()}')
features.shape
features.sample(5)
features.describe()
# Separating train and test set.



train = features.iloc[:len(y), :]

test = features.iloc[len(train):, :]
correlations = train.join(y).corrwith(train.join(y)['SalePrice']).iloc[:-1].to_frame()

correlations['Abs Corr'] = correlations[0].abs()

sorted_correlations = correlations.sort_values('Abs Corr', ascending=False)['Abs Corr']

fig, ax = plt.subplots(figsize=(12,12))

sns.heatmap(sorted_correlations.to_frame()[sorted_correlations>=.5], cmap='coolwarm', annot=True, vmin=-1, vmax=1, ax=ax);

def plot_dist3(df, feature, title):

    # Creating a customized chart. and giving in figsize and everything.

    fig = plt.figure(constrained_layout=True, figsize=(12, 8))

    

    # Creating a grid of 3 cols and 3 rows.

    

    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)



    # Customizing the histogram grid.

    

    ax1 = fig.add_subplot(grid[0, :2])

    # Set the title.

    

    ax1.set_title('Histogram')

    

    # Plot the histogram.

    

    sns.distplot(df.loc[:, feature],

                 hist=True,

                 kde=True,

                 fit=norm,

                 ax=ax1,

                 color='#e74c3c')

    ax1.legend(labels=['Normal', 'Actual'])



    # Customizing the QQ_plot.

    

    ax2 = fig.add_subplot(grid[1, :2])

    

    # Set the title.

    

    ax2.set_title('Probability Plot')

    

    # Plotting the QQ_Plot.

    

    stats.probplot(df.loc[:, feature].fillna(np.mean(df.loc[:, feature])),

                   plot=ax2)

    ax2.get_lines()[0].set_markerfacecolor('#e74c3c')

    ax2.get_lines()[0].set_markersize(12.0)



    # Customizing the Box Plot.

    

    ax3 = fig.add_subplot(grid[:, 2])

    # Set title.

    ax3.set_title('Box Plot')

    

    # Plotting the box plot.

    

    sns.boxplot(df.loc[:, feature], orient='v', ax=ax3, color='#e74c3c')

    ax3.yaxis.set_major_locator(MaxNLocator(nbins=24))



    plt.suptitle(f'{title}', fontsize=24)
# Checking target variable.



plot_dist3(train.join(y), 'SalePrice', 'Sale Price Before Log Transformation')
# Setting model data:



X = train

X_test = test

y = np.log1p(y)
plot_dist3(train.join(y), 'SalePrice', 'Sale Price After Log Transformation')
# Loading neccesary packages for modelling:



from sklearn.model_selection import cross_val_score, KFold, cross_validate

from sklearn.preprocessing import RobustScaler

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV, TweedieRegressor

from sklearn.experimental import enable_hist_gradient_boosting

from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor

from sklearn.svm import SVR

from sklearn.pipeline import make_pipeline

from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from mlxtend.regressor import StackingCVRegressor
# Setting kfold for future use.



kf = KFold(10, random_state=42)
alphas_alt = [15.5, 15.6, 15.7, 15.8, 15.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]

alphas2 = [

    5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008

]

e_alphas = [

    0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007

]

e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]



# ridge_cv:



ridge = make_pipeline(RobustScaler(), RidgeCV(

    alphas=alphas_alt,

    cv=kf,

))



# lasso_cv:



lasso = make_pipeline(

    RobustScaler(),

    LassoCV(max_iter=1e7, alphas=alphas2, random_state=42, cv=kf))



# elasticnet_cv:



elasticnet = make_pipeline(

    RobustScaler(),

    ElasticNetCV(max_iter=1e7,

                 alphas=e_alphas,

                 cv=kf,

                 random_state=42,

                 l1_ratio=e_l1ratio))



# svr:



svr = make_pipeline(RobustScaler(),

                    SVR(C=21, epsilon=0.0099, gamma=0.00017, tol=0.000121))



# gradientboosting:



gbr = GradientBoostingRegressor(n_estimators=2900,

                                learning_rate=0.0161,

                                max_depth=4,

                                max_features='sqrt',

                                min_samples_leaf=17,

                                loss='huber',

                                random_state=42)



# lightgbm:



lightgbm = LGBMRegressor(objective='regression',

                         n_estimators=3500,

                         num_leaves=5,

                         learning_rate=0.00721,

                         max_bin=163,

                         bagging_fraction=0.35711,

                         n_jobs=-1,

                         bagging_seed=42,

                         feature_fraction_seed=42,

                         bagging_freq=7,

                         feature_fraction=0.1294,

                         min_data_in_leaf=8)



# xgboost:



xgboost = XGBRegressor(

    learning_rate =0.0139,

    n_estimators =4500,

    max_depth =4,

    min_child_weight =0,

    subsample =0.7968,

    colsample_bytree =0.4064,

    nthread =-1,

    scale_pos_weight =2,

    seed=42,

)





# histgradientboost:



hgrd= HistGradientBoostingRegressor(    loss= 'least_squares',

    max_depth = 2,

    min_samples_leaf = 40,

    max_leaf_nodes = 29,

    learning_rate = 0.15,

    max_iter = 225,

                                    random_state=42)



#tweedie regresson:



tweed = make_pipeline(RobustScaler(),TweedieRegressor(alpha=0.005))





# stacking regressor:

stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, gbr,

                                            xgboost, lightgbm,hgrd, tweed),

                                meta_regressor=xgboost,

                                use_features_in_secondary=True)
def model_check(X, y, estimators, cv):

    

    ''' A function for testing multiple estimators.'''

    

    model_table = pd.DataFrame()



    row_index = 0

    for est, label in zip(estimators, labels):



        MLA_name = label

        model_table.loc[row_index, 'Model Name'] = MLA_name



        cv_results = cross_validate(est,

                                    X,

                                    y,

                                    cv=cv,

                                    scoring='neg_root_mean_squared_error',

                                    return_train_score=True,

                                    n_jobs=-1)



        model_table.loc[row_index, 'Train RMSE'] = -cv_results[

            'train_score'].mean()

        model_table.loc[row_index, 'Test RMSE'] = -cv_results[

            'test_score'].mean()

        model_table.loc[row_index, 'Test Std'] = cv_results['test_score'].std()

        model_table.loc[row_index, 'Time'] = cv_results['fit_time'].mean()



        row_index += 1



    model_table.sort_values(by=['Test RMSE'],

                            ascending=True,

                            inplace=True)



    return model_table
# Setting list of estimators and labels for them.



estimators = [ridge, lasso, elasticnet, gbr, xgboost, lightgbm, svr, hgrd, tweed]

labels = [

    'Ridge', 'Lasso', 'Elasticnet', 'GradientBoostingRegressor',

    'XGBRegressor', 'LGBMRegressor', 'SVR', 'HistGradientBoostingRegressor','TweedieRegressor'

]
# Executing cross validation.



raw_models = model_check(X, y, estimators, kf)

display(raw_models.style.background_gradient(cmap='summer_r'))
# Fitting the models on train data:



print('=' * 20, 'START Fitting', '=' * 20)

print('=' * 55)



print(datetime.now(), 'StackingCVRegressor')

stack_gen_model = stack_gen.fit(X.values, y.values)



print(datetime.now(), 'Elasticnet')

elastic_model_full_data = elasticnet.fit(X, y)



print(datetime.now(), 'Lasso')

lasso_model_full_data = lasso.fit(X, y)



print(datetime.now(), 'Ridge')

ridge_model_full_data = ridge.fit(X, y)



print(datetime.now(), 'SVR')

svr_model_full_data = svr.fit(X, y)



print(datetime.now(), 'GradientBoosting')

gbr_model_full_data = gbr.fit(X, y)



print(datetime.now(), 'XGboost')

xgb_model_full_data = xgboost.fit(X, y)



print(datetime.now(), 'Lightgbm')

lgb_model_full_data = lightgbm.fit(X, y)



print(datetime.now(), 'Hist')

hist_full_data = hgrd.fit(X, y)



print(datetime.now(), 'Tweed')

tweed_full_data = tweed.fit(X, y)



print('=' * 20, 'FINISHED Fitting', '=' * 20)

print('=' * 58)
# Blending models by assigning weights.



def blend_models_predict(X):

    return ((0.1 * elastic_model_full_data.predict(X)) +

            (0.1 * lasso_model_full_data.predict(X)) +

            (0.1 * ridge_model_full_data.predict(X)) +

            (0.1 * svr_model_full_data.predict(X)) +

            (0.05 * gbr_model_full_data.predict(X)) +

            (0.1 * xgb_model_full_data.predict(X)) +

            (0.05 * lgb_model_full_data.predict(X)) +

            (0.05 * hist_full_data.predict(X)) +

            (0.1 * tweed_full_data.predict(X)) +

            (0.25 * stack_gen_model.predict(X.values)))
submission = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')



# Inversing and flooring log scaled sale price predictions to see real prices again.



submission['SalePrice'] = np.floor(np.expm1(blend_models_predict(X_test)))



# Creating submission dataframe.



submission = submission[['Id', 'SalePrice']]
# Saving submission dataframe as csv file:



submission.to_csv('mysubmission.csv', index=False)



print(

    'Saving submission.',

    datetime.now(),

)

submission.head()