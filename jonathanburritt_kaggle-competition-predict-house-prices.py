# Standard libaries



# For data structures



import pandas as pd



# For basic plottting



from matplotlib import pyplot as plt

%matplotlib inline



# For advanced plotting



import seaborn as sns



# For scientific computing



import numpy as np



# Config option for displaying output

# Default is 'last_expr'



# from IPython.core.interactiveshell import InteractiveShell

# InteractiveShell.ast_node_interactivity = "all"



# Set display options to max column width and all columns and rows



pd.set_option('display.max_colwidth', None)

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)



# For python system functions



import sys



# For listing files in current working directory



# import os

# pd.DataFrame([os.getcwd()], columns=['Current Working Directory:'])

# pd.DataFrame(os.listdir(), columns=['Current Working Directory Contents:'])



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Additional Libraries



# For unziping files



from zipfile import ZipFile



# For plotting style



plt.style.use('fivethirtyeight')



# For models and evaluation



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso



from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_squared_log_error



from scipy import stats





# Load data



df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df_comp = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

# Explore training dataset



df_train.head()

# Explore training dataset



df_train.shape

# Explore competition dataset



df_comp.head()

# Explore competition dataset



df_comp.shape

# Compare features between datasets



lst_train_diff_features = list(set(df_train.columns.values)-set(df_comp.columns.values))

lst_train_diff_features

# Compare features between datasets



lst_comp_diff_features = list(set(df_comp.columns.values)-set(df_train.columns.values))

lst_comp_diff_features

# Explore data format training dataset



df_train.info(verbose=True)

# Explore data format competition dataset



df_comp.info(verbose=True)

# Explore missing data in training dataset



null_data_train = df_train.columns[df_train.isnull().any()]

df_train[null_data_train].isnull().sum()

# Explore missing data in competition dataset



null_data_comp = df_comp.columns[df_comp.isnull().any()]

df_comp[null_data_comp].isnull().sum()

    
# Drop features with high level of missing values from training dataset



df_train.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)

df_train.shape

# Drop features with high level of missing values from the competition dataset



df_comp.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)

df_comp.shape

# Explore missing data in training dataset



null_data_train = df_train.columns[df_train.isnull().any()]

df_train[null_data_train].isnull().sum()

# Explore missing data in competition dataset



null_data_comp = df_comp.columns[df_comp.isnull().any()]

df_comp[null_data_comp].isnull().sum()

# Reverify features between datasets



df_train.head()

# Reverify features between datasets



df_train.shape
# Reverify features between datasets



df_comp.head()

# Reverify features between datasets



df_comp.shape

# Reverify features between datasets



lst_train_diff_features = list(set(df_train.columns.values)-set(df_comp.columns.values))

lst_train_diff_features

# Reverify features between datasets



lst_comp_diff_features = list(set(df_comp.columns.values)-set(df_train.columns.values))

lst_comp_diff_features

# Create new dataframe of categorical variables for training dataset



df_train_cat_var = df_train[['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',

                       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',

                       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 

                       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',

                       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 

                       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 

                       'Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 

                       'PavedDrive', 'SaleType', 'SaleCondition']]

df_train_cat_var.head()

df_train_cat_var.shape

# Loop through categorical variables and convert to numeric values for training dataset



for column in df_train_cat_var:

    df_dummy_col = pd.get_dummies(df_train[column], prefix=column)

    df_train = pd.concat([df_train, df_dummy_col], axis=1)

    df_train.drop(column, axis=1, inplace=True)  



# Convert all data types to floats



df_train = df_train.astype(float)



# Explore the new dataset



df_train.head()

# Create new dataframe of categorical variables for competition dataset



df_comp_cat_var = df_comp[['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',

                       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',

                       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 

                       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',

                       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 

                       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 

                       'Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 

                       'PavedDrive', 'SaleType', 'SaleCondition']]

df_comp_cat_var.head()

# Loop through categorical variables and convert to numeric values for competition dataset



for column in df_comp_cat_var:

    df_dummy_col = pd.get_dummies(df_comp[column], prefix=column)

    df_comp = pd.concat([df_comp, df_dummy_col], axis=1)

    df_comp.drop(column, axis=1, inplace=True)  



# Convert all data types to floats



df_comp = df_comp.astype(float)



# Explore the new dataset



df_comp.head()

df_comp.shape

# Create difference list of features uncommon between training and comp datasets



lst_train_diff_features = list(set(df_train.columns.values)-set(df_comp.columns.values))

lst_train_diff_features

# Drop SalesPrice from difference list



lst_train_diff_features.remove('SalePrice')

lst_train_diff_features

# Drop difference list from training dataset



df_train.drop(lst_train_diff_features, axis=1, inplace=True)

# Reverify difference between training and comp datasets



lst_train_diff_features = list(set(df_train.columns.values)-set(df_comp.columns.values))

lst_train_diff_features

# Reverify difference between training and comp datasets



lst_comp_diff_features = list(set(df_comp.columns.values)-set(df_train.columns.values))

lst_comp_diff_features

# Sort features alphabetically and explore new datasets



df_train = df_train.reindex(sorted(df_train.columns), axis=1)

df_train.head()

# Sort features alphabetically and explore new datasets



df_train.shape

# Sort features alphabetically and explore new datasets



df_comp = df_comp.reindex(sorted(df_comp.columns), axis=1)

df_comp.head()

# Sort features alphabetically and explore new datasets



df_comp.shape

# Explore missing data in training dataset



null_data_train = df_train.columns[df_train.isnull().any()]

df_train[null_data_train].isnull().sum()

# Explore missing data in competition dataset



null_data_comp = df_comp.columns[df_comp.isnull().any()]

df_comp[null_data_comp].isnull().sum()

# Loop through training dataset and replace missing values with feature mean



lst_null_train = ['GarageYrBlt', 'LotFrontage', 'MasVnrArea']



for n in lst_null_train:

    df_train[n].fillna(df_train[n].mean(), axis=0, inplace=True)

# Loop through competition dataset and replace missing values with feature mean



lst_null_comp = ['GarageYrBlt', 'LotFrontage', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',

            'BsmtFullBath', 'BsmtHalfBath', 'BsmtUnfSF', 'GarageArea', 'GarageCars', 

            'TotalBsmtSF', 'GarageYrBlt']



for n in lst_null_comp:

    df_comp[n].fillna(df_comp[n].mean(), axis=0, inplace=True)

# Explore missing data in training dataset



null_data_train = df_train.columns[df_train.isnull().any()]

df_train[null_data_train].isnull().sum()

# Explore missing data in competition dataset



null_data_comp = df_comp.columns[df_comp.isnull().any()]

df_comp[null_data_comp].isnull().sum()

# Reverify features between datasets



df_train.head()

# Reverify features between datasets



df_train.shape

# Reverify features between datasets



df_comp.head()

# Reverify features between datasets



df_comp.shape

# Reverify features between datasets



lst_train_diff_features = list(set(df_train.columns.values)-set(df_comp.columns.values))

lst_train_diff_features

# Reverify features between datasets



lst_comp_diff_features = list(set(df_comp.columns.values)-set(df_train.columns.values))

lst_comp_diff_features

# Export training dataset features for easy input into model



lst_train_col = df_train.columns

lst_train_features = []



for f in lst_train_col:

    f = "'" + f + "'"

    lst_train_features.append([f])



pd.DataFrame(lst_train_features).to_csv('/kaggle/working/training_features.csv', index=False)

# Explore correlation of training dataset



df_train_corr = df_train.corr()

df_train_corr_sale = df_train_corr[['SalePrice']]

df_train_corr_sale = df_train_corr_sale.sort_values(['SalePrice'], ascending=False).head(10)

df_train_corr_sale

# Determine key features



lst_ind_var = ['BedroomAbvGr',  

                'BsmtCond_Fa', 

                'BsmtCond_Gd', 

                'BsmtCond_Po', 

                'BsmtCond_TA', 

                'ExterCond_Ex', 

                'ExterCond_Fa', 

                'ExterCond_Gd',

                'ExterCond_Po', 

                'ExterCond_TA', 

                'ExterQual_Ex', 

                'ExterQual_Fa',

                'ExterQual_Gd', 

                'ExterQual_TA', 

                'FullBath', 

                'GarageCars', 

                'GarageQual_Fa',

                'GarageQual_Gd',

                'GarageQual_Po',

                'GarageQual_TA',

                'GrLivArea',

                'HalfBath',

                'KitchenQual_Ex',

                'KitchenQual_Fa',

                'KitchenQual_Gd',

                'KitchenQual_TA',

                'LotArea',

                'MSSubClass',

                'MSZoning_C (all)',

                'MSZoning_FV',

                'MSZoning_RH',

                'MSZoning_RL',

                'MSZoning_RM',

                'Neighborhood_Blmngtn',

                'Neighborhood_Blueste',

                'Neighborhood_BrDale',

                'Neighborhood_BrkSide',

                'Neighborhood_ClearCr',

                'Neighborhood_CollgCr',

                'Neighborhood_Crawfor',

                'Neighborhood_Edwards',

                'Neighborhood_Gilbert',

                'Neighborhood_IDOTRR',

                'Neighborhood_MeadowV',

                'Neighborhood_Mitchel',

                'Neighborhood_NAmes',

                'Neighborhood_NPkVill',

                'Neighborhood_NWAmes',

                'Neighborhood_NoRidge',

                'Neighborhood_NridgHt',

                'Neighborhood_OldTown',

                'Neighborhood_SWISU',

                'Neighborhood_Sawyer',

                'Neighborhood_SawyerW',

                'Neighborhood_Somerst',

                'Neighborhood_StoneBr',

                'Neighborhood_Timber',

                'Neighborhood_Veenker',

                'OverallCond',

                'OverallQual',

                'Street_Grvl',

                'Street_Pave',

              ]



# Root Mean Squared Error 39735.084240

# R2 score 0.771371

# Plot key features



# Plot the data



fig, axes = plt.subplots(2, 2, sharey=True, figsize=(10, 6))

fig.subplots_adjust(wspace=0.4, hspace=0.4)



lst_plot_features = ['GrLivArea','LotArea', 'OverallQual', 'OverallCond']



for ax, features in zip(axes.ravel(), lst_plot_features):

    ax.scatter(df_train[features], df_train['SalePrice'])

    ax.set_title(features, fontsize=10)

    ax.set_xticklabels([])

    ax.set_ylabel('Sale Price', fontsize=10)

    ax.set_yticklabels([])



fig.suptitle('Plot Key Features Compared to Sale Price', fontsize=20)



# Show plots



plt.show()

# Prepare the data



lm = LinearRegression()



X_data = df_train[lst_ind_var]

y_data = df_train['SalePrice']



# Train and fit the model



X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=.20, random_state=0)

lm.fit(X_train, y_train)

y_hat = lm.predict(X_test)

print('Training set coef: ', lm.coef_)

print('Training set intercept: ', lm.intercept_)

print('Training set y_hat: ', y_hat[0:5])

df_coef = pd.DataFrame(lm.coef_, index=X_data.columns, columns=['Coefficients'])

df_coef

# Explore training model actual vs predicted house prices



df_train_comp = pd.DataFrame({'Actual (y_test)':y_test, 'Predicted (y_hat)':y_hat})

df_train_comp.head()

# Plot training model actual vs predicted house prices



# Number of records to display



n = 10



# Colors for bars



blue = '#0000ff'

gold = '#ffd700'



# Plot data



fig, ax = plt.subplots(figsize=(10, 6))



act_hse_price = df_train_comp['Actual (y_test)'].head(n)

pred_hse_price = df_train_comp['Predicted (y_hat)'].head(n)



bar_width = 0.25

x_bar_pos = np.arange(1, n+1)

x_label = df_train_comp.head(n).index

max_hse_price = df_train.SalePrice.max()



ax.bar(x_bar_pos, act_hse_price, color=blue, width=bar_width, label='Actual (y_test)')

ax.bar(x_bar_pos+bar_width, pred_hse_price, color=gold, width=bar_width, label='Predicted (y_hat)')



ax.set_xlabel('House ID', fontsize=10)

ax.set_xticks(x_bar_pos+(.5*bar_width))

ax.set_xticklabels(x_label, fontsize=10)



ax.set_ylabel('House Price', fontsize=10)

ax.set_yticks([0, 500000,  900000])

ax.set_ylim([0, 900000])



ax.set_title('Training Model: Actual vs Predicted Price (Sample)', fontsize=20)

ax.legend()



# Show plot



plt.show()

# Evaluate the model with Root Mean Squared Error, R2 score, Mean Squared Log Error, and Root Mean Squared Log Error



lst_eval = [np.sqrt(mean_squared_error(y_test, y_hat)), r2_score(y_test, y_hat), mean_squared_log_error(y_test, y_hat), np.sqrt(mean_squared_log_error(y_test,y_hat))]



df_eval = pd.DataFrame({'Evaluation':lst_eval}, index=['Root Mean Squared Error', 'R2 score', 'Mean Squared Log Error', 'Root Mean Squared Log Error'])

df_eval

# Use Ridge Regression to improve model



rr = Ridge(alpha=0.10)

rr.fit(X_train, y_train)



# Evaluate the model with Root Mean Squared Error, R2 score, Mean Squared Log Error, and Root Mean Squared Log Error



lst_eval = [np.sqrt(mean_squared_error(y_test, y_hat)), r2_score(y_test, y_hat), mean_squared_log_error(y_test, y_hat), np.sqrt(mean_squared_log_error(y_test,y_hat))]



df_eval = pd.DataFrame({'Evaluation':lst_eval}, index=['Root Mean Squared Error', 'R2 score', 'Mean Squared Log Error', 'Root Mean Squared Log Error'])

df_eval

# Use lasso regression to improve model



lasso = Lasso()

lasso.fit(X_train,y_train)



# Evaluate the model with Root Mean Squared Error, R2 score, Mean Squared Log Error, and Root Mean Squared Log Error



lst_eval = [np.sqrt(mean_squared_error(y_test, y_hat)), r2_score(y_test, y_hat), mean_squared_log_error(y_test, y_hat), np.sqrt(mean_squared_log_error(y_test,y_hat))]



df_eval = pd.DataFrame({'Evaluation':lst_eval}, index=['Root Mean Squared Error', 'R2 score', 'Mean Squared Log Error', 'Root Mean Squared Log Error'])

df_eval

# Loop through all elements in key independent variables list and print Pearson Coefficent and P-value



lst_pearson = []

for i in lst_ind_var:

    pearson_coef, p_value = stats.pearsonr(df_train[i], df_train['SalePrice'])

    lst_pearson.append([i, pearson_coef, p_value])



df_pearson = pd.DataFrame(lst_pearson, columns=['Key Indpendent Variable', 'Pearson Coefficient', 'P-value'])

df_pearson = df_pearson.set_index(['Key Indpendent Variable'])

df_pearson

# Descriptive stats of training set saleprice



df_train_desc = df_train.describe()

df_train_desc = df_train_desc[['SalePrice']]

df_train_desc

# Predict sale prices using comp data



y_hat_comp = lm.predict(df_comp[lst_ind_var])



# Load the results into dataframe



df_comp_sub = pd.DataFrame({'Id':df_comp.Id, 'SalePrice':y_hat_comp})

# Explore submission dataset



df_comp_sub['Id'] = df_comp_sub['Id'].astype(int)

df_comp_sub.head()

# Explore submission dataset



df_comp_sub.shape

# Export sales price data to submission csv file



df_comp_sub.to_csv('/kaggle/working/house_prices_submission.csv', index=False)
