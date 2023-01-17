# Standard Tools
from scipy import stats
import pandas as pd
import numpy as np
import pickle
import os

# Visualisation Tools
import matplotlib.pyplot as plt 
import seaborn as sns

# Modelling Tools
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import statsmodels as sm

# Misc
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
plt.style.use('fivethirtyeight')

# Setting the random state
np.random.seed(8888)
SEED = 8888

# # File names 
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
# Loading our data
df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

print(f'The dataset has {df.shape[0]} rows and {df.shape[1]} columns.')
# Getting a feel for how our data looks like
df.head()
# Dropping column Id as we will use the dataframe indexing
df = df.drop('Id', axis=1)
# Quick scan for missing values and wrong datatypes
df.info()
df.describe()
# Helper function to help check for missing data
def variable_missing_percentage(df, save_results=False):
    '''
    Function that shows variables that have missing values and the percentage of total observations that are missing.
    
    Arguments:
        df : Pandas DataFrame
        save_results : bool, default is False
            Set as True to save the Series with the missing percentages.
    
    Returns:
        percentage_missing : Pandas Series
            Series with variables and their respective missing percentages.
    '''
    percentage_missing = df.isnull().mean().sort_values(ascending=False) * 100
    percentage_missing = percentage_missing.loc[percentage_missing > 0].round(2)
    missing_variables = len(percentage_missing)
    
    if len(percentage_missing) > 0:
        print(f'There are a total of {missing_variables} variables with missing values. Percentage of total missing:')
        print()
        print(percentage_missing)
    
    else:
        print('The dataframe has no missing values in any column.')
    
    if save_results:
        return percentage_missing
variable_missing_percentage(df)
def drop_missing_variables(df, threshold, verbose=True):
    '''Function that removes variables that have missing percentages above a threshold.
    
    Arguments:
        df : Pandas DataFrame
        threshold : float
            Threshold missing percentage value in decimals.
        verbose : bool, default is True
            Prints the variables that were removed.
            
    Returns:
        df : Pandas DataFrame with variables removed
    '''
    shape_prior = df.shape
    vars_to_remove = df.columns[df.isnull().mean() > threshold].to_list()
    df = df.drop(vars_to_remove, axis=1)
    shape_post = df.shape
    
    print(f'The original DataFrame had {shape_prior[1]} variables.')
    print(f'The returned DataFrame has {shape_post[1]} variables.')
    
    if verbose:
        print()
        print('The following variables were removed:')
        print(vars_to_remove)
        
    return df
df = drop_missing_variables(df, 0.8)
# Dropping features that are related to the ones we just removed
df = df.drop(['MiscVal', 'PoolArea'], axis=1)
df.FireplaceQu = df.FireplaceQu.fillna('NoFirePlace')

basement_variables = ['BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtCond', 'BsmtQual']
df[basement_variables] = df[basement_variables].fillna('NoBasement')

garage_variables = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
df[garage_variables] = df[garage_variables].fillna('NoGarage')
# Checking which variables are still missing
variable_missing_percentage(df)
df.YearBuilt.corr(df.GarageYrBlt).round(2)
df = df.drop('GarageYrBlt', axis=1)
df = df.dropna(how='any', subset=['MasVnrType', 'MasVnrArea', 'Electrical'])
# Installing missingpy package, remember to turn internet on in the settings! 
!pip install missingpy

from missingpy import KNNImputer
knn_imputer = KNNImputer(n_neighbors=5, weights='distance', metric='masked_euclidean')

df.LotFrontage = knn_imputer.fit_transform(np.array(df.LotFrontage).reshape(-1,1))
# Making sure we have tackled all missing variables
variable_missing_percentage(df)
# Changing numeric variables to categorical
df = df.replace({
    'MSSubClass' : {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45", 50 : "SC50", 60 : "SC60",
                    70 : "SC70", 75 : "SC75", 80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120", 
                    150 : "SC150", 160 : "SC160", 180 : "SC180", 190 : "SC190"},
    'MoSold' : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 6 : "Jun",
                7 : "Jul", 8 : "Aug", 9 : "Sep", 10 : "Oct", 11 : "Nov", 12 : "Dec"}
})
# Converting categorical variables to an interval scale as they are ordinal in nature.
df = df.replace({
    'ExterQual' : {'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5},
    'ExterCond' : {'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5},
    'BsmtQual' : {'NoBasement' : 0, 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5},
    'BsmtCond' : {'NoBasement' : 0, 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5},
    'BsmtExposure' : {'NoBasement' : 0, 'No' : 1, 'Mn' : 2, 'Av' : 3, 'Gd' : 4},
    'BsmtFinType1' : {'NoBasement' : 0, 'Unf' : 1, 'LwQ' : 2, 'Rec' : 3, 'BLQ' : 4, 'ALQ' : 5, 'GLQ' : 6},
    'BsmtFinType2' : {'NoBasement' : 0, 'Unf' : 1, 'LwQ' : 2, 'Rec' : 3, 'BLQ' : 4, 'ALQ' : 5, 'GLQ' : 6},
    'HeatingQC' : {'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5},
    'KitchenQual' : {'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5},
    'FireplaceQu' : {'NoFirePlace' : 0, 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5},
    'GarageFinish' : {'NoGarage' : 0, 'Unf' : 1, 'RFn' : 2, 'Fin' : 3},
    'GarageQual' : {'NoGarage' : 0, 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5},
    'GarageCond' : {'NoGarage' : 0, 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5},
})

# Creating a list of our ordinal variables
ordinal_vars = [
    'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
    'BsmtFinType1', 'BsmtFinType2','HeatingQC', 'KitchenQual',
    'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond'
]
# Checking our datatypes once again
df.info()
# Changing features to their correct data types
df.BsmtCond = df.BsmtCond.astype('int64')
df.BsmtFinType2 = df.BsmtFinType2.astype('int64')
df.FireplaceQu = df.FireplaceQu.astype('int64')
def change_variables_to_categorical(df, vars_to_change=[]):
    '''Function that changes all non-numeric variables to categorical datatype.
    
    Arguments:
        df : Pandas DataFrame
        vars_to_change : list, default is an empty list
            If a non-empty list is passed, only the variables in the list are converted to 
            categorical datatype.
    
    Returns:
        df : Pandas DataFrame with categorical datatypes converted.
    '''
    categorical_variables = df.select_dtypes(exclude='number').columns.to_list()
    
    if len(vars_to_change) > 0:
        categorical_variables = vars_to_change
    
    for var in categorical_variables:
        df[var] = df[var].astype('category')
        
    return df
def numerical_categorical_split(df):
    '''Function that creates a list for numerical and categorical variables respectively.
    '''
    numerical_var_list = df.select_dtypes(include='number').columns.to_list()
    categorical_var_list = df.select_dtypes(exclude='number').columns.to_list()
    
    return numerical_var_list, categorical_var_list
# Changing datatypes from 'objects' to 'category' --> More memory efficient
df = change_variables_to_categorical(df)
# Creating lists of numerical and categorical features
numerical_vars, categorical_vars = numerical_categorical_split(df)

# Splitting 2 dataframes, one for numeric variables and another for categorical
numerical_df = df[numerical_vars]
categorical_df = df[categorical_vars]
# Checking if both have same number of observations
print(numerical_df.shape, categorical_df.shape)
any(numerical_df.SalePrice <= 0)
print(f'Skewness of SalePrice : {round(stats.skew(df.SalePrice),2)}')

fig = plt.figure(figsize=(7,4))
ax = sns.distplot(numerical_df.SalePrice, fit=stats.norm)
ax.set_title('Distribution of SalePrice', size=18, y=1.05)
plt.show();
numerical_df['LogSalePrice'] = np.log(numerical_df.SalePrice)

print(f'Skewness of LogSalePrice : {round(stats.skew(numerical_df.LogSalePrice),2)}')

fig = plt.figure(figsize=(7,4))
ax = sns.distplot(numerical_df.LogSalePrice, fit=stats.norm)
ax.set_title('Distribution of LogSalePrice', size=18, y=1.05)
plt.show();
numerical_df = numerical_df.drop('SalePrice', axis=1)
def check_variable_skew(df, threshold=1, verbose=True):
    '''Function that checks each variable in the dataframe for their skewness.
    
    Arguments:
        df : Pandas DataFrame
        threshold : int, default = 1
            The threshold that we allow for skewness within the variable.
        verbose : bool, default = True
            Prints out highly skewed variables and their values.
        
    Returns:
        highly_skewed_vars_list : list
    '''
    skewness = df.apply(lambda x : np.abs(stats.skew(x)))
    skewed_vars = skewness.loc[skewness >= threshold].sort_values(ascending=False).round(2)
    
    if len(skewed_vars) == 0:
        print('There are no variables that are highly skewed.')
        return []
    
    skewed_vars_list = skewed_vars.index.to_list()
    
    print(f'The following {len(skewed_vars_list)} variables are highly skewed:')
    print()
    for var in skewed_vars_list:
        print(var, '\t', skewed_vars.loc[var])
      
    return skewed_vars_list
def skewness_subplots(df, skewed_vars_list, n_cols=4, fig_size=(18,12)):
    '''Function that plots the distribution of each variable within a grid.
    
    Arguments:
        df : Pandas DataFrame
        skewed_vars_list : list
            List of variables to plot histograms for.
        n_cols : int, default = 4
            Number of columns for the grid
    '''
    num_vars = len(skewed_vars_list)
    n_rows = int(np.ceil(num_vars / n_cols))
    df_skewed_vars = df[skewed_vars_list]
    
    fig = plt.figure(figsize=fig_size)
    plt.suptitle('Distributions for Highly Skewed Variables', y=1.03, size=18)

    for i, col in enumerate(skewed_vars_list):
        skew = np.round(stats.skew(df[col]), 2)
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        sns.distplot(df[col], ax=ax, kde=False, bins=50)
        ax.set_title(f'Skew : {skew}', size=16)
    
    plt.tight_layout()    
    plt.show();
# Creating a list of variables that are highly skewed
highly_skewed_vars = check_variable_skew(numerical_df)
skewness_subplots(numerical_df, highly_skewed_vars, fig_size=(18,18))
def most_frequent_value_proportion(df, threshold=0.8, verbose=True):
    '''Function that returns series with variables and their most frequent values respectively.
    
    Arguments:
        df : Pandas DataFrame
        threshold : float
            Threshold for the maximum allowed proportion of a single value/class. 
            
    Returns:
        most_frequent_series : Pandas Series
            Variables as index and values as proportions for their most common value.
    '''
    most_frequent_pct = []
    for col in df.columns:
        most_frequent = df[col].value_counts(normalize=True).sort_values(ascending=False).iloc[0]
        most_frequent_pct.append(np.round(most_frequent,2))
    
    most_frequent_series = pd.Series(most_frequent_pct, index=df.columns)
    most_frequent_series = most_frequent_series.loc[most_frequent_series >= threshold]
    most_frequent_series = most_frequent_series.sort_values(ascending=False)
    
    if verbose:
        print(f'The following {len(most_frequent_series)} variables have a high concentration (>{threshold*100}%) of their values in one value only.')
        print()
        print(most_frequent_series)
    
    return most_frequent_series
narrow_dist_vars = most_frequent_value_proportion(numerical_df, threshold=0.8)
# Dropping narrowly distributed variables
numerical_df = numerical_df.drop(narrow_dist_vars.index.to_list(), axis=1)
# List of positvely skewed variables
pos_skewed_vars = list(set(highly_skewed_vars) - set(narrow_dist_vars.index.to_list()) - set(ordinal_vars))
def make_log_variables(df, variables_list, drop=False):
    '''Function to make new columns of the logarithmic transformation of a list of variables.
    Arguments:
        df : Pandas DataFrame
        variables_list : list
            List of variables to log-transform.
        drop : bool, default = False
            Pass as true to drop the original variables.
    Returns:
        df : Pandas DataFrame with new variables.
        log_var_list : list
            List of the log-transformed variable names.
    '''
    # Checking for negative values for each variable
    any_neg_value = np.sum((df[variables_list] < 0).all(axis=0))
    if any_neg_value:
        raise ValueError('There are one or more columns with negative values and cannot be log-transformed.')
    
    log_var_list = []
    
    for var in variables_list:
        log_var_name = 'Log' + var
        df[log_var_name] = np.log1p(df[var])
        log_var_list.append(log_var_name)
    
    if drop:
        df = df.drop(variables_list, axis=1)
    
    return df, log_var_list
# Creating log-transformations for our highly skewed variables and saving the new variables in a list
numerical_df, log_var_list = make_log_variables(numerical_df, pos_skewed_vars, drop=False)
print(f'Prior to log-transformation, there were {len(pos_skewed_vars)} variables that were highly positively skewed.')
highly_skewed_vars = check_variable_skew(numerical_df[log_var_list])
# Dropping original variables
pos_skewed_vars.remove('TotalBsmtSF')
numerical_df = numerical_df.drop(pos_skewed_vars, axis=1)
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
ax1.set_title(f'Skew : {stats.skew(numerical_df.TotalBsmtSF):.2f}', y=1.03)
sns.distplot(numerical_df.TotalBsmtSF, ax=ax1, bins=50)

ax2.set_title(f'Skew : {stats.skew(numerical_df.LogTotalBsmtSF):.2f}', y=1.03)
sns.distplot(numerical_df.LogTotalBsmtSF, ax=ax2, bins=50)

fig.tight_layout()
plt.show();
fig = sns.scatterplot(numerical_df.TotalBsmtSF, numerical_df.LogSalePrice)
# Removing the observation from both our numerical and caregorical data frames
outliers = numerical_df.loc[numerical_df.TotalBsmtSF > 5000].index.to_list()
# Dropping the outlier from both numerical and categorical data frames
numerical_df = numerical_df.drop(outliers, axis=0)
categorical_df = categorical_df.drop(outliers, axis=0)
print(numerical_df.shape, categorical_df.shape)
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
ax1.set_title(f'Skew : {stats.skew(numerical_df.TotalBsmtSF):.2f}', y=1.03)
sns.distplot(numerical_df.TotalBsmtSF, ax=ax1, bins=50)

ax2.set_title(f'Skew : {stats.skew(numerical_df.LogTotalBsmtSF):.2f}', y=1.03)
sns.distplot(numerical_df.LogTotalBsmtSF, ax=ax2, bins=50)

fig.tight_layout()
plt.show();
numerical_df = numerical_df.drop('LogTotalBsmtSF', axis=1)
# Rearranging our dataframe for easier interpretation of heatmap
log_sale_price = numerical_df.LogSalePrice
numerical_df = numerical_df.drop('LogSalePrice', axis=1)
numerical_df['LogSalePrice'] = log_sale_price
sns.set(style="white")

# Compute the correlation matrix
corr = numerical_df.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(20, 16))
plt.title('Correlation Heatmap', size=20)

cmap = sns.diverging_palette(220, 10, as_cmap=True)
heatmap = sns.heatmap(corr, mask=mask, cmap=cmap, fmt='.2f', vmin=-1, vmax=1.0, center=0, square=True,
                      linewidths=.5, cbar_kws={"shrink": .7}, annot=True, annot_kws={"size": 8})

bottom, top = ax.get_ylim()
heatmap.set_ylim(bottom + 0.5, top - 0.5)
numerical_df = numerical_df.drop(['OverallCond', 'YrSold', 'GarageCars'], axis=1)
corr_matrix_unstacked = corr.unstack().sort_values(ascending=False).drop_duplicates()
correlated_pairs = corr_matrix_unstacked.loc[corr_matrix_unstacked >= 0.75].index.to_list()
correlated_pairs
def scatter_subplots(df, target, hue=None, n_cols=4, fig_size=(12,12)):
    '''Function that plots the scatterplots of each variable against the target variable within a grid.
    
    Arguments:
        df : Pandas DataFrame with target variable included
        target : str
            Target feature name
        hue : str, default = None
            Column in the data frame that should be used for colour encoding
        n_cols : int, default = 4
            Number of columns for the grid
    '''
    independent_vars_list = list(df.columns)
    independent_vars_list.remove(target)
    num_vars = len(independent_vars_list)
    n_rows = int(np.ceil(num_vars / n_cols))
    
    plt.style.use('fivethirtyeight')
    fig = plt.figure(figsize=fig_size)
    plt.suptitle(f'Scatterplots of Independent Variables against {target}', y=1.02, size=18)

    for i, col in enumerate(independent_vars_list):
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        sns.scatterplot(x=col, y=target, hue=hue, data=df, ax=ax)
    
    plt.tight_layout()
    plt.show();
scatter_subplots(numerical_df, 'LogSalePrice', fig_size=(16,24))
# Adding the target variable to the categorical dataframe
categorical_df['LogSalePrice'] = numerical_df.LogSalePrice
print(categorical_df.shape, numerical_df.shape)
def annotate_plot(ax, dec_places=1, annot_size=14):
    '''Function that annotates plots with their value labels.
    Arguments:
        ax : Plot Axis.
        dec_places : int
            Number of decimal places for annotations.
        annot_size : int
            Font size of annotations.
    '''
    for p in ax.patches:
        ax.annotate(
            format(p.get_height(), '.{}f'.format(dec_places)),
            (p.get_x() + p.get_width() / 2., p.get_height(),),
            ha='center', va='center',
            xytext=(0,10), textcoords='offset points', size=annot_size
        )
def var_categories_countplots(df, n_cols=3, orientation='v', x_rotation=45, y_rotation=0, palette='pastel', fig_size=(18,12)):
    '''Function that plots the class distribution for categorical variables.
    
    Arguments:
        df : Pandas DataFrame
        n_cols : int, default = 3
            Number of columns for the subplot grid.
        orientation : str, default = 'v'
            Plot orientation, with 'v' for vertical and 'h' for horizontal.
        x_rotation : int, default = 45
            Rotation of the x-axis labels.
        palette : str, default = 'pastel'
            Seaborn color palette for plotting.
    '''
    categorical_vars = df.select_dtypes(exclude='number').columns.to_list()
    num_vars = len(categorical_vars)
    n_rows = int(np.ceil(num_vars / n_cols))
    
    fig = plt.figure(figsize=fig_size)
    plt.suptitle('Class Distributions for Categorical Variables', y=1.01, size=24)
    
    for i, col in enumerate(categorical_vars):
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        sns.countplot(x=df[col], ax=ax, orient=orientation, palette=palette)
        ax.set_ylabel('Frequency')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=x_rotation)
        
        annotate_plot(ax, dec_places=0, annot_size=12) # Annotating plot with count labels
    
    plt.tight_layout()
    plt.show();
# Finding variables that have more than 80% of their values in one category
highly_imbalanced_vars = most_frequent_value_proportion(categorical_df, threshold=0.8, verbose=False)
highly_imbalanced_vars_list = highly_imbalanced_vars.index.to_list()
print(f'The following {len(highly_imbalanced_vars)} variables have more than 80% of their data concentrated in only one class:')
print()
print(highly_imbalanced_vars)

# Note: Function was defined earlier in univariate analysis of numerical variables
var_categories_countplots(categorical_df[highly_imbalanced_vars_list], fig_size=(16,16))
categorical_df = categorical_df.drop(highly_imbalanced_vars_list, axis=1)
var_categories_countplots(categorical_df, fig_size=(20,16))
categorical_df.MSZoning = categorical_df.MSZoning.apply(lambda x :
                                                        x if x == 'RL'
                                                        else x if x == 'RM'
                                                        else 'Others')

categorical_df.LotShape = categorical_df.LotShape.apply(lambda x :
                                                        x if x =='Reg'
                                                        else 'Irregular')

categorical_df.LotConfig = categorical_df.LotConfig.apply(lambda x :
                                                          x if x == 'Inside'
                                                          else x if x == 'CulDSac'
                                                          else x if x == 'Corner'
                                                          else 'FR')

categorical_df.RoofStyle = categorical_df.RoofStyle.apply(lambda x :
                                                          x if x =='Gable'
                                                          else x if x == 'Hip'
                                                          else 'Others')

categorical_df.MasVnrType = categorical_df.MasVnrType.apply(lambda x :
                                                            x if x == 'None'
                                                            else x if x == 'Stone'
                                                            else 'Brk')

categorical_df.Foundation = categorical_df.Foundation.apply(lambda x :
                                                            x if x =='BrkTil'
                                                            else x if x == 'CBlock'
                                                            else x if x == 'PConc'
                                                            else 'Others')

categorical_df.GarageType = categorical_df.GarageType.apply(lambda x : 
                                                            x if x == 'Attchd'
                                                            else x if x == 'BuiltIn'
                                                            else x if x == 'Detchd'
                                                            else x if x == 'NoGarage'
                                                            else 'Others')
def var_categories_boxplots(df, target, hue=None, n_cols=3, orientation='v', x_rotation=45, y_rotation=0, palette='pastel', fig_size=(18,12)):
    '''Function that plots the class distribution for categorical variables against target variable.
    
    Arguments:
        df : Pandas DataFrame
        target : str
            Target variable name.
        hue : str, default = None
            Column in the data frame that should be used for colour encoding.
        n_cols : int, default = 3
            Number of columns for the subplot grid.
        orientation : str, default = 'v'
            Plot orientation, with 'v' for vertical and 'h' for horizontal.
        x_rotation : int, default = 45
            Rotation of the x-axis labels.
        palette : str, default = 'pastel'
            Seaborn color palette for plotting.
    '''
    categorical_vars = df.select_dtypes(exclude='number').columns.to_list()
    num_vars = len(categorical_vars)
    n_rows = int(np.ceil(num_vars / n_cols))
    
    fig = plt.figure(figsize=fig_size)
    plt.suptitle('Categorical Variables vs Target', y=1.01, size=24)
    
    for i, col in enumerate(categorical_vars):
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        sns.boxplot(x=df[col], y=df[target], ax=ax, hue=hue, orient=orientation, palette=palette)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=x_rotation)
    
    plt.tight_layout()
    plt.show();
var_categories_boxplots(categorical_df, 'LogSalePrice', n_cols=3, fig_size=(20,30))
df = pd.concat([categorical_df.drop('LogSalePrice', axis=1), numerical_df], axis=1).reset_index(drop=True)

# One Hot Encoding
df = pd.get_dummies(df)

y = df.LogSalePrice
X = df.drop('LogSalePrice', axis=1)
print(X.shape, y.shape)
X.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
kfold = KFold(n_splits=5, shuffle=True, random_state=SEED)
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Creating a sklearn scorer instance
rmse_scorer = {'RMSE' : make_scorer(rmse, greater_is_better=False, needs_proba=False, needs_threshold=False)}
# Kwargs for cross_validate method
cv_kwargs = {
    'scoring' : rmse_scorer,
    'cv' : kfold,
    'n_jobs' : -1,
    'return_train_score' : True,
    'verbose' : False,
    'return_estimator' : True
}
def save_cv_results(cv_results, scoring_name='score', verbose=True):
    '''Function to save the training and testing results from cross validation into a dataframe.
    
    Arguments:
        cv_results : dict
            Dictionary of results from scikit-learn's cross_validate method.
        scoring_name : str, default = 'score'
            Name of scorer used in the cross_validate method. If no custom scorer was passed, default should be 'score'.
            In the cv_results dictionary, there should be keys 'train_score' and 'test_score'
            If custom scorer was passed as the scoring method, the cv_results dictionary should have 'train_scoring_name'.
        verbose : bool, default = True
            Prints the mean training and testing scores and fitting times. 
            
    Returns:
        results_df : Pandas DataFrame with training and test scores.
    
    '''
    train_key = 'train_' + scoring_name
    test_key = 'test_' + scoring_name
    
    # Sklearn scorer flips the sign to negative so we need to flip it back
    train_scores = [-result for result in cv_results[train_key]]
    test_scores = [-result for result in cv_results[test_key]]
    
    indices = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']
    
    results_df = pd.DataFrame({'TrainScores' : train_scores, 'TestScores' : test_scores}, index=indices)
    
    if verbose:
        avg_train_score = np.mean(train_scores)
        avg_test_score = np.mean(test_scores)
        avg_training_time = np.mean(cv_results['fit_time'])
        avg_predict_time = np.mean(cv_results['score_time'])
        
        title = 'Cross Validation Results Summary'
        print(title)
        print('=' * len(title))
        print(f'Avg Training {scoring_name}', '\t', '{:.6f}'.format(avg_train_score))
        print(f'Avg Testing {scoring_name}', '\t', '{:.6f}'.format(avg_test_score))
        print()
        print('Avg Fitting Time', '\t', '{:.4f}s'.format(avg_training_time))
        print('Avg Scoring Time', '\t', '{:.4f}s'.format(avg_predict_time))
    
    return results_df
def training_vs_testing_plot(results, fig_size=(5,5), title_fs=18, legend_fs=12):
    '''Function that plots the training and testing scores obtained from cross validation.'''
    
    fig = plt.figure(figsize=fig_size)
    plt.style.use('fivethirtyeight')
    plt.title('Cross Validation : Training and Testing Scores', y=1.03, x=0.6, size=title_fs)
    plt.plot(results.TrainScores, color='b', label='Training')
    plt.plot(results.TestScores, color='r', label='Testing')
    plt.legend(loc='center left', bbox_to_anchor=(1.02,0.5), ncol=1, fontsize=legend_fs)
    plt.show();
def get_best_estimator(cv_results, scoring_name='score'):
    ''' Function that returns the best estimator found during cross valiation.
    Arguments:
        cv_results : dict
            Results from Sklearn's cross_validate method.
        scoring_name : str, default = 'score'
            Custom scoring name if a custom scorer was passed during cross validation.
            Default 'score' should be used when using sci-kit learn's  scoring metrics. 
    
    Returns:
        best_estimator : Sklearn estimator object
            Best estimator found during cross validation.
    '''
    test_key = 'test_' + scoring_name
    
    # Sklearn flips the sign during scoring so we need to flip it back
    scores = [-result for result in cv_results[test_key]]
    max_score_index = scores.index(max(scores))
    best_estimator = cv_results['estimator'][max_score_index]
    
    return best_estimator
linreg = LinearRegression(fit_intercept=True, normalize=False, n_jobs=-1)

linreg_cv_results = cross_validate(linreg, X_train, y_train, **cv_kwargs)
linreg_cv_results
# Saving cross validation results to a dataframe
linreg_cv_scores = save_cv_results(linreg_cv_results, scoring_name='RMSE')

# Plotting training vs testing RMSE scores
training_vs_testing_plot(linreg_cv_scores)
# Saving best estimator from cross validation
best_linreg = get_best_estimator(linreg_cv_results, scoring_name='RMSE')
def holdout_set_evaluation(model, X_train, y_train, X_test, y_test, model_name, scoring_name):
    '''Function that evaluates the performance on the holdout dataset.
    
    Arguments:
        model : sklearn estimator object
        model_name : str
            String to be passed as the index for the dataframe.
        scoring_name : str
            Evluation metric used as column header.
    
    Returns:
        rmse_score : Pandas DataFrame
            Column is the scoring_name, index is the model_name and value is the model performance on the holdout dataset.
    '''    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    rmse_score = rmse(y_test, y_pred) # calls rmse function
    rmse_score = pd.DataFrame({scoring_name : [rmse_score]}, index=[model_name])
    
    return rmse_score.round(4)
# Saving our result to a dataframe
linreg_result = holdout_set_evaluation(best_linreg, X_train, y_train, X_test, y_test, model_name='Linear', scoring_name='RMSE')
linreg_result
# Creating a new dataframe to store model results
model_results = linreg_result.copy(deep=True)
print('Highly Correlated Pairs of Variables:')
print(correlated_pairs)
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

vif_X = add_constant(X)
vifs = pd.Series([
    variance_inflation_factor(vif_X.values, i) for i in range(vif_X.shape[1])],
    index=vif_X.columns
)

# Dropping inf values due to dummy variables
vifs = vifs.loc[vifs != np.inf].sort_values(ascending=False)
vifs.loc[vifs > 5]
# Range of alphas to iterate over
alphas_vector = np.arange(1,200)
def alpha_tuning_results(alphas_vector, X_train, y_train, cv_kwargs, model_name='Ridge', scoring_name='score', x_axis_log_scale=False, fig_size=(5,5)):
    ''' Function to obtain the average training and testing RMSE across different alpha values.
    
    Arguments:
        alphas_vector : array
            array of alpha values to iterate through and fit the model
        cv_kwargs : dict
            kwargs for the cross_validate method
        model_name : str, default = 'Ridge'
            Type of model to fit training data. Any other str input will fit a Lasso model.
        scoring_name : str, default = 'score'
            Scoring used in sklearn's cross_validate method. Use default value if no custom scorer was used.
            Else, enter the name of the scorer used when making scorer.
        x_axis_log_scale : bool, default = False
            Set the X-axis to log scale. Useful when tuning Lasso Alpha.
        
    Returns:
        results_df : Pandas DataFrame with average training and testing RMSE per alpha. 
            
    '''
    results_df = pd.DataFrame(columns=['Avg_Train_RMSE', 'Avg_Test_RMSE'])

    for alpha in alphas_vector:
        if model_name == 'Ridge':
            model = Pipeline(steps=[
                ('Standardise',  StandardScaler()),
                ('Ridge', Ridge(alpha=alpha, fit_intercept=True, random_state=SEED))
            ])
        else:
            model = Pipeline(steps=[
                ('Standardise', StandardScaler()),
                ('Lasso', Lasso(alpha=alpha, fit_intercept=True, random_state=SEED))
            ])
        
        cv_results = cross_validate(model, X_train, y_train, **cv_kwargs)
        train_key = 'train_' + scoring_name
        test_key = 'test_' + scoring_name
        # Sklearn scorer flips the sign to negative so we need to flip it back
        train_scores = [-result for result in cv_results[train_key]]
        test_scores = [-result for result in cv_results[test_key]]
        avg_train_rmse = np.mean(train_scores)
        avg_test_rmse = np.mean(test_scores)
        
        results_df.loc[alpha] = [avg_train_rmse, avg_test_rmse]
    
    # Visualising the results
    fig = plt.figure(figsize=fig_size)
    plt.style.use('fivethirtyeight')
    plt.title(f'Training and Testing {scoring_name} for Different Alpha Values', y=1.03, x=0.6, size=16)
    plt.ylabel(f'{scoring_name}', size=14)
    plt.xlabel('Alpha', size=14)
    
    if x_axis_log_scale:
        plt.xscale('log')
    
    plt.plot(results_df.Avg_Train_RMSE, color='b', label='Training')
    plt.plot(results_df.Avg_Test_RMSE, color='r', label='Testing')
    plt.legend(loc='center left', bbox_to_anchor=(1.02,0.5), ncol=1, prop={'size': 14})
    plt.plot();
    
    return results_df
# Function to perform cross validation for each alpha value and plot the average RMSE obtained for each alpha value
ridge_results_df = alpha_tuning_results(np.arange(1,200), X_train, y_train, cv_kwargs, model_name='Ridge', scoring_name='RMSE')
# Getting the alpha which has the lowest testing RMSE
optimal_ridge_alpha = ridge_results_df.Avg_Test_RMSE.idxmin()
print(f'The optimal alpha from cross validation : {optimal_ridge_alpha}')
# Creating model with optimal alpha
ridge = Pipeline(steps=[
    ('Standardise',  StandardScaler()),
    ('Ridge', Ridge(alpha=optimal_ridge_alpha, fit_intercept=True, random_state=SEED))
])

# Cross validation results
ridge_cv_results = cross_validate(ridge, X_train, y_train, **cv_kwargs)

# Saving scores from cv results
ridge_cv_scores = save_cv_results(ridge_cv_results, scoring_name='RMSE')

# Plotting training vs testing RMSE scores
training_vs_testing_plot(ridge_cv_scores)
# Saving best estimator from cross validation
best_ridge = get_best_estimator(ridge_cv_results, scoring_name='RMSE')

# Evaluating resutls 
ridge_result = holdout_set_evaluation(best_ridge, X_train, y_train, X_test, y_test, model_name='Ridge', scoring_name='RMSE')
model_results = model_results.append(ridge_result)
model_results
# Using a GridSearch to find the optimal alpha for the ridge regression 
ridge_pipe = Pipeline(steps=[
    ('Standardise', StandardScaler()),
    ('Ridge', Ridge(fit_intercept=True, random_state=SEED))
])

ridge_params = {'Ridge__alpha' : np.arange(1,200)}

ridge_gscv = GridSearchCV(ridge_pipe, ridge_params, scoring=rmse_scorer['RMSE'], n_jobs=-1, cv=kfold, return_train_score=True)
ridge_gscv.fit(X_train, y_train)
# Best alpha 
ridge_gscv.best_params_
# Range of alphas to iterate over
alphas_vector = np.logspace(-6,0,7)

# Function to perform cross validation for each alpha value and plot the average RMSE obtained for each alpha value
lasso_results_df = alpha_tuning_results(alphas_vector, X_train, y_train, cv_kwargs, model_name='Lasso', scoring_name='RMSE', x_axis_log_scale=True)
lasso_results_df.Avg_Test_RMSE.idxmin()
alphas_vector = np.linspace(0.0001, 0.01, 100)

# Function to perform cross validation for each alpha value and plot the average RMSE obtained for each alpha value
lasso_results_df = alpha_tuning_results(alphas_vector, X_train, y_train, cv_kwargs, model_name='Lasso', scoring_name='RMSE', x_axis_log_scale=True)
# Getting the alpha which has the lowest testing RMSE
optimal_lasso_alpha = lasso_results_df.Avg_Test_RMSE.idxmin()
optimal_lasso_alpha
lasso_pipe = Pipeline(steps=[
    ('Standardise', StandardScaler()),
    ('Lasso', Lasso(alpha=optimal_lasso_alpha, fit_intercept=True, random_state=SEED))
])

lasso_cv_results = cross_validate(lasso_pipe, X_train, y_train, **cv_kwargs)

# Saving cross validation results 
lasso_cv_scores = save_cv_results(lasso_cv_results, scoring_name='RMSE')

# Plotting training vs testing RMSE scores
training_vs_testing_plot(lasso_cv_scores)
# Saving best model from cross validation
best_lasso = get_best_estimator(lasso_cv_results, scoring_name='RMSE')

# Evaluating Lasso performance
lasso_result = holdout_set_evaluation(best_lasso, X_train, y_train, X_test, y_test, model_name='Lasso', scoring_name='RMSE')
model_results = model_results.append(lasso_result)
model_results
best_lasso.fit(X_train, y_train)

# We need to access the 'named_steps' to get our Lasso estimator as we are using a Pipeline
lasso_model = best_lasso.named_steps['Lasso']
lasso_coefs = pd.Series(lasso_model.coef_, index=X.columns)
zero_coefs = lasso_coefs.loc[lasso_coefs == 0].index.to_list()

print(f'Original training data frame had {X.shape[1]} variables.')
print(f'Lasso selection removed {len(zero_coefs)} variables.')
print(f'Number of variables with non-zero coefficients : {X.shape[1] - len(zero_coefs)}')
# Creating new training and testing datasets after removing variables
small_df = df.drop(zero_coefs, axis=1)

Xsmall = small_df.drop('LogSalePrice', axis=1)
Xsmall_train, Xsmall_test, y_train, y_test = train_test_split(Xsmall, y, test_size=0.2, random_state=SEED)
print(Xsmall_train.shape, Xsmall_test.shape)
linreg_small = LinearRegression(fit_intercept=True, n_jobs=-1)

# Cross validation 
linreg_small_cv_results = cross_validate(linreg_small, Xsmall_train, y_train, **cv_kwargs)

# Saving cross validation results
linreg_small_cv_scores = save_cv_results(linreg_small_cv_results, scoring_name='RMSE')

# Plotting training vs testing RMSE scores
training_vs_testing_plot(linreg_small_cv_scores)
# Saving best estimator from cross validation
best_linreg_small = get_best_estimator(linreg_small_cv_results, scoring_name='RMSE')

# Saving our result to a dataframe so it is easier to append the performances of other models
linreg_small_result = holdout_set_evaluation(best_linreg_small, Xsmall_train, y_train, Xsmall_test, y_test, model_name='Linear_SmallDf', scoring_name='RMSE')

# Evaluating results
model_results = model_results.append(linreg_small_result)
model_results
# Baseline Untuned Random Forest Model
baseline_rforest = RandomForestRegressor(random_state=SEED)
baseline_rf_cv_results = cross_validate(baseline_rforest, Xsmall_train, y_train, **cv_kwargs)

# Saving cross validation results
baseline_rf_cv_scores = save_cv_results(baseline_rf_cv_results, scoring_name='RMSE')

# Plotting training vs testing RMSE scores
training_vs_testing_plot(baseline_rf_cv_scores)
# Saving best estimator from cross validation
best_baseline_rf = get_best_estimator(baseline_rf_cv_results, scoring_name='RMSE')

# Saving our result to a dataframe so it is easier to append the performances of other models
baseline_rf_result = holdout_set_evaluation(best_baseline_rf, Xsmall_train, y_train, Xsmall_test, y_test, model_name='BaselineRF', scoring_name='RMSE')
model_results = model_results.append(baseline_rf_result)
model_results
def rforest_tuning_scores(model, X_train, y_train, parameter, param_range, scorer, cv, flip_scores=True):
    
    gridsearch = GridSearchCV(model, param_grid={parameter : param_range}, scoring=scorer,
                              cv=cv, n_jobs=-1, return_train_score=True, verbose=False)
    
    gridsearch.fit(X_train, y_train)
    cv_results = gridsearch.cv_results_
    
    if flip_scores:
        train_scores = [-result for result in cv_results['mean_train_score']]
        test_scores = [-result for result in cv_results['mean_test_score']]
    else:
        train_scores = [cv_results['mean_train_score']]
        test_scores = [cv_results['mean_test_score']]
        
    results_df = pd.DataFrame({'TrainScores' : train_scores, 'TestScores' : test_scores}, index=param_range)
    
    return results_df
param_grid = {
    'max_features' : ['auto', 'sqrt', 'log2'],
    'max_depth' : [1, 2, 5, 10, 20, 30, 50, None],
    'min_samples_split' : np.arange(2, 30, step=2),
    'min_samples_leaf' : np.arange(1, 20, step=1)
}
n_cols = 2
n_vars = len(param_grid)
n_rows = int(np.ceil(n_vars / n_cols))
index = 0

fig = plt.figure(figsize=(12,6))
plt.suptitle('Training and Test Scores for Different Hyper Parameters', y=1.03, size=20)

for parameter, param_range in dict.items(param_grid):
    results_df = rforest_tuning_scores(baseline_rforest, Xsmall_train, y_train, parameter=parameter, param_range=param_range,
                                       scorer=rmse_scorer['RMSE'], cv=kfold, flip_scores=True)

    ax = fig.add_subplot(n_rows, n_cols, index+1)
    plt.plot(results_df.TrainScores, color='b', label='Training')
    plt.plot(results_df.TestScores, color='r', label='Testing')
    
    plt.xlabel(parameter, size=14)
    plt.legend(loc='center left', bbox_to_anchor=(1.02,0.5), ncol=1, prop={'size': 14})
    
    index += 1

plt.tight_layout()
plt.show();
# New parameter tuning grid
rforest_tuning_grid = {
    'max_depth' : np.arange(8,13 , step=1),
    'max_features' : ['auto', 'sqrt', 'log2'],
    'min_samples_split' : np.arange(2, 6, step=1),
    'min_samples_leaf' : np.arange(1, 6, step=1),
}
# Dictionary where we will store the optimal kwargs for the tuned random forest model
rforest_kwargs = {
    'n_estimators' : 100,
    'criterion' : 'mse',
    'max_features' : 'sqrt',
    'bootstrap' : True,
    'n_jobs' : -1,
    'random_state' : SEED, 
}
np.random.seed(8888)
gs_rforest = GridSearchCV(estimator=RandomForestRegressor(**rforest_kwargs), param_grid=rforest_tuning_grid,
                          scoring=rmse_scorer['RMSE'], n_jobs=-1, cv=kfold, refit=True)

gs_rforest.fit(Xsmall_train, y_train)
# Saving the best estimator from the gird search
tuned_rforest = gs_rforest.best_estimator_

# Updating our random forest kwargs with the optimal hyper parameter values
rforest_kwargs.update(gs_rforest.best_params_)
rforest_kwargs
# Evaluating tuned random forest results
tuned_rf_result = holdout_set_evaluation(tuned_rforest, Xsmall_train, y_train, Xsmall_test, y_test, model_name='TunedRF', scoring_name='RMSE')
model_results = model_results.append(tuned_rf_result)
model_results