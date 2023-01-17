import numpy as np

import pandas as pd



# displays 300 columns of pandas dataframe

pd.options.display.max_columns = 300



# model selection

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV



# metrics

from sklearn.metrics import mean_absolute_error



# models

from sklearn.ensemble import GradientBoostingRegressor

from xgboost import XGBRegressor



train_name = '/kaggle/input/home-data-for-ml-course/train.csv'

test_name = '/kaggle/input/home-data-for-ml-course/test.csv'
def read_csvs(csv):

    return pd.read_csv(csv)

train = read_csvs(train_name)

test = read_csvs(test_name)
train.head()
train.describe()
train.groupby(['OverallQual'])['SalePrice'].mean()
# Example plot

train.plot(x='OverallQual', y='SalePrice', style='o')
def find_nan(df):

    """ 

    Finds the columns that have NaN entries

    """

    null_series = df.isnull().sum()

    return [col_name for col_name in null_series.index if null_series[col_name] != 0]
# Number of NaN entries in each column that has NaN entries

for col in find_nan(train):

    print(col, str(train[col].isnull().sum()))
def fill_nan(df):

    """ 

    Fills columns containing NaN

    """

    # Series of columns with NaN entries and each column's datatype

    dtypes = df[find_nan(df)].dtypes

    

    # Fills NaN entries in Object columns with 'None'

    none_cols = [col_name for col_name in dtypes.index if dtypes[col_name] == 'O']

    for col in none_cols:

        df[col] = df[col].fillna('None')

    

    # Fills most numerical columns with 0

    zero_cols = ['LotFrontage', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 

                 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 

                 'GarageCars', 'GarageArea']

    for col in zero_cols:

        df[col] = df[col].fillna(0)

        

    # Fills year garage built with year house built, the closest approximation

    df.GarageYrBlt = df.GarageYrBlt.fillna(df.YearBuilt)

    

    return df

train = fill_nan(train)

test = fill_nan(test)
find_nan(train), find_nan(test)
train.FireplaceQu
train.MasVnrArea
def fill_ord_cols(df):

    """ 

    This method sets ordinal columns to numerical rankings



    For example, BsmtCond (Basement Condition) looks like this at the start:

    Ex - Excellent

    Gd - Good

    TA - Typical/Average

    Fa - Fair

    Po - Poor

    None - No Basement



    Let's instead change to be:

    5 - Excellent

    4 - Good

    3 - Typical/Average

    2 - Fair

    1 - Poor

    0 - No Basement

    """

    # Creates dictionaries for conversion and lists of certain columns to convert

    grading_dict = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}

    grading_list = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 

                    'HeatingQC', 'KitchenQual', 'GarageQual', 'GarageCond', 

                    'PoolQC', 'FireplaceQu']

    bsmtFinType_dict = {'GLQ': 2, 'ALQ': 2, 'Unf': 2, 'LwQ': 1,

                        'BLQ': 1, 'Rec': 1, 'None': 0}

    BsmtFinType_list = ['BsmtFinType1', 'BsmtFinType2']

    

    utilities_dict = {'AllPub': 4, 'NoSewr': 3, 'NoSeWa': 2, 'ELO': 1}

    bsmtExposure_dict = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'None': 0}

    centralAir_dict = {'Y': 1, 'N': 0}

    street_dict = {'Pave': 1, 'Grvl': 0}

    fence_dict = {'None': 5, 'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1}

    pavedDrive_dict = {'Y': 2, 'P': 1, 'N': 0}

    garage_dict = {'Fin':3, 'RFn': 2, 'Unf': 1, 'None': 0}

    

    # Coverts the columns

    for col in grading_list:

        df[col] = df[col].map(grading_dict)

    for col in BsmtFinType_list:

        df[col] = df[col].map(bsmtFinType_dict)

        

    df.BsmtExposure = df.BsmtExposure.map(bsmtExposure_dict)

    df.CentralAir = df.CentralAir.map(centralAir_dict)

    df.PavedDrive = df.PavedDrive.map(pavedDrive_dict)

    df.Street = df.Street.map(street_dict)

    df.GarageFinish = df.GarageFinish.map(garage_dict)

    df.Fence = df.Fence.map(fence_dict)

    df.Utilities = df.Utilities.map(utilities_dict)

    return df



train = fill_ord_cols(train)

test = fill_ord_cols(test)
train.BsmtExposure
def make_dummies(df):

    """

    Returns a dataframe with remaining Object columns into dummy columns

    """

    return pd.get_dummies(df)

train = make_dummies(train)

test = make_dummies(test)
train.head()
train.shape
def get_best_cols(num_cols):

    """

    Returns a list of the best columns ranking from highest to lowest

    in terms of their correlation to SalePrice using the train dataframe

    """

    correlations = train[train.columns[1:]].corr()['SalePrice'].abs()

    correlations = correlations.nlargest(num_cols)

    return list(correlations.index)

best_cols = get_best_cols(125)[1:]
# Top 10 columns most correlated to SalePrice and the correlations

for col in best_cols[:10]:

    print (col, train[col].corr(train.SalePrice))
X = train[best_cols]

y = train.SalePrice

test = test[best_cols]
X
train_features, test_features, train_labels, test_labels = train_test_split(X, y)
#Gradient Boost trial

grad_params = {

    'n_estimators': [5000],

    'learning_rate': [0.035, 0.05, 0.065],

    'loss': ['ls'],

    'random_state': [0]

}



grad_model = GridSearchCV(

    GradientBoostingRegressor(),

    scoring='neg_mean_absolute_error',

    param_grid=grad_params,

    cv=5,

    n_jobs=-1,

    verbose=1,

).fit(train_features, train_labels)



print(grad_model.best_estimator_)



mean_absolute_error(test_labels, grad_model.predict(test_features))
#XGBoost Regressor

xgb_model = XGBRegressor(

    n_estimators=10000,

    max_depth=5,

    min_child_weight=0,

    learning_rate=0.0031,

    subsample=0.2,

    random_state=0).fit(train_features, train_labels)



mean_absolute_error(test_labels, xgb_model.predict(test_features))
model = XGBRegressor(

    n_estimators=10000,

    max_depth=5,

    min_child_weight=0,

    learning_rate=0.0031,

    subsample=0.2,

    random_state=0

).fit(X, y)
predictions = model.predict(test)
id_series = read_csvs(test_name).Id

id_series
df = pd.DataFrame({'Id': id_series, 'SalePrice': predictions})
df.to_csv('submission.csv', index=False) 