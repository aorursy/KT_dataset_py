import numpy as np

import pandas as pd

import seaborn as sns

import os
# CONSTS USED

# pinned random state

RANDOM_STATE = 1313361

TEST_SET_SIZE = 0.2



OCEAN_PROX = 'ocean_proximity'



RIDGE = 'RidgeRegressor'

LASSO = 'LassoRegressor'

ELASTIC = 'ElasticNetRegressor'



MEDIAN_HOUSE = 'median_house_value'

MAX_HP = 500000 # max house price

MIN_HP = 15000 # min house price

ALPHA_VALUES = [0.1, 0.001]

from sklearn.preprocessing import OrdinalEncoder

from sklearn.impute import SimpleImputer



pd.options.mode.chained_assignment = None



def encode_categorical(df_col):

    '''

    Encodes a categorical dataframe column into numeric values.

    

    Args:

        df_col (DataFrame): A dataframe column which is categorical.

    '''

    encoder = OrdinalEncoder()

    encoded_col = encoder.fit_transform(df_col)

    return pd.DataFrame(encoded_col, columns=df_col.columns)



def impute_values(df):

    '''Imputes missing values for passed in DataFrame'''

    imputer = SimpleImputer(strategy='median')

    imputer.fit(df)

    df_transformed = imputer.transform(df)

    return pd.DataFrame(df_transformed, columns=df.columns)

    

    
input_csv_loc = '../input/housing.csv'

housing_raw = pd.read_csv(input_csv_loc)



# housing = housing_raw.dropna()

housing_raw[OCEAN_PROX] = encode_categorical(housing_raw[[OCEAN_PROX]])

housing = impute_values(housing_raw)

housing.head()

from sklearn.model_selection import train_test_split

X = housing.drop(MEDIAN_HOUSE, axis=1)

y = housing[MEDIAN_HOUSE].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SET_SIZE, random_state=RANDOM_STATE)
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt



def run_reg(regressor, X_train, X_test, y_train, y_test, model_name=''):

    '''

    Trains the passed in regressor with training data.

    Plots actual vs predicted results as well as the

    trained models coeffecients.

    '''

    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test).clip(MIN_HP, MAX_HP)

    mae = mean_absolute_error(y_test, y_pred)



    if model_name:

        f1_title = '%s MAE = %s' % (model_name, mae)

        f2_title = '%s model coeffecients' % model_name

    else:

        f1_title = 'MAE = %s' % mae

        f2_title = 'model coeffecients'



    # actual vs predicted plot

    plt.plot(y_test, y_pred, 'o', markersize=2)

    plt.suptitle(f1_title)

    plt.ylabel('actual house prices')

    plt.xlabel('predicted house prices')

    plt.show()

    

    # model coefs plot

    plt.plot(regressor.coef_)

    plt.suptitle(f2_title)

    plt.show()

    return mae
from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet



def get_models_with_alpha(alpha_value, ridge_solver='cholesky', elastic_l1=0.5):

    '''

    Creates the following regressors, with alpha set:

        - Ridge

        - Lasso

        - ElasticNet

    '''

    return {

        RIDGE: Ridge(alpha=alpha_value, solver=ridge_solver),

        LASSO: Lasso(alpha=alpha_value),

        ELASTIC: ElasticNet(alpha=alpha_value, l1_ratio=elastic_l1),

    }



def run_models(X_train, X_test, y_train, y_test):

    maes = []

    for alpha in ALPHA_VALUES:

        models = get_models_with_alpha(alpha)

        for model in models:

            print('Model: %s, Alpha: %s' % (model, alpha))

            maes.append(run_reg(models[model], X_train, X_test, y_train, y_test, model_name=model))

    print(maes)



run_models(X_train, X_test, y_train, y_test)

from sklearn.preprocessing import PolynomialFeatures

d3_poly_f = PolynomialFeatures(degree=3)

X_train_poly3 = d3_poly_f.fit_transform(X_train)

X_test_poly3 = d3_poly_f.fit_transform(X_test)
run_models(X_train_poly3, X_test_poly3, y_train, y_test)