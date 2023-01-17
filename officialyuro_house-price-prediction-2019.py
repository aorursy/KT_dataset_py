'''STEP 1: Load Data'''



import numpy as np

import pandas as pd



sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

test_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
# '''STEP 2: Exploratory Data Analysis'''



# # Eyeball the data...

# train_data.head()

# train_data.columns # Features

# train_data.info()



# # Distribution Plots

# import matplotlib.pyplot as plt

# import seaborn as sns

# from bokeh.io import output_file, show

# from bokeh.plotting import figure



# num_f = [ feature for feature in  train_data.columns if (train_data[feature].count() == 1460 and train_data[feature].dtype != 'O') ]



# num_nf = [ feature for feature in  train_data.columns if (train_data[feature].count() != 1460 and train_data[feature].dtype != 'O') ]



# # %matplotlib inline

# for feat in num_f:

#     sns.scatterplot ( train_data[ feat ], train_data[ 'SalePrice' ] )

#     plt.title ( feat )

#     plt.show()









# # Observations from Plots

# train = train_data.copy( deep=True )

# num_to_cat = [ 'MSSubClass', 'OverallQual', 'OverallCond', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars' ]



# for feat in num_to_cat:

#      train [ feat ] = train [ feat ].astype ( 'category' )



# feat_with_outliers = [ 'LotArea', 'BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'GarageArea', ]



# to_date = [ 'YearBuilt', 'YearRemodAdd', 'MoSold', 'YrSold', 'GarageYrBlt' ]





# # plot = figure(plot_width=400, tools='pan,box_zoom')

# # # plot.circle ( ... )

# # plot.output_file('DistPlot.html')

# # show (plot)



# train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
'''STEP 3: Data Transformation, Feature Engineering Process'''



def sort_features ( _df ):

    '''

    Funtion to sort the features of a given dataframe into numerical, and object-type features.

    '''

    df = _df.copy()

    numerical_features = []

    object_features =[]

    counter = 0

    for i in df.dtypes:

        if i == 'O': object_features.append( df.columns[ counter ] )

        else: numerical_features.append( df.columns[ counter ] )

        counter += 1

    

    return numerical_features, object_features



def drop_columns ( _df, feat_to_drop_list ):

    '''

    Function to discard a list of unwanted data categories.

    '''

    df = _df.copy()

    for feature in feat_to_drop_list:

        df = df.drop ( feature, axis=1 )

    return df



# from sklearn.preprocessing import OneHotEncoder

# encoder = OneHotEncoder() 

def to_category ( _df, feat_list ):

    '''

    Funtion to transform object-type variables to categorical variables.

    '''

    df = _df.copy()

    for feature in feat_list:

        obj_feature = df[ feature ]

        df [ feature ] = obj_feature.astype ( 'category' )

        df [ feature ] = df[ feature ].cat.codes

    return df



from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder



def transform_data ( _df ):

    '''

    Function to handle missing data in a given dataframe.

    '''

    df = _df.copy()

    cols = df.columns

    num, cat = sort_features ( df )

    count = 0

    

    for feature in df:

        num_of_nulls = df [ feature ].isnull().sum()

        perc_null = round((100 * ( num_of_nulls / 1460 )), 2)



        if df.dtypes[ count ] == 'O':

            if perc_null >= 90:

                drop_columns ( df, [feature] )



            elif perc_null < 5:

                df [ feature ].dropna( inplace=True )



            else:

                df [ feature ] = df [ feature ].astype ( 'category' )

                df [ feature ].fillna( method='bfill', inplace=True )



        count += 1



    imp = SimpleImputer( strategy = 'most_frequent' )

    imp.fit( df.values )

    df = imp.transform( df )



    df = pd.DataFrame( df, columns=cols )

    

    for feature in num:

        df[ feature ] = df[ feature ].apply( lambda x: float(x) )

    

    df = to_category( df, cat )

    

    return df

    

filled_data = transform_data ( train_data )

# filled_data = filled_data.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

#...

# We will have to do OneHotEncoding for all the object_features.

# set(list(train_data.Street)) # For viewing the categories
'''STEP 4: Building and Testing different Model.'''



# Import the needed libraries...

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, Ridge, LassoCV

from sklearn.preprocessing import PolynomialFeatures

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR

import lightgbm

import xgboost

from mlxtend.regressor import StackingCVRegressor

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import mean_squared_error

import math

import warnings



warnings.filterwarnings('ignore')

warnings.filterwarnings('ignore', category=DeprecationWarning)



SEED=0





# Linear Regression Model

lr_pipe = Pipeline(

    steps = [

        ( 'scaler', StandardScaler() ),

        ( 'linreg', LinearRegression(

                        fit_intercept=True,

                        normalize=True,

                        copy_X=True,

                        n_jobs=-1

        ) )

    ]

)



# Ridge Regression Model

ri_pipe = Pipeline(

    steps = [

        ( 'scaler', StandardScaler() ),

        ( 'ridge', Ridge(

                    alpha=1.0,

                    fit_intercept=True,

                    normalize=True,

                    copy_X=True,

                    max_iter=None,

                    tol=0.001,

                    solver='lsqr',

                    random_state=SEED

        ) )

    ]

)



# Lasso Regression Model

la_pipe = Pipeline(

    steps = [

        ( 'scaler', StandardScaler() ),

        ( 'lasso', LassoCV(

                    eps=0.001,

                    n_alphas=1000,

                    alphas=None,

                    fit_intercept=True,

                    normalize=False,

                    precompute='auto',

                    max_iter=1000,

                    tol=0.0001,

                    copy_X=True,

                    cv=8,

                    verbose=0,

                    n_jobs=-1,

                    positive=False,

                    random_state=SEED,

                    selection='random'

        ) )

    ]

)



# SVR Linear Model

svr_pipe = Pipeline(

    steps = [

        ( 'scaler', StandardScaler() ),

        ( 'svr', SVR(

                    kernel='rbf',

                    degree=3,

                    gamma='auto_deprecated',

                    coef0=0.0,

                    tol=0.001,

                    C=1.0,

                    epsilon=0.1,

                    shrinking=True,

                    cache_size=200,

                    verbose=False,

                    max_iter=-1

        ) )

    ]

)



# Random Regression Model

rf_pipe = Pipeline(

    steps = [

        ( 'scaler', StandardScaler() ),

        ( 'rf', RandomForestRegressor(

                    n_estimators=1800,

                    criterion='mse',

                    max_depth=15,

                    min_samples_split=2,

                    min_samples_leaf=1,

                    min_weight_fraction_leaf=0.0,

                    max_features='auto',

                    max_leaf_nodes=None,

                    min_impurity_decrease=0.0,

                    min_impurity_split=None,

                    bootstrap=True,

                    oob_score=False,

                    n_jobs=-1,

                    random_state=SEED,

                    verbose=0,

                    warm_start=False,



        ) )

    ]

)





# LGB Regression Model

lg_pipe = Pipeline(

    steps = [

        ( 'scaler', StandardScaler() ),

        ( 'lg', lightgbm.LGBMRegressor(

                        n_estimators=5000,

                        objective='poisson',

                        boosting='gbdt',

                        learning_rate=0.095                      

        ) )

    ]

)



# XGB Regression Model

xgb_pipe = Pipeline(

    steps = [

        ( 'scaler', StandardScaler() ),

        ( 'xgb', xgboost.XGBRFRegressor(

                                max_depth=15,

                                learning_rate=1,

                                n_estimators=5000,

                                verbosity=0,

                                silent=None,

                                objective='reg:squarederror',

                                n_jobs=-1,

                                nthread=None,

                                gamma=0,

                                min_child_weight=1,

                                max_delta_step=0,

                                subsample=0.8,

                                colsample_bytree=1,

                                colsample_bylevel=1,

                                colsample_bynode=0.8,

                                reg_alpha=0,

                                reg_lambda=1,

                                scale_pos_weight=1,

                                base_score=0.5,

                                random_state=SEED,

                                seed=None,

                                missing=None

        ) )

    ]

)





# Testiong the models above...





def test_models ( models, training_data, target_variable ):

    '''

    Function to evaluate the efficiency of a dictionary of models.

    The parameter 'models' is a dictionary of the models to be tested where the keys are merely strings of the model_title, and the values are the actual models themselves.

    '''



    X = training_data.drop([ target_variable ], axis=1)

    y = training_data[ target_variable ]

    X_train, X_test, y_train, y_test = train_test_split(

                X,

                y,

                test_size=0.3,

                random_state=SEED

    )

    results = {}



    for model_title, model in models.items():

        _ = model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        error = mean_squared_error(y_test, y_pred)



        results.update( { model_title : round (math.log10(error), 5 ) } )



    return results



models_to_test = {  'LR': lr_pipe,

                    'RI': ri_pipe,

                    'LA': la_pipe,

                    'RF': rf_pipe,

                    'LG': lg_pipe,

                    'SVR': svr_pipe,

                    'XGB': xgb_pipe  }



MSE_results = test_models ( models_to_test, filled_data, 'SalePrice' )

print (MSE_results)



# Stacking Regressors



# stack_pipe = Pipeline(

#     steps = [

#         ( 'scaler', StandardScaler() ),

#         ( 'stack', StackingCVRegressor(

#                         regressors=[ lg_pipe, rf_pipe, xgb_pipe ], 

#                         meta_regressor=lg_pipe,

#                         cv=10,

#                         shuffle=True,

#                         random_state=SEED, 

#                         verbose=0,

#                         refit=True,

#                         use_features_in_secondary=False,

#                         store_train_meta_features=True,

#                         n_jobs=-1,

#                         pre_dispatch='2*n_jobs'

#         ) )

#     ]

# )

# STACK_result = test_models ( {'STACK': stack_pipe}, filled_data, 'SalePrice' )

# print (STACK_result)

# Not an improvement over lg_pipe...
'''STEP 5: Prepare Submission.'''

_ = lg_pipe.fit(

            X = filled_data.drop([ 'SalePrice' ], axis=1),

            y = filled_data[ 'SalePrice' ]

)



test_data['SalePrice'] = lg_pipe.predict( transform_data(test_data) )



submission = test_data[['Id', 'SalePrice']]

submission.to_csv('submission.csv', index=False)