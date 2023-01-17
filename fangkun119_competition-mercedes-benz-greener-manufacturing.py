# Essentials

import numpy                 as np

import pandas                as pd

import datetime

import random



# Plots

import seaborn               as sns

import matplotlib.pyplot     as plt



# Feature Engineering

from sklearn.random_projection import GaussianRandomProjection

from sklearn.random_projection import SparseRandomProjection

from sklearn.decomposition     import PCA

from sklearn.decomposition     import FastICA

from sklearn.decomposition     import TruncatedSVD



# Models

from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor

from sklearn.kernel_ridge    import KernelRidge

from sklearn.linear_model    import Ridge, RidgeCV, Lasso, LassoCV

from sklearn.linear_model    import ElasticNet, ElasticNetCV

from sklearn.svm             import SVR

from sklearn.tree            import DecisionTreeRegressor

from mlxtend.regressor       import StackingCVRegressor

import lightgbm              as     lgb

from lightgbm                import LGBMRegressor

from xgboost                 import XGBRegressor



# Stats

from scipy.stats             import skew, norm

from scipy.special           import boxcox1p

from scipy.stats             import boxcox_normmax 



# Misc

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics         import r2_score

from sklearn.metrics         import mean_squared_error

from sklearn.preprocessing   import OneHotEncoder

from sklearn.preprocessing   import LabelEncoder

from sklearn.pipeline        import make_pipeline, Pipeline

from sklearn.preprocessing   import scale

from sklearn.preprocessing   import StandardScaler

from sklearn.preprocessing   import RobustScaler

pd.set_option('display.max_columns', None)

# RobustScaler 



# Ignore useless warnings

import warnings

warnings.filterwarnings(action="ignore")

pd.options.display.max_seq_items = 8000

pd.options.display.max_rows      = 8000
testing_set_scores = pd.DataFrame(data=[

            ['predection_output_stack_gen.csv (the final stacked model)', 0.55600],     

            ['prediction_abr_lasso.csv', 0.54510], 

            ['prediction_abr_tree.csv', 0.54816], 

            ['prediction_gbr.csv', 0.54475], 

            ['prediction_gbr_2.csv', 0.54645], 

            ['prediction_lasso.csv', 0.54711], 

            ['prediction_lgm.csv', 0.54998], 

            ['prediction_lightgbm_2.csv', 0.54869],

            ['prediction_lightgbm_goss.csv', 0.55056],

            ['prediction_rf.csv', 0.55127], 

            ['prediction_rf_2.csv', 0.55164], 

            ['prediction_ridge.csv', 0.54308] 

        ],columns=['model','score']).sort_index(axis=0, ascending=False)

plt.barh(testing_set_scores['model'], testing_set_scores['score'])

plt.xlim(0.5405,0.5576)

for score, model in zip(testing_set_scores['score'], testing_set_scores['model']):

    plt.text(x=score, y=model, s=score, ha='left', va='center', fontsize=9)
# Data files

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Load

train = pd.read_csv('/kaggle/input/train.csv', index_col=0)   #Y, Features

test  = pd.read_csv('/kaggle/input/test.csv', index_col=0)    #Features

print("train.shape:", train.shape, "; test.shape:",test.shape, "; columns only in train.csv:", set(train.columns) - set(test.columns))

# Sample rows

train.head(n=2)
(train.y // 5 * 5).value_counts().sort_index().plot.bar()
desc_y_le_170 = train[train.y <= 170].describe()

desc_y_gt_170 = train[train.y  > 170].describe()

desc_joined   = desc_y_le_170.join(desc_y_gt_170, lsuffix='_le_170', rsuffix='_gt_170')

desc_joined[['y_le_170', 'y_gt_170']]
def despine_plot_compare_norm_fit(data_series, var_name, plot_title):

    sns.set_style("white")

    sns.set_color_codes(palette='deep')

    f, ax = plt.subplots(figsize=(8, 4))

    sns.distplot(data_series, fit=norm, color="b"); 

    (mu, sigma) = norm.fit(data_series) 

    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')

    ax.xaxis.grid(False)

    ax.set(ylabel="Frequency")

    ax.set(xlabel=var_name)

    ax.set(title=plot_title)

    sns.despine(trim=True, left=True)

    plt.show()

despine_plot_compare_norm_fit(train.y, "y", "y distribution")
print("Skewness: %f" % train[train.y <= 170]['y'].skew())

print("Kurtosis: %f" % train[train.y <= 170]['y'].kurt())

print("N Count: %d" % train.y.isnull().sum())
train.describe()
category_cols = [col for col in train.columns if col != 'y' and train[col].dtype == 'object']

train[category_cols].head(n=2)
binary_cols = [col for col in train.columns if col != 'y' and train[col].dtype != 'object']

train[binary_cols].head(n=2)
%%time

# Visualising Categorical Columns

def visualize_categorical_columns():

    fig, axs = plt.subplots(ncols=2, nrows=0, figsize=(12, 12))

    plt.subplots_adjust(right=2)

    plt.subplots_adjust(top=2)

    sns.color_palette("husl", 8)

    for i, col in enumerate(list(train[category_cols]), 1):

        plt.subplot(len(list(category_cols)), 1, i)

        sns.violinplot(x=col, y='y', data=train[train.y <= 170])

        plt.xlabel('{}'.format(col), size=15, labelpad=12.5)

        plt.ylabel('y', size=15, labelpad=12.5)

        for j in range(2):

            plt.tick_params(axis='x', labelsize=12)

            plt.tick_params(axis='y', labelsize=12)

        plt.legend(loc='best', prop={'size': 10})

    plt.show()

visualize_categorical_columns()
%%time

# Visualising Binary Columns

def visualize_binary_columns():

    fig, axs = plt.subplots(ncols=2, nrows=0, figsize=(12, 360))

    plt.subplots_adjust(right=2)

    plt.subplots_adjust(top=2)

    sns.color_palette("husl", 8)

    for i, col in enumerate(list(train[binary_cols]), 1):

        plt.subplot(len(list(binary_cols)), 6, i)

        sns.violinplot(x=col, y='y', data=train[train.y <= 170])

        plt.xlabel('{}'.format(col), size=15,labelpad=12.5)

        plt.ylabel('y', size=15, labelpad=12.5)

        for j in range(2):

            plt.tick_params(axis='x', labelsize=12)

            plt.tick_params(axis='y', labelsize=12)

        plt.legend(loc='best', prop={'size': 10})

    plt.show()

#visualize_binary_columns()
train_labels   = train['y'].reset_index(drop=True)

train_features = train.drop(['y'], axis=1)

test_features  = test



print("train_labels.shape", train_labels.shape)

print("train_features.shape", train_features.shape)

print("test_features.shape", test_features.shape)
def percent_missing(df):

    data    = pd.DataFrame(df)

    df_cols = list(pd.DataFrame(data))

    dict_x  = {}

    for i in range(0, len(df_cols)):

        dict_x.update({df_cols[i]: round(data[df_cols[i]].isnull().mean()*100,2)})

    return dict_x



train_features_missing = percent_missing(train_features)

print('Top 5 percent train_features missing', sorted(train_features_missing.items(), key=lambda x: x[1], reverse=True)[:5])

test_features_missing  = percent_missing(test_features)

print('Top 5 percent train_features missing', sorted(test_features_missing.items(), key=lambda x: x[1], reverse=True)[:5])
# categorical feature names and non-categorical (binary) feature names

category_feature_names = [fea for fea in train_features.columns if fea != 'y' and train_features[fea].dtype == 'object']

binary_feature_names   = [fea for fea in train_features.columns if fea != 'y' and train_features[fea].dtype != 'object']
print(category_feature_names)
print(binary_feature_names[:5], "...", binary_feature_names[-5:])
%%time

# value_counts of each categorical feature

def plot_categorical_feature_coverage():

    fig, axs = plt.subplots(ncols=2, nrows=0, figsize=(12, 5))

    plt.subplots_adjust(right=2)

    plt.subplots_adjust(top=2)

    sns.color_palette("husl", 8)

    for i, feature_name in enumerate(list(category_feature_names), 1):

        plt.subplot(len(list(category_feature_names)), 1, i)

        sns.countplot(train_features[feature_name])

        plt.xlabel('{}'.format(feature_name), size=15, labelpad=12.5)

        plt.ylabel('y', size=15, labelpad=12.5)

        for j in range(2):

            plt.tick_params(axis='x', labelsize=12)

            plt.tick_params(axis='y', labelsize=12)

        plt.legend(loc='best', prop={'size': 10})

    plt.show()

plot_categorical_feature_coverage()
%%time

# value_counts of each binary feature

def plot_binary_features_coverage():

    bin_fea_value_count = pd.DataFrame(train_features[binary_feature_names[0]].value_counts().sort_index())

    for feature_name in binary_feature_names[1:]:

        col_to_append = train_features[feature_name].value_counts().sort_index()

        bin_fea_value_count = bin_fea_value_count.join(col_to_append, rsuffix=feature_name, how='outer')



    bin_fea_value_count = bin_fea_value_count.fillna(0)

    print(bin_fea_value_count.shape)

    plt.figure(figsize=(15,80))

    sns.barplot(y = bin_fea_value_count.columns, x = (bin_fea_value_count.loc[1] / bin_fea_value_count.sum()))

    plt.show()

# plot_binary_features_coverage()
def encoding_features_with_expend(train_features, test_features, n_comp=12):

    # prepare

    print('input: train shape={}; test shape={}'.format(train_features.shape, test_features.shape)) 

    category_feature_names = [fea for fea in train_features.columns if train_features[fea].dtype == 'object']

    # encoding for categorical features

    oh_encoder = OneHotEncoder(handle_unknown='ignore')

    oh_encoder.fit(list(train_features[category_feature_names].values) + list(test_features[category_feature_names].values))

    # categorical feature to be one-hot-coded

    train_cat_fea_oh_ndarray = oh_encoder.transform(train_features[category_feature_names]).toarray()

    test_cat_fea_oh_ndarray  = oh_encoder.transform(test_features[category_feature_names]).toarray()

    train_cat_fea_oh = pd.DataFrame(data=train_cat_fea_oh_ndarray)

    test_cat_fea_oh  = pd.DataFrame(data=test_cat_fea_oh_ndarray)

    print("train_cat_fea_oh.shape:", train_cat_fea_oh.shape)

    train_cat_fea_oh.index = train_features.index

    test_cat_fea_oh.index  = test_features.index

    # non-categorical features: 

    train_non_cat_fea      = train_features.drop(category_feature_names, axis=1)

    test_non_cat_fea       = test_features.drop(category_feature_names, axis=1) 

    # merge them all

    train_fea_oh = pd.concat([train_cat_fea_oh, train_non_cat_fea], axis=1)

    test_fea_oh  = pd.concat([test_cat_fea_oh, test_non_cat_fea], axis=1)  

    # return

    print('after one-hot: train shape={}; test shape={}'.format(train_fea_oh.shape, test_fea_oh.shape)) 

    return train_fea_oh, test_fea_oh



train_features_oh, test_features_oh = encoding_features_with_expend(train_features, test_features)
train_features, test_features = train_features_oh, test_features_oh 
# model training parameters

BASE_MODEL_KF  = 7

STACK_MODEL_KF = 7



# "True == is_verify_code" to check code before running on the full data

is_verify_code = False 



# re-split the train and test features

X = train_features  

y = train_labels

X_test = test_features

if is_verify_code:

    X = X.head(n=BASE_MODEL_KF * STACK_MODEL_KF)

    y = y.head(n=BASE_MODEL_KF * STACK_MODEL_KF)

    X_test = X_test.head(n=BASE_MODEL_KF * STACK_MODEL_KF)



# shape

X.shape, y.shape, X_test.shape
from sklearn.model_selection import GridSearchCV



kf = KFold(n_splits=BASE_MODEL_KF, random_state=37, shuffle=True)



def cv_rmse(model, X=X, y=y):

    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kf))

    return (rmse)



def cv_r2(model, X=X, y=y, print_params=True):

    r2 = cross_val_score(model, X, y, scoring="r2", cv=kf)

    print("---"*20)

    if print_params: 

        print("cv_r2: mean=", r2.mean(), "; std=", r2.std())

        print("--- model params: -------\n")

        print(model)

    else:

        print("cv_r2: ", r2.mean(), ";", r2.std())

    print("---"*20)

    return(r2)



def plot_grid_cv_results(grid_with_train_score_returned):

    grid       = grid_with_train_score_returned

    cv_results = grid.cv_results_

    rst = pd.DataFrame(cv_results)[['mean_test_score', 'mean_train_score', 'std_test_score', 'std_train_score']]

    rst[['mean_test_score', 'mean_train_score']].plot.bar(figsize=(50,5), grid=True)

    rst[['std_test_score', 'std_train_score']].plot.bar(figsize=(50,5), grid=True)

    print('---'*20)

    print(rst)

    print('---'*20)

    for i in range(0, len(cv_results['params'])): 

        print(i, "\t:", cv_results['params'][i])

    print('---'*20)

    print('best_index:', grid.best_index_ )

    print('best_score:', grid.best_score_ )

    print('best_param:', grid.best_params_)

    return rst
%%time

# Grid Search for Lasso Regressor

# lasso = Pipeline([('scaler', RobustScaler()),('lasso',  Lasso(alpha=0.024, normalize=False, random_state=42))])

lasso = Pipeline([('lasso',  Lasso(alpha=0.024, normalize=False, random_state=42))])



def grid_search_lasso(model = lasso):

    params = {

        #'lasso__alpha':[1e-4,1e-3,0.01,0.015,0.016,0.017,0.018,0.019,0.02,0.021,0.022,0.023,0.024,0.025,0.026,0.028,0.03,0.05,0.1,1,10]

        'lasso__alpha':[0.005, 0.015, 0.02, 0.021, 0.022, 0.023, 0.024, 0.025, 0.026, 0.028, 0.029, 0.03, 0.031, 0.05, 0.1]

    }

    grid = GridSearchCV(model, params, cv=kf, scoring="r2", return_train_score=True)

    grid.fit(X, y)

    plot_grid_cv_results(grid)

    return grid.best_estimator_



lasso = grid_search_lasso()

cv_r2(lasso, X=X, y=y, print_params=False)
%%time

# Grid Search for Ridge Regressor

# ridge = Pipeline([('scaler',RobustScaler()), ('ridge', Ridge(alpha=45, normalize=False, random_state=42))])

ridge = Pipeline([('ridge', Ridge(alpha=45, normalize=False, random_state=42))])



#normalize_alpha_list = list(range(10,25,3)) + list(range(25,35,2)) + \

#                       list(range(35,45,1)) + list(range(45,55,2)) + list(range(55,70,3))

normalize_alpha_list = list(range(25,35,2)) + list(range(35,45,1)) + list(range(45,55,2))



def grid_search_ridge(model=ridge):

    #no_exp_fea: best_cv_score=0.56218; alpha=42; noralize=False)

    params = [

        {'ridge__alpha':normalize_alpha_list, 'ridge__normalize':[False]}

     # ,{'ridge__alpha':[1e-2, 0.1, 0.2, 0.39, 0.4, 0.41, 0.5, 0.6, 1, 5, 10], 'ridge__normalize' : [True]}

    ]

    grid = GridSearchCV(model, params, cv=kf, scoring="r2", return_train_score=True)

    grid.fit(X, y)

    plot_grid_cv_results(grid)

    return grid.best_estimator_



ridge = grid_search_ridge(ridge)

cv_r2(ridge, X=X, y=y, print_params=False)
%%time

# Random Forest Regressor

rf = RandomForestRegressor(

                    max_depth         = 4,

                    n_estimators      = 1000,

                    min_impurity_decrease = 0.175,   

                    criterion         = "mse",

                    min_samples_split = 2,

                    min_samples_leaf  = 1,

                    min_weight_fraction_leaf = 0.02, 

                    max_features      = 0.88,

                    oob_score         = True,

                    n_jobs            = -1,

                    warm_start        = False,

                    random_state      = 42)



# Grid Search for Random Forest: 0.57, no overfitting in cv

def grid_search_random_forest(model = rf):       # 0.57, 0.571

    params = {"max_depth": [4],                  # 3,    4

              "max_features": [0.88],            # 1,    0.88

              "min_weight_fraction_leaf":[0.02], # 0.02, 0.02  #0.02 is better than 0.01, obviousely better than 0.03

              "min_impurity_decrease":[0.175, 0.2, 0.25], # 0.02, 0.05  #larger might be better

              "n_estimators":[1000]

             } 

    grid = GridSearchCV(model, params, scoring="r2", return_train_score=True, cv=kf)

    grid.fit(X, y)

    plot_grid_cv_results(grid)  

    return grid.best_estimator_



rf = grid_search_random_forest()

#cv_r2(rf, X=X, y=y, print_params=False)
%%time

# Random Forest Regressor

rf_2 = RandomForestRegressor(

                    n_estimators      = 900,

                    max_features      = None, 

                    max_depth         = 5,

                    min_samples_split = 5,

                    min_samples_leaf  = 5,

                    oob_score         = True,

                    random_state = 42)



def grid_search_random_forest_2(model = rf_2):

    params = {"n_estimators":[900],

              "max_depth":[4,5,6],

              "min_samples_split":[5],

              "min_samples_leaf":[5]

             } 

    grid = GridSearchCV(model, params, scoring="r2", return_train_score=True, cv=kf)

    grid.fit(X, y)

    plot_grid_cv_results(grid)  

    return grid.best_estimator_



rf_2 = grid_search_random_forest_2()

#cv_r2(rf_2, X=X, y=y, print_params=False)
%%time

# Grid Search for GradientBoostingRegressor

gbr = GradientBoostingRegressor(                 # score: 0.560 -> ... -> 0.571

                    n_estimators      = 1000, 

                    learning_rate     = 0.006,   # better than 0.005, 0.01

                    max_features      = 1.0,    

                    min_impurity_decrease = 0.5, # 0.1 -> ... -> 0.5 better than 0.6

                    subsample         = 1,       # over fitting for 0.66, 0.88,

                    max_depth         = 3,       # better than 4

                    max_leaf_nodes    = None, 

                    min_samples_leaf  = 1,

                    min_samples_split = 2, 

                    loss              = 'huber', #["ls", "lad", "huber", "quantile"]

                    random_state      = 42)



def grid_search_gradient_boosting_regressor(model = gbr):                       

    params = [{                            

        "max_depth":[None], "max_leaf_nodes":[6] # test=0.570521, train=0.581021

               

    }, {

        "max_depth":[3], "max_leaf_nodes":[None] # test=0.567940, train=0.600836

    }]

    grid = GridSearchCV(model, params, scoring="r2", return_train_score=True, cv=kf)

    grid.fit(X, y)

    plot_grid_cv_results(grid) 

    return grid.best_estimator_



gbr = grid_search_gradient_boosting_regressor()

#cv_r2(gbr, X=X, y=y, print_params=False)
%%time

# Grid Search for GradientBoostingRegressor

gbr_2 = GradientBoostingRegressor(

                    n_estimators      = 500,   

                    learning_rate     = 0.01,   

                    max_depth         = 4,

                    max_features      = 0.6,

                    min_samples_leaf  = 4,

                    min_samples_split = 40, 

                    loss              = 'huber',

                    random_state      = 42)



def grid_search_gradient_boosting_regressor_2(model = gbr_2): 

    params = {                                 

        "n_estimators":[500],        # 3000  -> 2000  -> 500

        "learning_rate":[0.01],

        "max_features":[0.8, 0.85],  # 0.8 is bettern than 0.6, 1.0        

        "min_samples_split":[110, 120, 130],  # 15    -> 20    -> 40  -> 50 -> 70        

        "min_samples_leaf":[4],      # 5     -> 5 .   -> 4

    }

    grid = GridSearchCV(model, params, scoring="r2", return_train_score=True, cv=kf)

    grid.fit(X, y)

    plot_grid_cv_results(grid) 

    return grid.best_estimator_



gbr_2 = grid_search_gradient_boosting_regressor_2()

#cv_r2(gbr_2, X=X, y=y, print_params=False)
%%time

# Light Gradient Boosting Regressor

# https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html#lightgbm.LGBMRegressor

# https://lightgbm.readthedocs.io/en/latest/Parameters.html

lightgbm = LGBMRegressor(

    boosting_type = 'gbdt',           # 'gbdt'(default); ‘dart’; ‘goss’; ‘rf’

    objective     = 'regression',     # 'regression', 'binary', 'multiclass', 'lambdarank'

    num_leaves    = 31,               # default 31, max tree leaves number

    max_depth     = 3,                # default -1(no limit), tree depth limit

    learning_rate = 0.002,         

    n_estimators  = 1000,

    min_split_gain = 0.1,

    max_bin       = 40,               # default 255?, number of samples to construct bins

    reg_alpha     = 0,                # default 0, L1 regularization term on weight

    reg_lambda    = 0,                # default 0, L2 regularization term on weight

    bagging_freq  = 4,                # default 0 (no bagging), perform bagging every k iteration 

    bagging_fraction = 1.0,           # 

    feature_fraction = 1.0,           # 

    min_sum_hessian_in_leaf = 0.001,  # default 1e-3, minimal sum hessian in one leaf. 

                                      # like min_data_in_leaf, it can be used to deal with over-fitting

    n_jobs=-1, verbose = -1, random_state = 42)



#learning_rates  = [0.0015, 0.002, 0.0025] #, 0.0026, 0.0027, 0.0028]   # 0.003 is slightly overfiting, 0.005 is over fitting

#hessian_in_leaf = [0.001, 10, 30]                    # list(np.array(range(0,65,5)) / 10) #print(hessian_in_leaf)



def grid_search_light_gbm(model = lightgbm):          ## learning_rate=0.003 is slightly overfiting, 0.005 is over fitting                                

    params = {                                        # 0.561

        "n_estimators": [1000],                       # 1000 

        "learning_rate": [0.002, 0.00225, 0.0025, 0.00275, 0.003], # 0.002 (better then 0.0018, 0.0016, 0.001)

        "max_bin": [40],                              # 10(10 is best, but not too much effect)

        "max_depth" : [3]                             # 3(no overfitting), 4(overfitting but has higher cv-score)

        #"min_sum_hessian_in_leaf" : hessian_in_leaf, # no difference between [0, 6]

        #"reg_alpha": [0, 64]                         # 0     -> 0      -> 0      -> 0       -> 0     -> 64

    }

    grid = GridSearchCV(model, params, scoring="r2", return_train_score=True, cv=kf)

    grid.fit(X, y)

    plot_grid_cv_results(grid)

    return grid.best_estimator_



lightgbm = grid_search_light_gbm()

#cv_r2(lightgbm, X=X, y=y, print_params=False)
%%time

# Light Gradient Boosting Regressor

# https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html#lightgbm.LGBMRegressor

# https://lightgbm.readthedocs.io/en/latest/Parameters.html

lightgbm_2 = LGBMRegressor(    

    boosting_type = 'gbdt',       # 'gbdt'(default); ‘dart’; ‘goss’; ‘rf’

    objective     = 'regression', # 'regression', 'binary', 'multiclass', 'lambdarank'

    num_leaves    = 60,           # default 31, max tree leaves number

    max_depth     = -1,           # default -1(no limit), tree depth limit

    learning_rate = 0.01,         

    n_estimators  = 350,          # 800 -> 350

    max_bin       = 90,           # default 255

    bagging_fraction = 0.35,      # randomly select part of data without resampling: 0.35 better than 0.3 and 0.45 in CV

    reg_alpha     = 0,            # default 0, L1 regularization term on weight

    reg_lambda    = 0,            # default 0, L2 regularization term on weight

    bagging_freq  = 4,            # default 0 (no bagging), perform bagging every k iteration 

    bagging_seed  = 8,            # default 3, random seed for bagging

    feature_fraction = 0.8,       # randomly select part of features on each iteration

    feature_fraction_seed   = 8,  # default 2, random seed for feature_fraction

    #min_sum_hessian_in_leaf= 11, # default 1e-3, minimal sum hessian in one leaf. 

                                  # like min_data_in_leaf, it can be used to deal with over-fitting

    verbose = -1, random_state = 42)



def grid_search_light_gbm_2(model = lightgbm_2):                               

    params = [{ 

        "num_leaves": [None], "max_depth":[3], 

        "feature_fraction": [0.8, 0.84, 0.88, 0.92]    #0.5 -> 0.8

    }, {

        "num_leaves": [6], "max_depth":[None], 

        "feature_fraction": [0.8, 0.84, 0.88, 0.92]    #0.5 -> 0.8

    }]

    grid = GridSearchCV(model, params, scoring="r2", return_train_score=True, cv=kf)

    grid.fit(X, y)

    plot_grid_cv_results(grid)

    return grid.best_estimator_



lightgbm_2 = grid_search_light_gbm_2()

#cv_r2(lightgbm_2, X=X, y=y, print_params=False)
%%time

# Grid Serch for Adaboost Regressor with Decision Tree

# https://www.programcreek.com/python/example/86712/sklearn.ensemble.AdaBoostRegressor

abr_tree = AdaBoostRegressor(

                    base_estimator    = DecisionTreeRegressor(

                                            max_depth         = 4,

                                            min_samples_split = 2, 

                                            min_samples_leaf  = 1, 

                                            min_weight_fraction_leaf = 0.0, 

                                            min_impurity_decrease = 0.4, 

                                            random_state      = 53),  

                    n_estimators      = 1000,

                    learning_rate     = 0.001,

                    loss              = "exponential",          #'linear', 'square', 'exponential'

                    random_state      = 42)



def grid_search_ada_boost_tree_regressor(model = abr_tree):                            

    params = {                                              # 0.568 -> 0.569 no overfitting if learning rate is low

        "base_estimator__max_depth": [4],                   # 3      # 9,5: over fitting; 3: under fitting 

        "base_estimator__min_impurity_decrease":[0.55,0.6], # 0.3, 0.4, 0.5, 0.6: the lower the better score but severe over-fitting

        "base_estimator__min_weight_fraction_leaf":[0.0],   # 0.0    # not try other values yet

        "learning_rate" : [0.001],                          # 0.001  # better than 0.02,0.03

        "loss" : ['exponential']                            # expon  # 

    }

    #8 	: {'base_estimator__max_depth': 3, 'base_estimator__min_impurity_decrease': 0.5, 'base_estimator__min_weight_fraction_leaf': 0.0, 'learning_rate': 0.001, 'loss': 'exponential'}

    grid = GridSearchCV(model, params, scoring="r2", return_train_score=True, cv=kf)

    grid.fit(X, y)

    plot_grid_cv_results(grid)

    return grid.best_estimator_



abr_tree = grid_search_ada_boost_tree_regressor()

#cv_r2(abr_tree, X=X, y=y, print_params=False)
%%time

lightgbm_goss = LGBMRegressor(    

    boosting_type = 'goss',           # 'gbdt'(default); ‘dart’; ‘goss’; ‘rf’

    objective     = 'regression',     # 'regression', 'binary', 'multiclass', 'lambdarank'

    num_leaves    = 31,               # default 31, max tree leaves number

    max_depth     = 3,               # default -1(no limit), tree depth limit

    learning_rate = 0.005,         

    n_estimators  = 1000,

    min_split_gain = 0.1,

    max_bin       = 40,               # ? default 255, number of samples to construct bins

    reg_alpha     = 0,                # default 0, L1 regularization term on weight

    reg_lambda    = 0,                # default 0, L2 regularization term on weight

    bagging_freq  = 4,                # default 0 (no bagging), perform bagging every k iteration 

    bagging_fraction = 1.0,           # 

    feature_fraction = 1.0,           # 

    min_sum_hessian_in_leaf = 0.001,  # default 1e-3, minimal sum hessian in one leaf. 

                                      # like min_data_in_leaf, it can be used to deal with over-fitting

    n_jobs=-1, verbose = -1, random_state = 42

)



def grid_search_light_gbm_goss(model = lightgbm_goss):                          

    params = {

        "boosting_type":["goss"],

        "n_estimators": [1000],

        "learning_rate": [0.002],

        "max_bin": [80],

        "max_depth" : [3],

        "num_leaves" : [6,7]

    }

    grid = GridSearchCV(model, params, scoring="r2", return_train_score=True, cv=kf)

    grid.fit(X, y)

    plot_grid_cv_results(grid)

    return grid.best_estimator_

lightgbm_goss = grid_search_light_gbm_goss()
%%time

# Grid Serch for Adaboost Regressor

# https://www.programcreek.com/python/example/86712/sklearn.ensemble.AdaBoostRegressor

alpha_list = [0.02, 0.025, 0.03, 0.05, 0.1, 0.5, 1, 1.5, 3, 6, 10, 13, 16, 20, 23, 26, 30, 33, 36, 40, 43, 46, 50]

abr_lasso = AdaBoostRegressor(

                    base_estimator    = LassoCV(alphas=alpha_list, normalize=False, random_state=42), 

                    n_estimators      = 24,

                    learning_rate     = 0.0005,

                    loss              = "exponential",   #'linear', 'square', 'exponential'

                    random_state      = 42)



def grid_search_abr_lasso(model = abr_lasso):                  # overfitting, 0.570581:0.590737

    params = {                              # 

        "n_estimators" : [18,12],           # 

        "learning_rate" : [0.0005, 0.001],  # 0.005 -> 0.001 -> 0.0005 (not to )

        #"loss" : ['exponential','square']  # no obviously difference 

    }

    grid = GridSearchCV(model, params, scoring="r2", return_train_score=True, cv=kf)

    grid.fit(X, y)

    plot_grid_cv_results(grid)

    return grid.best_estimator_



abr_lasso = grid_search_abr_lasso(abr_lasso)

#cv_r2(abr, X=X, y=y, print_params=False)
%%time

# StackingCVRegressor: 

# http://rasbt.github.io/mlxtend/user_guide/classifier/StackingCVClassifier/

alphas_list_1 = [0.001, 0.003, 0.005, 0.007, 0.009]

alphas_list_2 = [0.1,   0.3,   0.5,   0.7,   0.9]

alphas_list_3 = [1,     3,     5,     7,     9]

alphas_list_4 = [10,    13,    15,    17,    19,  21,    23,    25,    27,    29]

alphas_list_5 = [31,    33,    35,    37,    39,  41,    43,    45,    47,    49]

alphas_list_6 = [51,    53,    55,    57,    59,  61,    63,    65,    67,    70]

alphas_list_7 = [75,    80,    85,    90,    95,  100,   200,   400,   800,   1600]

kf_stack_gen = KFold(n_splits=STACK_MODEL_KF, random_state=37, shuffle=True)

stack_gen    = StackingCVRegressor(

                    regressors = (gbr, rf, abr_tree, lightgbm, lasso, lightgbm_goss, rf_2, gbr_2, ridge, lightgbm_2, abr_lasso),

                    meta_regressor = make_pipeline(RobustScaler(), RidgeCV(scoring='r2', alphas=alphas_list_4)), 

                    cv = kf_stack_gen,

                    n_jobs = 8, 

                    use_features_in_secondary=False)



def grid_search_stack_gen(meta_model_list, use_2nd_fea_list=[False], model=stack_gen):

    params = [{'meta_regressor': meta_model_list, 'use_features_in_secondary': use_2nd_fea_list}]

    grid_search = GridSearchCV(model, params, cv=3, scoring='r2', return_train_score=True) 

    grid_search.fit(X, y)

    print("grid_search.best_params_: ", grid_search.best_params_)

    print("grid_search.best_estimator_: ", grid_search.best_estimator_)

    print("means test scores:")

    cvres = grid_search.cv_results_

    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):

        print(mean_score, params)

    print("means training scores:")

    for mean_score, params in zip(cvres["mean_train_score"], cvres["params"]):

        print(mean_score, params)

    print("details: -------------------------------------")

    print(cvres)

    return grid_search.best_estimator_



#stack_gen = grid_search_stack_gen([

#                make_pipeline(RobustScaler(),RidgeCV(alphas=alphas_list_4, scoring='r2'))

#            ])
%%time

print('fit stack_gen with full training data')

stack_gen_model = stack_gen.fit(X,y)



# Read in sample_submission dataframe

submission = pd.read_csv("/kaggle/input/sample_submission.csv")

print("submission.shape:", submission.shape, "; submission.columns: ", submission.columns)

print("X_test.shape:", X_test.shape)



submission.iloc[:,1] = stack_gen_model.predict(X_test)

submission.to_csv("predection_output_stack_gen.csv", index=False)
%%time

base_models_fitted = {}



def pred_testset_with_base_model():

    base_model_list = [('lasso', lasso), ('ridge', ridge), ('rf', rf), ('gbr',gbr), 

                       ('lgm', lightgbm), ('abr_tree', abr_tree), ('abr_lasso', abr_lasso),

                       ('lightgbm_goss', lightgbm_goss), ('rf_2', rf_2), ('gbr_2', gbr_2), 

                       ('lightgbm_2', lightgbm_2)]

    for name, model in base_model_list:

        file_name            = "prediction_" + name + ".csv"

        trained_model        = model.fit(X,y)

        submission.iloc[:,1] = trained_model.predict(X_test)

        submission.to_csv(file_name, index=False)

        

        base_models_fitted[name] = trained_model

        print(file_name)

pred_testset_with_base_model()