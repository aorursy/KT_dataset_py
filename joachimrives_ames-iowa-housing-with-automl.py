# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# %load ../input/ames-iowa-house-prices-handmade-data/Ames Iowa Housing Kaggle Edition/standard_pipeline.py

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder

import numpy as np
import matplotlib.pyplot as plt

scope_names = dir()

def make_name_from_estimator(obj_estimator):
    
    name = str(obj_estimator).split('(')
    name = name[0].strip('(')
    return name

def return_repeatedstratifiedkfold(splits = 5, repeats = 5, rand_state = 88):        
    from sklearn.model_selection import RepeatedStratifiedKFold
    
    cv_splitter = RepeatedStratifiedKFold(
        n_splits = splits,
        n_repeats = repeats,
        random_state = rand_state
        )
    
    return cv_splitter

def run_rfecv(df, lst_X, y, estimator, scoring_method, cv_splitter = None):
    
    if "RFECV" not in scope_names:
        from sklearn.feature_selection import RFECV    
    
    if cv_splitter == None:
        cv_splitter = return_repeatedstratifiedkfold()

    rfecv = RFECV(
        estimator = estimator,
        cv = cv_splitter,
        scoring = scoring_method,
        n_jobs = -1
        )
    
    rfecv.fit(X = df[lst_X], y = df[y])
    
    results = {
        "features": df[lst_X].columns[rfecv.support_],
        "split_scores": rfecv.grid_scores_,
        "fit_estimator": rfecv.estimator_,
        "rankings": rfecv.ranking_
        }
    
    return results

def run_kbest(feature_data, target_values, n_passers = "all", score_func = None):
#     if (score_func == None) and ("mutual_info_regression" not in scope_names):
#         from skelarn.feature_selection import mutual_info_regression
#         score_func = mutual_info_regression
    
#     if "SelectKBest" not in scope_names:
#         from sklearn.feature_selection import SelectKBest

    from sklearn.feature_selection import SelectKBest
    
    kbest = SelectKBest(score_func = score_func, k = n_passers)
    kbest.fit(X = feature_data, y = target_values)
        
    results = {
        "kbest_scores": kbest.scores_,
        "kbest_pvalues": kbest.pvalues_,
        "kbest_params": kbest.get_params(),
        #"passing_features": kbest.get_support()
        }
    
    return results

def manyset_manymethod_kbest(
    dict_datasets,
    dict_targets,
    lst_score_funcs,
    y = "is_benign",    
    n_passers = "all"    
    ):
    
    result_dicts = {}
    
    for alias, dataset in dict_datasets.items():
        set_results = {}
        
        for method in lst_score_funcs:
            try:
                rslt = run_kbest(
                    feature_data = dataset,
                    target_values = dict_targets[alias],                    
                    n_passers = n_passers,
                    score_func = method
                )
                
                set_results[ str(method).split(' ')[1] ] = rslt
            except Exception as e:
                set_results[ str(method).split(' ')[1] ] = {"Error":str(e)}
        
        result_dicts[alias] = set_results
    
    return result_dicts

def run_randsearch(estimator, X_data, y_data, dict_params,
                   scoring_method = None,
                   n_combinations = 128,
                   cv_splitter = None,
                   get_trainset_scores = False
                   ):
    
    if cv_splitter == None:
        cv_splitter = return_repeatedstratifiedkfold()
    
    randomized_search = RandomizedSearchCV(
        estimator = estimator,
        param_distributions = dict_params,
        n_iter = n_combinations,
        scoring = scoring_method,
        n_jobs = -1,
        cv = cv_splitter,
        return_train_score = get_trainset_scores
        )        
    
    randomized_search.fit(X = X_data, y = y_data)
    dict_cv_results = randomized_search.cv_results_
    
    results = {
        "cv_results": dict_cv_results,
        "cv_results_best": dict_cv_results['params'][randomized_search.best_index_],
        "best_params": randomized_search.best_params_,
        "best_score": randomized_search.best_score_,
        "best_estimator": randomized_search.best_estimator_,
        "refit_time": randomized_search.refit_time_
        }
    
    return results

def manymodel_manyfeatureset_randsearch(
        dict_estimators_params,
        dict_feature_sets,
        dict_target_features,
        scoring_method = None,
        n_combinations = 128,
        cv_splitter = None,
        get_trainset_scores = False        
        ):
    
    model_results = {}
    
    for estimator, param_grid in dict_estimators_params.items():
        name = make_name_from_estimator(estimator)        
        featureset_results = {}
        
        for alias, feature_data in dict_feature_sets.items():
            featureset_results[alias] = run_randsearch(
                estimator = estimator,
                X_data = feature_data,
                y_data = dict_target_features[alias],
                dict_params = param_grid)
        
        model_results[name] = featureset_results
    
    return model_results

def run_gridsearch(
        estimator, X_data, y_data, dict_params,
        scoring_method = None,
        cv_splitter = None,
        get_trainset_scores = False
        ):
    if cv_splitter == None:
        cv_splitter = return_repeatedstratifiedkfold()

    grid_search = GridSearchCV(
        estimator = estimator,
        param_grid = dict_params,
        scoring = scoring_method,
        n_jobs = -1,
        cv = cv_splitter,
        return_train_score = get_trainset_scores
        )
    
    grid_search.fit(X = X_data, y = y_data)
    cv_results = grid_search.cv_results_
    
    results = {
        "best_params": grid_search.best_params_,
        "best_score": grid_search.best_score_,
        "best_estimator": grid_search.best_estimator_,
        "cv_results": cv_results,
        "best_of_cv_results": cv_results['params'][grid_search.best_index_]
        }    
    
    return results

def manymodel_manyfeatureset_hparam_gridsearch(
    dict_estimators_params,
    dict_X_combinations,
    dict_y_data,
    scoring_method,
    cv_splitter = return_repeatedstratifiedkfold(),
    get_trainset_scores = False):
    
    dict_manymodel_gridsearch = {}
    
    for model, hparam_grid in dict_estimators_params.items():
        name = make_name_from_estimator(model)
        rslt = {}
        
        for alias, feature_combination in dict_X_combinations.items():
            rslt[alias] = run_gridsearch(
                estimator = model,
                dict_params = hparam_grid,                
                y_data = dict_y_data[alias],
                X_data = feature_combination,
                scoring_method = scoring_method,
                cv_splitter = cv_splitter,
                get_trainset_scores = get_trainset_scores
            )            
            
        dict_manymodel_gridsearch[name] = rslt
        
    return dict_manymodel_gridsearch
    
def run_crossvalscore(
        estimator, train_data, target_feature,
        scoring_method = None,
        cv_splitter = None
        ):
    """    

    Parameters
    ----------
    estimator : MODEL OR ESTIMATOR
        DESCRIPTION.
    train_data : TYPE
        DESCRIPTION.
    target_feature : TYPE
        DESCRIPTION.
    scoring_method : TYPE, optional
        DESCRIPTION. The default is None.
    cv_splitter : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    results : TYPE
        DESCRIPTION.

    """
    
    if (cv_splitter == None) and ("RepeatedStratifiedKFold" not in scope_names):
        cv_splitter = return_repeatedstratifiedkfold()
    
    cvscores = cross_val_score(
        estimator = estimator,
        X = train_data,
        y = target_feature,
        scoring = scoring_method,
        cv = cv_splitter,
        n_jobs = -1
        )    
    
    mean_of_scores_withnan = np.mean(cvscores)
    nanmean_of_scores = np.nanmean(cvscores)
    standard_deviation = np.std(cvscores)
    variance = np.var(cvscores)    
    
    results = {
        "mean_score": mean_of_scores_withnan,
        "nanmean_score": nanmean_of_scores,
        "std": standard_deviation,
        "var": variance
        }    
    
    return results

def manymodel_manyfeatureset_cvs(
        lst_estimators,
        dict_featuresets,        
        scoring_method,
        cv_splitter = return_repeatedstratifiedkfold(),
        ):
    """
    Parameters
    ----------
    lst_stimators: List-like of Estimator Objects/Models
        A list with properly instantiated estimators or algorithm models, ex:
            MLPRegressor(hidden_layer_sizes = (64, 64)).
    
    dict_featuresets: Dictionary
        A dictionary. The keys are arbitrary names/aliases used to identify the
        different sets of training data. The values are the corresponding list with
        feature data to be passed "as-is" to an estimator placed on index 0. The
        target data will be placed on index 1. THe input may be dataframes,
        series, or NumPy n-dimensional arrays. Format:
            { "dataset_name": [features, target_values] }
    
    scoring_method: String
        A string giving the scoring metric or criteria, such as root mean squared error
        or accuracy score.
    
    cv_splitter: CV Splitter Object, default RepeatedStratifiedKFold(n_repeats = 5, n_splits = 5)
        An instance of an SKLearn cross-validation splitter.
    
    Returns
    -------
    dict_results: Dictionary
        A dictionary whose keys are the estimator names with parenthesis and parameters removed.
        The values are also dictionaries.
        The "second layer" dictionaries corresponding to the estimator names
        have the feature set aliases as keys and
        dictionaries with summary statistics about the cross-validation scores.
        Sample Structure:
        {
            "LinearRegression":
                {
                    "df_raw":
                        {"mean_score":NaN, "nanmean_score":0.53, "std":3, "var":9},
                    "df_processed":
                        {"mean_score": 0.75, "nanmean_score":0.75, "std":0.5, "var":0.70710}
             },
            "MLPRegression":
                {
                    "df_raw":
                        { "mean_score":NaN, "nanmean_score":0.83, "std":3, "var":9 },
                    "df_processed":
                        {"mean_score": 0.85, "nanmean_score":0.95, "std":0.1, "var":0.31622}
                }    
        }

    """
    
    dict_results = dict()
    
    for estimator in lst_estimators:
        name = make_name_from_estimator(estimator)        
        featureset_rslt = dict()
        
        for alias, dataset in dict_featuresets.items():
            featureset_rslt[alias] = run_crossvalscore(
                estimator = estimator,
                train_data = dataset[0],
                target_feature = dataset[1],
                scoring_method = scoring_method,
                cv_splitter = cv_splitter
                )
        
        dict_results[name] = featureset_rslt
    
    return dict_results


def make_traintest(df, train_fraction = 0.7, random_state_val = 88):
    df = df.copy()
    df_train = df.sample(frac = train_fraction, random_state = random_state_val)    
    bmask_istrain = df.index.isin(df_train.index.values)
    df_test = df.loc[ ~bmask_istrain ]
    
    #return (df_train, df_test)
    
    return {
        "train":df_train,
        "test":df_test
        }

def train_models(
        dict_model_and_dataset
        ):
    """

    Parameters
    ----------
    dict_model_and_dataset : TYPE
        DESCRIPTION: A dictionary with an arbitrary alias/name for the model and data set
        as its keys and the corresponding estimator's data set as a list. The list
        must have the estimator object at index 0,
        features' data at index 1 and the target variables' data at index 2.
        Ex:
            {"MLPRegressor_baseline": [regressor_object,
                                       train[list_of_features],
                                       train["target_feature"]
                                       ]
             }

    Returns
    -------
    fit_models : DICTIONARY
        DESCRIPTION: A dictionary with the estimator objects after their `.fit()`
        method has been called with the proper corresponding data as `fit` method
        parameters.
        In case of failure to train an estimator, the estimator object is set to
        `None`.
        The aliases supplied to this function instance are used as keys.

    """
    
    fit_models = {}
    
    for alias, model_and_data in dict_model_and_dataset.items():
        name = alias
        estimator = model_and_data[0]
        try:
            estimator.fit(X = model_and_data[1], y = model_and_data[2])
        except:
            estimator = None
        finally:
            fit_models[name] = estimator
        
    return fit_models
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RepeatedStratifiedKFold
#!pip install pip --upgrade
!python -m pip install --upgrade pip
!pip install -U tensorflow
import tensorflow as tf
tf.__version__
import pandas as pd, numpy as np, datetime as dt
import seaborn as sbrn, matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

path_to_train = "https://raw.githubusercontent.com/SS-Runen/Kaggle-Competition-Data/master/Ames%20Iowa%20Housing%20Kaggle%20Edition/train.csv"
df_train = pd.read_csv(filepath_or_buffer = path_to_train, index_col = "Id")

path_to_holdout = "https://github.com/SS-Runen/Kaggle-Competition-Data/raw/master/Ames%20Iowa%20Housing%20Kaggle%20Edition/test.csv"
df_holdout = pd.read_csv(filepath_or_buffer = path_to_holdout, index_col = "Id")

## ** Remove Data Leakers

leakers = [
    "MoSold",
    "YrSold",
    "SaleType",
    "SaleCondition",
    "SaleType",
    "SaleCondition"
]
df_train.drop(columns = leakers, inplace = True)

for item in leakers:
    try:
        df_holdout.drop(columns = item, inplace = True)
    except Exception as e:
        print("Error occured processing", item, ':')
        print('\n', e)
        
## ** Change numeric features to their proper data types.
lst_true_integers = [    
    "YearBuilt",
    "YearRemodAdd",    
    "BsmtFullBath",
    "BsmtHalfBath",
    "FullBath",
    "HalfBath",
    "BedroomAbvGr",
    "KitchenAbvGr",
    "TotRmsAbvGrd",
    "Fireplaces",
    "GarageYrBlt",
    "GarageCars",    
]
lst_true_integers.sort()

lst_true_floats = [
    "LotFrontage",
    "LotArea",
    "MasVnrArea",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "1stFlrSF",
    "2ndFlrSF",
    "LowQualFinSF",
    "GrLivArea",
    "GarageArea",
    "WoodDeckSF",
    "OpenPorchSF",
    "PoolArea",
    "EnclosedPorch",
    "3SsnPorch",
    "ScreenPorch",
    "MiscVal",    
]
lst_true_floats.sort()

### *** Make list of features and their data types.

lst_numerics = [n for n in lst_true_integers]

for item in lst_true_floats:
    lst_numerics.append(item)
    #lst_numeric_dtypes.append(item)

for feature in lst_true_integers:
    if feature not in lst_numerics:
        print("Integer Feature missing from Numerics List:", feature)
else:
    print("All integer features added.")
        
for feature in lst_true_floats:
    if feature not in lst_numerics:
        print("Float Feature missing from Numerics List:", feature)
else:
    print("All floating numeral features added.")

## ** Situation-specific Cleaning
def make_age_from_year(integer_year):
    return int(dt.datetime.now().year) - integer_year

lst_year_columns = ["GarageYrBlt", "YearBuilt", "YearRemodAdd"]
for feature in lst_year_columns:
    df_train[feature] = df_train[feature].apply(func = make_age_from_year)
    df_holdout[feature] = df_holdout[feature].apply(func = make_age_from_year)
    
lst_categoricals = []
for feature in df_train.columns.values:
    if (feature not in lst_numerics) and (str(feature) != "SalePrice"):
        lst_categoricals.append(feature)        

df_train["SalePrice"] = df_train["SalePrice"].astype(dtype = np.float64)
lst_categoricals.sort()

## ** Data Visualization
### *** Rough View: Distributions

lst_floats_and_target = [ n for n in lst_true_floats ]
lst_floats_and_target.append("SalePrice")
lst_categoricals_and_target = [n for n in lst_categoricals]
lst_categoricals_and_target.append("SalePrice")

df_corr_floats = df_train[ lst_floats_and_target ].corr()
df_train[lst_floats_and_target].hist(bins = 5, xrot = 90, figsize = (12,12), layout = (4,5))
#df_train[lst_true_integers].hist(bins = 8, xrot = 45)
#df_train[lst_categoricals].hist(xrot = 45)
df_train[lst_floats_and_target].plot(kind='density', subplots=True, layout=(4,5), sharex=False, figsize = (12,12) )
df_train[lst_floats_and_target].plot(kind='box', subplots=True, layout=(4,5), sharex=False, sharey=False, figsize = (12,12))
#PyPlot correlation matrix
fig_corrmat = plt.figure()
ax = fig_corrmat.add_subplot(1, 1, 1)
matrix_axis = ax.matshow(df_train[lst_floats_and_target].corr(), vmin=-1, vmax=1)
fig_corrmat.colorbar(matrix_axis)
ticks = np.arange(0, len(df_train[lst_floats_and_target].corr().columns.values), 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(df_train[lst_floats_and_target].corr().columns.values)
ax.set_yticklabels(df_train[lst_floats_and_target].corr().index.values)
plt.xticks(rotation = 90)
plt.show()
#Seaborn bersion of correlation matrix
sbrn.heatmap( df_corr_floats )
help(scatter_matrix)
#Scatter plot with PyPlot
scatter_matrix( df_train[lst_floats_and_target], figsize = (24,24) )
plt.show()
hist_axis = plt.axes()
hist_axis.plot("X", "Y", df_train[["SalePrice", "LotArea"]])
def get_incomplete_features(df, threshold = 0.05):
    row_count = df.shape[0]
    sr_null_counts = df.isnull().sum()
    lst_drop = []
    lst_have_nulls = sr_null_counts.index.values.tolist()
    end_decorator = "\n******"
    for feature in lst_have_nulls:
        threshold = 0.05
        null_percent = (sr_null_counts[feature] / row_count)

        if null_percent > threshold:
            print("To Drop: ", feature, type(df_train[feature]), type(df_train[feature].iloc[0]) )
            lst_drop.append(feature)            
                
    return lst_drop

lst_missing_value_failers = get_incomplete_features(df = df_train, threshold = 0.05)

df_train.drop(columns = lst_missing_value_failers, inplace = True)

# dict_drop_errors = {}
# for feature in lst_missing_value_failers:
#     try:
#         df_holdout.drop(inplace = True, columns = feature)
#     except Exception as e:
#         dict_drop_errors["holdout"] = {feature:e}
    
#     try:
#         df_test.drop(inplace = True, columns = feature)
#     except Exception as e:
#         dict_drop_errors["test"] = {feature:e}

lst_features = df_train.columns.tolist()
lst_features.remove("SalePrice")
df_holdout = df_holdout[ lst_features ]

for feature in lst_missing_value_failers:
    if feature in lst_numerics:
        lst_numerics.remove(feature)
    if feature in lst_categoricals:
        lst_categoricals.remove(feature)
temp = pd.Series(data = df_train.select_dtypes(include = "number").isnull().sum() )
temp[ (temp > 0) ]
df_holdout.shape
df_train.shape
def var_threshold(df, minimum_variance = 0.20):
    dict_variances = {}
    lst_drop = []
    df = df.copy()
    
    for feature in df.columns.values:
        name = feature
        var = df[name].var()        
        dict_variances[name] = var
        
        if var < minimum_variance:
            lst_drop.append(name)
    
    return {"variances":dict_variances, "failers":lst_drop}
def make_traintest(df, train_fraction = 0.7, random_state_val = 88):
    df = df.copy()
    df_train = df.sample(frac = train_fraction, random_state = random_state_val)    
    bmask_istrain = df.index.isin(df_train.index.values)
    df_test = df.loc[ ~bmask_istrain ]
    
    #return (df_train, df_test)
    
    return {
        "train":df_train,
        "test":df_test
        }
split = make_traintest(df_train)
df_housing_clean = df_train.copy()

df_train = split["train"]
df_test = split["test"]

df_train_unimputed, df_test_unimputed, df_holdout_unimputed = df_train.copy(), df_test.copy(), df_holdout.copy()

df_train_modes = df_train.mode()
for feature in df_train_modes.columns.values:
    df_train[feature] = df_train[feature].fillna(value = df_train_modes[feature].iloc[0])
#Note: When the following line was used to impute, it triggered confusing error messages.
#df_train[feature].fillna(inplace = True, value = df_train_modes[feature].iloc[0])

df_test_modes = df_test.mode()
for feature in df_test_modes.columns.values:
    df_test[feature] = df_test[feature].fillna(value = df_test_modes[feature].iloc[0])
    
df_holdout_modes = df_holdout.mode()
for feature in df_holdout_modes.columns.values:
    df_holdout[feature] = df_holdout[feature].fillna(value = df_holdout_modes[feature].iloc[0])
from sklearn.impute import SimpleImputer

imputer_mean = SimpleImputer(
    strategy = "mean",
    verbose = True,
    add_indicator = False,
    missing_values = np.nan
)

lst_current_numeric_columns = df_train.select_dtypes(include = "number").columns.tolist()
df_train[lst_numerics] = imputer_mean.fit_transform(X = df_train[lst_numerics])
df_train["SalePrice"] = imputer_mean.fit_transform(X = df_train[["SalePrice"]])

lst_current_numeric_columns.remove("SalePrice")
df_test[lst_numerics] = imputer_mean.fit_transform(X = df_test[lst_numerics])
df_test["SalePrice"] = imputer_mean.fit_transform(X = df_test[["SalePrice"]])

df_holdout[lst_numerics] = imputer_mean.fit_transform(X = df_holdout[lst_numerics])
temp = pd.Series(data = df_train.select_dtypes(include = "number").isnull().sum() )
temp[ (temp > 0) ]
temp = pd.Series(data = df_test.select_dtypes(include = "number").isnull().sum() )
temp[ (temp > 0) ]
temp = pd.Series(data = df_holdout.select_dtypes(include = "number").isnull().sum() )
temp[ (temp > 0) ]
#Old process was too complicated. These functions are here for reference but are not used.
##=========================================================================================

#     label_encoder = LabelEncoder()    
    
#     encoded_data = label_encoder.fit_transform( sr.dropna().astype(dtype = str) )    
    
#     value_counts = sr.astype(dtype = str).value_counts().sort_values(ascending = False)    
#     mode = value_counts.iloc[0]
    
#     bmask_isnull = sr.isnull()
#     sr_string_version = sr.astype(dtype = str)
#     sr_string_version[bmask_isnull] = mode
    
#     ndarr_clean = label_encoder.transform(sr_string_version)
#     ndarr_clean = label_encoder.inverse_transform(encoded_data) 
    
#     return {"data":ndarr_clean, "impute_value":mode}

# #Calling the hand-made functions.
# for feature in lst_categoricals:
#     try:
#         impute_results = impute_categorical(df_train[feature])
#         df_train[feature] = impute_results["data"]
        
# #         bmask_isnull_test = df_test[feature].isnull()
# #         df_test.loc[bmask_isnull_test, feature] = impute_results["impute_value"]
        
    
#         impute_results = impute_categorical(df_test[feature])
#         df_test[feature] = impute_results["data"]
        
#     except Exception as e:
#         print("Exception in imputing", feature, ":")
#         print(e)
# dict_classes = {}
# for feature in lst_categoricals:
#     impute_results = impute_categorical(df_train[feature])
#     df_train[feature] = impute_results["data"]
    
#     encoding_result = label_encode(df_train[feature])
#     try:
#         dict_classes[feature] = encoding_result["encoder_object"].classes_
#     except:
#         pass
    
# #       bmask_isnull_test = df_test[feature].isnull()
# #       df_test.loc[bmask_isnull_test, feature] = impute_results["impute_value"]
        
    
#     impute_results = impute_categorical(df_test[feature])
#     df_test[feature] = impute_results["data"]
df_train_modes = df_train.mode()
for feature in df_train_modes.columns.values:
    df_train[feature] = df_train[feature].fillna(value = df_train_modes[feature].iloc[0])
    #df_train[feature].fillna(inplace = True, value = df_train_modes[feature].iloc[0])

df_test_modes = df_test.mode()
for feature in df_test_modes.columns.values:
    df_test[feature] = df_test[feature].fillna(value = df_test_modes[feature].iloc[0])
    
df_holdout_modes = df_holdout.mode()
for feature in df_holdout_modes.columns.values:
    df_holdout[feature] = df_holdout[feature].fillna(value = df_holdout_modes[feature].iloc[0])
#Label-encoding categorical features.
label_encoder = LabelEncoder()
df_labeled_categoricals = df_train[lst_categoricals]

dict_classes = {}

for feature in df_labeled_categoricals.columns.values:
    df_labeled_categoricals[feature] = label_encoder.fit_transform(df_labeled_categoricals[feature].dropna())    
    dict_classes[feature] = label_encoder.classes_    

#Scaling features.
minmax_scaler = MinMaxScaler()
df_scaled_numerics = df_train[lst_numerics].copy()
df_scaled_categoricals = df_labeled_categoricals[lst_categoricals].copy()

#Min-max-scaled data frame with all true numeric data-types.
df_scaled_numerics[lst_numerics] = minmax_scaler.fit_transform(X = df_train[lst_numerics])

#Separate data frame for categorical features.
df_scaled_categoricals[lst_categoricals] = minmax_scaler.fit_transform(X = df_labeled_categoricals[lst_categoricals])

#Get list of features that lack variance.
results = var_threshold(df = df_scaled_numerics, minimum_variance = 0.03)
dict_variances, lst_variance_failers = results["variances"], results["failers"]

categorical_results = var_threshold(df = df_scaled_categoricals, minimum_variance = 0.03)
dict_categoricals_variances, lst_variance_failers_categorical = categorical_results["variances"], categorical_results["failers"]

for failer in lst_variance_failers_categorical:
    lst_variance_failers.append(failer)
    
#Drop features from lists of features.
for feature in lst_variance_failers:
    try:
        lst_numerics.remove(feature)       
    except Exception as e:
        print("Error removing", feature, "from list of true numerics/numeric data types.", '\n', e)

for feature in lst_variance_failers:
    try:
        lst_categoricals.remove(feature)
    except Exception as e:
        print("Error removing", feature, "from lst_categoricals.", '\n', e)

#Drop features from data frames.
for feature in lst_variance_failers:
    try:
        df_train.drop(columns = feature, inplace = True)
    except Exception as e:
        print("Error removing", feature, "from training data.", '\n', e)
else:
    print("=====\nAll variance failers removed from train data.\n=====")
        
for feature in lst_variance_failers:
    try:
        df_holdout.drop(columns = feature, inplace = True)
    except Exception as e:
        print("Error removing", feature, "from holdout data.", '\n', e)
else:
    print("=====\nAll variance failers removed from holdout data.\n=====")

for feature in lst_variance_failers:
    try:
        df_test.drop(columns = feature, inplace = True)
    except Exception as e:
        print("Error removing", feature, "from test data.", '\n', e)
else:
    print("=====\nAll variance failers removed from test data.\n=====")
print("Nulls in train set:", max(df_train.isnull().sum()) )
print("Nulls in test set:", max(df_test.isnull().sum()) )
print("Nulls in holdout set:", max(df_holdout.isnull().sum()) )
df_train.shape
df_train.head(2)
df_test.shape
df_test.head(2)
df_holdout.shape
df_holdout.head(2)
#Get dummies. Done to solve error from TPOT complaining about receiving classification data.
df_train_onehot = df_train.copy()
df_train_onehot = df_train_onehot.append(df_test)
df_train_onehot = df_train_onehot.append(df_holdout)

df_train_onehot = pd.get_dummies(df_train_onehot)
print(df_train_onehot.index)

df_holdout_onehot = df_train_onehot[1461: ]
df_train_onehot.drop(inplace = True, index = df_holdout_onehot.index)
df_holdout_onehot.drop(columns = "SalePrice", inplace = True)

split_onehot = make_traintest(df_train_onehot)
df_train_onehot = split_onehot["train"]
df_test_onehot = split_onehot["test"]
df_train_onehot.shape
df_train_onehot.head(2)
df_test_onehot.shape
df_test_onehot.head(2)
df_holdout_onehot.shape
df_holdout_onehot.head(2)
for column in df_train_onehot.columns.values:
    if column not in df_holdout_onehot.columns.values:
        print(column)
def save_submission_file(dict_data, filename):
    df = pd.DataFrame(data = dict_data)
    timestamp = str( dt.datetime.now() )
    name = filename + '-' + timestamp + ".csv"
    
    df.to_csv(
        path_or_buf = name,
        index = False
    )
    
    return None
url_to_raw = "https://github.com/SS-Runen/Kaggle-Competition-Data/raw/master/Ames%20Iowa%20Housing%20Kaggle%20Edition/df_train_statclean.csv"
df_train_statclean = pd.read_csv(
    filepath_or_buffer = url_to_raw,
    index_col = "Id"
)
max(df_train_statclean.isnull().sum())
df_train_statclean.head()
url_to_raw = "https://github.com/SS-Runen/Kaggle-Competition-Data/raw/master/Ames%20Iowa%20Housing%20Kaggle%20Edition/df_test_statclean.csv"
df_test_statclean = pd.read_csv(
    filepath_or_buffer = url_to_raw,
    index_col = "Id"
)

max(df_test_statclean.isnull().sum())
df_test_statclean.head()
url_to_raw = "https://github.com/SS-Runen/Kaggle-Competition-Data/raw/master/Ames%20Iowa%20Housing%20Kaggle%20Edition/df_holdout_statclean.csv"
df_holdout_statclean = pd.read_csv(
    filepath_or_buffer = url_to_raw,
    index_col = "Id"
)

max(df_holdout_statclean.isnull().sum())
df_holdout_statclean.head()
#!pip install -U keras-tuner
!pip install git+https://github.com/keras-team/keras-tuner.git@1.0.2rc2
!pip install autokeras
from autokeras import StructuredDataRegressor

print("Search Started.")
# define the search
search = StructuredDataRegressor(max_trials = 4, overwrite = True, loss = 'mean_squared_error')

# perform the search
# search.fit(
#     x = df_train_onehot.drop(columns = "SalePrice"),
#     y = df_train_onehot["SalePrice"],
#     verbose=0)
search.fit(
    x = df_train_statclean.drop(columns = "SalePrice"),
    y = df_train_statclean["SalePrice"],
    epochs = 4)
print("Search Finished.")
print("Started AutoKeras evaluation.")
try:    
    mse = search.evaluate(
        x = df_test_statclean.drop(columns = "SalePrice"),
        y = df_test_statclean["SalePrice"],
        )
    print('RMSE:{}'.format(np.sqrt(mse)))
except Exception as e:
        print("Error thrown attempting to evaluate AutoKeras model:\n", e)
# use the model to make a prediction
yhat = search.predict(df_holdout_statclean)
print('First Prediction: %.3f' % yhat[0] )

# get the best performing model
model = search.export_model()

# summarize the loaded model
print("Model Summary")
model.summary()

# save the best performing model to file
model.save('mdl_autokeras_statclean')
url_to_raw_holdout = "https://raw.githubusercontent.com/SS-Runen/Kaggle-Competition-Data/master/Ames%20Iowa%20Housing%20Kaggle%20Edition/df_holdout_statclean.csv"
yhat = model.predict(url_to_raw_holdout)

save_submission_file(
    dict_data = {"Id":df_holdout_statclean.index.tolist(), "SalePrice":yhat.reshape(yhat.size, )},
    filename = "AutoKeras on Stat-cleaned from Traditional Notebook"
)
# !pip install auto-sklearn
# import autosklearn
# autosklearn.__version__
# #Install autosklearn on Kaggle
# #URL: https://www.kaggle.com/general/146842
# !apt-get remove swig
# !apt-get install swig3.0 build-essential -y
# !ln -s /usr/bin/swig3.0 /usr/bin/swig
# !apt-get install build-essential
# !pip install --upgrade setuptools
# !pip install auto-sklearn
!pip install tpot
import tpot
from tpot import TPOTRegressor
tpot.__version__
print(df_train_onehot.shape)
# #Reduced to 3 because error: pipeline failed to complete.
# #Failed because n_splits being greater than the amount of values/samples in a class.
# cv_rskfold = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 5, random_state = 88)

# mdl_tpot_regressor = TPOTRegressor(
#     n_jobs = -1,
#     generations = 8,
#     population_size = 32,
#     cv = cv_rskfold
# )

# mdl_tpot_regressor.fit(
#     features = df_train_onehot.drop(columns = "SalePrice"),
#     target = df_train_onehot["SalePrice"])
from tpot import TPOTRegressor
# define evaluation procedure
cv_rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=88)
# define search
model = TPOTRegressor(
    generations=4,
    population_size=8,
    scoring='neg_mean_absolute_error',
    cv=cv_rskf,
    verbosity=2,
    random_state=1,
    n_jobs=-1
)
print(df_train_onehot.isnull().sum().sort_values())
# perform the search
model.fit(
    df_train_onehot.drop(columns = "SalePrice"),
    imputer_mean.fit_transform(df_train_onehot[["SalePrice"]]).reshape(
        len(df_train_onehot["SalePrice"]), )
)
# export the best model
model.export('TPOT Baseline Model, Onehot.py')
results_tpot_clean = model.predict(features = df_holdout_onehot)
save_submission_file(
    dict_data = {"Id":df_holdout.index.tolist(), "SalePrice":results_tpot_clean},
    filename = "TPOTRegressor Lifted from Docs, Stat-clean"
)
mdl_tpot_regressor.fit(
    features = df_train_unimputed.drop(columns = "SalePrice"),
    target = df_train_unimputed["SalePrice"])

results_tpot_unclean = mdl_tpot_regressor.predict(features = df_holdout_unimputed)
mdl_tpot_regressor.score(df_test.drop(columns = "SalePrice"), df_test["SalePrice"])
mdl_tpot_regressor.export('tpot_pipeline_unclean.py')
save_submission_file(
    dict_data = {"Id":df_holdout.index.tolist(), "SalePrice":results_tpot_unclean},
    filename = "Baseline TPOT on Unclean Data"
)
!pip install hyperopt
!git clone https://github.com/hyperopt/hyperopt-sklearn.git
#!cd hyperopt
!pip install -e .
!pip show 
!pip show hpsklearn
from sklearn.svm import SVC, LinearSVC, LinearSVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from hpsklearn import HyperoptEstimator
import xgboost

mdl_svc, mdl_linSVC = SVC(), LinearSVC()
mdl_randforest_class = RandomForestClassifier(
    n_jobs = -1,
    random_state=88,
    max_samples = 0.70,
    min_samples_leaf=0.05
    )
lst_classifiers = [mdl_svc, mdl_linSVC, mdl_randforest_class]

mdl_svr, mdl_xgb_regress = LinearSVR(), xgboost.XGBRegressor()
mdl_randforest_regress = RandomForestClassifier(
    n_jobs = -1,
    random_state=88,
    max_samples=0.70,
    min_samples_leaf=0.05
    )
#lst_regressors = [mdl_xgb_regress, mdl_svr, mdl_randforest_regress]
lst_regressors = [mdl_xgb_regress, mdl_svr]

prep_pca, prep_onehot = PCA(), OneHotEncoder()
imputer_mean = SimpleImputer()
lst_preprocessors = [prep_pca, prep_onehot, imputer_mean]

estimator_hyperopt = HyperoptEstimator(
    regressor=lst_regressors,
    preprocessing=lst_preprocessors,
    max_evals=32,
    trial_timeout=300)
#Used to be:
#imputer_mean.fit_transform(df_train_onehot[["SalePrice"]]).reshape(len(df_train_onehot["SalePrice"]), )
df_train_onehot_saleprice = pd.DataFrame(data = imputer_mean.fit_transform(df_train_onehot[["SalePrice"]]), index = df_train_onehot.index)
# estimator_hyperopt.fit(
#     X = df_train_onehot.drop(columns = "SalePrice"),
#     y = df_train_onehot_saleprice.values
# )

# test_score = estimator_hyperopt.score(
#     X = df_test.drop(columns = "SalePrice").values,
#     y = df_test["SalePrice"].values
#     )

# mdl_best_hyperopt = estimator_hyperopt.best_model()
# print("Best HyperOpt model:\n", mdl_best_hyperopt)

estimator_hyperopt.fit(
    X = df_train_statclean.drop(columns = "SalePrice").values,
    y = df_train_statclean["SalePrice"]
)

test_score = estimator_hyperopt.score(
    X = df_test_statclean.drop(columns = "SalePrice").values,
    y = df_test_statclean["SalePrice"].values
    )

mdl_best_hyperopt = estimator_hyperopt.best_model()
print("Best HyperOpt model:\n", mdl_best_hyperopt)
import xgboost as xgb
xgb.__version__
DMatrix_train = xgb.DMatrix(
    df_train_statclean.drop(columns = "SalePrice"),
    label = df_train_statclean["SalePrice"].values
)

DMatrix_test = xgb.DMatrix(
    df_test_statclean.drop(columns = ["SalePrice", "MSSubClass"]),
    label = df_test_statclean["SalePrice"].values
)

parameters = {
    "verbosity": 3,
    "max_depth": 8,
    "subsample": 0.7,
    "seed": 105,
    "booster": "gbl",
    "eta": 0.1
}

rounds = 2048

eval_sets = [(DMatrix_train, "train-on-train"), (DMatrix_test, "test_set")]

dict_evaluation_results = {}


mdl_xgb = xgb.train(
    params = parameters,
    dtrain = DMatrix_train,
    num_boost_round = rounds,
    evals = eval_sets,
    evals_result = dict_evaluation_results
)
dict_evaluation_results
DMatrix_holdout = xgb.DMatrix(df_holdout_statclean)
xgb_predictions = mdl_xgb.predict(DMatrix_holdout)

save_submission_file(
    dict_data = {"Id":df_holdout_statclean.index.tolist(), "SalePrice":xgb_predictions},
    filename = "Baseline XGBoost on Stat-cleaned Data v1.5 GBL"
)
print("Done")
