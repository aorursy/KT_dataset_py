import pandas as pd, matplotlib.pyplot as plt, numpy as np, datetime as dt, seaborn as sbrn
#Model Testers
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
#MSE metric, as stated in Kaggle Documentation.
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
#Estimators/models.
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
#Feature Selection Functions
from sklearn.feature_selection import SelectKBest, SelectFwe, RFE, RFECV
from sklearn.feature_selection import f_regression, mutual_info_regression, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.datasets import make_sparse_uncorrelated
from sklearn.metrics import make_scorer
scope_names = dir()
#scope_names
#Try replacing with %load or %loadpy and use python sript.

#%load standard_functions.py

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
df_train = pd.read_csv(
    #sep = '\t',
    filepath_or_buffer = "/kaggle/input/house-prices-advanced-regression-techniques/train.csv",
    #filepath_or_buffer = "Ames Iowa Housing Data Kaggle/train.csv",
    index_col = "Id"    
)

# df_test = pd.read_csv(
#     filepath_or_buffer = "/kaggle/input/house-prices-advanced-regression-techniques/test.csv",
#     #filepath_or_buffer = "Ames Iowa Housing Data Kaggle/train.csv",
#     index_col = "Id"
# )

df_holdout = pd.read_csv(
    filepath_or_buffer = "/kaggle/input/house-prices-advanced-regression-techniques/test.csv",
    #filepath_or_buffer = "Ames Iowa Housing Data Kaggle/train.csv",
    index_col = "Id"
)
df_train.head()
#df_train.columns.tolist()
df_train.shape
df_holdout.shape
# Features below will not be available for unsold houses. They might also "leak" information that would not be present in
# actual data sets of unsold houses.
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
print("Count of true integers and true floats:", "Ints:", len(lst_true_integers), "Floats:", len(lst_true_floats))

lst_numerics = [n for n in lst_true_integers]
#print(len(lst_numerics))
#lst_numeric_dtypes = [n for n in lst_true_integers]

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
#lst_numeric_dtypes.append("SalePrice")
print(len(lst_numerics))
def make_int(df):
    df_result = pd.DataFrame(index = df.index)
    
    for feature in df.columns.values:
        try:
            df_result[feature] = pd.to_numeric(arg = df[feature], errors = "coerce", downcast = "integer")
        except Exception as e:
            print("Failed to transform", feature, ",\n", e)
    
    return df_result
def make_float64(df):
    df_result = pd.DataFrame(index = df.index)
    
    for feature in df.columns.values:
        try:
            df_result[feature] = df[feature].astype(dtype = np.float64)
        except Exception as e:
            print("Failed to transform", feature, ",\n", e)
    
    return df_result
ConvertToInt = FunctionTransformer(func = make_int)
ConvertToFloat64 = FunctionTransformer(func = make_float64)
df_train[lst_true_integers] = ConvertToInt.transform(X = df_train[lst_true_integers])
df_train[lst_true_floats] = ConvertToFloat64.transform(X = df_train[lst_true_floats])

df_holdout[lst_true_floats] = ConvertToInt.transform(X = df_holdout[lst_true_floats])
df_holdout[lst_true_floats] = ConvertToFloat64.transform(X = df_holdout[lst_true_floats])
lst_categoricals = []
for feature in df_train.columns.values:
    if (feature not in lst_numerics) and (str(feature) != "SalePrice"):
        #df_train[feature] = df_train[feature].astype(dtype = str)
        #df_train[feature] = df_train[feature].astype(dtype = "category")
        lst_categoricals.append(feature)        

df_train["SalePrice"] = df_train["SalePrice"].astype(dtype = np.float64)
lst_categoricals.sort()
df_train.head()
df_train[lst_numerics].info()
def make_age_from_year(integer_year):
    return int(dt.datetime.now().year) - integer_year
lst_year_columns = ["GarageYrBlt", "YearBuilt", "YearRemodAdd"]
for feature in lst_year_columns:
    df_train[feature] = df_train[feature].apply(func = make_age_from_year)
    df_holdout[feature] = df_holdout[feature].apply(func = make_age_from_year)
df_train[lst_year_columns].head()
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
df_train.head()
# for feature in lst_categoricals:
#     df_train[feature] = df_train[feature].astype(dtype = str)
# df_dummy = df_housing.drop(columns = ["SalePrice"])
# lst_features = df_dummy.columns.values.tolist()
# df_dummy["SalePrice"] = df_housing["SalePrice"]

# df_dummy[lst_numerics] = slctr_var_threshold.fit_transform(X = df_dummy[lst_numerics])
# slctr_var_threshold.variances_
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
def label_encode(feature_data):
    label_encoder = LabelEncoder()    
    label_encoded = label_encoder.fit_transform(feature_data)    
    
    return {"encoded_data":label_encoded, "encoder_object":label_encoder}
#Process is sub-obtimal because it performs some steps of transform before error stops execution.
#Ideally, target column would be passed in or categoricals would be automatically targeted.
def label_encode(df):
    for feature in df:
        try:
            label_encoder = LabelEncoder()
            label_encoded = label_encoder.fit_transform(df_feature)
            df[feaure] = label_encoded
        except Exception as e:
            print("Could not label-encode", feature, ".\n", e)
    
    return df
label_encoder = LabelEncoder()
df_categoricals_train = df_train[lst_categoricals].dropna()

dict_classes = {}

for feature in df_categoricals_train.columns.values:
    df_categoricals_train[feature] = label_encoder.fit_transform(df_categoricals_train[feature].astype(str))    
    dict_classes[feature] = label_encoder.classes_
# for feature, classes in dict_classes.items():
#     print(feature, "\n", classes)
df_categoricals_train.head()
lst_features = df_train.columns.tolist()
lst_features.remove("SalePrice")

lst_current_numeric_features = df_train.select_dtypes(include = "number").columns.tolist()
lst_current_numeric_features.remove("SalePrice")

lst_current_numeric_columns = df_train.select_dtypes(include = "number").columns.tolist()

df_numeric_features = pd.DataFrame(
    data = minmax_scale(X = df_train[lst_numerics]),
    index = df_train.index,
    columns = df_train[lst_numerics].columns
)
df_numeric_features.head()
df_categoricals_train[lst_categoricals] = minmax_scale(X = df_categoricals_train[lst_categoricals])
df_categoricals_train.head()
results = var_threshold(df = df_numeric_features[lst_numerics], minimum_variance = 0.05)
dict_variances, lst_variance_failers = results["variances"], results["failers"]
categorical_results = var_threshold(df = df_categoricals_train, minimum_variance = 0.05)
dict_categoricals_variances, lst_variance_failers_categorical = categorical_results["variances"], categorical_results["failers"]
for failer in lst_variance_failers_categorical:
    lst_variance_failers.append(failer)
print(len(lst_variance_failers))
print(len(lst_features))
for feature in lst_variance_failers:
    try:
        lst_numerics.remove(feature)
        lst_numeric_dtypes.remove(feature)        
    except Exception as e:
        print("Error removing", feature, "from list of true numerics/numeric data types.", '\n', e)
        
for feature in lst_variance_failers:
    try:
        df_train.drop(columns = feature, inplace = True)
    except Exception as e:
        print("Error removing", feature, "from training data.", '\n', e)
        
for feature in lst_variance_failers:
    try:
        df_holdout.drop(columns = feature, inplace = True)
    except Exception as e:
        print("Error removing", feature, "from holdout data.", '\n', e)
        
for feature in lst_variance_failers:
    try:
        lst_categoricals.remove(feature)
    except Exception as e:
        print("Error removing", feature, "from lst_categoricals.", '\n', e)
lst_variance_failers.sort()
lst_variance_failers
split = make_traintest(df_train)
df_housing_clean = df_train.copy()

df_train = split["train"]
df_test = split["test"]
df_train.shape
df_test.shape
df_holdout.shape
def impute_categorical(sr):
    label_encoder = LabelEncoder()    
    
    encoded_data = label_encoder.fit_transform( sr.dropna().astype(dtype = str) )    
    
    value_counts = sr.astype(dtype = str).value_counts().sort_values(ascending = False)    
    mode = value_counts.iloc[0]
    
    bmask_isnull = sr.isnull()
    sr_string_version = sr.astype(dtype = str)
    sr_string_version[bmask_isnull] = mode
    
    ndarr_clean = label_encoder.transform(sr_string_version)
    ndarr_clean = label_encoder.inverse_transform(encoded_data) 
    
    return {"data":ndarr_clean, "impute_value":mode}
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
df_train.isnull().sum()
df_test.isnull().sum()
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
max(df_test[lst_categoricals].isnull().sum())
max(df_train[lst_categoricals].isnull().sum())
max(df_holdout.isnull().sum())
df_train.head(2)
df_train.shape
df_test.head(2)
df_test.shape
df_holdout.head(2)
df_holdout.shape
max(df_train.isnull().sum())
max(df_test.isnull().sum())
max(df_holdout.isnull().sum())
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
temp = pd.Series(data = df_holdout.select_dtypes(include = "number").isnull().sum() )
temp[ (temp > 0) ]
def print_result_dict(dict_result):
    
    for dataset, dataset_results in dict_result.items():
        print("\n=====\n", "Start", "\n=====\n", dataset)
        
        for method, result_dict in dataset_results.items():
            print("\n-----\n", method)
            for result_name, result_value in result_dict.items():
                print(":", result_name, ":")
                try:
                    sr_values = pd.Series(data = result_value)
                    print(sr_values.sort_values(ascending = False))
                except:
                    print(result_value)
                finally:
                    print("\n*** End Section***")
        print("\n=== End of Results ===\n")
    return None
# def manyset_manymethod_kbest(
#     dict_datasets,
#     dict_targets,
#     lst_score_funcs,
#     y = "is_benign",    
#     n_passers = "all"    
#     ):

dict_kbest_results = manyset_manymethod_kbest(
    dict_datasets = {"df_train" : df_train[lst_numerics] },
    dict_targets = {"df_train" : df_train["SalePrice"] },
    lst_score_funcs = [f_regression, mutual_info_regression],
    n_passers = 0
)
print_result_dict(dict_kbest_results)
def filter_by_threshold(dict_value_series, remove_low = True, dict_thresholds = {"default":np.nan}):
    lst_failers = []    
    dict_exceptions = {}
    
    if remove_low == True: 
        try:
            for alias, series in dict_value_series.items():
                for feature in series.index.tolist():
                    if (series[feature] <= dict_thresholds[alias]) and (feature not in lst_failers):
                        lst_failers.append(feature)
        except Exception as e:                
                dict_exceptions["alias"] = str(e)            
                    
    else:        
        for alias, series in dict_value_series.items():
            try:
                for feature in series.index.tolist():                
                    if (series[feature] <= dict_thresholds[alias]) and (feature not in lst_failers):
                        lst_failers.append(feature)            
            except Exception as e:                
                dict_exceptions["alias"] = str(e)
    
    return lst_failers
def extract_result_dict(dict_result):
    
    extracted = {}
    
    for dataset, dataset_results in dict_result.items():
        #print("\n=====\n", "Start", "\n=====\n", dataset)
        
        for method, result_dict in dataset_results.items():
            #print("\n-----\n", method)
            for result_name, result_value in result_dict.items():
                item_name = method + '-' + result_name
                try:
                    sr_values = pd.Series(data = result_value)
                    extracted[item_name] = sr_values.sort_values(ascending = False)
                except Exception as e:
                    print("Error occured processing:", item_name, e)
                    continue        
                    
    return extracted
def find_kbest_statistics_max(dictionary_of_series):
    dict_maximums = {}
    for alias, series in dictionary_of_series.items():
        try:
            dict_maximums[alias] = np.nanmax(series)
        except:
            dict_maximums[alias] = np.nan
    
    return dict_maximums
def find_kbest_statistics_min(dictionary_of_series):
    dict_minimums = {}
    for alias, series in dictionary_of_series.items():
        try:
            dict_minimums[alias] = np.nanmin(series)
        except:
            dict_minimums[alias] = np.nan
            
    return dict_minimums
#Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_sparse_uncorrelated.html
#The first four features are useful for regression. The rest are not.
feature_count = len(lst_numerics) + 4
row_count = len(df_train["SalePrice"])
sparse_uncorrelated_X, sparse_uncorrelated_y = make_sparse_uncorrelated(
    n_samples = row_count,
    n_features = feature_count,
    random_state = 0
)
df_sparse_uncorr = pd.DataFrame(data = sparse_uncorrelated_X)
df_sparse_uncorr = df_sparse_uncorr.loc[ : , 4: ]
df_sparse_uncorr["SalePrice"] = sparse_uncorrelated_y
df_sparse_uncorr.head()
df_sparse_uncorr.shape
dict_sparse_uncorr = manyset_manymethod_kbest(
    dict_datasets = {"df_sparse_uncorr" : df_sparse_uncorr.drop(columns = ["SalePrice"]) },
    dict_targets = {"df_sparse_uncorr" : df_sparse_uncorr["SalePrice"] },
    lst_score_funcs = [f_regression, mutual_info_regression]  
)
print_result_dict(dict_sparse_uncorr)
kbest_stats_for_synthetic = extract_result_dict(dict_sparse_uncorr)
kbest_stats_for_synthetic.keys()
kbest_stats_for_train = extract_result_dict(dict_kbest_results)
#del(kbest_stats_for_train["f_regression-kbest_pvalues"])
kbest_stats_for_train.keys()
sparse_uncorr_failers = set(
    filter_by_threshold(
        dict_value_series = kbest_stats_for_train,
        dict_thresholds = {
            "f_regression-kbest_scores": np.max(kbest_stats_for_synthetic["f_regression-kbest_scores"]),
            "mutual_info_regression-kbest_scores": np.max(kbest_stats_for_synthetic["mutual_info_regression-kbest_scores"])
        }
    )
)
sparse_uncorr_failers
lst_synthetic_data_failers = list(df_train[lst_numerics].columns[list(sparse_uncorr_failers)])
#List of failers for `mutual info regression` and `f_regression` tests.
lst_synthetic_data_failers
#Consistency check. The list `lst_numerics` was used to slice the data frame and get the statistics for the train dataset.
#The list is sorted, as are columns of the dataframes. DF columns are automatically sorted.
def check_iloc_in_lst_numerics(iterable_iloc_indexer, index_names):
    for iloc_index in iterable_iloc_indexer:
        print(index_names[ int(iloc_index)] )
check_iloc_in_lst_numerics( list(sparse_uncorr_failers), lst_numerics)
FWE = SelectFwe(score_func = f_regression, alpha = 0.01)
FWE.fit_transform(X = df_train[lst_numerics], y = df_train["SalePrice"])
bmask_fwe_passers = FWE.get_support()
bmask_fwe_failers = ~bmask_fwe_passers
lst_fwe_passers = df_train[lst_numerics].columns[bmask_fwe_passers]
lst_fwe_passers
lst_fwe_failers = df_train[lst_numerics].columns[bmask_fwe_failers].tolist()
lst_fwe_failers.sort()
lst_fwe_failers
decomposer_pca = PCA(n_components = None)
decomposer_pca.fit(df_sparse_uncorr)
uncorrelated_max_explained_variance = max(decomposer_pca.explained_variance_)
uncorrelated_max_explained_variance_ratio = max(decomposer_pca.explained_variance_ratio_)
label_encoder = LabelEncoder()
df_categoricals_label_encoded = pd.DataFrame(
    columns = df_train[lst_categoricals].columns,
    index = df_train[lst_categoricals].index
)
for feature in lst_categoricals:
    df_categoricals_label_encoded[feature] = label_encoder.fit_transform(y = df_train[feature])

decomposer_pca.fit(df_categoricals_label_encoded)
lst_pca_failers = []
lst_explained_variances = decomposer_pca.explained_variance_
lst_explained_variance_ratios = decomposer_pca.explained_variance_ratio_

for feature_iloc in range( len(decomposer_pca.explained_variance_) ):
    feature = lst_categoricals[feature_iloc]    
    
    if (lst_explained_variances[feature_iloc] <= uncorrelated_max_explained_variance) and (feature not in lst_pca_failers):
        lst_pca_failers.append(feature)
    
    if (lst_explained_variance_ratios[feature_iloc] <= uncorrelated_max_explained_variance) and (feature not in lst_pca_failers):
        lst_pca_failers.append(feature)
        
lst_pca_failers        
lst_statistical_failers = []

for item in lst_fwe_failers:
    if item not in lst_statistical_failers:
        lst_statistical_failers.append(item)

for item in lst_synthetic_data_failers:
    if item not in lst_statistical_failers:
        lst_statistical_failers.append(item)

for item in lst_pca_failers:
    if item not in lst_statistical_failers:
        lst_statistical_failers.append(item)
df_train = df_train.append(df_test)
df_train = df_train.append(df_holdout)
df_train = pd.get_dummies(data = df_train)
df_holdout = df_train.loc[1461: , : ]
df_holdout.drop(columns = "SalePrice", inplace = True)
df_train.drop(index = df_train.loc[1461: , : ].index, inplace = True)

split = make_traintest(df = df_train)
df_train = split["train"]
df_test = split["test"]
df_train.shape
df_test.shape
df_test["SalePrice"]
df_holdout.shape
df_train_statclean = df_train.copy()
df_test_statclean = df_test.copy()
df_holdout_statclean = df_holdout.copy()

for feature in lst_statistical_failers:
    try:
        df_train_statclean.drop(columns = feature, inplace = True)
    except Exception as e:
        print("Error dropping from train", feature, '\n', e)

for feature in lst_statistical_failers:
    try:
        df_test_statclean.drop(columns = feature)
    except Exception as e:
        print("Error dropping from test", feature, '\n', e)
        
for feature in lst_statistical_failers:
    try:
        df_holdout_statclean.drop(inplace = True, columns = feature)
    except Exception as e:
        print("Error dropping from holdout", feature, '\n', e)
mdl_linreg = LinearRegression()
mdl_randforest = RandomForestRegressor()
mdl_nnregressor = MLPRegressor(hidden_layer_sizes = (287, 64, 64) )
mdl_linreg.fit(X = df_train.drop(columns = "SalePrice"), y = df_train["SalePrice"])
mdl_randforest.fit(X = df_train.drop(columns = "SalePrice"), y = df_train["SalePrice"])
mdl_nnregressor.fit(X = df_train.drop(columns = "SalePrice"), y = df_train["SalePrice"])
mse_scorer = make_scorer(mean_squared_error)
# dict_baseline_cvs = manymodel_manyfeatureset_cvs(
#     lst_estimators = [mdl_linreg, mdl_randforest, mdl_nnregressor],
#     dict_featuresets = {
#         "df_train":[
#             df_train.drop(columns = "SalePrice"),
#             df_train["SalePrice"]
#         ],
#         "df_test":[
#             df_test.drop(columns = "SalePrice"),
#             df_test["SalePrice"]
#         ]
#     },
#     scoring_method = mse_scorer
# )
def print_cv_result_dict(dict_result):
    
    for dataset, dataset_results in dict_result.items():
        print("\n=====\n", "Start", "\n=====\n", dataset)
        
        for method, result_dict in dataset_results.items():
            print("\n-----\n", method)
            for result_name, result_value in result_dict.items():
                print(":", result_name, ":")
                res = "{:.3f}".format( np.sqrt(result_value) )
                try:                    
                    sr_values = pd.Series(data = res)
                    print(sr_values.sort_values(ascending = False))
                except:
                    print( res )
                finally:
                    print("\n*** End Section***")
        print("\n=== End of Results ===\n")
    return None
#print_cv_result_dict(dict_baseline_cvs)
max(df_train_statclean.isnull().sum())
max(df_test_statclean.isnull().sum())
max(df_holdout_statclean.isnull().sum())
mdl_randforest_statcleaned = RandomForestRegressor()
mdl_randforest_statcleaned.fit(
    X = df_train_statclean.drop(columns = ["SalePrice"]),
    y = df_train_statclean["SalePrice"])
mdl_nnregressor_statcleaned = MLPRegressor(hidden_layer_sizes = (277, 128, 128), max_iter = 2054)
mdl_nnregressor_statcleaned.fit(X = df_train_statclean.drop(columns = ["SalePrice"]), y = df_train_statclean["SalePrice"])
# dict_crossval_results = manymodel_manyfeatureset_cvs(
#     lst_estimators = [mdl_randforest_statcleaned, mdl_nnregressor_statcleaned],
#     dict_featuresets = {
#         "df_train_statclean": [
#             df_train_statclean.drop(columns = ["SalePrice"]),
#             df_train_statclean["SalePrice"]
#         ],
#         "df_test_statclean": [
#             df_test_statclean.drop(columns = "SalePrice"),
#             df_test_statclean["SalePrice"]
#         ],
#         "df_train_statclean_numeric":[
#             df_train_statclean.select_dtypes(include = "number").drop(columns = "SalePrice"),
#             df_train_statclean["SalePrice"]
#         ],
#         "df_test_statclean_numeric": [
#             df_test_statclean.select_dtypes(include = "number").drop(columns = "SalePrice"),
#             df_test_statclean["SalePrice"]
#         ]
#     },
#     scoring_method = mse_scorer
# )
#print_cv_result_dict(dict_crossval_results)
def save_submission_file(dict_data, filename):
    df = pd.DataFrame(data = dict_data)
    timestamp = str( dt.datetime.now() )
    name = filename + '-' + timestamp + ".csv"
    
    df.to_csv(
        path_or_buf = name,
        index = False
    )
    
    return None
nn_results = mdl_nnregressor_statcleaned.predict(X = df_holdout_statclean)
#ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 307 is different from 242)
randforest_results = mdl_randforest_statcleaned.predict(X = df_holdout_statclean)
#ValueError: Number of features of the model must match the input. Model n_features is 223 and input n_features is 208 
baseline_randforest = mdl_randforest.predict(X = df_holdout)
baseline_nnregressor = mdl_nnregressor.predict(X = df_holdout)
save_submission_file(
    dict_data = {"Id":df_holdout.index.tolist(), "SalePrice":baseline_randforest},
    filename = "Baseline Random Forest Regressor"
)
save_submission_file(
    dict_data = {"Id":df_holdout.index.tolist(), "SalePrice":baseline_nnregressor},
    filename = "Baseline Neural Net Regressor"
)
save_submission_file(
    dict_data = {"Id":df_holdout.index.tolist(), "SalePrice":nn_results},
    filename = "Neural Network Regressor, Stat Clean"
)
save_submission_file(
    dict_data = {"Id":df_holdout.index.tolist(), "SalePrice":randforest_results},
    filename = "Random Forest Regressor, Stat Clean"
)
!pip install -U tensorflow
import tensorflow as tf
tf.__version__
!pip install git+https://github.com/keras-team/keras-tuner.git@1.0.2rc2
!pip install autokeras
#autokeras.__version__
from autokeras import StructuredDataRegressor
# define the search
search = StructuredDataRegressor(max_trials = 8, loss='mean_squared_error')
print("Started search.")
# perform the search
search.fit(
    x = df_train_statclean.drop(columns = "SalePrice"),
    y = df_train_statclean["SalePrice"]
)
# evaluate the model
mse = search.evaluate(
    x = df_test_statclean.drop(columns = "SalePrice"),
    y = df_test_statclean["SalePrice"],
    verbose = 0)
print('RMSE: {.3f}'.format(np.sqrt(mse)))

# use the model to make a prediction
yhat = search.predict(df_holdout_statclean)
print('Predicted: %.3f' % yhat[0])

# get the best performing model
model = search.export_model()

# summarize the loaded model
model.summary()

# save the best performing model to file
model.save('mdl_autokeras_baseline')
from tpot import TPOTRegressor
# define evaluation procedure
cv_rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=88)
# define search
model_tpotreg = TPOTRegressor(
    generations=4,
    population_size=8,
    scoring='neg_mean_absolute_error',
    cv=cv_rskf,
    verbosity=2,
    random_state=88,
    n_jobs=-1
)
# perform the search
model_tpotreg.fit(df_train_statclean.drop(columns = "SalePrice"), df_train_statclean["SalePrice"])
# export the best model
model_tpotreg.export('44201762 TPOT Baseline gen=4, popsize=8.py')
tpot_baseline = model_tpotreg.predict(df_holdout_statclean)

save_submission_file(
    dict_data = {"Id":df_holdout.index.tolist(), "SalePrice":tpot_baseline},
    filename = "44201762 TPOT Baseline, RidgeCV(RobustScaler(input_matrix))"
)
from tpot import TPOTRegressor

tpot = TPOTRegressor(
    generations = 32,
    population_size = 256,
    verbosity = 3,
    random_state = 88)

tpot.fit(df_train_statclean.drop(columns = "SalePrice"), df_train_statclean["SalePrice"])
print(tpot.score(df_train_statclean.drop(columns = "SalePrice"), df_train_statclean["SalePrice"]))
tpot.export('44201762 TPOTRegressor gen 32 pop 256 randstate 88.py')
tpot_results = tpot.predict(df_holdout_statclean)
save_submission_file(
    dict_data = {"Id":df_holdout_statclean.index.tolist(), "SalePrice":tpot_results},
    filename = "44201762 TPOT TPOTRegressor gen 32 pop 256 randstate 88"
)
df_train_statclean.to_csv(path_or_buf = "/kaggle/working/df_train_statclean.csv" )
df_test_statclean.to_csv(path_or_buf = "/kaggle/working/df_test_statclean.csv")
df_holdout_statclean.to_csv(path_or_buf = "/kaggle/working/df_holdout_statclean.csv")
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
estimator_hyperopt.fit(
    X = df_train_statclean.drop(columns = "SalePrice").values,
    y = df_train_statclean["SalePrice"].values)

test_score = estimator_hyperopt.score(
    X = df_test_statclean.drop(columns = "SalePrice").values,
    y = df_test_statclean["SalePrice"].values
    )

mdl_best_hyperopt = estimator_hyperopt.best_model()
print("Best HyperOpt model:\n", mdl_best_hyperopt)
ndarr_predictions_hyperopt = mdl_best_hyperopt.predict(X = df_holdout.values)
save_submission_file(
    dict_data = {"Id":df_holdout.index.tolist(), "SalePrice":ndarr_predictions_hyperopt},
    filename = "Neural Network Regressor, Stat Clean"
)
print("Done")
