from os import walk

for (dirpath, dirnames, filenames) in walk("../input"):

    print("Directory path: ", dirpath)

    print("Folder name: ", dirnames)

    print("File name: ", filenames)
import pandas as pd, matplotlib.pyplot as plt, numpy as np, datetime as dt, seaborn as sbrn
#Model Testers

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score

from sklearn.model_selection import RepeatedStratifiedKFold
#MSE metric, as stated in Kaggle Documentation.

from sklearn.metrics import mean_squared_error
#Estimators/models.

from sklearn.neural_network import MLPRegressor

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor
#Feature Selection Functions

from sklearn.feature_selection import SelectKBest, RFE, RFECV

from sklearn.feature_selection import f_regression, mutual_info_regression
scope_names = dir()

scope_names
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



def run_kbest(lst_X, y, df, n_passers = "all", score_func = None):

    if (score_func == None) and ("mutual_info_regression" not in scope_names):

        from skelarn.feature_selection import mutual_info_regression

        score_func = mutual_info_regression

    

    if "SelectKBest" not in scope_names:

        from sklearn.feature_selection import SelectKBest

    

    kbest = SelectKBest(score_func = score_func, k = n_passers)

    kbest.fit(X = df[lst_X], y = df[y])

        

    results = {

        "kbest_scores": kbest.scores_,

        "kbest_pvalues": kbest.pvalues_,

        "kbest_params": kbest.get_params(),

        "passing_features": kbest.get_support()

        }

    

    return results



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

        estimator, df, lst_X, y, dict_params,

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

    

    grid_search.fit(X = df[lst_X], y = df[y])

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

    df,

    y,

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

                df = df,

                y = y,

                lst_X = feature_combination,

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

    

    mean_of_scores = np.nanmean(cvscores)

    mean_of_scores_withnan = np.mean(cvscores)

    standard_deviation = np.std(cvscores)

    variance = np.var(cvscores)    

    

    results = {

        "mean_score": mean_of_scores,

        "nanmean_score": mean_of_scores_withnan,

        "std": standard_deviation,

        "var": variance

        }    

    

    return results



def manymodel_manyfeatureset_cvs(

        lst_estimators,

        dict_featuresets,

        dict_target_features,

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

        different sets of training data. The values are the corresponding set of

        feature data to be passed "as-is" to an estimator. Examples are

        dataframes, series, or NumPy n-dimensional arrays.

        

    dict_target_features: Dictionary

        A Python dictionary whose key-value pairs are a name and target feature data set

        stored as a series or dataframe. The keys, i.e. names/aliases, must be the same

        as the corresponding target feature's learning input data. For example,

            `dict_featuresets` contains the values ["df_classif": df_classify,"df_reg":df_regress],

            `dict_target_features` must contain ["df_classif":Beats_Marketprice, "df_reg":Price]

    

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

        

        for alias, X in dict_featuresets.items():

            featureset_rslt[alias] = run_crossvalscore(

                estimator = estimator,

                train_data = X,

                target_feature = dict_target_features[alias],

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

    

    return {

        "train":df_train,

        "test":df_test

        }
df_housing_tsv = pd.read_csv(filepath_or_buffer = "../input/ames-iowa-housing-tsv/AmesHousingTabSep.tsv",

                         sep = '\t',

                         index_col = "Order")
df_housing_tsv.head()
df_housing = pd.read_csv(filepath_or_buffer = "../input/housing/AmesHousing.csv",  

                         index_col = "Order")
df_housing.head()
original_shape = df_housing.shape

original_shape
df_housing.info()
# Features below will not be available for unsold houses. They might also "leak" information that would not be present in

# actual data sets of unsold houses.

leakers = [

    "Mo Sold",

    "Yr Sold",

    "Sale Type",

    "Sale Condition"

]

df_housing.drop(columns = leakers, inplace = True)
cols = df_housing.columns.values

cols.sort()

cols
lst_true_integers = [    

    "Year Built",

    "Year Remod/Add",    

    "Bsmt Full Bath",

    "Bsmt Half Bath",

    "Full Bath",

    "Half Bath",

    "Bedroom AbvGr",

    "Kitchen AbvGr",

    "TotRms AbvGrd",

    "Fireplaces",

    "Garage Yr Blt",

    "Garage Cars",    

]

lst_true_integers.sort()
lst_true_floats = [

    "Lot Frontage",

    "Lot Area",

    "Mas Vnr Area",

    "BsmtFin SF 1",

    "BsmtFin SF 2",

    "Bsmt Unf SF",

    "Total Bsmt SF",

    "1st Flr SF",

    "2nd Flr SF",

    "Low Qual Fin SF",

    "Gr Liv Area",

    "Garage Area",

    "Wood Deck SF",

    "Open Porch SF",

    "Pool Area",

    "Enclosed Porch",

    "3Ssn Porch",

    "Screen Porch",

    "Misc Val"

]

lst_true_floats.sort()
df_housing.info()
#Columns with suspiciously low number of non-null values/too many null values.

#When run, columns that do have values return several non-null values in the resulting dictionary.

#To be studied later.



# null_counts = df_housing.isnull().sum()

# print("Null Counts from `df.isnull().sum()`:\n", type(null_counts), null_counts, "\n=====")

# dict_null_counts = dict(zip(null_counts.index.values, null_counts.values))

# print("Results of `null_counts.index.values:\n", type(null_counts.index.values), null_counts, "\n=====")

# print("Results of `null_counts.values:\n", type(null_counts.index.values), null_counts, "\n=====")

# for column, null_count in dict_null_counts.items():

#     if null_count > 0:

#         print(column, ":\t", null_count)
#Columns with suspiciously low number of non-null values/too many null values.

null_counts = df_housing.isnull().sum()

dict_nulls = dict(zip(null_counts.keys(), null_counts.values))

for feature, null_count in dict_nulls.items():

    print(feature, ':\t', null_count)
df_housing.info()
lst_have_nulls = []

for feature in df_housing.columns.values.tolist():

    nullcount = df_housing[feature].isnull().sum()

    if nullcount > 0:

        lst_have_nulls.append(feature)

        print(feature, "\n=====\nNull Count:\t", nullcount,'\n*****')
dict_traintest = make_traintest(df = df_housing)

df_train = dict_traintest["train"]

df_test = dict_traintest["test"]
from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df_housing_tsv, train_size = 0.70, test_size = 0.30, random_state = 88)
df_train.head()
df_train.head()
drop_columns = df_train.columns[df_train.isna().sum() / df_train.shape[0] > 0.05]

df_train.drop(drop_columns, axis=1)



num_columns = []

cat_columns = []



for col, dtype in df_train.dtypes.iteritems():

    if dtype in [np.int64, np.float64]:

        num_columns.append(col)

    else:

        cat_columns.append(col)

        

df_train[num_columns] = df_train[num_columns].fillna(df_train[num_columns].mean())

df_train[cat_columns] = df_train[cat_columns].fillna(df_train[cat_columns].mode())
df_train.head()
row_count = df_train.shape[0]

sr_null_counts = df_train.isnull().sum()

lst_drop = []

end_decorator = "\n******"

for feature in lst_have_nulls:

    threshold = 0.05

    null_percent = (sr_null_counts[feature] / row_count)

    

    if null_percent > 0.05:

        print("To Drop: ", feature, type(df_train[feature]), type(df_train[feature].iloc[0]) )

        lst_drop.append(feature)

    else:

        print("Imputed:", str(type(df_train[feature]) ), type(df_train[feature].iloc[0]) )

        data = df_train[feature]

        df_train[feature] = impute_series(sr_values = data, feature_name = feature)

        

    print(end_decorator)

    

df_train.drop(columns = lst_drop, inplace = True)    
df_train.head()
lst_have_nulls.sort()

lst_have_nulls
df_train.info()