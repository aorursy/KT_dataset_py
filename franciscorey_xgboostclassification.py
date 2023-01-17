import pickle

from datetime import datetime



import numpy as np

import pandas as pd

pd.options.display.float_format = "{:,.4f}".format

import matplotlib.pyplot as plt

plt.style.use('seaborn')
DATA_FRACTION = 1.0



data = pd.read_csv("/kaggle/input/projetmachinelearning/cac40_v3.csv").drop('Unnamed: 0', axis=1)

data = data.sample(frac=DATA_FRACTION, axis='index', random_state=1).sort_values(

                                                                        by=['TICKER', 'annee', 'mois', 'jour']).reset_index(drop=True)

data
LABEL = "RDMT_M"



numeric_dependent_variables = ["RDMT_J", "RDMT_S", "RDMT_M"]

numeric_exogenous_variables = ["OP", "UP", "DO", "CL", "VO"] + [f"{var}_{x}" for x in ["J", "S", "M"] for var in ["HISTO", "VOL", "UP", "DO"]]

numeric_variables = numeric_dependent_variables + numeric_exogenous_variables

non_numeric_variables = data.columns[~data.columns.isin(numeric_variables)]

words = non_numeric_variables[4:]

descriptive_variables = non_numeric_variables[:4]
from collections import namedtuple



MIN_WORD_COUNT = 100 * DATA_FRACTION



Row = namedtuple('Row', ['nb_appearances', 'average_return'])



frequent_word_returns = {}

for word in words:

    if data[word].sum() > MIN_WORD_COUNT:

        mean_return = np.mean(data[LABEL].loc[data[word] == 1])

        num_app = data[word].sum()

        if mean_return > 0.01:

            frequent_word_returns[word] = Row(num_app, mean_return)



frequent_word_returns = pd.DataFrame(data=frequent_word_returns.values(), index=frequent_word_returns.keys())

frequent_word_returns
keepers = list(frequent_word_returns.index)

data["FLAG_KEEP"] = data[keepers].sum(axis=1)

data = data.loc[~(data["FLAG_KEEP"]==0)].drop("FLAG_KEEP", axis=1).reset_index(drop=True)
data.head()
data.shape
CORRELATION_THRESHOLD = 0.75



correlation_matrix = data.loc[:,~data.columns.isin(["TICKER", "annee", "mois", "jour"])].corr().abs()

upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool))

high_corr_vars = [col for col in upper_triangle.columns if any(upper_triangle[col] > CORRELATION_THRESHOLD)]
data.drop(high_corr_vars, axis=1, inplace=True)

data
data.shape
LABEL_THRESH = 0.02



X = data.loc[:, ~data.columns.isin(numeric_dependent_variables + ["TICKER"])].copy()

y = data[LABEL].copy()

y = pd.DataFrame({'y':[1 if v > LABEL_THRESH else -1 for v in y]})
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train.head()
from random import choice

from scipy import stats

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score, precision_score, recall_score

from sklearn.preprocessing import normalize

from xgboost.sklearn import XGBClassifier

from imblearn.over_sampling import SMOTE



def mean_encode(X:pd.DataFrame, y:pd.Series, feature:str) -> pd.Series:

    """ function computes the mean encoder to apply to the feature 

    """

    tmp = X.merge(y, right_index=True, left_index=True)

    mean_encoder = tmp.groupby(feature).apply(np.mean)["y"].copy()

    return mean_encoder



def classification_metrics(type_model:str, parameters:dict,

                          y_cv:list, predictions:list) -> tuple:

    """ function computes classification metrics AUC, Precision and Recall

        for given classification results

    """

    auc = round(roc_auc_score(y_cv, predictions), 4)

    precision = round(precision_score(y_cv, predictions), 4)

    recall = round(recall_score(y_cv, predictions), 4)

    return ([{'algorithm':type_model,

             'parameters':parameters,

             'metrics':{

                 'auc':auc,

                 'precision':precision,

                 'recall':recall

             }}])

                       

def handle_outliers(data:pd.DataFrame, features:list, how:str) -> pd.DataFrame:

    """ function handles the outliers in the specified features either by 

        capping or removal of observations

    """

    if how=='remove':

        mask = np.where(abs(stats.zscore(data[features])) > 3)

        return data.reset_index().drop(mask[0], axis=0).set_index('index')

    elif how=='cap':

        data.loc[:,features].clip(lower=data[features].quantile(0.25),

                                  upper=data[features].quantile(0.75), 

                                  inplace=True, axis='columns')

        return data

                     

def tune_XGB_hyper_params(hyper_parameter_grid:list, X:pd.DataFrame, 

                          y:pd.DataFrame, num_fold:int, num_trials:int) -> dict:

    """ function performs a Random Grid Search for 

        the best hyper-parameters and returns a dictionnary containing the tuning results

        

        Parameters

        ----------

        hyper_parameter_grid: list of hyper-parameter lists

            - [0]: learning_rates

            - [1]: max_depth

            - [2]: n_estimators

            - [3]: subsample

        

    """

    results = []

    kf = KFold(n_splits=num_fold)

    for n in range(num_trials):

        hyper_param_set = choice(hyper_parameter_grid)

        

        param_learning_rate = hyper_param_set[0]

        param_max_depth = hyper_param_set[1]

        param_n_estimators = hyper_param_set[2]

        param_subsample = hyper_param_set[3]

        param_colsample_bytree = hyper_param_set[4]

        param_colsample_bylevel = hyper_param_set[5]

        

        y_cv = []

        predictions = []

        for train_index, valid_index in kf.split(X, y):

            # Define model

            xgb = XGBClassifier(booster='gbtree', objective='binary:logistic', njobs=4,

                                learning_rate=param_learning_rate,

                                max_depth=param_max_depth,

                                n_estimators=param_n_estimators,

                                subsample=param_subsample,

                                colsample_bytree=param_colsample_bytree, 

                                colsample_bylevel=param_colsample_bylevel

                                )

            

            # Select training data of this fold

            X_train, X_valid = X.iloc[train_index].copy(), X.iloc[valid_index].copy()

            y_train, y_valid = y.iloc[train_index].copy(), y.iloc[valid_index].copy()

            

            

            # Apply preprocessing steps to data

            #     1) handle the outliers found in volatility

            #X_train = handle_outliers(X_train, VOL_PARAMS, 'remove')

            #y_train = y_train.loc[y_train.index.isin(X_train.index)]

            #X_valid = handle_outliers(X_valid, VOL_PARAMS, 'remove')

            #y_valid = y_valid.loc[y_valid.index.isin(X_valid.index)]

            

            #     2) mean encode the year, month and dates (learning only from training set): 

            #          -> +5% AUC for dates, -2% AUC for tickers

            year_encoder = mean_encode(X_train, y_train, "annee")

            month_encoder = mean_encode(X_train, y_train, "mois")

            day_encoder = mean_encode(X_train, y_train, "jour")

            

            def apply_encoding(x, encoder):

                try:

                    return encoder[x]

                except KeyError as e:

                    print(e)

                    return encoder.mean()

                

            X_train.loc[:,"annee"] = X_train.loc[:,"annee"].apply(lambda x: apply_encoding(x, year_encoder))

            X_train.loc[:,"mois"] = X_train.loc[:,"mois"].apply(lambda x: apply_encoding(x, month_encoder))

            X_train.loc[:,"jour"] = X_train.loc[:,"jour"].apply(lambda x: apply_encoding(x, day_encoder))

            

            X_valid.loc[:,"annee"] = X_valid.loc[:,"annee"].apply(lambda x: apply_encoding(x, year_encoder))

            X_valid.loc[:,"mois"] = X_valid.loc[:,"mois"].apply(lambda x: apply_encoding(x, month_encoder))

            X_valid.loc[:,"jour"] = X_valid.loc[:,"jour"].apply(lambda x: apply_encoding(x, day_encoder))

            

            #      3) Normalize input variables

            #X_train.loc[:,X_train.columns.isin(numeric_exogenous_variables)] = \

            #            normalize(X_train.loc[:,X_train.columns.isin(numeric_exogenous_variables)])

            #X_valid.loc[:,X_valid.columns.isin(numeric_exogenous_variables)] = \

            #            normalize(X_valid.loc[:,X_valid.columns.isin(numeric_exogenous_variables)])

            

            #      4) rebalance the label by SMOTE over-sampling of training data (do not touch test data)

            oversample = SMOTE()

            X_train, y_train = oversample.fit_resample(X_train, y_train)

            

            # fit the model and make cross-validation predictions

            xgb_model = xgb.fit(X_train, y_train.y)

            pred_valid = xgb_model.predict(X_valid)

            

            # store cross-validation results

            y_cv = y_cv + y_valid.y.to_list()

            predictions = predictions + list(pred_valid)

            

        results = results + classification_metrics("XGBClassifier",

                                                  {'learning_rate':param_learning_rate,

                                                  'max_depth':param_max_depth,

                                                  'n_estimators':param_n_estimators,

                                                  'subsample':param_subsample,

                                                  'colsample_bytree':param_colsample_bytree,

                                                  'colsample_bylevel':param_colsample_bylevel},

                                                  y_cv, predictions)

    return results

            
import itertools



hyper_learning_rate = np.logspace(-4,-2, 4)

hyper_max_depth = [8, 9, 10, 11, 12]

hyper_n_estimators = [100, 200, 500]

hyper_subsample = [0.7, 0.8, 0.9, 1]

hyper_colsample_bytree = [0.7, 0.8, 0.9, 1]

hyper_colsample_bylevel = [0.7, 0.8, 0.9, 1]



hyper_parameter_list = [hyper_learning_rate, hyper_max_depth, 

                        hyper_n_estimators, hyper_subsample, 

                        hyper_colsample_bytree, hyper_colsample_bylevel]

hyper_parameter_grid = list(itertools.product(*hyper_parameter_list))

len(hyper_parameter_grid)
#VOL_PARAMS = ["VO"] + [f"VOL_{x}" for x in ["J", "S", "M"]]



# We don't train on outliers, but we test on them since we haven't defined a strategy for innovation detection

#X_train = handle_outliers(X_train, VOL_PARAMS, 'remove')

#y_train = y_train.loc[y_train.index.isin(X_train.index)] --> Didn't work



results = tune_XGB_hyper_params(hyper_parameter_grid, X_train, y_train, 4, 500)
with open("/kaggle/working/results_4_200_outlier_removal_before_tune.pickle", "wb") as file:

    pickle.dump(results, file)



results[np.argmax([x['metrics']['auc'] for x in results])]
def plot_tuning_results(results:list, param_name:str):

    ''' Plots the results of hyper parameter tuning for required parameter

    '''

    auc = [res['metrics']['auc'] for res in results]

    precision = [res['metrics']['precision'] for res in results]

    recall = [res['metrics']['recall'] for res in results]

    param = [res['parameters'][param_name] for res in results]



    fig, ax = plt.subplots(1,3, figsize=(15,5))

    ax[0].scatter(param, auc)

    ax[0].set_xlabel(param_name)

    ax[0].set_ylabel('Validation AUC')

    ax[1].scatter(param, precision)

    ax[1].set_xlabel(param_name)

    ax[1].set_ylabel('Validation Precision')

    ax[2].scatter(param, recall)

    ax[2].set_xlabel(param_name)

    ax[2].set_ylabel('Validation Recall');



    return
plot_tuning_results(results, 'learning_rate')
plot_tuning_results(results, 'max_depth')
plot_tuning_results(results, 'n_estimators')
plot_tuning_results(results, 'subsample')
auc = [x['metrics']['auc'] for x in results]

recall = [x['metrics']['recall'] for x in results]

precision = [x['metrics']['precision'] for x in results]

metrics = pd.DataFrame({'auc':auc, 'recall':recall, 'precision':precision})

metrics.head()
import plotly.express as px



fig = px.scatter_3d(metrics, x='recall', y='precision', z='auc')



fig.update_layout(title="Classification metrics")

fig