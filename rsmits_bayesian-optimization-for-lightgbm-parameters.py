# Import Modules

import pandas as pd

import numpy as np

import gc

import random

import lightgbm as lgbm



import warnings

warnings.filterwarnings("ignore")



import os

print(os.listdir("../input"))
# Import modules specific for Bayesian Optimization

from bayes_opt import BayesianOptimization

from bayes_opt.logger import JSONLogger

from bayes_opt.event import Events
# Specify some constants

seed = 4249

folds = 5

number_of_rows = 1000000
# Select Features

features = ['AVProductStatesIdentifier',

            'AVProductsInstalled', 

            'Census_ProcessorModelIdentifier',

            'Census_TotalPhysicalRAM',

            'Census_PrimaryDiskTotalCapacity',

            'EngineVersion',

            'Census_SystemVolumeTotalCapacity',

            'Census_InternalPrimaryDiagonalDisplaySizeInInches',

            'Census_OSBuildRevision',

            'AppVersion',

            'Census_OEMNameIdentifier',

            'Census_InternalPrimaryDisplayResolutionVertical',

            'Census_ProcessorCoreCount',

            'Census_OEMModelIdentifier',

            'CountryIdentifier',

            'LocaleEnglishNameIdentifier',

            'GeoNameIdentifier',

            'Census_InternalPrimaryDisplayResolutionHorizontal',

            'IeVerIdentifier',

            'HasDetections']
# Load Data with selected features

X = pd.read_csv('../input/train.csv', usecols = features, nrows = number_of_rows)
# Labels

Y = X['HasDetections']



# Remove Labels from Dataframe

X.drop(['HasDetections'], axis = 1, inplace = True)
# Factorize Some Columns

X['EngineVersion'] = pd.to_numeric(pd.factorize(X['EngineVersion'])[0])

X['AppVersion'] = pd.to_numeric(pd.factorize(X['AppVersion'])[0])
# Final Data Shapes

print(X.shape)

print(Y.shape)
# Create LightGBM Dataset

lgbm_dataset = lgbm.Dataset(data = X, label = Y)
# Specify LightGBM Cross Validation function

def lgbm_cv_evaluator(learning_rate, num_leaves, feature_fraction, bagging_fraction, max_depth):

    # Setup Parameters

    params = {  'objective':            'binary',

                'boosting':             'gbdt',

                'num_iterations':       1250, 

                'early_stopping_round': 100, 

                'metric':               'auc',

                'verbose':              -1

            }

    params['learning_rate'] =       learning_rate

    params['num_leaves'] =          int(round(num_leaves))

    params['feature_fraction'] =    feature_fraction

    params['bagging_fraction'] =    bagging_fraction

    params['max_depth'] =           int(round(max_depth))

        

    # Run LightGBM Cross Validation

    result = lgbm.cv(params, lgbm_dataset, nfold = folds, seed = seed, 

                     stratified = True, verbose_eval = -1, metrics = ['auc']) 

    

    # Return AUC

    return max(result['auc-mean'])
def display_progress(event, instance):

    iter = len(instance.res) - 1

    print('Iteration: {} - AUC: {} - {}'.format(iter, instance.res[iter].get('target'), instance.res[iter].get('params')))
def bayesian_parameter_optimization(init_rounds = 1, opt_rounds = 1):    

    

    # Initialize Bayesian Optimization

    optimizer = BayesianOptimization(f = lgbm_cv_evaluator, 

                                    pbounds = { 'learning_rate':        (0.02, 0.06),

                                                'num_leaves':           (20, 100),

                                                'feature_fraction':     (0.25, 0.75),

                                                'bagging_fraction':     (0.75, 0.95),

                                                'max_depth':            (8, 15) },

                                    random_state = seed, 

                                    verbose = 2)

    

    # Subscribe Logging to file for each Optimization Step

    logger = JSONLogger(path = 'parameter_output.json')

    optimizer.subscribe(Events.OPTMIZATION_STEP, logger)

    

    # Subscribe the custom display_progress function for each Optimization Step

    optimizer.subscribe(Events.OPTMIZATION_STEP, " ", display_progress)



    # Perform Bayesian Optimization. 

    # Modify acq, kappa and xi to change the behaviour of Bayesian Optimization itself.

    optimizer.maximize(init_points = init_rounds, n_iter = opt_rounds, acq = "ei", kappa = 2, xi = 0.1)

    

    # Return Found Best Parameter values and Target

    return optimizer.max
# Configure and Perform Bayesian Optimization 

max_params = bayesian_parameter_optimization(init_rounds = 15, opt_rounds = 15)



print('================= Results')

print('Found Max AUC: {} with the following Parameters: '.format(max_params.get('target')))

print(max_params.get('params'))