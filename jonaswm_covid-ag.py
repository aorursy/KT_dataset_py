!pip install mxnet autogluon
import pandas as pd

import numpy as np

import shutil, os

from autogluon import TabularPrediction as task



directory = '../input/covid19-global-forecasting-week-2/'



label_cases = 'ConfirmedCases' # name of target variable to predict in this competition

label_fatalities = 'Fatalities'

outputdir_cases = 'AGmodels_' + label_cases + '/' # where to store trained models

outputdir_fatalities = 'AGmodels_' + label_fatalities + '/' # where to store trained models



if os.path.exists(outputdir_cases):

    shutil.rmtree(outputdir_cases)



if os.path.exists(outputdir_fatalities):

    shutil.rmtree(outputdir_fatalities)



train_data = task.Dataset(file_path=directory+'train.csv')

train_data.drop(["Id"], axis=1, inplace=True)

log_cases_vals = np.log(train_data[label_cases] + 1)

log_fatalities_vals = np.log(train_data[label_fatalities] + 1)

train_data[label_fatalities] = log_fatalities_vals

train_data[label_cases] = log_cases_vals



train_data_cases = train_data.drop([label_fatalities], axis=1)

train_data_fatalities = train_data.drop([label_cases], axis=1)

train_data.head()
time_limits = 5 * 60 * 2

stack = False

num_bagging_folds = 10

hyperparams = {

    'NN': {'num_epochs': 500, 'dropout_prob': 0.2, 'weight_decay': 1e-5, 

           'activation': 'softrelu', 'epochs_wo_improve': 50, 'use_batchnorm': False,

           'layers': [2048], 'numeric_embed_dim': 2048,

           'y_range': (0.0,np.inf), 'y_range_extend': 1.0},

    'CAT': {'iterations': 10000},

    'GBM': {'num_boost_round': 10000},

}



predictor_cases = task.fit(train_data=train_data_cases, label=label_cases,

                           output_directory=outputdir_cases, problem_type='regression',

                           auto_stack=stack, num_bagging_folds=num_bagging_folds, 

                           time_limits=time_limits/2, hyperparameters=hyperparams)



predictor_fatalities = task.fit(train_data=train_data_fatalities, label=label_fatalities,

                                output_directory=outputdir_fatalities, problem_type='regression',

                                auto_stack=stack, num_bagging_folds=num_bagging_folds,

                                time_limits=time_limits/2, hyperparameters=hyperparams)
test_data = task.Dataset(file_path=directory+'test.csv')



pred_cases_log = predictor_cases.predict(test_data)

pred_fatalities_log = predictor_fatalities.predict(test_data)

pred_cases = np.exp(pred_cases_log) - 1.0

pred_fatalities = np.exp(pred_fatalities_log) - 1.0

pred_cases[pred_cases < 0.99] = 0

pred_fatalities[pred_fatalities < 1] = 0



"""

if os.path.exists(outputdir_cases):

    shutil.rmtree(outputdir_cases)



if os.path.exists(outputdir_fatalities):

    shutil.rmtree(outputdir_fatalities)

"""



submission = pd.read_csv(directory+'submission.csv')

submission[label_cases] = pred_cases

submission[label_fatalities] = pred_fatalities

submission.head()



submission_filename = 'submission.csv'

submission.to_csv(submission_filename, index=False)

print("Prediction file generated:")

submission.head()