# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import os
import gc

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install pycaret
# Use regression
from pycaret.regression import *

basedir = "/kaggle/input/trends-assessment-prediction/"
os.listdir(basedir)
#===========================================================
# Config
#===========================================================
OUTPUT_DICT = ''

ID = 'Id'
TARGET_COLS = ['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']
SEED = 42
# seed_everything(seed=SEED)

N_FOLD = 5
## Data Load
train = pd.read_csv(basedir+'train_scores.csv', dtype={'Id':str})\
            .dropna().reset_index(drop=True) # to make things easy
reveal_ID = pd.read_csv(basedir+'reveal_ID_site2.csv', dtype={'Id':str})
ICN_numbers = pd.read_csv(basedir+'ICN_numbers.csv')
loading = pd.read_csv(basedir+'loading.csv', dtype={'Id':str})
fnc = pd.read_csv(basedir+'fnc.csv', dtype={'Id':str})
sample_submission = pd.read_csv(basedir+'sample_submission.csv', dtype={'Id':str})
train.head()
reveal_ID.head()
ICN_numbers.head()
loading.head()
fnc.head()
sample_submission.head()
sample_submission['ID_num'] = sample_submission[ID].apply(lambda x: int(x.split('_')[0]))
test = pd.DataFrame({ID: sample_submission['ID_num'].unique().astype(str)})
del sample_submission['ID_num']; gc.collect()
test.head()
# merge
train = train.merge(loading, on=ID, how='left')
train = train.merge(fnc, on=ID, how='left')
train.head()
# merge
test = test.merge(loading, on=ID, how='left')
test = test.merge(fnc, on=ID, how='left')
test.head()
reg = {}
exp = {}
pred_dict = {}

%%time
TARGET = "age"
reg = {}

# Setting
x = train.drop([x for x in TARGET_COLS if x is not TARGET], axis=1)
x = x.drop(ID, axis=1)

exp[TARGET] = setup(x, target=TARGET,
                    train_size = 0.8,
                    normalize = True, normalize_method="minmax",
#                     feature_interaction = True,
                    combine_rare_levels = True, 
                    feature_selection = True, feature_selection_threshold=0.9,
                    remove_multicollinearity = True, multicollinearity_threshold=0.9)
%%time

model_names = ["br"]
for model in model_names:
#     reg[model] = create_model(model, fold=N_FOLD)
    reg[model] = tune_model(model, fold=N_FOLD, optimize='mse')
%%time
# generate predictions on holdout
predictions_holdout = predict_model(reg[model])

%%time
est= finalize_model(reg[model])
%%time
pred = predict_model(est, data=test.drop(ID, axis=1))
pred_dict[TARGET] = pred["Label"]
pred_dict[TARGET].to_csv('{}.csv'.format(TARGET), index=False)
print(pred_dict[TARGET])
pred_dict[TARGET].describe()
%%time

TARGET = "domain1_var1"
reg = {}

# Setting
x = train.drop([x for x in TARGET_COLS if x is not TARGET], axis=1)
x = x.drop(ID, axis=1)

exp[TARGET] = setup(x, target=TARGET,
                    train_size = 0.8,
                    normalize = True, normalize_method="minmax",
#                     feature_interaction = True,
                    combine_rare_levels = True, 
                    feature_selection = True, feature_selection_threshold=0.9,
                    remove_multicollinearity = True, multicollinearity_threshold=0.9)
%%time

model_names = ["br"]
for model in model_names:
#     reg[model] = create_model(model, fold=N_FOLD)
    reg[model] = tune_model(model, fold=N_FOLD, optimize='mse')
%%time
# generate predictions on holdout
predictions_holdout = predict_model(reg[model])
%%time
est= finalize_model(reg[model])
pred = predict_model(est, data=test.drop(ID, axis=1))
pred_dict[TARGET] = pred["Label"]
pred_dict[TARGET].to_csv('{}.csv'.format(TARGET), index=False)

print(pred_dict[TARGET])
pred_dict[TARGET].describe()
%%time

TARGET = "domain1_var2"
reg = {}

# Setting
x = train.drop([x for x in TARGET_COLS if x is not TARGET], axis=1)
x = x.drop(ID, axis=1)

exp[TARGET] = setup(x, target=TARGET,
                    train_size = 0.8,
                    normalize = True, normalize_method="minmax",
#                     feature_interaction = True,
                    combine_rare_levels = True, 
                    feature_selection = True, feature_selection_threshold=0.9,
                    remove_multicollinearity = True, multicollinearity_threshold=0.9)
%%time

model_names = ["br"]
for model in model_names:
#     reg[model] = create_model(model, fold=N_FOLD)
    reg[model] = tune_model(model, fold=N_FOLD, optimize='mse')

%%time
# generate predictions on holdout
predictions_holdout = predict_model(reg[model] )
%%time
est= finalize_model(reg[model])
pred = predict_model(est, data=test.drop(ID, axis=1))
pred_dict[TARGET] = pred["Label"]
pred_dict[TARGET].to_csv('{}.csv'.format(TARGET), index=False)

print(pred_dict[TARGET])
pred_dict[TARGET].describe()
%%time

TARGET = "domain2_var1"
reg = {}

# Setting
x = train.drop([x for x in TARGET_COLS if x is not TARGET], axis=1)
x = x.drop(ID, axis=1)

exp[TARGET] = setup(x, target=TARGET,
                    train_size = 0.8,
                    normalize = True, normalize_method="minmax",
#                     feature_interaction = True,
                    combine_rare_levels = True, 
                    feature_selection = True, feature_selection_threshold=0.9,
                    remove_multicollinearity = True, multicollinearity_threshold=0.9)
%%time

model_names = ["br"]
for model in model_names:
#     reg[model] = create_model(model, fold=N_FOLD)
    reg[model] = tune_model(model, fold=N_FOLD, optimize='mse')
%%time
# generate predictions on holdout
predictions_holdout = predict_model(reg[model])
est= finalize_model(reg[model])
pred = predict_model(est, data=test.drop(ID, axis=1))
pred_dict[TARGET] = pred["Label"]
pred_dict[TARGET].to_csv('{}.csv'.format(TARGET), index=False)

print(pred_dict[TARGET])
pred_dict[TARGET].describe()
%%time

TARGET = "domain2_var2"
reg = {}

# Setting
x = train.drop([x for x in TARGET_COLS if x is not TARGET], axis=1)
x = x.drop(ID, axis=1)

exp[TARGET] = setup(x, target=TARGET,
                    train_size = 0.8,
                    normalize = True, normalize_method="minmax",
#                     feature_interaction = True,
                    combine_rare_levels = True, 
                    feature_selection = True, feature_selection_threshold=0.9,
                    remove_multicollinearity = True, multicollinearity_threshold=0.9)

%%time

model_names = ["br"]
for model in model_names:
    reg[model] = tune_model(model, fold=N_FOLD, optimize='mse')
#     reg[model] = tune_model(model, fold=N_FOLD, optimize='mse')
    
%%time
# generate predictions on holdout
predictions_holdout = predict_model(reg[model])
est= finalize_model(reg[model])
pred = predict_model(est, data=test.drop(ID, axis=1))
pred_dict[TARGET] = pred["Label"]
pred_dict[TARGET].to_csv('{}.csv'.format(TARGET), index=False)

print(pred_dict[TARGET])
pred_dict[TARGET].describe()
pred_df = pd.DataFrame()

for TARGET in TARGET_COLS:
    tmp = pd.DataFrame()
    tmp[ID] = [f'{c}_{TARGET}' for c in test[ID].values]
    tmp['Predicted'] = pred_dict[TARGET]
    pred_df = pd.concat([pred_df, tmp])

print(pred_df.shape)
print(sample_submission.shape)

pred_df.head()
submission = sample_submission.drop(columns='Predicted').merge(pred_df, on=ID, how='left')
print(submission.shape)
submission.to_csv('/kaggle/working/submission.csv', index=False)
submission.head()
