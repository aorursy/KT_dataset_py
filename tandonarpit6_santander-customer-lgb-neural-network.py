import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import os

print(os.listdir("../input"))



%matplotlib inline



import gc

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
# Look at first 10 records of the train dataset

train.head(n=10).T
# Check out the shape of the train and test sets

print('Train:', train.shape)

print('Test:', test.shape)
# Check the target variable distribution

train['target'].value_counts()
# Imports for Modeling



from sklearn.model_selection import KFold, train_test_split

from sklearn.metrics import roc_curve, roc_auc_score, auc, confusion_matrix, classification_report

import lightgbm as lgb
# Target variable from the Training Set

Target = train['target']



# Input dataset for Train and Test 

train_inp = train.drop(columns = ['target', 'ID_code','var_10','var_124','var_185','var_103','var_7','var_129','var_17','var_16',

                                  'var_117','var_161','var_100','var_96','var_30','var_136','var_27','var_98','var_29','var_38','var_183','var_182',

                                 'var_158','var_41','var_126','var_73','var_160','var_46','var_189','var_39','var_79','var_47',

                                 'var_69','var_176','var_42','var_101','var_84','var_3','var_61','var_19','var_59','var_37'])

X_test = test.drop(columns = ['ID_code','var_10','var_124','var_185','var_103','var_7','var_129','var_17','var_16',

                                  'var_117','var_161','var_100','var_96','var_30','var_136','var_27','var_98','var_29','var_38','var_183','var_182',

                             'var_158','var_41','var_126','var_73','var_160','var_46','var_189','var_39','var_79','var_47',

                                 'var_69','var_176','var_42','var_101','var_84','var_3','var_61','var_19','var_59','var_37'])



train= train.drop(columns=['ID_code','var_10','var_124','var_185','var_103','var_7','var_129','var_17','var_16',

                                  'var_117','var_161','var_100','var_96','var_30','var_136','var_27','var_98','var_29','var_38','var_183','var_182',

                          'var_158','var_41','var_126','var_73','var_160','var_46','var_189','var_39','var_79','var_47',

                                 'var_69','var_176','var_42','var_101','var_84','var_3','var_61','var_19','var_59','var_37'])

test= test.drop(columns=['ID_code','var_10','var_124','var_185','var_103','var_7','var_129','var_17','var_16',

                                  'var_117','var_161','var_100','var_96','var_30','var_136','var_27','var_98','var_29','var_38','var_183','var_182',

                        'var_158','var_41','var_126','var_73','var_160','var_46','var_189','var_39','var_79','var_47',

                                 'var_69','var_176','var_42','var_101','var_84','var_3','var_61','var_19','var_59','var_37'])



# List of feature names

features = list(train_inp.columns)
# Split the Train Dataset into training and validation sets for model building. 



X_train, X_val, Y_train, Y_val = train_test_split(train_inp, Target, test_size= 0.1)
# check the split of train and validation

print('Train:',X_train.shape)

print('Validation:',X_val.shape)
#custom function to build the LightGBM model.

def run_lgb(X_train, Y_train, X_val, Y_val, X_test):

    params = {

        "objective" : "binary",

        "metric" : "auc",

        "num_leaves" : 600,

        "learning_rate" : 0.01,

        "verbosity" : -1,

        "boosting":"gbdt",

        "max_depth":-1,

        "scale_pos_weight":2

    }

    

    lgtrain = lgb.Dataset(X_train, label=Y_train)

    lgval = lgb.Dataset(X_val, label=Y_val)

    evals_result = {}

    model_lgb = lgb.train(params, lgtrain, 2800 , valid_sets=[lgval], 

                      early_stopping_rounds=100, verbose_eval=100, evals_result=evals_result)

    

    Y_pred_lgb = model_lgb.predict(X_val, num_iteration=model_lgb.best_iteration)

    return Y_pred_lgb, model_lgb, evals_result
# Training the model 

Y_pred_lgb, model_lgb, evals_result = run_lgb(X_train, Y_train, X_val, Y_val, X_test)
# Extract feature importances

feature_importance_values = model_lgb.feature_importance()

feature_importances = pd.DataFrame({'feature': features, 'importance': feature_importance_values})

feature_importances.sort_values(by='importance', ascending=True)
# Submission dataframe



Y_pred_lgb= model_lgb.predict(X_test)



submit_file_lgb = pd.read_csv('../input/sample_submission.csv')

submit_file_lgb['target'] = Y_pred_lgb

submit_file_lgb.to_csv('Light GBM.csv', index=False)



print ("Light GBM prediction file successfully generated.")
from fastai.tabular import *



procs = [Normalize]

valid_idx = range(len(train)- 10000, len(train))

data = TabularDataBunch.from_df(path = '.',df=train,dep_var='target',valid_idx = valid_idx, procs = procs,test_df=test)
learn = tabular_learner(data,layers=[150,100],metrics=accuracy)

learn.fit_one_cycle(16,0.001)



#learn.lr_find()

#learn.recorder.plot()
test_predicts = learn.get_preds(ds_type=DatasetType.Test)

Y_pred_nn = to_np(test_predicts[0])

Y_pred_nn = Y_pred_nn[:,1]
submit_file_nn = pd.read_csv('../input/sample_submission.csv')

submit_file_nn['target'] = Y_pred_nn

submit_file_nn.to_csv('Neural Network.csv', index=False)



print ("Neural Network prediction file successfully generated.")
Y_pred = (0.5 * Y_pred_nn) + (0.5 * Y_pred_lgb)



submit_file = pd.read_csv('../input/sample_submission.csv')

submit_file['target'] = Y_pred

submit_file.to_csv('LGB and NN.csv', index=False)



print ("Combination of LGB and Neural Network prediction file successfully generated.")
gc.collect()