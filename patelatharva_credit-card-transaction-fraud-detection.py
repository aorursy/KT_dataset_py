# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import gc

import hyperopt

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from hyperopt import space_eval

import time

import math

from hyperopt.pyll.base import scope

from sklearn.model_selection import train_test_split

from tqdm import tqdm_notebook as tqdm

import lightgbm as lgb

import pprint

pp = pprint.PrettyPrinter(indent=4)

from imblearn.over_sampling import SMOTE

from sklearn.metrics import roc_auc_score, f1_score

from sklearn.model_selection import KFold
data_dir= "/kaggle/input/creditcardfraud"
df = pd.read_csv(data_dir + "/" + "creditcard.csv")
df.head()
input_cols = ["V" + str(x) for x in range(1,29)] + ["Amount"]
X = df[input_cols]
y = df["Class"]
y.value_counts()
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=7)
# Balance dataset with SMOTE

sm = SMOTE(random_state=7)

X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

X_train_bal = pd.DataFrame(X_train_bal, columns=input_cols)

y_train_bal = pd.Series(y_train_bal)
number_of_evals = 300

def find_best_params_for_lgb(X, y):

    evaluated_point_scores = {}

    

    def objective(params):

        garbage=gc.collect()

        if (str(params) in evaluated_point_scores):

            return evaluated_point_scores[str(params)]

        else:          

            kf = KFold(n_splits=2, random_state=7)

            scores = []

            for train_index, test_index in kf.split(X.values):                

                X_train, X_val = X.values[train_index], X.values[test_index]

                y_train, y_val = y.values.ravel()[train_index], y.values.ravel()[test_index]

            

                train_data = lgb.Dataset(X_train, 

                                label=y_train,

                                feature_name=list(X.columns),

                                )

                

                validation_data = lgb.Dataset(X_val, 

                                label=y_val,

                                feature_name=list(X.columns),

                                )

                

                evals_result = {}

                bst = lgb.train(params, train_data, 

                                valid_sets=[train_data, validation_data], 

                                valid_names=['train', 'val'], 

                                evals_result=evals_result, 

                                num_boost_round=10000,

                                early_stopping_rounds=100,

                                verbose_eval=None,

                               )



                y_val_preds = np.where(bst.predict(X_val) > 0.5, 1, 0)

                score = f1_score(y_val, y_val_preds)

                scores.append(score)

                

#             print("Evaluating params:")

#             pp.pprint(params)

            socre=np.mean(scores).item(0)

#             print("f1: " + str(score))

            evaluated_point_scores[str(params)] = -score

            return -score

    param_space = {

            'objective': hp.choice("objective", ["binary"]),        

            "max_depth": scope.int(hp.quniform("max_depth", 50, 60, 1)),

            "learning_rate": hp.choice("learning_rate", [0.2]),

            "num_leaves": scope.int(hp.quniform("num_leaves", 32, 1024, 10)),   

            "max_bin": scope.int(hp.quniform("max_bin", 50, 250, 10)),

            "bagging_fraction": hp.quniform('bagging_fraction', 0.70, 1.0, 0.05),

            "feature_fraction": hp.uniform("feature_fraction", 0.90, 1.0),

            "bagging_freq": hp.choice("bagging_freq", [1]),

            "lambda_l1": hp.quniform('lambda_l1', 1, 10, 1),        

            "lambda_l2": hp.quniform('lambda_l2', 1, 100, 5),

            "loss_function": hp.choice("loss_function", ["binary_error"]), 

            "eval_metric": hp.choice("eval_metric", ["binary_error"]),

            "metric": hp.choice("metric", ["binary_error"]),

            "random_state": hp.choice("random_state", [7]),

            "verbose": hp.choice("verbose", [None])

        }



    best_params = space_eval(

        param_space, 

        fmin(objective, 

             param_space, 

             algo=hyperopt.tpe.suggest,

             max_evals=number_of_evals))    

    

    

    # Finding best number of iterations with learning rate 0.1

    best_params["learning_rate"] = 0.1



    kf = KFold(n_splits=5)



    num_iterations_array = []

    for train_index, test_index in kf.split(X.values):                

        X_train, X_val = X.values[train_index], X.values[test_index]

        y_train, y_val = y.values.ravel()[train_index], y.values.ravel()[test_index]



        train_data = lgb.Dataset(X_train, 

                        label=y_train,

                        feature_name=list(X.columns),

                        )



        validation_data = lgb.Dataset(X_val, 

                        label=y_val,

                        feature_name=list(X.columns),

                        )



        evals_result = {}

        bst = lgb.train(best_params, train_data, 

                        valid_sets=[train_data, validation_data], 

                        valid_names=['train', 'val'], 

                        evals_result=evals_result, 

                        num_boost_round=10000,

                        early_stopping_rounds=100,

                        verbose_eval=None,

                       )



        num_iterations_array.append(bst.best_iteration)        



    best_params["num_iterations"] = int(np.mean(num_iterations_array).item(0))        

    print ("Best Hyperparameters found:")

    pp.pprint(best_params)

    return best_params
best_params = find_best_params_for_lgb(X=X_train_bal, y=y_train_bal)
train_data = lgb.Dataset(X_train_bal.values, 

                            label=y_train_bal.values.ravel(),

                            feature_name=list(X_train_bal.columns),

                        )
bst = lgb.train(best_params, train_data)
y_probs = bst.predict(X_test)
test_score = roc_auc_score(y_test, y_probs)
test_score
y_preds = np.where(y_probs > 0.5, 1, 0)
f1 = f1_score(y_test, y_preds)
f1