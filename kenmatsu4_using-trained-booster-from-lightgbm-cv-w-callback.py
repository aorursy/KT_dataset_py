#!/usr/bin/env python

# -*- coding: utf-8 -*-



import lightgbm as lgb

from sklearn import datasets

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

import numpy as np

import pandas as pd

from sklearn.model_selection import StratifiedKFold, KFold

import numpy.random as rd
class ModelExtractionCallback(object):

    """Callback class for retrieving trained model from lightgbm.cv()

    NOTE: This class depends on '_CVBooster' which is hidden class, so it might doesn't work if the specification is changed.

    """



    def __init__(self):

        self._model = None



    def __call__(self, env):

        # Saving _CVBooster object.

        self._model = env.model



    def _assert_called_cb(self):

        if self._model is None:

            # Throw exception if the callback class is not called.

            raise RuntimeError('callback has not called yet')



    @property

    def boosters_proxy(self):

        self._assert_called_cb()

        # return Booster object

        return self._model



    @property

    def raw_boosters(self):

        self._assert_called_cb()

        # return list of Booster

        return self._model.boosters



    @property

    def best_iteration(self):

        self._assert_called_cb()

        # return boosting round when early stopping.

        return self._model.best_iteration



def loading_dataset():

    # Loading Iris Dataset

    iris = datasets.load_iris()

    X, y = iris.data, iris.target



    # Split dataset for this demonstration.

    train, test, y_train, y_test = train_test_split(X, y,

                                                    shuffle=True,

                                                    random_state=42)

    return train, test, y_train, y_test
rd.seed(123)



# Loading Sample Dataset.

train, test, y_train, y_test = loading_dataset()



# one hot representation of y_train

max_class_num = y_train.max()+1

y_train_ohe = np.identity(max_class_num)[y_train]

    

# Create LightGBM dataset for train.

lgb_train = lgb.Dataset(train, y_train)



# Create callback class for retrieving trained model.

extraction_cb = ModelExtractionCallback()

callbacks = [

    extraction_cb,

]



# LightGBM parameter

lgbm_params = {

    'objective': 'multiclass',

    'num_class': 3,

    'verbosity' : 1,

}



# Training settings

FOLD_NUM = 5

fold_seed = 71

folds = StratifiedKFold(n_splits=FOLD_NUM, shuffle=True, random_state=fold_seed)



# Fitting

ret = lgb.cv(params=lgbm_params,

               train_set=lgb_train,

               folds=folds,

               num_boost_round=1000,

               verbose_eval = 10,

               early_stopping_rounds=10,

               callbacks=callbacks,

               )

df_ret = pd.DataFrame(ret)

df_ret
# Retrieving booster and training information.

proxy = extraction_cb.boosters_proxy

boosters = extraction_cb.raw_boosters

best_iteration = extraction_cb.best_iteration
# Create oof prediction result

fold_iter = folds.split(train, y_train)

oof_preds = np.zeros_like(y_train_ohe)

for n_fold, ((trn_idx, val_idx), booster) in enumerate(zip(fold_iter, boosters)):

    print(val_idx)

    valid = train[val_idx]

    oof_preds[val_idx] = booster.predict(valid, num_iteration=best_iteration)

print(f"accuracy on oof preds: {accuracy_score(y_train, np.argmax(oof_preds, axis=1))}")
# Averaging prediction result for test data.

y_pred_proba_list = proxy.predict(test, num_iteration=best_iteration)

y_pred_proba_avg = np.array(y_pred_proba_list).mean(axis=0)

y_pred = np.argmax(y_pred_proba_avg, axis=1)

accuracy = accuracy_score(y_test, y_pred)

print('Averaging accuracy:', accuracy)

# Predicting with test data of each CV separately.

for i, booster in enumerate(boosters):

    y_pred_proba = booster.predict(test,

                                   num_iteration=best_iteration)

    y_pred = np.argmax(y_pred_proba, axis=1)

    accuracy = accuracy_score(y_test, y_pred)

    print('Model {0} accuracy: {1}'.format(i, accuracy))