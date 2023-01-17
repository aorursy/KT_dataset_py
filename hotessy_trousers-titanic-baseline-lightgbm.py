import numpy as np 

import pandas as pd 

from path import Path

import os

import lightgbm as lgb

from tqdm.notebook import tqdm

import warnings

from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score, classification_report, confusion_matrix

from sklearn.model_selection import KFold, StratifiedKFold

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from sklearn.model_selection import train_test_split

from hyperopt.pyll import scope





warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', 1000)

tqdm.pandas()

np.random.seed(592)

base_path = Path('/kaggle/input/titanic/')
features = {

    'Pclass': 'category', 

    'Sex': 'category', 

    'Age': float, 

    'SibSp': 'category',

    'Parch': 'category', 

    'Fare': float, 

    'Embarked': 'category'}



target = 'Survived'
submission = pd.read_csv(base_path/'gender_submission.csv')

train = pd.read_csv(base_path/'train.csv', dtype=features, usecols=list(features.keys()) + [target])

test = pd.read_csv(base_path/'test.csv', dtype=features, usecols=features.keys())
class Base_Model(object):

    

    def __init__(self, train_df, test_df, features, evaluator, params=None, categoricals=[], n_splits=5, verbose=True):

        self.train_df = train_df

        self.test_df = test_df

        self.features = features

        self.n_splits = n_splits

        self.categoricals = categoricals

        self.target = 'Survived'

        self.cv = self.get_cv()

        self.verbose = verbose

        self.params = params # if params not None else self.get_params()

        self.evaluator = evaluator

#         self.oof_pred, self.y_pred, self.score, self.model = self.fit()

        

    def __call__(self):

        self.oof_pred, self.y_pred, self.score, self.model = self.fit()

        return self

        

    def train_model(self, train_set, val_set):

        raise NotImplementedError

        

    def get_cv(self):

        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True)

        return cv.split(self.train_df, self.train_df[self.target])

    

    def get_params(self):

        raise NotImplementedError

        

    def convert_dataset(self, x_train, y_train, x_val, y_val):

        raise NotImplementedError

        

    def convert_x(self, x):

        return x

        

    def fit(self):

        oof_pred = np.zeros((len(self.train_df), 2))

        y_pred = np.zeros((len(self.test_df), 2))

        for fold, (train_idx, val_idx) in enumerate(self.cv):

            x_train, x_val = self.train_df[self.features].iloc[train_idx], self.train_df[self.features].iloc[val_idx]

            y_train, y_val = self.train_df[self.target][train_idx], self.train_df[self.target][val_idx]

            train_set, val_set = self.convert_dataset(x_train, y_train, x_val, y_val)

            model = self.train_model(train_set, val_set)

            conv_x_val = self.convert_x(x_val)

            oof_pred[val_idx] = model.predict(conv_x_val).reshape(oof_pred[val_idx].shape)

            x_test = self.convert_x(self.test_df[self.features])

            y_pred += model.predict(x_test).reshape(y_pred.shape) / self.n_splits

#             print(f'Partial score of fold {fold} is: {self.evaluator(y_val, oof_pred[val_idx])}')

            

        loss_score = self.evaluator(self.train_df[self.target], oof_pred)



        if self.verbose:

            print(f'oof {self.evaluator.__name__} score is {loss_score}')

        return oof_pred, y_pred, loss_score, model
class Lgb_Model(Base_Model):

    

    def train_model(self, train_set, val_set):

        verbosity = 100 if self.verbose else 0

        return lgb.train(self.params, train_set, valid_sets=[train_set, val_set], verbose_eval=verbosity)

        

    def convert_dataset(self, x_train, y_train, x_val, y_val):

        train_set = lgb.Dataset(x_train, y_train, categorical_feature=self.categoricals)

        val_set = lgb.Dataset(x_val, y_val, categorical_feature=self.categoricals)

        return train_set, val_set
def hyperopt(param_space, X_train, y_train, X_test, y_test, categoricals, num_eval):

    

    train_set = lgb.Dataset(X_train, y_train, categorical_feature=categoricals, free_raw_data=False)

    val_set = lgb.Dataset(X_test, y_test, categorical_feature=categoricals, free_raw_data=False)

    

    def objective_function(params):

        clf = lgb.train(params, train_set)

        y_pred = clf.predict(X_test)

        score = f1_score(y_test, y_pred.argmax(axis=1))

        return {'loss': 1-score, 'status': STATUS_OK}



    trials = Trials()

    best_param = fmin(objective_function, 

                      param_space, 

                      algo=tpe.suggest, 

                      max_evals=num_eval, 

                      trials=trials)

    

#     loss = [x['result']['loss'] for x in trials.trials]

#     print("##### Results")

#     print("Score best parameters: ", min(loss)*-1)

#     print("Best parameters: ", best_param)

#     print("Parameter combinations evaluated: ", num_eval)

    

    return trials, best_param
max_depth = scope.int(hp.quniform('max_depth', 5, 30, 1))

bagging_fraction = hp.uniform('bagging_fraction', 0.6, 1.0)

feature_fraction = hp.uniform('feature_fraction', 0.6, 1.0)

boosting = hp.choice('boosting', ['gbdt', 'dart'])

learning_rate = hp.loguniform('learning_rate', np.log(0.01), np.log(1))

num_leaves = scope.int(hp.quniform('num_leaves', 5, 50, 1))

n_estimators = scope.int(hp.quniform('n_estimators', 100, 500, 5))



param_hyperopt= {

    'objective': 'multiclass',

    'num_classes': 2,

    'learning_rate': learning_rate,

    'max_depth': max_depth,

    'n_estimators': n_estimators,

    'num_leaves': num_leaves,

    'bagging_fraction': bagging_fraction,

    'feature_fraction': feature_fraction,

    'boosting': boosting,

    'verbosity': -1

}
x_train, x_val, y_train, y_val = train_test_split(

    train.drop(columns='Survived'),

    train['Survived'],

    test_size=0.15

)

categoricals = list(train.dtypes[train.dtypes == 'category'].index)



num_eval = 100

results_hyperopt, para = hyperopt(param_hyperopt, x_train, y_train.astype(int), x_val, y_val.astype(int), categoricals, num_eval)
def process_params(param):

    new_param = {}

    dtypes = {

        'bagging_fraction': float,

        'boosting': str,

        'feature_fraction': float,

        'learning_rate': float,

        'max_depth': int,

        'n_estimators': int,

        'num_leaves': int

    }

        

    for k, v in param.items():

        new_param[k] = dtypes[k](param[k])

        

    new_param['boosting'] = 'gbdt' if param['boosting'] == 0 else 'dart'         

    return new_param
evaluator = lambda y_true, y_pred: f1_score(y_true, y_pred.argmax(axis=1))

params = {**process_params(para), **{'objective': 'multiclass', 'num_classes': 2}}

print(params)



lgb_model = Lgb_Model(train, test, features.keys(), categoricals=categoricals, params=params, evaluator=evaluator, verbose=True, n_splits=2)()

final_pred = lgb_model.y_pred.argmax(axis=1)
submission['Survived'] = final_pred
submission.to_csv('submission.csv', index=False)