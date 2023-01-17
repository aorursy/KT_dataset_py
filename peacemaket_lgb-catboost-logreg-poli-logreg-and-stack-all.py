import os

import pandas as pd

import numpy as np

import pickle as pkl

import dill

import category_encoders

import gc

import time



from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, PolynomialFeatures, OrdinalEncoder

from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit

from sklearn.pipeline import Pipeline

from sklearn.metrics import auc, precision_recall_curve, f1_score, recall_score, precision_score, accuracy_score, roc_auc_score

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier



from lightgbm import LGBMClassifier, Dataset

import lightgbm as lgb

from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

import catboost



if not os.path.exists('./best_models'):

    os.mkdir('./best_models')

if not os.path.exists('./submits'):

    os.mkdir('./submits')
train = pd.read_csv('../input/cat-in-the-dat-ii/train.csv')

train.drop('id', axis=1, inplace=True)

train.head()
label_encoding_features = []

object_cols = train.select_dtypes('object').columns

for col in object_cols:

    if col[0] != 'o':# and col not in target_encoder_features:# special transform for ord features

        label_encoding_features.append(col)

label_encoding_features
class Preprocessor():

    

    def __init__(self, label_encoding_features=[], target_encoder_features=[], ohe_features=[], min_max_features=[], custom_transform={}, drop_columns=[]):

        self.label_encoding_features = label_encoding_features

        self.target_encoder_features = target_encoder_features

        self.ohe_features = ohe_features

        self.min_max_features = min_max_features

        self.custom_transform = custom_transform

        self.drop_columns = drop_columns

        self.isTrain = True

        self.y = None

        

        self.le = []

        self.ohe = []

        self.mm = []

        self.te = []

        

    def fit(self, X, y=None):

        X = X.copy()

        

        for col in self.label_encoding_features:

            self.le.append(OrdinalEncoder())

            X.loc[~X[col].isna(), col] = self.le[-1].fit_transform(X.loc[~X[col].isna(), col].values.reshape(-1, 1))

                

        for col in self.custom_transform:

            if type(self.custom_transform[col]) is dict:

                X.loc[~X[col].isna(), col] = X.loc[~X[col].isna(), col].replace(self.custom_transform[col])

            elif type(self.custom_transform[col]) is list:

                for sub_col, func in self.custom_transform[col]:

                    X[sub_col] = -1

                    X.loc[~X[col].isna(), sub_col] = X.loc[~X[col].isna(), col].apply(func)

            else:

                X.loc[~X[col].isna(), col] = X.loc[~X[col].isna(), col].apply(self.custom_transform[col])

                

        for column in self.min_max_features:

            self.mm.append(MinMaxScaler())

            X.loc[~X[column].isna(), column] = self.mm[-1].fit_transform(X.loc[~X[column].isna(), column].values.reshape(-1, 1))

                

        if self.target_encoder_features:

            self.y = y

            for train_ind, val_ind in StratifiedKFold(shuffle=True, random_state=123).split(X, y):

                self.te.append(category_encoders.TargetEncoder(cols=self.target_encoder_features, handle_missing='return_nan'))

                self.te[-1].fit(X.loc[train_ind, self.target_encoder_features], X.loc[train_ind, 'target'].values.reshape(-1, 1))



            self.te.append(category_encoders.TargetEncoder(cols=self.target_encoder_features, handle_missing='return_nan'))#, smoothing=0.25))

            self.te[-1].fit(X[self.target_encoder_features], y)

        

        return self

    

    def transform(self, X):

        X = X.copy()

        

        for ind, col in enumerate(self.label_encoding_features):

            X.loc[~X[col].isin(list(self.le[ind].categories_[0])), col] = np.nan

            X.loc[~X[col].isna(), col] = self.le[ind].transform(X.loc[~X[col].isna(), col].values.reshape(-1, 1))#.astype(int)

                

        for col in self.custom_transform:

            if type(self.custom_transform[col]) is dict:

                X.loc[~X[col].isna(), col] = X.loc[~X[col].isna(), col].replace(self.custom_transform[col])

            elif type(self.custom_transform[col]) is list:

                for sub_col, func in self.custom_transform[col]:

                    X[sub_col] = -1

                    X.loc[~X[col].isna(), sub_col] = X.loc[~X[col].isna(), col].apply(func)

            else:

                X.loc[~X[col].isna(), col] = X.loc[~X[col].isna(), col].apply(self.custom_transform[col])

                

        for ind, column in enumerate(self.min_max_features):

            X.loc[~X[column].isna(), column] = self.mm[ind].transform(X.loc[~X[column].isna(), column].values.reshape(-1, 1))

                

        if self.target_encoder_features:

            if self.isTrain: #train-val

                for ind, (train_ind, val_ind) in enumerate(StratifiedKFold(shuffle=True, random_state=123).split(X, self.y)):

                    X.loc[val_ind, self.target_encoder_features] = self.te[ind].transform(X.loc[val_ind, self.target_encoder_features])

            else: # test

                X[self.target_encoder_features] = self.te[-1].transform(X[self.target_encoder_features])

            

        if self.drop_columns:

            X = X.drop(self.drop_columns, axis=1)

            

        return X
class NanImputer():

    

    def __init__(self, mode):

        self.mode = mode # ('ohe',) or ('fillna', -1)

        

    def fit(self, X, y=None):

        return self

        

    def transform(self, X, y=None):

        X = X.copy()

        if self.mode[0] == 'fillna':

            X.fillna(self.mode[1], inplace=True)

        elif self.mode[0] == 'ohe':

            nan_columns = X.isna().sum()

            nan_columns = nan_columns[nan_columns > 0].index.values

            for column in nan_columns:

#                 X[f'{column}_isNaN'] = X[column].isna() * 1

#                 X.loc[X[column].isna(), column] = X.loc[~X[column].isna(), column].value_counts().index.values.mean()

                X.loc[X[column].isna(), column] = X.loc[~X[column].isna(), column].values.mean()

                X[column] = X[column].astype(float)

        return X

        

def cross_validation(cv, model, X, y, metrics=[roc_auc_score], verbose=True, train_params={}):

    

    scores = {}

    for metric in metrics:

        scores[metric.__name__] = {'train': [], 'val': []}

    modeltype = train_params.pop('modeltype', None)

    cat_features = train_params.pop('cat_features', None)

        

    for train_index, val_index in cv.split(X, y):

        X_train, X_val, y_train, y_val = X.loc[train_index], X.loc[val_index], y.loc[train_index], y.loc[val_index]

        

        if modeltype == 'lgb':

            train_dataset = Dataset(X_train, y_train, free_raw_data=False)

            val_dataset = Dataset(X_val, y_val, free_raw_data=False)



            model = lgb.train(train_set=train_dataset, valid_sets=[val_dataset], **train_params)



            train_predictions_proba = model.predict(X_train)

            val_predictions_proba = model.predict(X_val)



        elif modeltype == 'catboost':

            train_dataset = catboost.Pool(X_train, y_train, cat_features=cat_features, feature_names=list(X_train.columns), thread_count=1)

            val_dataset = catboost.Pool(X_val, y_val, cat_features=cat_features, feature_names=list(X_train.columns), thread_count=1)



            model = catboost.CatBoostClassifier(**train_params['params'])

            model.fit(train_dataset, eval_set=val_dataset, **train_params['fit_params'])



            train_predictions_proba = model.predict_proba(X_train).T[1]

            val_predictions_proba = model.predict_proba(X_val).T[1]

        else:

            model.fit(X_train, y_train)

            

            train_predictions_proba = model.predict_proba(X_train).T[1]

            val_predictions_proba = model.predict_proba(X_val).T[1]



        train_predictions = np.round(train_predictions_proba)

        val_predictions = np.round(val_predictions_proba)



        # metric calculation

        for index, metric in enumerate(metrics):

            if metric.__name__ in ['precision_recall_curve', 'roc_curve']:

                train_score = auc(*metric(y_train, train_predictions_proba)[:2][::-1])

                val_score = auc(*metric(y_val, val_predictions_proba)[:2][::-1])

            elif metric.__name__ == 'roc_auc_score':

                train_score = metric(y_train, train_predictions_proba)

                val_score = metric(y_val, val_predictions_proba)

            else:

                train_score = metric(y_train, train_predictions)

                val_score = metric(y_val, val_predictions)



            scores[metric.__name__]['train'].append(train_score)

            scores[metric.__name__]['val'].append(val_score)

            

    for metric in metrics:

        if verbose:

            print(metric.__name__)

        for key in ['train', 'val']:

            scores[metric.__name__][key] = np.round(scores[metric.__name__][key], 5)

            scores[metric.__name__][f'{key}_mean'] = round(np.mean(scores[metric.__name__][key]), 5)

            if verbose:

                print(f"{key.upper()}: {scores[metric.__name__][key]} ({scores[metric.__name__][key+'_mean']})")

    

    return scores, model

    
def hyperparameters_optimization(X, y, model, space_search, max_evals, base_params={}, loss=''):



    modeltype = base_params.get('modeltype', None)

    

    def objective(space_search):

        if model is not None:

            model.set_params(**space_search)

        else:

            if 'params' in base_params:

                base_params['params'].update(space_search)

            else:

                base_params.update(space_search)

            base_params['modeltype'] = modeltype

#         print(space_search, model)

        scores = cross_validation(cv, model, X, y, verbose=True, train_params=base_params)[0];

        if loss == 'overfit':

            return {'loss': -scores['roc_auc_score']['val_mean'] + max(0, (scores['roc_auc_score']['train_mean'] - scores['roc_auc_score']['val_mean'])), 

                    'status': STATUS_OK, 'scores': scores, 'params': space_search}

        return {'loss': -scores['roc_auc_score']['val_mean'], 'status': STATUS_OK, 'scores': scores, 'params': space_search}

    

    trials = Trials()

    best = fmin(fn=objective,

                space=space_search,

                algo=tpe.suggest,

                max_evals=max_evals,

                trials=trials)

    

    return best, sorted(trials.results, key=lambda x: x['loss'])
target_encoder_features = [f'nom_{i}' for i in range(4, 10)] + ['ord_5']

preproc_params = {

    'label_encoding_features': label_encoding_features,

    'target_encoder_features': target_encoder_features,

    'custom_transform': {

        'ord_1': {'Novice': 0, 'Contributor': 1, 'Expert': 2, 'Master': 3, 'Grandmaster': 4},

        'ord_2': {'Freezing': 0, 'Cold': 1, 'Warm': 2, 'Hot': 3, 'Boiling Hot': 4, 'Lava Hot': 5},

        'ord_3': lambda x: ord(x) - 97,

        'ord_4': lambda x: ord(x) - 65,

    },

}

preproc_pipepline = Pipeline([

    ('preprocessor', Preprocessor(**preproc_params)),

    ('nan_imputer', NanImputer(('fillna', -1))),

])



preproc_pipepline[0].isTrain = True

train_preproc = preproc_pipepline.fit_transform(train, train.target)

# X_columns = [column for column in train_preproc.columns if column != 'target']

# X, y = train_preproc[X_columns], train_preproc.target



# train_params = {'params': {

#                     'num_leaves': 18,

#                     'min_data_in_leaf': 10, 

#                     'objective':'binary',

#                     'reg_alpha': 1,

#                     'reg_lambda': 1,

#                     'learning_rate': 0.1,

#                     "boosting": "gbdt",

#                     "feature_fraction": 0.85,

#                     "bagging_freq": 1,

#                     "bagging_fraction": 0.95 ,

#                     "seed": 123,

#                     'num_threads': 1,

#                     'is_unbalance': True,

#                     'boost_from_average': False,

#                     "metric": 'auc',

#                     "verbosity": -1

#                     },

#                 'num_boost_round': 3000,

#                 'verbose_eval': 1000,

#                 'early_stopping_rounds': 50,

#                 'modeltype': 'lgb',

# }



# # define cross_validation

# cv_params = {

#     'n_splits': 4,

#     'shuffle': True,

#     'random_state': 234,

# }

# cv = StratifiedKFold(**cv_params)



# # hyperparameters tuning

# search_space = {

#     'num_leaves': hp.uniformint('num_leaves', 6, 32), 

#     'min_data_in_leaf': hp.uniformint('min_data_in_leaf', 10, 1000),

#     'feature_fraction': hp.uniform('feature_fraction', 0.05, 1.0),

#     'bagging_fraction': hp.uniform('bagging_fraction', 0.6, 1.0),

# }

# max_eval = 30

# best_params, hp_tuning_results = hyperparameters_optimization(X, y, None, search_space, max_eval, train_params, loss='')

# best_params, hp_tuning_results
X_columns = [column for column in train_preproc.columns if column != 'target']

rs = 234

test_size = 0.15

train_X, val_X, train_y, val_y = train_test_split(train_preproc.loc[:, X_columns], train_preproc.target, 

                                                  test_size=test_size, stratify=train_preproc.target, 

                                                  random_state=rs)

print(train_X.shape, val_X.shape)
train_dataset = Dataset(train_X, train_y, free_raw_data=False)#, categorical_feature=categorical_features)

val_dataset = Dataset(val_X, val_y, free_raw_data=False)#, categorical_feature=categorical_features)

param = {

                'learning_rate': 0.1,

                'num_leaves': 11,

                'min_data_in_leaf': 141, 

                'objective':'binary',

                'reg_alpha': 1,

                'reg_lambda': 1,

                "boosting": "gbdt",

                "feature_fraction": 0.11159440461908189,

                "bagging_fraction": 0.7092434829167672,

                "seed": 123,

                'num_threads': 1,

                'is_unbalance': True,

                "metric": 'auc',

                "verbosity": -1

}



clf = lgb.train(param, train_dataset, num_boost_round=500, 

                valid_sets=[val_dataset], #[val_dataset, train_dataset], 

                verbose_eval=50, 

                early_stopping_rounds=50

               )
with open('./best_models/lgb.params', 'w') as f:

    f.write(str(param))



clf.save_model('./best_models/lgb.model')

with open('./best_models/lgb_preproc_pipeline.ppln', 'wb') as f:

    dill.dump(preproc_pipepline, f)
test = pd.read_csv('../input/cat-in-the-dat-ii/test.csv')

preproc_pipepline['preprocessor'].isTrain = False

test = preproc_pipepline.transform(test)

test.head()
predictions = clf.predict(test.iloc[:, 1:])

predictions
submission = pd.DataFrame.from_dict({

    'id': test.id,

    'target': predictions

})

submission.to_csv('./submits/best_lgb.csv', index=False)
target_encoder_features = [f'nom_{i}' for i in range(4, 10)] + ['ord_5']

preproc_params = {

    'label_encoding_features': label_encoding_features,

    'target_encoder_features': target_encoder_features,

    'custom_transform': {

        'ord_1': {'Novice': 0, 'Contributor': 1, 'Expert': 2, 'Master': 3, 'Grandmaster': 4},

        'ord_2': {'Freezing': 0, 'Cold': 1, 'Warm': 2, 'Hot': 3, 'Boiling Hot': 4, 'Lava Hot': 5},

        'ord_3': lambda x: ord(x) - 97,

        'ord_4': lambda x: ord(x) - 65,

    },

}

preproc_pipepline = Pipeline([

    ('preprocessor', Preprocessor(**preproc_params)),

    ('nan_imputer', NanImputer(('fillna', -1))),

])



preproc_pipepline[0].isTrain = True

train_preproc = preproc_pipepline.fit_transform(train, train.target)



cat_features = [column for column in train_preproc.columns if ('nom' in column or 'ord' in column) and column not in target_encoder_features] + ['day', 'month']

train_preproc[cat_features] = train_preproc[cat_features].astype(int)
# X_columns = [column for column in train_preproc.columns if column != 'target']

# cat_features_ind = [ind for ind, col in enumerate(X_columns) if col in cat_features]

# X, y = train_preproc[X_columns], train_preproc.target



# train_params = {'params': {

#                     'depth': 6,

#                     'num_leaves': 18,

#                     'min_data_in_leaf': 10, 

#                     'loss_function': 'Logloss',

#                     'iterations': 1500,

#                     'early_stopping_rounds': 50,

#                     'l2_leaf_reg': 30,

#                     'learning_rate': 0.05,

#                     'bagging_temperature': 0.8,

#                     'random_strength': 0.8,

#                     'task_type': "GPU",

#                     'grow_policy': 'Lossguide',

#                     "random_seed": 123,

#                     'thread_count': 1,

#                     "eval_metric": 'AUC',

#                     "verbose": False,

#                     'use_best_model': True

#                     },

#                 'fit_params': {'verbose_eval': 1000, 'use_best_model': True},

#                 'modeltype': 'catboost',

#                 'cat_features': cat_features_ind,

# }



# # define cross_validation

# cv_params = {

#     'n_splits': 4,

#     'test_size': 0.2,

#     'random_state': 123,

# }

# cv = StratifiedShuffleSplit(**cv_params)



# # hyperparameters tuning

# search_space = {

#     'num_leaves': hp.uniformint('num_leaves', 4, 32), 

#     'min_data_in_leaf': hp.uniformint('min_data_in_leaf', 10, 1000),

#     'random_strength': hp.uniform('random_strength', 0.1, 1.0),

#     'bagging_temperature': hp.uniform('bagging_temperature', 0.5, 1.0),

# }

# max_eval = 30

# best_params, hp_tuning_results = hyperparameters_optimization(X, y, None, search_space, max_eval, train_params, loss='overfit')

# best_params
X_columns = [column for column in train_preproc.columns if column != 'target']

cat_features_ind = [ind for ind, col in enumerate(X_columns) if col in cat_features]

rs = 123

test_size = 0.2

train_X, val_X, train_y, val_y = train_test_split(train_preproc.loc[:, X_columns], train_preproc.target, 

                                                  test_size=test_size, stratify=train_preproc.target, 

                                                  random_state=rs)

print(train_X.shape, val_X.shape)
train_dataset = catboost.Pool(train_X, train_y, cat_features=cat_features_ind, feature_names=list(train_X.columns), thread_count=1)

val_dataset = catboost.Pool(val_X, val_y, cat_features=cat_features_ind, feature_names=list(train_X.columns), thread_count=1)

param = {'params': {

                    'depth': 6,

                    'num_leaves': 18,

                    'min_data_in_leaf': 10, 

                    'l2_leaf_reg': 30,

                    'learning_rate': 0.05,

                    'bagging_temperature': 0.8,

                    'random_strength': 0.8,

                    'task_type': "GPU",

                    'grow_policy': 'Lossguide',

                    'iterations': 1500,

                    'early_stopping_rounds': 50,

                    "random_seed": 123,

                    'thread_count': 1,

                    "eval_metric": 'AUC',

                    "verbose": False,

                    'use_best_model': True

                    },

         'fit_params': {'verbose_eval': 100,},

}

param['params'].update({'bagging_temperature': 0.7497082074820156,

 'min_data_in_leaf': 67.0,

 'num_leaves': 4.0,

 'random_strength': 0.2017357950398055})



clf = catboost.CatBoostClassifier(**param['params'])#)dtrain=train_dataset, eval_set=val_dataset, **param)

clf.fit(train_dataset, eval_set=val_dataset, **param['fit_params'])
with open('./best_models/catboost.params', 'w') as f:

    f.write(str(param))



clf.save_model('./best_models/catboost.model')

with open('./best_models/catboost_preproc_pipeline.ppln', 'wb') as f:

    dill.dump(preproc_pipepline, f)
test = pd.read_csv('../input/cat-in-the-dat-ii/test.csv')

preproc_pipepline['preprocessor'].isTrain = False

test = preproc_pipepline.transform(test)

test[cat_features] = test[cat_features].astype(int)

test.head()
predictions = clf.predict_proba(catboost.Pool(test.iloc[:, 1:], cat_features=cat_features_ind, feature_names=list(test.columns[1:]), thread_count=1)).T[1]

predictions
submission = pd.DataFrame.from_dict({

    'id': test.id,

    'target': predictions

})

submission.to_csv('./submits/best_cat.csv', index=False)
minmax_features = [f'ord_{i}' for i in range(5)]

target_encoder_features = [i for i in train.columns if i not in minmax_features and i != 'target']

preproc_params = {

    'label_encoding_features': label_encoding_features,

    'target_encoder_features': target_encoder_features,

    'min_max_features': minmax_features,

    'custom_transform': {

        'ord_1': {'Novice': 0, 'Contributor': 1, 'Expert': 2, 'Master': 3, 'Grandmaster': 4},

        'ord_2': {'Freezing': 0, 'Cold': 1, 'Warm': 2, 'Hot': 3, 'Boiling Hot': 4, 'Lava Hot': 5},

        'ord_3': lambda x: ord(x) - 97,

        'ord_4': lambda x: ord(x) - 65,

    },

}

preproc_pipepline = Pipeline([

    ('preprocessor', Preprocessor(**preproc_params)),

    ('nan_imputer', NanImputer(('ohe', -1))),

])



preproc_pipepline[0].isTrain = True

train_preproc = preproc_pipepline.fit_transform(train, train.target)

X_columns = [column for column in train_preproc.columns if column != 'target' and 'NaN' not in column]

X, y = train_preproc[X_columns], train_preproc.target



train_params = {

    'random_state': 1, 

    'solver': 'lbfgs', 

    'max_iter': 2020, 

#     'penalty': 'l2',

#     'C': 1,

    'verbose': 0,

    'n_jobs': 1

}



# define cross_validation

cv_params = {

    'n_splits': 4,

    'shuffle': True,

    'random_state': 123,

}

cv = StratifiedKFold(**cv_params)



cross_validation(cv, LogisticRegression(**train_params), X, y, verbose=True)
X_columns = [column for column in train_preproc.columns if column != 'target' and 'NaN' not in column]

X, y = train_preproc[X_columns], train_preproc.target



train_params = {

    'random_state': 1, 

    'solver': 'lbfgs', 

    'max_iter': 2020, 

#     'penalty': 'l2',

#     'C': 1,

    'verbose': 0,

    'n_jobs': 1

}



clf = LogisticRegression(**train_params)

clf.fit(X, y)
with open('./best_models/logreg.params', 'w') as f:

    f.write(str(train_params))



with open('./best_models/logreg.model', 'wb') as f:

    pkl.dump(clf, f)

with open('./best_models/logreg_preproc_pipeline.ppln', 'wb') as f:

    dill.dump(preproc_pipepline, f)
test = pd.read_csv('../input/cat-in-the-dat-ii/test.csv')

preproc_pipepline['preprocessor'].isTrain = False

test = preproc_pipepline.transform(test)

test.head()
predictions = clf.predict_proba(test.iloc[:, 1:]).T[1]

predictions
submission = pd.DataFrame.from_dict({

    'id': test.id,

    'target': predictions

})

submission.to_csv('./submits/best_logreg.csv', index=False)
compresed_xy = None

for tr_ind, val_ind in StratifiedKFold(10, shuffle=True, random_state=123).split(X, y):

    compresed_xy = (X.iloc[val_ind], y.iloc[val_ind])

    break

compresed_xy[0].shape, compresed_xy[1].shape
poly = PolynomialFeatures(2, interaction_only=True)

poly_X = pd.DataFrame(poly.fit_transform(compresed_xy[0].reset_index(drop=True)))

# poly_X = pd.DataFrame(poly.fit_transform(train_preproc[X_columns]))



train_params = {

    'random_state': 1, 

    'solver': 'lbfgs', 

    'max_iter': 2020, 

    'penalty': 'l2',

    'C': 1,

    'verbose': 0,

    'n_jobs': 1

}



cv_params = {

    'n_splits': 5,

    'shuffle': True,

    'random_state': 123,

}

cv = StratifiedKFold(**cv_params)

cross_validation(cv, LogisticRegression(**train_params), 

                 poly_X, 

                 compresed_xy[1].reset_index(drop=True), verbose=True)

poly = PolynomialFeatures(2, interaction_only=True)

poly_X = pd.DataFrame(poly.fit_transform(compresed_xy[0].reset_index(drop=True)))

train_params = {

    'random_state': 1, 

    'solver': 'lbfgs', 

    'max_iter': 500, 

    'penalty': 'l2',

    'C': 1,

    'verbose': 0,

    'n_jobs': 1

}



clf = LogisticRegression(**train_params)

clf.fit(poly_X, compresed_xy[1].reset_index(drop=True))
with open('./best_models/poly_logreg.params', 'w') as f:

    f.write(str(train_params))



with open('./best_models/poly_logreg.model', 'wb') as f:

    pkl.dump(clf, f)

with open('./best_models/poly_logreg_preproc_pipeline.ppln', 'wb') as f:

    dill.dump(preproc_pipepline, f)
test = pd.read_csv('../input/cat-in-the-dat-ii/test.csv')

preproc_pipepline['preprocessor'].isTrain = False

test = preproc_pipepline.transform(test)

test.head()
predictions = clf.predict_proba(poly.transform(test.iloc[:, 1:])).T[1]

predictions
submission = pd.DataFrame.from_dict({

    'id': test.id,

    'target': predictions

})

submission.to_csv('./submits/best_poly_logreg.csv', index=False)
minmax_features = [f'ord_{i}' for i in range(5)]# + [f'nom_{i}' for i in range(4)] + ['day', 'month']

target_encoder_features = [i for i in train.columns if i not in minmax_features and i != 'target']

preproc_params = {

    'label_encoding_features': label_encoding_features,

    'target_encoder_features': target_encoder_features,

    'min_max_features': minmax_features,

    'custom_transform': {

        'ord_1': {'Novice': 0, 'Contributor': 1, 'Expert': 2, 'Master': 3, 'Grandmaster': 4},

        'ord_2': {'Freezing': 0, 'Cold': 1, 'Warm': 2, 'Hot': 3, 'Boiling Hot': 4, 'Lava Hot': 5},

        'ord_3': lambda x: ord(x) - 97,

        'ord_4': lambda x: ord(x) - 65,

    },

}

preproc_pipepline = Pipeline([

    ('preprocessor', Preprocessor(**preproc_params)),

    ('nan_imputer', NanImputer(('ohe', -1))),

])



preproc_pipepline[0].isTrain = True

train_preproc = preproc_pipepline.fit_transform(train, train.target)



X_columns = [column for column in train_preproc.columns if column != 'target' and 'NaN' not in column]

X, y = train_preproc[X_columns], train_preproc.target



train_params = {

    'random_state': 1, 

    'solver': 'lbfgs', 

    'max_iter': 2020, 

    'verbose': 0,

    'n_jobs': 1

}

log_model = LogisticRegression(**train_params)

log_model.fit(X, y)

X *= abs(log_model.coef_[0])
compresed_xy = None

for tr_ind, val_ind in StratifiedKFold(10, shuffle=True, random_state=123).split(X, y):

    compresed_xy = (X.iloc[val_ind], y.iloc[val_ind])

    break

compresed_xy[0].shape, compresed_xy[1].shape
train_params = {'n_neighbors': 188, 'p': 2, 'weights': 'uniform'}



cv_params = {

    'n_splits': 1,

    'test_size': 0.2,

    'random_state': 123,

}

cv = StratifiedShuffleSplit(**cv_params)



cross_validation(cv, KNeighborsClassifier(**train_params), 

                 compresed_xy[0].reset_index(drop=True), 

                 compresed_xy[1].reset_index(drop=True), verbose=True)#[0], verbose=True)#[0]

label_encoding_features = []

object_cols = train.select_dtypes('object').columns

for col in object_cols:

    if col[0] != 'o':# and col not in target_encoder_features:# special transform for ord features

        label_encoding_features.append(col)

print(label_encoding_features)

X_columns = [column for column in train.columns if column != 'target']

test = pd.read_csv('../input/cat-in-the-dat-ii/test.csv')



# lgb preproc

print('lgb preproc')

if os.path.exists('./best_models/lgb_preproc_pipeline.ppln'):

    print('load existsing preproc_pipeline...')

    with open('./best_models/lgb_preproc_pipeline.ppln', 'rb') as f:

        lgb_preproc_pipepline = dill.load(f)

    lgb_preproc_pipepline[0].isTrain = True

    lgb_train_preproc = lgb_preproc_pipepline.transform(train)

else:

    target_encoder_features = [f'nom_{i}' for i in range(4, 10)] + ['ord_5']

    preproc_params = {

        'label_encoding_features': label_encoding_features,

        'target_encoder_features': target_encoder_features,

        'custom_transform': {

            'ord_1': {'Novice': 0, 'Contributor': 1, 'Expert': 2, 'Master': 3, 'Grandmaster': 4},

            'ord_2': {'Freezing': 0, 'Cold': 1, 'Warm': 2, 'Hot': 3, 'Boiling Hot': 4, 'Lava Hot': 5},

            'ord_3': lambda x: ord(x) - 97,

            'ord_4': lambda x: ord(x) - 65,

        },

    }

    lgb_preproc_pipepline = Pipeline([

        ('preprocessor', Preprocessor(**preproc_params)),

        ('nan_imputer', NanImputer(('fillna', -1))),

    ])



    lgb_preproc_pipepline[0].isTrain = True

    lgb_train_preproc = lgb_preproc_pipepline.fit_transform(train, train.target)

    

lgb_params = {

    'learning_rate': 0.1,

    'num_leaves': 11,

    'min_data_in_leaf': 141, 

    'objective':'binary',

    'reg_alpha': 1,

    'reg_lambda': 1,

    "boosting": "gbdt",

    "feature_fraction": 0.11159440461908189,

    "bagging_fraction": 0.7092434829167672,

    "seed": 123,

    'num_threads': 1,

    'is_unbalance': True,

    "metric": 'auc',

    "verbosity": -1

}

lgb_preproc_pipepline['preprocessor'].isTrain = False

lgb_test = lgb_preproc_pipepline.transform(test)





# catboost preproc

print('catboost preproc')

if os.path.exists('./best_models/catboost_preproc_pipeline.ppln'):

    print('load existsing preproc_pipeline...')

    with open('./best_models/catboost_preproc_pipeline.ppln', 'rb') as f:

        catboost_preproc_pipepline = dill.load(f)

    catboost_preproc_pipepline[0].isTrain = True

    catboost_train_preproc = catboost_preproc_pipepline.transform(train)

else:

    target_encoder_features = [f'nom_{i}' for i in range(4, 10)] + ['ord_5']

    preproc_params = {

        'label_encoding_features': label_encoding_features,

        'target_encoder_features': target_encoder_features,

        'custom_transform': {

            'ord_1': {'Novice': 0, 'Contributor': 1, 'Expert': 2, 'Master': 3, 'Grandmaster': 4},

            'ord_2': {'Freezing': 0, 'Cold': 1, 'Warm': 2, 'Hot': 3, 'Boiling Hot': 4, 'Lava Hot': 5},

            'ord_3': lambda x: ord(x) - 97,

            'ord_4': lambda x: ord(x) - 65,

        },

    }

    catboost_preproc_pipepline = Pipeline([

        ('preprocessor', Preprocessor(**preproc_params)),

        ('nan_imputer', NanImputer(('fillna', -1))),

    ])



    catboost_preproc_pipepline[0].isTrain = True

    catboost_train_preproc = catboost_preproc_pipepline.fit_transform(train, train.target)

    

cat_features = [column for column in catboost_train_preproc.columns if ('nom' in column or 'ord' in column) and column not in target_encoder_features] + ['day', 'month']

catboost_train_preproc[cat_features] = catboost_train_preproc[cat_features].astype(int)

cat_features_ind = [ind for ind, col in enumerate(X_columns) if col in cat_features]

catboost_params = {

    'params': {

        'depth': 6,

        'num_leaves': 4,

        'min_data_in_leaf': 67, 

        'l2_leaf_reg': 30,

        'learning_rate': 0.05,

        'bagging_temperature': 0.7497082074820156,

        'random_strength': 0.2017357950398055,

        'task_type': "GPU",

        'grow_policy': 'Lossguide',

        'iterations': 1500,

        'early_stopping_rounds': 50,

        "random_seed": 123,

        'thread_count': 1,

        "eval_metric": 'AUC',

        "verbose": False,

        'use_best_model': True

    },

    'fit_params': {'verbose_eval': 100,},

}

catboost_preproc_pipepline['preprocessor'].isTrain = False

catboost_test = catboost_preproc_pipepline.transform(test)

catboost_test[cat_features] = catboost_test[cat_features].astype(int)





# log_reg preproc

print('log_reg preproc')

if os.path.exists('./best_models/logreg_preproc_pipeline.ppln'):

    print('load existsing preproc_pipeline...')

    with open('./best_models/logreg_preproc_pipeline.ppln', 'rb') as f:

        logreg_preproc_pipepline = dill.load(f)

    logreg_preproc_pipepline[0].isTrain = True

    logreg_train_preproc = logreg_preproc_pipepline.transform(train)

else:

    minmax_features = [f'ord_{i}' for i in range(5)]

    target_encoder_features = [i for i in train.columns if i not in minmax_features and i != 'target']

    preproc_params = {

        'label_encoding_features': label_encoding_features,

        'target_encoder_features': target_encoder_features,

        'min_max_features': minmax_features,

        'custom_transform': {

            'ord_1': {'Novice': 0, 'Contributor': 1, 'Expert': 2, 'Master': 3, 'Grandmaster': 4},

            'ord_2': {'Freezing': 0, 'Cold': 1, 'Warm': 2, 'Hot': 3, 'Boiling Hot': 4, 'Lava Hot': 5},

            'ord_3': lambda x: ord(x) - 97,

            'ord_4': lambda x: ord(x) - 65,

        },

    }

    logreg_preproc_pipepline = Pipeline([

        ('preprocessor', Preprocessor(**preproc_params)),

        ('nan_imputer', NanImputer(('ohe', -1))),

    ])



    logreg_preproc_pipepline[0].isTrain = True

    logreg_train_preproc = logreg_preproc_pipepline.fit_transform(train, train.target)

    

logreg_params = {

    'random_state': 1, 

    'solver': 'lbfgs', 

    'max_iter': 2020, 

    'verbose': 0,

    'n_jobs': 1

}

logreg_preproc_pipepline['preprocessor'].isTrain = False

logreg_test = logreg_preproc_pipepline.transform(test)





# polynimal logreg

polylogreg_params = {

    'random_state': 1, 

    'solver': 'lbfgs', 

    'max_iter': 2020, 

    'penalty': 'l2',

    'C': 1,

    'verbose': 0,

    'n_jobs': 1

}





stack = pd.DataFrame(index=train.index)

for modelname in ['lgb', 'catboost', 'logreg', 'poly_logreg',]:

    stack[modelname] = 0.5

stack['target'] = logreg_train_preproc.target.values

test_pred = []

    

cv_params = {

    'n_splits': 5,

    'shuffle': True,

    'random_state': 123,

}

cv = StratifiedKFold(**cv_params)



n_fold = 1

for tr_ind, val_ind in cv.split(train, train.target):

    print(f'n_fold={n_fold}')

    n_fold += 1

    # lgb

    train_X, train_y = lgb_train_preproc.iloc[tr_ind][X_columns], lgb_train_preproc.iloc[tr_ind].target

    val_X, val_y = lgb_train_preproc.iloc[val_ind][X_columns], lgb_train_preproc.iloc[val_ind].target

    

    train_dataset = Dataset(train_X, train_y, free_raw_data=False)

    val_dataset = Dataset(val_X, val_y, free_raw_data=False)



    clf = lgb.train(lgb_params, train_dataset, num_boost_round=500, 

                    valid_sets=[val_dataset],

                    verbose_eval=50, 

                    early_stopping_rounds=50

                   )

    

    stack.iloc[val_ind, 0] = clf.predict(val_X)

    # test prediction

    test_pred.append(clf.predict(lgb_test.iloc[:, 1:]).tolist())

    

    # catboost

    train_X, train_y = catboost_train_preproc.iloc[tr_ind][X_columns], catboost_train_preproc.iloc[tr_ind].target

    val_X, val_y = catboost_train_preproc.iloc[val_ind][X_columns], catboost_train_preproc.iloc[val_ind].target

    train_dataset = catboost.Pool(train_X, train_y, cat_features=cat_features_ind, feature_names=list(train_X.columns), thread_count=1)

    val_dataset = catboost.Pool(val_X, val_y, cat_features=cat_features_ind, feature_names=list(train_X.columns), thread_count=1)



    clf = catboost.CatBoostClassifier(**catboost_params['params'])#)dtrain=train_dataset, eval_set=val_dataset, **param)

    clf.fit(train_dataset, eval_set=val_dataset, **catboost_params['fit_params'])

    

    stack.iloc[val_ind, 1] = clf.predict_proba(val_X).T[1]

    # test prediction

    test_pred.append(clf.predict_proba(catboost.Pool(catboost_test.iloc[:, 1:], 

                                                     cat_features=cat_features_ind, 

                                                     feature_names=list(catboost_test.columns[1:]), 

                                                     thread_count=1)

                                      ).T[1].tolist())

    

    # logreg

    train_X, train_y = logreg_train_preproc.iloc[tr_ind][X_columns], logreg_train_preproc.iloc[tr_ind].target

    val_X, val_y = logreg_train_preproc.iloc[val_ind][X_columns], logreg_train_preproc.iloc[val_ind].target



    clf = LogisticRegression(**logreg_params)

    clf.fit(train_X, train_y)

    

    stack.iloc[val_ind, 2] = clf.predict_proba(val_X).T[1]

    # test prediction

    test_pred.append(clf.predict_proba(logreg_test.iloc[:, 1:]).T[1].tolist())

    

    # polynomial logreg

    tm = time.time()

    train_X, train_y = logreg_train_preproc.iloc[tr_ind][X_columns], logreg_train_preproc.iloc[tr_ind].target

    val_X, val_y = logreg_train_preproc.iloc[val_ind][X_columns], logreg_train_preproc.iloc[val_ind].target

    compresed_xy = None

    for tr_ind1, val_ind1 in StratifiedShuffleSplit(1, test_size=0.2, random_state=123).split(train_X, train_y):

        compresed_xy = (train_X.iloc[val_ind1], train_y.iloc[val_ind1])

    poly = PolynomialFeatures(2, interaction_only=True)

    train_X = pd.DataFrame(poly.fit_transform(compresed_xy[0].reset_index(drop=True)))

    train_y = compresed_xy[1]



    clf = LogisticRegression(**polylogreg_params)

    clf.fit(train_X, train_y)

    

    stack.iloc[val_ind, 3] = clf.predict_proba(poly.transform(val_X)).T[1]

    # test prediction

    test_pred.append(clf.predict_proba(poly.transform(logreg_test.iloc[:, 1:])).T[1].tolist())

    
logreg_params = {

    'random_state': 1, 

    'solver': 'lbfgs', 

    'max_iter': 2020, 

    'verbose': 0,

    'n_jobs': 1

}



cv_params = {

    'n_splits': 5,

    'shuffle': True,

    'random_state': 321,

}

cv = StratifiedKFold(**cv_params)

cross_validation(cv, LogisticRegression(**logreg_params), stack.iloc[:, :stack.shape[1]-1], stack.target, verbose=True)
train_params = {'params': {

                    'num_leaves': 18,

                    'min_data_in_leaf': 10, 

                    'objective':'binary',

                    'learning_rate': 0.1,

                    "boosting": "gbdt",

                    "seed": 123,

                    'num_threads': 1,

                    'is_unbalance': True,

                    'boost_from_average': False,

                    "metric": 'auc',

                    "verbosity": -1

                    },

                'num_boost_round': 3000,

                'verbose_eval': 50,

                'early_stopping_rounds': 50,

                'modeltype': 'lgb',

}



# define cross_validation

cv_params = {

    'n_splits': 5,

    'shuffle': True,

    'random_state': 321,

}

cv = StratifiedKFold(**cv_params)

cross_validation(cv, None, stack.iloc[:, :stack.shape[1]-1], stack.target, verbose=True, train_params=train_params)
logreg_params = {

    'random_state': 1, 

    'solver': 'lbfgs', 

    'max_iter': 2020, 

    'verbose': 0,

    'n_jobs': 1

}



stack_logreg = LogisticRegression(**logreg_params)

stack_logreg.fit(stack.iloc[:, :stack.shape[1]-1], stack.target)

stack_logreg.coef_[0] # logreg coeficients
with open('./best_models/agregate_logreg.params', 'w') as f:

    f.write(str(logreg_params))



with open('./best_models/agregate_logreg.model', 'wb') as f:

    pkl.dump(stack_logreg, f)
test_stack = pd.DataFrame(index=test.index)

for index, modelname in enumerate(['lgb', 'catboost', 'logreg', 'poly_logreg',]):

    test_stack[modelname] = np.mean(test_pred[index::4], axis=0) # where 4 - number of models

test_stack.describe()
predictions = stack_logreg.predict_proba(test_stack).T[1]

predictions
submission = pd.DataFrame.from_dict({

    'id': test.id,

    'target': predictions

})

# submission.to_csv('./submits/best_logreg_stack_with_poly.csv', index=False)

submission.to_csv('submission.csv', index=False)