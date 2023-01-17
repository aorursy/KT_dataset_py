import numpy as np

import pandas as pd

from sklearn import preprocessing

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score

import time

import pickle

import gc

from tqdm.notebook import tqdm



import lightgbm as lgb

import xgboost as xgb

import catboost as catbst

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.neural_network import MLPClassifier





import riiideducation
# You can only call make_env() once, so don't lose it!

env = riiideducation.make_env()
from pandas.api.types import is_datetime64_any_dtype as is_datetime

from pandas.api.types import is_categorical_dtype

def reduce_mem_usage(df, use_float16=False, log_name=None):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))



    for col in tqdm(df.columns):

        if is_datetime(df[col]) or is_categorical_dtype(df[col]) or str(df[col].dtype)=='timedelta64[ns]':

            # skip datetime type or categorical type

            continue

        col_type = df[col].dtype



        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)

            else:

                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    # if log_name is not None:

    #     get_logger(log_name).info('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    #     get_logger(log_name).info('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    # else:

    #     print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    #     print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    print(

        'Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(

        100 * (start_mem - end_mem) / start_mem))



    return df
train_df = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv',

                       low_memory=False,

                       nrows=10**7,

                       usecols=[

                           #'timestamp', 

                           'user_id', 

                           'content_id', 

                           #'content_type_id', 

                           'task_container_id', 

                           'user_answer', 

                           'answered_correctly', 

                           #'prior_question_elapsed_time', 

                           #'prior_question_had_explanation'

                       ],

                       dtype={

                        #'row_id': 'int64',

                        #'timestamp': 'int64',

                        'user_id': 'int32',

                        'content_id': 'int16',

                        #'content_type_id': 'int8',

                        'task_container_id': 'int16',

                        'user_answer': 'int8',

                        'answered_correctly': 'int8',

                        #'prior_question_elapsed_time': 'float32',

                        #'prior_question_had_explanation': 'boolean',

                             }

                      )



train_df = reduce_mem_usage(train_df)

train_df = train_df.query('answered_correctly != -1').reset_index(drop=True)

#train_df['prior_question_had_explanation'] = train_df['prior_question_had_explanation'].astype(float)
train_df
y_train = train_df['answered_correctly']

X_train = train_df.drop(['answered_correctly', 'user_answer'], axis=1)
TRAIN_FEATS = list(X_train.columns)

TRAIN_FEATS
models = [

    #{'model_name': 'KNeighborsClassifier', 'model_instance': KNeighborsClassifier(3), 'model_weight': None}, # This performs low score

    #{'model_name': 'SVC_linear', 'model_instance': SVC(kernel="linear", C=0.025), 'model_weight': 0.05}, # This took much time

    #{'model_name': 'SVC_gamma', 'model_instance': SVC(gamma=2, C=1), 'model_weight': None}, # This took much time

    #{'model_name': 'GaussianProcessClassifier', 'model_instance': GaussianProcessClassifier(1.0 * RBF(1.0)), 'model_weight': None}, # This cause memory leak

    #{'model_name': 'DecisionTreeClassifier', 'model_instance': DecisionTreeClassifier(max_depth=5), 'model_weight': None},

    #{'model_name': 'GaussianProcessClassifier', 'model_instance': GaussianProcessClassifier(1.0 * RBF(1.0)), 'model_weight': None},# This cause memory leak

    #{'model_name': 'RandomForestClassifier', 'model_instance': RandomForestClassifier(max_depth=5, n_estimators=100, max_features=1), 'model_weight': None},

    #{'model_name': 'MLPClassifier', 'model_instance': MLPClassifier(alpha=1, max_iter=1000), 'model_weight': None}, # This performs low score

    #{'model_name': 'AdaBoostClassifier', 'model_instance': AdaBoostClassifier(), 'model_weight': None},

    #{'model_name': 'GaussianNB', 'model_instance': GaussianNB(), 'model_weight': None}, # This performs low score

    #{'model_name': 'QuadraticDiscriminantAnalysis', 'model_instance': QuadraticDiscriminantAnalysis(), 'model_weight': None}, # This performs low score

    #{'model_name': 'lgb', 'model_instance': None, 'model_weight': 0.3},

    {'model_name': 'xgb', 'model_instance': None, 'model_weight': 0.5},

    {'model_name': 'catbst', 'model_instance': None, 'model_weight': 0.5},

]





def arrange_weightNone(models):

    weights = [m['model_weight'] if m['model_weight'] else 0 for m in models]

    sum_weights = sum(weights)

    null_weights = weights.count(0)

    if null_weights > 0:

        res_weights = (1 - sum_weights) / null_weights

        weights = [res_weights if w==0 else w for w in weights]

        for i in range(len(models)):

            models[i]['model_weight'] = weights[i]

    return models



models = arrange_weightNone(models)

models
def save_model(model, save_path):

    with open(save_path, 'wb') as f:

        pickle.dump(model, f)

    del model

    gc.collect()

    return save_path



def read_model(model_path):

    with open(model_path, 'rb') as f:

        model = pickle.load(f)

    return model
trained_models = {}



oof_train = np.zeros((len(X_train),))

N_SPLITS = 5 

cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=0)



categorical_features = [

    'user_id', 

    'content_type_id', 

    'task_container_id', 

    'prior_question_had_explanation'

]

categorical_features = [c for c in categorical_features if c in TRAIN_FEATS]



PRINT_FEATURE_IMPORTANCE = True



for m in tqdm(models):

    model_name = m['model_name']

    model_weight = m['model_weight']

    print (f'******************************************************************')

    print (f'*********** MODEL = {model_name} (weight = {model_weight}) start')

    start_time = time.time()

    

    trained_models_sub = []

    oof_train_sub = np.zeros((len(X_train),))



    for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train)):

        model_path = f'./{model_name}_{fold_id}.pkl'

        

        X_tr = X_train.iloc[train_index, :]

        X_val = X_train.iloc[valid_index, :]

        y_tr = y_train[train_index]

        y_val = y_train[valid_index]



        if model_name == 'lgb':

            params = {

                'objective': 'binary',

                'metric': 'auc',

                'learning_rate': 0.05,

                "max_depth": 7,

                "min_data_in_leaf": 50, 

                "reg_alpha": 0.1, 

                "reg_lambda": 1, 

                "num_leaves" : 31,

                "bagging_fraction" : 0.8,

                "feature_fraction" : 0.8,

                'seed': 123,

            }



            dtrain = lgb.Dataset(X_tr, y_tr, categorical_feature=categorical_features)

            dvalid = lgb.Dataset(X_val, y_val, reference=dtrain, categorical_feature=categorical_features)



            model = lgb.train(

                params, 

                dtrain,

                valid_sets=[dtrain, dvalid],

                verbose_eval=100,

                num_boost_round=1000,

                early_stopping_rounds=100

            )



            oof_train_sub[valid_index] = model.predict(X_val, num_iteration=model.best_iteration)

            

            if PRINT_FEATURE_IMPORTANCE:

                feature_importance = sorted(zip(model.feature_name(), model.feature_importance(importance_type='gain')),key=lambda x: x[1], reverse=True)[:]

                for i, item in enumerate(feature_importance[:]):

                    print('Feature importance {}: {}'.format(i, str(item)))

                #feature_importance = pd.DataFrame(feature_importance, columns=['feature', 'importance'])

                

            model_path = save_model(model, model_path)

            trained_models_sub.append(model_path)

            

        elif model_name == 'xgb':

            params = {

                'booster': 'gbtree',

                'task': 'train',

                'eval_metric': 'auc',

                'objective': 'binary:logistic',

                'base_score': 0.5,

                'learning_rate': 0.05,

                'max_depth': 5,

                'alpha': 1,

                'lambda': 20,

                'gamma': 0.1,

                'colsample_bytree': 0.2, 

                'colsample_bynode': 1,

                'colsample_bylevel': 0.3,

                'subsample': 0.85,

                'scale_pos_weight': 10,

                'min_child_weight': 30,

                'seed': 123

            }

            

            dtrain = xgb.DMatrix(X_tr, label=y_tr)

            dvalid = xgb.DMatrix(X_val, label=y_val)

            

            model = xgb.train(

                params, 

                dtrain=dtrain,

                evals=[(dtrain, 'train'),(dvalid, 'eval')],

                verbose_eval=100,

                num_boost_round=1000,

                early_stopping_rounds=100

            )

            

            oof_train_sub[valid_index] = model.predict(xgb.DMatrix(X_val), ntree_limit=model.best_iteration)

            model_path = save_model(model, model_path)

            trained_models_sub.append(model_path)

            

        elif model_name == 'catbst':

            params = {

                'bootstrap_type': 'Bayesian',

                'boosting_type': 'Plain',

                'objective': 'Logloss',

                'eval_metric': 'AUC',

                'num_boost_round': 1000,

                'learning_rate': 0.05,

                'max_depth': 9,

                'colsample_bylevel': 0.1,

                'bagging_temperature': 0.3,

                'random_seed': 123,

                'use_best_model': True,

                'od_type': 'Iter',

                'od_wait': 100,

            }

            

            cat_feats_index = np.array([i for i in range(len(TRAIN_FEATS)) if TRAIN_FEATS[i] in categorical_features])

            X_tr[categorical_features] = X_tr[categorical_features].astype(str)

            X_val[categorical_features] = X_val[categorical_features].astype(str)

            

            dtrain = catbst.Pool(X_tr, label=y_tr)

            dvalid = catbst.Pool(X_val, label=y_val)

            

            model = catbst.CatBoostClassifier(**params, cat_features=cat_feats_index)

            model.set_feature_names(TRAIN_FEATS)

            model = model.fit(

                X_tr,

                y_tr,

                eval_set=(X_val, y_val),

                verbose=100,

                early_stopping_rounds=100,

            )

            

            oof_train_sub[valid_index] = model.predict_proba(X_val)[:, 1]

            model_path = save_model(model, model_path)

            trained_models_sub.append(model_path)

            

        else:

            model = m['model_instance']

            X_tr = X_tr.fillna(0)

            X_val = X_val.fillna(0)

            model = model.fit(X_tr, y_tr)

            if hasattr(model, "decision_function"):

                oof_train_sub[valid_index] = model.decision_function(X_val)

            else:

                oof_train_sub[valid_index] = model.predict_proba(X_val)[:, 1]

            model_path = save_model(model, model_path)

            trained_models_sub.append(model_path)

        

    oof_score = roc_auc_score(y_train, oof_train_sub)



    trained_models[model_name] = [trained_models_sub, float(model_weight)]

    

    oof_train += oof_train_sub * model_weight

    

    end_time = time.time()

    training_time = round((end_time - start_time)/60, 2)

    

    print (f'*********** MODEL = {model_name} end: score = {oof_score}, training_time = {training_time} min')

    print (f'******************************************************************')

    
oof_score = roc_auc_score(y_train, oof_train)

print (f'*********** ALL MODELS ENSEMBLE: score = {oof_score}')
iter_test = env.iter_test()
# user_id_train_last = train_df.groupby('user_id').last()[['user_id', 'answered_correctly_user_id_cumsum','answered_correctly_user_id_cummean']]

# content_id_train_last = train_df.groupby('content_id').last()[['content_id', 'answered_correctly_content_id_cumsum','answered_correctly_content_id_cummean']]
for (test_df, sample_prediction_df) in iter_test:

    y_preds = []

    #test_df['prior_question_had_explanation'] = test_df['prior_question_had_explanation'].astype(float)

    #test_df = test_df.merge(user_id_train_last, on='user_id', how='left')

    #test_df = test_df.merge(content_id_train_last, on='content_id', how='left')

    X_test = test_df[TRAIN_FEATS]

    

    for model_name, models in trained_models.items():

        model_weight = models[1]

        for model_path in models[0]:

            model = read_model(model_path)

            if model_name == 'lgb':

                y_pred = model.predict(X_test, num_iteration=model.best_iteration) * model_weight / len(models)

                y_preds.append(y_pred)

                

            elif model_name == 'xgb':

                y_pred = model.predict(xgb.DMatrix(X_test), ntree_limit=model.best_iteration) * model_weight / len(models)

                y_preds.append(y_pred)

                

            elif model_name == 'catbst':

                X_test[categorical_features] = X_test[categorical_features].astype(str)

                y_pred = model.predict_proba(X_test)[:, 1] * model_weight / len(models)

                y_preds.append(y_pred)

                

            else:

                X_test = X_test.fillna(0)

                if hasattr(model, "decision_function"):

                    y_pred = model.decision_function(X_test) * model_weight / len(models)

                else:

                    y_pred = model.predict_proba(X_test)[:, 1] * model_weight / len(models)

                y_preds.append(y_pred)



    y_preds = sum(y_preds)

    test_df['answered_correctly'] = y_preds

    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])