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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import re

from datetime import date, timedelta

import json

from copy import deepcopy



####################



from sklearn.model_selection import cross_val_score

from lightgbm import LGBMClassifier

import lightgbm as lgb

import xgboost as xgb





from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import SGDClassifier

from catboost import CatBoostClassifier

from xgboost import XGBClassifier
df = pd.read_csv('/kaggle/input/hse-practical-ml-1/car_loan_train.csv') 
features = list(set(df.columns) - set(['UniqueID', 'target']))

var_target = 'target'

id_cols = [i for i in df.columns if re.findall('id', i.lower())]

flag_cols = [i for i in df.columns if re.findall('flag', i.lower())]
def parse_date(row):

    return np.array(row.replace('yrs ', ' ').replace('mon', '').split(' ')).astype(int)
acc_parsed = df['AVERAGE.ACCT.AGE'].apply(parse_date)



acc_year = []

acc_month = []

acc_total = []

for val in acc_parsed.values:

    acc_year.append(val[0])

    acc_month.append(val[1])

    acc_total.append(val[0] + (val[1] / 12))

    

    

df['AVERAGE.ACCT.AGE.YEAR'] = acc_year

df['AVERAGE.ACCT.AGE.MONTHS'] = acc_month

df['AVERAGE.ACCT.AGE.TOTAL'] = acc_total



df.drop('AVERAGE.ACCT.AGE', axis=1, inplace=True)
cred_parsed = df['CREDIT.HISTORY.LENGTH'].apply(parse_date)



cred_year = []

cred_month = []

cred_total = []

for val in cred_parsed.values:

    cred_year.append(val[0])

    cred_month.append(val[1])

    cred_total.append(val[0] + (val[1] / 12))

    

    

df['CREDIT.HISTORY.LENGTH.YEAR'] = cred_year

df['CREDIT.HISTORY.LENGTH.MONTHS'] = cred_month

df['CREDIT.HISTORY.LENGTH.TOTAL'] = cred_total



df.drop('CREDIT.HISTORY.LENGTH', axis=1, inplace=True)
cols_dt = ['Date.of.Birth', 'DisbursalDate']

df['Date.of.Birth'] = pd.to_datetime(df['Date.of.Birth'])

df['DisbursalDate'] = pd.to_datetime(df['DisbursalDate'])



days = (df['DisbursalDate'] - df['Date.of.Birth']) / timedelta(1)

years = (df['DisbursalDate'] - df['Date.of.Birth']) / timedelta(365) # Полных лет

weeks = (df['DisbursalDate'] - df['Date.of.Birth']) / timedelta(7)



df['TDIST_DAYS'] = days

df['TDIST_YEARS'] = years.astype(int)

df['TDIST_WEEKS'] = weeks.astype(int)



df['Date.of.Birth.year'] = df['Date.of.Birth'].dt.year

df['Date.of.Birth.month'] = df['Date.of.Birth'].dt.month

df['Date.of.Birth.quarter'] = df['Date.of.Birth'].dt.quarter

df['Date.of.Birth.day'] = df['Date.of.Birth'].dt.day

df['Date.of.Birth.week'] = df['Date.of.Birth'].dt.week

df['Date.of.Birth.dayofweek'] = df['Date.of.Birth'].dt.dayofweek



df['DisbursalDate.year'] = df['DisbursalDate'].dt.year

df['DisbursalDate.month'] = df['DisbursalDate'].dt.month

df['DisbursalDate.quarter'] = df['DisbursalDate'].dt.quarter

df['DisbursalDate.day'] = df['DisbursalDate'].dt.day

df['DisbursalDate.week'] = df['DisbursalDate'].dt.week

df['DisbursalDate.dayofweek'] = df['DisbursalDate'].dt.dayofweek



df.drop(cols_dt, axis=1, inplace=True)
def cat_parse(row, names):

    for idx, value in enumerate(names): 

        if re.findall(value, row):

            return idx
# Уменьшение числа категорий

names = ['Not Scored', 'No Bureau', '-Very Low Risk', 'Low Risk', 'Medium Risk', '-Very High Risk', 'High Risk']

df['PERFORM_CNS.SCORE.DESCRIPTION.AGGR_1'] = df['PERFORM_CNS.SCORE.DESCRIPTION'].apply(cat_parse, **{'names': names})



names = ['Not Scored', 'No Bureau', 'Low Risk', 'Medium Risk', 'High Risk']

df['PERFORM_CNS.SCORE.DESCRIPTION.AGGR_2'] = df['PERFORM_CNS.SCORE.DESCRIPTION'].apply(cat_parse, **{'names': names})
cols_cat = ['Employment.Type', 'PERFORM_CNS.SCORE.DESCRIPTION']

#cols_cat = ['Employment.Type', 'PERFORM_CNS.SCORE.DESCRIPTION', 'AVERAGE.ACCT.AGE', 'CREDIT.HISTORY.LENGTH']
missed_value = 'MISSED'

n_features_limit = 800



total_encoders = {}



for col in cols_cat:

    print(col)

    top_values = df[col].fillna(missed_value).value_counts(ascending=False).index

    ## Топ выбранных

    

    try: # Возможны ошибки, когда ключ не формата str + float

        # Топ объектов по убыванию

        names = top_values[:n_features_limit].tolist() # .astype(float).astype(str).tolist() 

    except:

        names = top_values[:n_features_limit].tolist() # .astype(str).tolist()

        

    encoded_names = [i for i in range(n_features_limit)] # Закодирование представления

    ## Оставшиеся признаки (не топ)

    rest_names = top_values[n_features_limit:].tolist()

    rest_encoded_names = [-1 for i in range(len(rest_names))]



    dict_encoders = dict(zip(names + rest_names, encoded_names + rest_encoded_names)) # Словарь с кодировками

    df[col] = df[col].fillna(missed_value).apply(lambda x: dict_encoders[x]) # Применение словаря к столбцу



    ## Обновление общего словаря кодировок

    total_encoders[col] = dict_encoders



# Сохранение полного словаря кодировок

with open ('encoders.txt', 'w', encoding='utf-8') as file:

    file.write(json.dumps(total_encoders, ensure_ascii=False))
features_to_geterate = list(set(df.columns.values) - set(['target']) - set(id_cols) - set(flag_cols) - set(cols_cat))

print(len(features_to_geterate))



for col in features_to_geterate:

    if df[col].value_counts().shape[0] < 3:

        features_to_geterate.remove(col)

print(len(features_to_geterate))
def scoring(data, target):

    

    import lightgbm as lgb

    import numpy as np

    

    params = {}

    params['app'] = 'binary'

    params['learning_rate'] = 0.05

    params['metric'] = 'auc'

    params['seed'] = 0

    params['num_threads'] = 6



    xtr = lgb.Dataset(data, label=target)



    model = lgb.cv(

        params,

        xtr,

        num_boost_round=10000,

        early_stopping_rounds=201,

        verbose_eval=0

    )

    

    return np.max(model['auc-mean'])





def feature_generation(data, target, n_attempts=10):

    from copy import deepcopy

    from tqdm.notebook import tqdm_notebook

    

    base_score = scoring(data, target)

    base_columns = data.columns.values

    best_generations = {}

    

    

    for i in tqdm_notebook(range(n_attempts)):

        a, b = np.random.choice(base_columns, 2, replace=False)

        new_feature = data[a] / data[b]

        new_feature.replace([np.inf, -np.inf], np.nan, inplace=True)

        

        new_score = scoring(pd.concat((data, new_feature), axis=1), target)

        if new_score > base_score:

            best_generations[a + ' / ' + b] = new_score - base_score

            

    return dict(sorted(best_generations.items(), key=lambda x: x[1], reverse=True))



new_features = feature_generation(df[features_to_geterate], df[var_target], n_attempts=20)
new_features = list(new_features.keys())

len(new_features)
## Генерация новых признаков



for pair in new_features:

    a, _, b = pair.split(' ')

    df[pair] = df[a] / df[b]

    df[pair].replace([np.inf, -np.inf], np.nan, inplace=True)

    

    

features = list(set(df.columns) - set(['UniqueID', 'target']))

var_target = 'target'
def permutation_importance(X, y, cv=5, n_jobs=5, num_threads=2, n_steps=1):

        

        """

        Функция для расчёта feature_importances

    

        Parameters

        ----------

        X :

            pandas DataFrame. Выборка для обучения

        y : 

            array like. Таргет

        cv :

            int. Число фолдов

        n_jobs :

            int. Число процессов для параллельности cv

        num_threads:

            int. Число процессов на 1 n_jobs. Для этапа lgb.train



        Returns

        -------

        dict

            Словарь {название признака: важность признака}



        """



        from sklearn.metrics import roc_auc_score

        from sklearn.model_selection import StratifiedKFold

        from sklearn.model_selection import cross_val_score

        from sklearn.model_selection import train_test_split

        from sklearn.metrics import roc_auc_score

        from joblib import Parallel, delayed

        from copy import deepcopy

        import numpy as np

        import lightgbm as lgb

        from lightgbm import LGBMClassifier

               

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        xtr = lgb.Dataset(X_train, label=y_train)



        params = {}

        params['app'] = 'binary'

        params['learning_rate'] = 0.1

        params['metric'] = 'auc'

        params['seed'] = 0

        params['num_threads'] = num_threads

        

        model = lgb.train(

                            params,

                            xtr,

                            num_boost_round=1000, # ТУТ ПОМЕНЯТЬ

                            valid_sets=[xtr],

                            early_stopping_rounds=50, verbose_eval=False

        )

        

        model.params['num_threads'] = 2

        base_score = roc_auc_score(y_test, model.predict(X_test))

        print('Base score: %0.4f' % base_score)



        

        num_threads = 2

        def permutation(X_test, y_test, col, model, base_score):

            from copy import deepcopy

            X_val = deepcopy(X_test)



            X_val.loc[:, col] = np.random.permutation(X_val.loc[:, col].values)

            model.params['num_threads'] = 2

            pred = model.predict(X_val)

            score = roc_auc_score(y_test, pred)

            

            return base_score - score # если > 0, то переменная важна

        

        def gen_columns(df): # Генератор столбцов

            for col in df.columns:

                yield col



        parallel = Parallel(n_jobs=n_jobs, pre_dispatch='2*n_jobs')

        importance_blocks = parallel(delayed(permutation)(X_test, y_test, col, model, base_score)

                                     for col in gen_columns(X_test))

        

        return importance_blocks# dict(zip(X.columns, importance_blocks))
perm_scores = permutation_importance(df[features], df[var_target], n_jobs=2, num_threads=4)

importance_dict = dict(zip(features, perm_scores))



features_selected = np.array(features)[np.array(list(importance_dict.values())) >= 0]

print(len(features_selected))
perm_scores = permutation_importance(df[features_selected], df[var_target], n_jobs=2, num_threads=4)

importance_dict = dict(zip(features_selected, perm_scores))

features_selected = np.array(features_selected)[np.array(list(importance_dict.values())) >= 0]

print(len(features_selected))
# Проверка результатов (кросс-валидация)



params = {}

params['app'] = 'binary'

params['learning_rate'] = 0.05

params['num_leaves'] = 20

params['min_data'] = 200



params['metric'] = 'auc'

params['seed'] = 0

params['num_threads'] = 4



xtr = lgb.Dataset(df[features_selected],label= df['target'],)



model = lgb.cv(

                    params,

                    xtr,

                    num_boost_round=10000,

                    early_stopping_rounds=201,

    verbose_eval=100

)

print(np.max(model['auc-mean']))
dtrain = lgb.Dataset(df[features_selected],label= df['target'],)



from hyperopt import hp, tpe

from hyperopt.fmin import fmin

from sklearn.metrics import make_scorer

import lightgbm as lgbm

from sklearn.metrics import roc_auc_score



N_JOBS = 5



def lgb_auc_score(preds, dtrain):

    labels = dtrain.get_label()

    return 'auc', roc_auc_score(labels, preds), True



def objective(params):

    

    # Округление и форматирование в int нужных гиперпараметров

    params['num_leaves'] = int(round(params['num_leaves']))

    params['learning_rate'] = max(params['learning_rate'], 0.05)

    params['feature_fraction'] = max(min(params['feature_fraction'], 1), 0)

    params['bagging_fraction'] = max(min(params['bagging_fraction'], 1), 0)

    params['bagging_freq'] = int(round(params['bagging_freq']))

    params['max_bin'] = int(round(params['max_bin']))

    params['max_depth'] = int(round(params['max_depth']))

    #params['scale_pos_weight'] = int(round(scale_pos_weight))

    #params['min_split_gain'] =  params['min_split_gain']

    params['min_child_weight'] = params['min_child_weight']

    params['min_child_samples'] = int(round(params['min_child_samples']))

    params['gamma'] = params['gamma']

    params['reg_alpha'] = params['reg_alpha']

    params['reg_lambda'] = params['reg_lambda']

    params['num_threads'] = N_JOBS

    params['app'] = 'binary'

    params['application'] = 'binary'

            

    cv_results = lgb.cv(params, dtrain, nfold=5, stratified=True, 

                        verbose_eval=None, feval=lgb_auc_score) 



    print("auc {:.5f} ".format(np.max(cv_results['auc-mean'])))



    return - np.max(cv_results['auc-mean'])



space = {

    'num_leaves': hp.quniform('num_leaves', 4, 50, 1),

    'learning_rate' : hp.quniform('learning_rate', 0.01, 0.2, 0.05),

    'feature_fraction': hp.quniform('feature_fraction', 0.6, 0.95, 0.05), 

    'bagging_fraction': hp.quniform('bagging_fraction', 0.6, 0.95, 0.05), 

    'bagging_freq': hp.quniform('bagging_freq', 1, 10, 1), 

    'max_bin' : hp.quniform('max_bin', 10, 200, 5), 

    #'scale_pos_weight' : hp.normal(1, 3),  

    #'min_split_gain': hp.quniform('min_split_gain', 0.3, 0.5, 0.05),

    'min_child_weight': hp.quniform('min_child_weight', 0.01, 0.4, 0.05),

    'min_child_samples': hp.quniform('min_child_samples', 4, 12, 1),

    'gamma': hp.quniform('gamma', 1, 10, 0.5),

    'reg_alpha': hp.quniform('reg_alpha', 1, 20, 0.5),

    'reg_lambda': hp.quniform('reg_lambda', 1, 20, 0.5),

    'max_depth': hp.quniform('max_depth', 4, 40, 1)

}
## Подбор гиперпараметров

best = fmin(

    fn=objective,

    space=space,

    algo=tpe.suggest,

    max_evals=10

)



print(best)



params = best
params['num_leaves'] = int(round(params['num_leaves']))

params['learning_rate'] = max(params['learning_rate'], 0.05)

params['feature_fraction'] = max(min(params['feature_fraction'], 1), 0)

params['bagging_fraction'] = max(min(params['bagging_fraction'], 1), 0)

params['bagging_freq'] = int(round(params['bagging_freq']))

params['max_bin'] = int(round(params['max_bin']))

params['max_depth'] = int(round(params['max_depth']))

#params['scale_pos_weight'] = int(round(scale_pos_weight))

#params['min_split_gain'] =  params['min_split_gain']

params['min_child_weight'] = params['min_child_weight']

params['min_child_samples'] = int(round(params['min_child_samples']))

params['gamma'] = params['gamma']

params['reg_alpha'] = params['reg_alpha']

params['reg_lambda'] = params['reg_lambda']

params['num_threads'] = 4
params['app'] = 'binary'

params['application'] = 'binary'

params['metric'] = 'auc'

params['seed'] = 0
xtr = lgb.Dataset(df[features_selected],label= df['target'], )



model = lgb.cv(params,

               xtr,

               num_boost_round=10000,

               early_stopping_rounds=50,

               verbose_eval=0)

print(np.max(model['auc-mean']))
dtrain = xgb.DMatrix(df[features_selected],label= df['target'],)



from hyperopt import hp, tpe

from hyperopt.fmin import fmin

from sklearn.metrics import make_scorer

import xgboost as xgb

from sklearn.metrics import roc_auc_score



N_JOBS = 5



def lgb_auc_score(preds, dtrain):

    labels = dtrain.get_label()

    return 'auc', roc_auc_score(labels, preds)



def objective(params):

    

    # Округление и форматирование в int нужных гиперпараметров

    params['num_leaves'] = int(round(params['num_leaves']))

    params['learning_rate'] = max(params['learning_rate'], 0.05)

    params['feature_fraction'] = max(min(params['feature_fraction'], 1), 0)

    params['bagging_fraction'] = max(min(params['bagging_fraction'], 1), 0)

    params['bagging_freq'] = int(round(params['bagging_freq']))

    params['max_bin'] = int(round(params['max_bin']))

    params['max_depth'] = int(round(params['max_depth']))

    #params['scale_pos_weight'] = int(round(scale_pos_weight))

    #params['min_split_gain'] =  params['min_split_gain']

    params['min_child_weight'] = params['min_child_weight']

    params['min_child_samples'] = int(round(params['min_child_samples']))

    params['gamma'] = params['gamma']

    params['reg_alpha'] = params['reg_alpha']

    params['reg_lambda'] = params['reg_lambda']

    params['num_threads'] = N_JOBS

    params['app'] = 'binary'

    params['application'] = 'binary'

    params['bosster']='gblinear'

            

    cv_results = xgb.cv(params, dtrain, nfold=5, stratified=True, 

                        verbose_eval=None, feval=lgb_auc_score) 



    print("auc {:.5f} ".format(np.max(cv_results['test-auc-mean'])))



    return - np.max(cv_results['test-auc-mean'])



space = {

    'num_leaves': hp.quniform('num_leaves', 4, 50, 1),

    'learning_rate' : hp.quniform('learning_rate', 0.01, 0.2, 0.05),

    'feature_fraction': hp.quniform('feature_fraction', 0.6, 0.95, 0.05), 

    'bagging_fraction': hp.quniform('bagging_fraction', 0.6, 0.95, 0.05), 

    'bagging_freq': hp.quniform('bagging_freq', 1, 10, 1), 

    'max_bin' : hp.quniform('max_bin', 10, 200, 5), 

    #'scale_pos_weight' : hp.normal(1, 3),  

    #'min_split_gain': hp.quniform('min_split_gain', 0.3, 0.5, 0.05),

    'min_child_weight': hp.quniform('min_child_weight', 0.01, 0.4, 0.05),

    'min_child_samples': hp.quniform('min_child_samples', 4, 12, 1),

    'gamma': hp.quniform('gamma', 1, 10, 0.5),

    'reg_alpha': hp.quniform('reg_alpha', 1, 20, 0.5),

    'reg_lambda': hp.quniform('reg_lambda', 1, 20, 0.5),

    'max_depth': hp.quniform('max_depth', 4, 40, 1)

}
## Подбор гиперпараметров

best = fmin(

    fn=objective,

    space=space,

    algo=tpe.suggest,

    max_evals=3

)



print(best)
best['num_leaves'] = int(round(best['num_leaves']))

best['learning_rate'] = max(best['learning_rate'], 0.05)

best['feature_fraction'] = max(min(best['feature_fraction'], 1), 0)

best['bagging_fraction'] = max(min(best['bagging_fraction'], 1), 0)

best['bagging_freq'] = int(round(best['bagging_freq']))

best['max_bin'] = int(round(best['max_bin']))

best['max_depth'] = int(round(best['max_depth']))

#best['scale_pos_weight'] = int(round(scale_pos_weight))

#best['min_split_gain'] =  best['min_split_gain']

best['min_child_weight'] = best['min_child_weight']

best['min_child_samples'] = int(round(best['min_child_samples']))

best['gamma'] = best['gamma']

best['reg_alpha'] = best['reg_alpha']

best['reg_lambda'] = best['reg_lambda']

best['num_threads'] = 4
dtrain = xgb.DMatrix(df[features_selected],label= df['target'],)



cv_results = xgb.cv(best, dtrain, nfold=5, stratified=True, 

                    verbose_eval=None, feval=lgb_auc_score) 



print("auc {:.5f} ".format(np.max(cv_results['test-auc-mean'])))
test = pd.read_csv('/kaggle/input/hse-practical-ml-1/car_loan_test.csv')
acc_parsed = test['AVERAGE.ACCT.AGE'].apply(parse_date)



acc_year = []

acc_month = []

acc_total = []

for val in acc_parsed.values:

    acc_year.append(val[0])

    acc_month.append(val[1])

    acc_total.append(val[0] + (val[1] / 12))

    

    

test['AVERAGE.ACCT.AGE.YEAR'] = acc_year

test['AVERAGE.ACCT.AGE.MONTHS'] = acc_month

test['AVERAGE.ACCT.AGE.TOTAL'] = acc_total



test.drop('AVERAGE.ACCT.AGE', axis=1, inplace=True)



#######################################################################



cred_parsed = test['CREDIT.HISTORY.LENGTH'].apply(parse_date)



cred_year = []

cred_month = []

cred_total = []

for val in cred_parsed.values:

    cred_year.append(val[0])

    cred_month.append(val[1])

    cred_total.append(val[0] + (val[1] / 12))

    

    

test['CREDIT.HISTORY.LENGTH.YEAR'] = cred_year

test['CREDIT.HISTORY.LENGTH.MONTHS'] = cred_month

test['CREDIT.HISTORY.LENGTH.TOTAL'] = cred_total



test.drop('CREDIT.HISTORY.LENGTH', axis=1, inplace=True)



#######################################################################



test['Date.of.Birth'] = pd.to_datetime(test['Date.of.Birth'])

test['DisbursalDate'] = pd.to_datetime(test['DisbursalDate'])



days = (test['DisbursalDate'] - test['Date.of.Birth']) / timedelta(1)

years = (test['DisbursalDate'] - test['Date.of.Birth']) / timedelta(365) # Полных лет

weeks = (test['DisbursalDate'] - test['Date.of.Birth']) / timedelta(7)



test['TDIST_DAYS'] = days

test['TDIST_YEARS'] = years.astype(int)

test['TDIST_WEEKS'] = weeks.astype(int)



test['Date.of.Birth.year'] = test['Date.of.Birth'].dt.year

test['Date.of.Birth.month'] = test['Date.of.Birth'].dt.month

test['Date.of.Birth.quarter'] = test['Date.of.Birth'].dt.quarter

test['Date.of.Birth.day'] = test['Date.of.Birth'].dt.day

test['Date.of.Birth.week'] = test['Date.of.Birth'].dt.week

test['Date.of.Birth.dayofweek'] = test['Date.of.Birth'].dt.dayofweek



test['DisbursalDate.year'] = test['DisbursalDate'].dt.year

test['DisbursalDate.month'] = test['DisbursalDate'].dt.month

test['DisbursalDate.quarter'] = test['DisbursalDate'].dt.quarter

test['DisbursalDate.day'] = test['DisbursalDate'].dt.day

test['DisbursalDate.week'] = test['DisbursalDate'].dt.week

test['DisbursalDate.dayofweek'] = test['DisbursalDate'].dt.dayofweek



test.drop(cols_dt, axis=1, inplace=True)





# Уменьшение числа категорий

names = ['Not Scored', 'No Bureau', '-Very Low Risk', 'Low Risk', 'Medium Risk', '-Very High Risk', 'High Risk']

test['PERFORM_CNS.SCORE.DESCRIPTION.AGGR_1'] = test['PERFORM_CNS.SCORE.DESCRIPTION'].apply(cat_parse, **{'names': names})



names = ['Not Scored', 'No Bureau', 'Low Risk', 'Medium Risk', 'High Risk']

test['PERFORM_CNS.SCORE.DESCRIPTION.AGGR_2'] = test['PERFORM_CNS.SCORE.DESCRIPTION'].apply(cat_parse, **{'names': names})



#######################################################################\







for col_name in set(features_selected).intersection(set(total_encoders.keys())):

    test[col_name].fillna('MISSED', inplace=True) # Заполняем пропуски особой категорией

    test[col_name] = test[col_name].astype('str')

    

    # Проверка на новые уникальные значения столбца в данных

    value_difference = set(test[col_name].unique()) - set(np.unique(list(total_encoders[col_name].keys())))

    

    ## для обработки признака с более 800 уникальных значений

    if len(np.unique(list(total_encoders[col_name].keys()))) > 800:

        

        if len(value_difference) != 0: # Всем новым значениям присвоим значение категории MISSED

            for new_value in value_difference:

                total_encoders[col_name][new_value] = -1 # Интовое значение для редких столбцов

                

    else:

        if len(value_difference) != 0: # Всем новым значениям присвоим значение категории MISSED

            for new_value in value_difference:

                try: # Проверка на наличие категории MISSED

                    total_encoders[col_name][new_value] = total_encoders[col_name]['MISSED']

                    

                except: # Присваивание значения самой часто встречаемой категории

                    total_encoders[col_name][new_value] = 0

            

    # ! Возможна ошибка в том случае: 1 - в train не было missed, 2 - в train не было новых категорий и missed.

    

    # Применение словаря к столбцу

    

    #try: # Возможны ошибки - float - int

    test[col_name] = test[col_name].apply(lambda x: total_encoders[col_name][x])
## Генерация новых признаков



for pair in new_features:

    a, _, b = pair.split(' ')

    test[pair] = test[a] / test[b]

    test[pair].replace([np.inf, -np.inf], np.nan, inplace=True)
### Процесс обучения и усреднения результатов

import warnings

lgb_aucs = []

xgb_aucs = []

#lin_aucs = []



N_MODELS = 2

N_SPLITS = 3

count = 0

# test[features_selected]

test_xgb = xgb.DMatrix(test[features_selected])

results = np.zeros((test.shape[0], N_MODELS * N_SPLITS))



cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

for train, tst in cv.split(df[features_selected], df[var_target]):

    print(count + 1, end=' ')

    warnings.filterwarnings('ignore')

    xtr = lgb.Dataset(df[features_selected].iloc[train],label= df['target'].iloc[train])

    xtr_xgb = xgb.DMatrix(df[features_selected].iloc[train],label= df['target'].iloc[train])

    xts_xgb = xgb.DMatrix(df[features_selected].iloc[tst],label= df['target'].iloc[tst])

    #xts = lgb.Dataset(df[features_selected].iloc[test],label= df['target'].iloc[test])

    

    model_1 = lgb.train(params, xtr, num_boost_round=180)

    model_2 = xgb.train(best, xtr_xgb)

    #model_3 = LogisticRegression(tol=0.00000001, C=1000).fit(lin_df[features_selected].iloc[train], 

    #                                                         lin_df[var_target].iloc[train])

    

    preds_1 = model_1.predict(df[features_selected].iloc[tst])

    preds_2 = model_2.predict(xts_xgb)

    #preds_3 = model_3.predict_proba(df[features_selected].iloc[tst])[:, 1]

    

    lgb_aucs.append(roc_auc_score(df[var_target].iloc[tst], preds_1))

    xgb_aucs.append(roc_auc_score(df[var_target].iloc[tst], preds_2))

    #lin_aucs.append(roc_auc_score(lin_df[var_target].iloc[tst], preds_3))

    

    ######### Предсказание на тесте #########

    

    preds_1 = model_1.predict(test[features_selected])

    preds_2 = model_2.predict(test_xgb)

    #preds_3 = model_3.predict_proba(test_df[features_selected])[:, 1]

    

    

    results[:, count+0] = preds_1

    results[:, count+1] = preds_2

    #results[:, count+2] = preds_3

    

    count += 1

    

print('lgb: %0.5f' % np.mean(lgb_aucs))

print('xgb: %0.5f' % np.mean(xgb_aucs))

#print('log: %0.5f' % np.mean(lin_aucs))
preds = results.mean(axis=1)

assert np.sum(preds <= 0) == 0
submit = pd.DataFrame()

submit['ID'] = [i for i in range(len(test['UniqueID']))]

submit['Predicted'] = preds
submit.to_csv('submission.csv', index=False)