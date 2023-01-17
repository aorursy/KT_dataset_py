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



import lightgbm as lgb

from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

from sklearn.metrics import roc_auc_score



from tqdm import tqdm_notebook

import gc
train = pd.read_csv('/kaggle/input/widsdatathon2020/training_v2.csv')

test = pd.read_csv('/kaggle/input/widsdatathon2020/unlabeled.csv')

sample_submission = pd.read_csv('/kaggle/input/widsdatathon2020/samplesubmission.csv')

solution_template = pd.read_csv('/kaggle/input/widsdatathon2020/solution_template.csv')

train.shape, test.shape
train.sample(5)
test.head(5)
def make_submit(y_pred, filename='submission.csv'):

    solution_template['hospital_death'] = y_pred

    solution_template.to_csv(f'{filename}', index=False)

    print('solution file created. Commit notebook and submit file...')

    solution_template['hospital_death'].hist()

    
# LightGBM GBDT with KFold or Stratified KFold



def kfold_lightgbm(train, test, target_col, params, cols_to_drop=None, cat_features=None, num_folds=5, stratified = False, 

                   debug= False):

    

    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train.shape, test.shape))





    

    # Cross validation model

    if stratified:

        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=1)

    else:

        folds = KFold(n_splits= num_folds, shuffle=True, random_state=1)



    # Create arrays and dataframes to store results

    oof_preds = np.zeros(train.shape[0])

    sub_preds = np.zeros(test.shape[0])

    feature_importance_df = pd.DataFrame()

    if cols_to_drop == None:

        feats = [f for f in train.columns if f not in [target_col]]

    else:

        feats = [f for f in train.columns if f not in cols_to_drop+[target_col]]



    # k-fold

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train[feats], train[target_col])):

        train_x, train_y = train[feats].iloc[train_idx], train[target_col].iloc[train_idx]

        valid_x, valid_y = train[feats].iloc[valid_idx], train[target_col].iloc[valid_idx]



        # set data structure

        lgb_train = lgb.Dataset(train_x,

                                label=train_y,

                                categorical_feature=cat_features,

                                free_raw_data=False)

        lgb_test = lgb.Dataset(valid_x,

                               label=valid_y,

                               categorical_feature=cat_features,

                               free_raw_data=False)



        # params after optimization

        reg = lgb.train(

                        params,

                        lgb_train,

                        valid_sets=[lgb_train, lgb_test],

                        valid_names=['train', 'test'],

#                         num_boost_round=10000,

#                         early_stopping_rounds= 200,

                        verbose_eval=False

                        )



        roc_auc = []

        oof_preds[valid_idx] = reg.predict(valid_x, num_iteration=reg.best_iteration)

        sub_preds += reg.predict(test[feats], num_iteration=reg.best_iteration) / folds.n_splits



        fold_importance_df = pd.DataFrame()

        fold_importance_df["feature"] = feats

        fold_importance_df["importance"] = np.log1p(reg.feature_importance(importance_type='gain', 

                                                                           iteration=reg.best_iteration))

        fold_importance_df["fold"] = n_fold + 1

        

        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        print('Fold %2d ROC-AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))

        roc_auc.append(roc_auc_score(valid_y, oof_preds[valid_idx]))

        del reg, train_x, train_y, valid_x, valid_y

        gc.collect()

        

    print('Mean ROC-AUC : %.6f' % (np.mean(roc_auc)))

    return sub_preds
cat_features = [x for x in train.columns if train[x].dtype == 'object' ]
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()



# Для основных датасетов

for col in tqdm_notebook(cat_features):

    train[col] = train[col].astype('str')

    train[col] = le.fit_transform(train[col])

    

for col in tqdm_notebook(cat_features):

    test[col] = test[col].astype('str')

    test[col] = le.fit_transform(test[col])
params ={

    'objective': 'binary',

    'metric': 'roc_auc',

    'categorical_features': cat_features

                }

params_best = {

    'bagging_fraction': 0.15760757965010433, 

    'feature_fraction': 0.11161740354830015, 

    'learning_rate': 0.03, 

    'max_depth': 50, 

    'min_child_weight': 0.008857217513412136, 

    'min_data_in_leaf': 20, 

    'n_estimators': 815, 

    'num_leaves': 96, 

    'reg_alpha': 1.5292311993088907, 

    'reg_lambda': 1.903834634991243}
submit_best_params = kfold_lightgbm(train, test.drop('hospital_death', axis=1), cat_features=cat_features, 

                                 target_col='hospital_death', params=params_best)
make_submit(submit_best_params, 'submission.csv')
baseline_submit = kfold_lightgbm(train, test.drop('hospital_death', axis=1), cat_features=cat_features, 

                                 target_col='hospital_death', params=params)
def bayes_auc_lgb(

    n_estimators,

    learning_rate,

    num_leaves, 

    bagging_fraction,

    feature_fraction,

    min_child_weight, 

    min_data_in_leaf,

    max_depth,

    reg_alpha,

    reg_lambda):

    

    """

    До запуска надо переопределить ИСХОДНЫЙ ДАТАФРЕЙМ и КАТЕГОРИАЛЬНЫЕ ПРИЗНАКИ

    """

    

    # На вход LightGBM следующие парамерты должны подаваться в виде целых чисел. 

    n_estimators = int(n_estimators)

    num_leaves = int(num_leaves)

    min_data_in_leaf = int(min_data_in_leaf)

    max_depth = int(max_depth)

    

    assert type(n_estimators) == int

    assert type(num_leaves) == int

    assert type(min_data_in_leaf) == int

    assert type(max_depth) == int

    

    params = {

              'n_estimators': n_estimators,

              'num_leaves': num_leaves, 

              'min_data_in_leaf': min_data_in_leaf,

              'min_child_weight': min_child_weight,

              'bagging_fraction' : bagging_fraction,

              'feature_fraction' : feature_fraction,

              'learning_rate' : learning_rate,

              'max_depth': max_depth,

              'reg_alpha': reg_alpha,

              'reg_lambda': reg_lambda,

              'objective': 'binary',

              'save_binary': True,

              'seed': 1337,

              'feature_fraction_seed': 1337,

              'bagging_seed': 1337,

              'drop_seed': 1337,

              'data_random_seed': 1337,

              'boosting_type': 'gbdt',

              'verbose': 1,

              'is_unbalance': False,

              'boost_from_average': True,

              'metric':'f1'}



    

    # кросс-валидация

    folds = StratifiedKFold(n_splits= 5, shuffle=True, random_state=1)





    # Массивы для сохранения результатов

    oof_preds = np.zeros(df.shape[0])



    feats = [f for f in df.columns if f not in ['hospital_death']]



    # k-fold

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(df[feats], df['hospital_death'])):

        train_x, train_y = df[feats].iloc[train_idx], df['hospital_death'].iloc[train_idx]

        valid_x, valid_y = df[feats].iloc[valid_idx], df['hospital_death'].iloc[valid_idx]



        # датасеты для обучения

        lgb_train = lgbm.Dataset(train_x,

                                label=train_y,

                                categorical_feature=cat_f,

                                free_raw_data=False)

        lgb_test = lgbm.Dataset(valid_x,

                               label=valid_y,

                               categorical_feature=cat_f,

                               free_raw_data=False)



        # Обучение

        clf = lgbm.train(

                        params,

                        lgb_train,

                        valid_sets=[lgb_train, lgb_test],

                        verbose_eval=False

                        )



        auc = []

        oof_preds[valid_idx] = clf.predict(valid_x, num_iteration=clf.best_iteration)







        auc.append(roc_auc_score(valid_y, oof_preds[valid_idx]))



    return np.mean(auc)
# Границы параметров для поиска   

bounds_lgb = {

    'n_estimators': (10, 1000),

    'num_leaves': (10, 500), 

    'min_data_in_leaf': (20, 200),

    'bagging_fraction' : (0.1, 0.9),

    'feature_fraction' : (0.1, 0.9),

    'learning_rate': (0.01, 0.3),

    'min_child_weight': (0.00001, 0.01),   

    'reg_alpha': (1, 2), 

    'reg_lambda': (1, 2),

    'max_depth':(-1,50),

}
def bayes_lgb(score_func, bound_lgb, init_points: int = 10, n_iter:int = 100):

    """

    Поиск оптимальных гиперпараметров для алгоритма с использованием байесовской оптимизации. 

    :param score_func: функция для оптимизации. Определена отдельно.

    :param bounds_lgb: границы диапазона для поисков параметров (словарь). Задается отдельно

    :param n_iter: количество итераций

    :return: Оптимальные параметры в виде словаря

    """

    # инициализация оптимизатора

    lgb_bo = BayesianOptimization(score_func, bounds_lgb, verbose=0)

    

    # поиск

    lgb_bo.maximize(init_points=init_points, n_iter=n_iter, xi=0.0, alpha=1e-6)

    

    print("Максимальное значение метрики: ",lgb_bo.max['target'])

    print("Оптимальные параметры: ", lgb_bo.max['params'])

    

    return lgb_bo.max['params']
from bayes_opt import BayesianOptimization
cat_features.remove('icu_stay_type')
df = train.copy().drop(['encounter_id', 'patient_id', 'hospital_id', 'icu_stay_type', 'icu_id'], axis=1)

cat_f=cat_features



bo_best_params = bayes_lgb(bayes_auc_lgb, bounds_lgb)