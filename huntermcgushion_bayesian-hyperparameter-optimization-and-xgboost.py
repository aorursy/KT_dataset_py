###############################################
# Import Machine Learning Assets
###############################################
from bayes_opt import BayesianOptimization
from xgboost import XGBRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

###############################################
# Import Miscellaneous Assets
###############################################
import numpy as np
import pandas as pd
import warnings
from datetime import datetime
from functools import partial
from pprint import pprint as pp
from tqdm import tqdm, tqdm_notebook

pd.set_option('display.expand_frame_repr', False)

###############################################
# Declare Global Variables
###############################################
CROSS_VALIDATION_PARAMS = dict(n_splits=5, shuffle=True, random_state=32)
XGBOOST_REGRESSOR_PARAMS = dict(
    learning_rate=0.2, n_estimators=200, subsample=0.8, colsample_bytree=0.8, 
    max_depth=10, n_jobs=-1
)

BAYESIAN_OPTIMIZATION_MAXIMIZE_PARAMS = dict(
    init_points=1,  # init_points=20,
    n_iter=2,  # n_iter=60,
    acq='poi', xi=0.0
)
BAYESIAN_OPTIMIZATION_BOUNDARIES = dict(
    max_depth=(5, 12.99),
    gamma=(0.01, 5),
    min_child_weight=(0, 6),
    scale_pos_weight=(1.2, 5),
    reg_alpha=(4.0, 10.0),
    reg_lambda=(1.0, 10.0),
    max_delta_step=(0, 5),
    subsample=(0.5, 1.0),
    colsample_bytree=(0.3, 1.0),
    learning_rate=(0.0, 1.0)
)
BAYESIAN_OPTIMIZATION_INITIAL_SEARCH_POINTS = dict(
    max_depth=[5, 10],
    gamma=[0.1511, 3.8463],
    min_child_weight=[2.4073, 4.9954],
    scale_pos_weight=[2.2281, 4.0345],
    reg_alpha=[8.0702, 9.0573],
    reg_lambda=[2.0126, 3.5934],
    max_delta_step=[1, 2],
    subsample=[0.8, 0.8234],
    colsample_bytree=[0.8, 0.7903],
    learning_rate=[0.2, 0.1]
)

reserve_features = [
    'rs1_x', 'rs1_y', 'rs2_x', 'rs2_y', 'rv1_x', 'rv1_y', 'rv2_x', 'rv2_y',
    'total_reserve_dt_diff_mean', 'total_reserve_mean', 'total_reserve_sum'
]

BASE_ESTIMATOR = partial(XGBRegressor)
# train_df, test_df = None, None
# oof_predictions, test_predictions = None, None
# train_input = None
# best_round = None
# target = None
data = {
    'tra': pd.read_csv('../input/air_visit_data.csv'),
    'as': pd.read_csv('../input/air_store_info.csv'),
    'hs': pd.read_csv('../input/hpg_store_info.csv'),
    'ar': pd.read_csv('../input/air_reserve.csv'),
    'hr': pd.read_csv('../input/hpg_reserve.csv'),
    'id': pd.read_csv('../input/store_id_relation.csv'),
    'tes': pd.read_csv('../input/sample_submission.csv'),
    'hol': pd.read_csv('../input/date_info.csv').rename(columns={'calendar_date':'visit_date'})
}

data['hr'] = pd.merge(data['hr'], data['id'], how='inner', on=['hpg_store_id'])
for df in ['ar','hr']:
    data[df]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime'])
    data[df]['visit_dow'] = data[df]['visit_datetime'].dt.dayofweek
    data[df]['visit_datetime'] = data[df]['visit_datetime'].dt.date
    data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime'])
    data[df]['reserve_datetime'] = data[df]['reserve_datetime'].dt.date
    data[df]['reserve_datetime_diff'] = data[df].apply(
        lambda _: (_['visit_datetime'] - _['reserve_datetime']).days, axis=1
    )
    
    ###############################################
    # aharless's Same-Week Reservation Exclusion
    ###############################################
    data[df] = data[df][data[df]['reserve_datetime_diff'] > data[df]['visit_dow']]
    tmp1 = data[df].groupby(
        ['air_store_id','visit_datetime'], as_index=False
    )[['reserve_datetime_diff', 'reserve_visitors']].sum().rename(columns={
        'visit_datetime':'visit_date', 
        'reserve_datetime_diff': 'rs1', 
        'reserve_visitors':'rv1'
    })
    tmp2 = data[df].groupby(
        ['air_store_id','visit_datetime'], as_index=False
    )[['reserve_datetime_diff', 'reserve_visitors']].mean().rename(columns={
        'visit_datetime':'visit_date', 
        'reserve_datetime_diff': 'rs2', 
        'reserve_visitors':'rv2'
    })
    data[df] = pd.merge(tmp1, tmp2, how='inner', on=['air_store_id','visit_date'])

data['tra']['visit_date'] = pd.to_datetime(data['tra']['visit_date'])
data['tra']['dow'] = data['tra']['visit_date'].dt.dayofweek
data['tra']['year'] = data['tra']['visit_date'].dt.year
data['tra']['month'] = data['tra']['visit_date'].dt.month
data['tra']['visit_date'] = data['tra']['visit_date'].dt.date

data['tes']['visit_date'] = data['tes']['id'].map(lambda _: str(_).split('_')[2])
data['tes']['air_store_id'] = data['tes']['id'].map(lambda _: '_'.join(_.split('_')[:2]))
data['tes']['visit_date'] = pd.to_datetime(data['tes']['visit_date'])
data['tes']['dow'] = data['tes']['visit_date'].dt.dayofweek
data['tes']['year'] = data['tes']['visit_date'].dt.year
data['tes']['month'] = data['tes']['visit_date'].dt.month
data['tes']['visit_date'] = data['tes']['visit_date'].dt.date

unique_stores = data['tes']['air_store_id'].unique()
stores = pd.concat(
    [pd.DataFrame({
        'air_store_id': unique_stores, 
        'dow': [_] * len(unique_stores)
    }) for _ in range(7)], 
    axis=0, ignore_index=True
).reset_index(drop=True)

###############################################
# Jerome Vallet's Optimization
###############################################
tmp = data['tra'].groupby(['air_store_id', 'dow']).agg(
    {'visitors': [np.min, np.mean, np.median, np.max, np.size]}
).reset_index()
tmp.columns = [
    'air_store_id', 'dow', 'min_visitors', 'mean_visitors', 'median_visitors', 'max_visitors', 
    'count_observations'
]
stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
stores = pd.merge(stores, data['as'], how='left', on=['air_store_id'])

###############################################
# Georgii Vyshnia's Features
###############################################
stores['air_genre_name'] = stores['air_genre_name'].map(lambda _: str(str(_).replace('/',' ')))
stores['air_area_name'] = stores['air_area_name'].map(lambda _: str(str(_).replace('-',' ')))
lbl = LabelEncoder()
for i in range(10):
    stores['air_genre_name' + str(i)] = lbl.fit_transform(stores['air_genre_name'].map(
        lambda _: str(str(_).split(' ')[i]) if len(str(_).split(' ')) > i else ''
    ))
    stores['air_area_name' + str(i)] = lbl.fit_transform(stores['air_area_name'].map(
        lambda _: str(str(_).split(' ')[i]) if len(str(_).split(' ')) > i else ''
    ))
stores['air_genre_name'] = lbl.fit_transform(stores['air_genre_name'])
stores['air_area_name'] = lbl.fit_transform(stores['air_area_name'])

data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])
data['hol']['day_of_week'] = lbl.fit_transform(data['hol']['day_of_week'])
data['hol']['visit_date'] = data['hol']['visit_date'].dt.date
train = pd.merge(data['tra'], data['hol'], how='left', on=['visit_date']) 
test = pd.merge(data['tes'], data['hol'], how='left', on=['visit_date']) 

train = pd.merge(train, stores, how='inner', on=['air_store_id','dow']) 
test = pd.merge(test, stores, how='left', on=['air_store_id','dow'])

for df in ['ar','hr']:
    train = pd.merge(train, data[df], how='left', on=['air_store_id', 'visit_date']) 
    test = pd.merge(test, data[df], how='left', on=['air_store_id', 'visit_date'])

train['id'] = train.apply(
    lambda _: '_'.join([str(_['air_store_id']), str(_['visit_date'])]), axis=1
)

train['total_reserv_sum'] = train['rv1_x'] + train['rv1_y']
train['total_reserv_mean'] = (train['rv2_x'] + train['rv2_y']) / 2
train['total_reserv_dt_diff_mean'] = (train['rs2_x'] + train['rs2_y']) / 2

test['total_reserv_sum'] = test['rv1_x'] + test['rv1_y']
test['total_reserv_mean'] = (test['rv2_x'] + test['rv2_y']) / 2
test['total_reserv_dt_diff_mean'] = (test['rs2_x'] + test['rs2_y']) / 2

###############################################
# JMBULL's Features 
###############################################
train['date_int'] = train['visit_date'].apply(lambda _: _.strftime('%Y%m%d')).astype(int)
test['date_int'] = test['visit_date'].apply(lambda _: _.strftime('%Y%m%d')).astype(int)
train['var_max_lat'] = train['latitude'].max() - train['latitude']
train['var_max_long'] = train['longitude'].max() - train['longitude']
test['var_max_lat'] = test['latitude'].max() - test['latitude']
test['var_max_long'] = test['longitude'].max() - test['longitude']

###############################################
# Georgii Vyshnia's Features
###############################################
train['lon_plus_lat'] = train['longitude'] + train['latitude'] 
test['lon_plus_lat'] = test['longitude'] + test['latitude']

lbl = LabelEncoder()
train['air_store_id2'] = lbl.fit_transform(train['air_store_id'])
test['air_store_id2'] = lbl.transform(test['air_store_id'])

col = [_ for _ in train if _ not in ['id', 'air_store_id', 'visit_date','visitors']]
train = train.fillna(-1)
test = test.fillna(-1)

train_df = train[col]
test_df = test[col]
target = pd.DataFrame()
target['visitors'] = np.log1p(train['visitors'].values)
def search_node(**kwargs):
    # global train_df, test_df, train_input, oof, test_predictions, best_round, target
    global train_df, target

    ###############################################
    # Unify Parameters
    ###############################################
    received_params = dict(dict(
        n_estimators=200,
    ), **{_k: _v if _k not in ('max_depth') else int(_v) for _k, _v in kwargs.items()})
    
    current_params = dict(XGBOOST_REGRESSOR_PARAMS, **received_params)

    ###############################################
    # Initialize Folds and Result Placeholders
    ###############################################
    folds = KFold(**CROSS_VALIDATION_PARAMS)
    evaluation = np.zeros((current_params['n_estimators'], CROSS_VALIDATION_PARAMS['n_splits']))
    oof_predictions = np.empty(len(train_df))
    np.random.seed(32)

    progress_bar = tqdm_notebook(
        enumerate(folds.split(target, target)), 
        total=CROSS_VALIDATION_PARAMS['n_splits'], 
        leave=False
    )
    
    ###############################################
    # Begin Cross-Validation
    ###############################################
    for fold, (train_index, validation_index) in progress_bar:
        train_input, validation_input = train_df.iloc[train_index], train_df.iloc[validation_index]
        train_target, validation_target = target.iloc[train_index], target.iloc[validation_index]

        ###############################################
        # Initialize and Fit Model With Current Parameters
        ###############################################
        model = BASE_ESTIMATOR(**current_params)
        eval_set = [(train_input, train_target), (validation_input, validation_target)]
        model.fit(train_input, train_target, eval_set=eval_set, verbose=False)

        ###############################################
        # Find Best Round for Validation Set
        ###############################################
        evaluation[:, fold] = model.evals_result_["validation_1"]['rmse']
        best_round = np.argsort(evaluation[:, fold])[0]

        progress_bar.set_description('Fold #{}:   {:.5f}'.format(
            fold, evaluation[:, fold][best_round]
        ), refresh=True)

    ###############################################
    # Compute Mean and Standard Deviation of RMSLE
    ###############################################
    mean_eval, std_eval = np.mean(evaluation, axis=1), np.std(evaluation, axis=1)
    best_round = np.argsort(mean_eval)[0]
    search_value = mean_eval[best_round]

    ###############################################
    # Update Best Score and Return Negative Value
    # In order to minimize error, instead of maximizing accuracy
    ###############################################
    print(' Stopped After {} Epochs... Validation RMSLE: {:.6f} +- {:.6f}'.format(
        best_round, search_value, std_eval[best_round]
    ))

    return -search_value
bayes_opt = BayesianOptimization(search_node, BAYESIAN_OPTIMIZATION_BOUNDARIES)
bayes_opt.explore(BAYESIAN_OPTIMIZATION_INITIAL_SEARCH_POINTS)

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    bayes_opt.maximize(**BAYESIAN_OPTIMIZATION_MAXIMIZE_PARAMS)
print('Maximum Value: {}'.format(bayes_opt.res['max']['max_val']))
print('Best Parameters:')
pp(bayes_opt.res['max']['max_params'])
bayes_opt.points_to_csv('bayes_opt_search_points.csv')

best_params = dict(XGBOOST_REGRESSOR_PARAMS, **dict(
    n_estimators=200,
    learning_rate=bayes_opt.res['max']['max_params']['learning_rate'],
    max_depth=int(bayes_opt.res['max']['max_params']['max_depth']),
    gamma=bayes_opt.res['max']['max_params']['gamma'],
    min_child_weight=bayes_opt.res['max']['max_params']['min_child_weight'],
    max_delta_step=int(bayes_opt.res['max']['max_params']['max_delta_step']),
    subsample=bayes_opt.res['max']['max_params']['subsample'],
    colsample_bytree=bayes_opt.res['max']['max_params']['colsample_bytree'],
    scale_pos_weight=bayes_opt.res['max']['max_params']['scale_pos_weight'],
    reg_alpha=bayes_opt.res['max']['max_params']['reg_alpha'],
    reg_lambda=bayes_opt.res['max']['max_params']['reg_lambda']
))
def RMSLE(target, prediction):
    return metrics.mean_squared_error(target, prediction) ** 0.5

###############################################
# Initialize Folds and Result Placeholders
###############################################
folds = KFold(**CROSS_VALIDATION_PARAMS)
imp_df = np.zeros((len(train_df.columns), CROSS_VALIDATION_PARAMS['n_splits']))
best_evaluation = np.zeros((best_params['n_estimators'], CROSS_VALIDATION_PARAMS['n_splits']))
oof_predictions, test_predictions = np.empty(train_df.shape[0]), np.zeros(test_df.shape[0])
np.random.seed(32)

for fold, (train_index, validation_index) in enumerate(folds.split(target, target)):
    train_input, validation_input = train_df.iloc[train_index], train_df.iloc[validation_index]
    train_target, validation_target = target.iloc[train_index], target.iloc[validation_index]
    
    ###############################################
    # Initialize and Fit Model With Best Parameters
    ###############################################
    model = BASE_ESTIMATOR(**best_params)
    eval_set=[(train_input, train_target), (validation_input, validation_target)]
    model.fit(train_input, train_target, eval_set=eval_set, verbose=False)

    ###############################################
    # Record Feature Importances and Best OOF Round
    ###############################################
    imp_df[:, fold] = model.feature_importances_
    best_evaluation[:, fold] = model.evals_result_["validation_1"]['rmse']
    # best_round = np.argsort(xgb_evaluation[:, fold])[::-1][0]  # FLAG: ORIGINAL
    best_round = np.argsort(best_evaluation[:, fold])[0]  # FLAG: TEST

    ###############################################
    # Make OOF and Test Predictions With Best Round
    ###############################################
    oof_predictions[validation_index] = model.predict(validation_input, ntree_limit=best_round)
    test_predictions += model.predict(test_df, ntree_limit=best_round)

    ###############################################
    # Report Results for Fold
    ###############################################
    oof_rmsle = RMSLE(validation_target, oof_predictions[validation_index])
    print('Fold {}: {:.6f}     Best Score: {:.6f} @ {:4}'.format(
        fold, oof_rmsle, best_evaluation[best_round, fold], best_round
    ))

print('#' * 80 + '\n')
print('OOF RMSLE   {}'.format(RMSLE(target, oof_predictions)))

###############################################
# Compute Mean and Standard Deviation RMSLE
###############################################
test_predictions /= CROSS_VALIDATION_PARAMS['n_splits']
mean_eval, std_eval = np.mean(best_evaluation, axis=1), np.std(best_evaluation, axis=1)
best_round = np.argsort(mean_eval)[0]
print('Best Mean Score: {:.6f} +- {:.6f} @{:4}'.format(
    mean_eval[best_round], std_eval[best_round], best_round
))

importances = sorted(
    [(train_df.columns[i], imp) for i, imp in enumerate(imp_df.mean(axis=1))], 
    key=lambda x: x[1]
)

final_df = pd.DataFrame(
    data=list(zip(test['id'], np.expm1(test_predictions))), columns=['id', 'visitors']
).to_csv('submission_xgb-bayes-opt.csv', index=False, float_format="%.9f")

print('Feature Importances')
pp(importances)
