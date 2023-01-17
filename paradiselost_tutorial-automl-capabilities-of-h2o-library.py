!pip install h2o
import sys, os, os.path

import warnings

warnings.filterwarnings('ignore')



import numpy as np

import pandas as pd

import pickle



import h2o

from h2o.automl import H2OAutoML



h2o.init(

    nthreads=-1,     # number of threads when launching a new H2O server

    max_mem_size=12  # in gigabytes

)
train_df = pd.read_csv('../input/mlcourse/flight_delays_train.csv')

test_df = pd.read_csv('../input/mlcourse/flight_delays_test.csv')
print('train_df cols:', list(train_df.columns))

print('test_df cols: ', list(test_df.columns))

train_df.head()
train_df.dtypes
for df in [train_df, test_df]:

    df['Month'] = df['Month'].apply(lambda s: s.split('-')[1]).astype('int')

    df['DayofMonth'] = df['DayofMonth'].apply(lambda s: s.split('-')[1]).astype('int')

    df['DayOfWeek'] = df['DayOfWeek'].apply(lambda s: s.split('-')[1]).astype('int')

    

    df['HourFloat'] = df['DepTime'].apply(

        lambda t: (t // 100) % 24 + ((t % 100) % 60) / 60

    ).astype('float')
for df in [train_df, test_df]:

    df['Route'] = df[['Origin', 'Dest']].apply(

        lambda pair: ''.join([str(a) for a in pair]),

        axis='columns'

    ).astype('str')
target = train_df['dep_delayed_15min'].map({'Y': 1, 'N': 0})



feature_cols = [

    'Month', 'DayofMonth', 'DayOfWeek', 'HourFloat', 

    'UniqueCarrier', 'Origin', 'Dest', 'Route', 'Distance',]

train_df_modif = train_df[feature_cols]

test_df_modif = test_df[feature_cols]
N_train = train_df_modif.shape[0]

train_test_X = pd.concat([train_df_modif, test_df_modif], axis='index')



for feat in ['UniqueCarrier', 'Origin', 'Dest', 'Route']:

    train_test_X[feat] = train_test_X[feat].astype('category')
X_train = train_test_X[:N_train]

X_test = train_test_X[N_train:]

y_train = target
X_y_train_h = h2o.H2OFrame(pd.concat([X_train, y_train], axis='columns'))

X_y_train_h['dep_delayed_15min'] = X_y_train_h['dep_delayed_15min'].asfactor()

# ^ the target column should have categorical type for classification tasks

#   (numerical type for regression tasks)



X_test_h = h2o.H2OFrame(X_test)



X_y_train_h.describe()
aml = H2OAutoML(

    max_runtime_secs=(3600 * 8),  # 8 hours

    max_models=None,  # no limit

    seed=17

)
%%time



# aml.train(

#     x=feature_cols,

#     y='dep_delayed_15min',

#     training_frame=X_y_train_h

# )



# lb = aml.leaderboard

# model_ids = list(lb['model_id'].as_data_frame().iloc[:,0])

# out_path = "."



# for m_id in model_ids:

#     mdl = h2o.get_model(m_id)

#     h2o.save_model(model=mdl, path=out_path, force=True)



# h2o.export_file(lb, os.path.join(out_path, 'aml_leaderboard.h2o'), force=True)
models_path = "../input/h2o-automl-saved-models-classif/"



lb = h2o.import_file(path=os.path.join(models_path, "aml_leaderboard.h2o"))



lb.head(rows=10)

#lb.head(rows=lb.nrows)

# ^ to see the entire leaderboard
se_all = h2o.load_model(os.path.join(models_path, "StackedEnsemble_AllModels_AutoML_20190414_112210"))

# Get the Stacked Ensemble metalearner model

metalearner = h2o.get_model(se_all.metalearner()['name'])
%matplotlib inline

metalearner.std_coef_plot(num_of_features=20)

# ^ all importance values starting from the 16th are zero



#metalearner.coef_norm()

# ^ to see the table in the text form
se_best_of_family = h2o.load_model(os.path.join(models_path, "StackedEnsemble_BestOfFamily_AutoML_20190414_112210"))

# Get the Stacked Ensemble metalearner model

metalearner = h2o.get_model(se_best_of_family.metalearner()['name'])



%matplotlib inline

metalearner.std_coef_plot(num_of_features=10)

#metalearner.coef_norm()
from h2o.estimators.xgboost import H2OXGBoostEstimator



model_01 = h2o.load_model(os.path.join(models_path, "XGBoost_grid_1_AutoML_20190414_112210_model_19"))



excluded_params = ['model_id', 'response_column', 'ignored_columns']

model_01_actual_params = {k: v['actual'] for k, v in model_01.params.items() if k not in excluded_params}



reprod_model_01 = H2OXGBoostEstimator(**model_01_actual_params)

reprod_model_01.train(

    x=feature_cols,

    y='dep_delayed_15min',

    training_frame=X_y_train_h

)

reprod_model_01.auc(xval=True)

# ^ 0.749453, slightly worse compared to the leaderboard value
from h2o.estimators.gbm import H2OGradientBoostingEstimator



model_12 = h2o.load_model(os.path.join(models_path, "GBM_grid_1_AutoML_20190414_112210_model_85"))



excluded_params = ['model_id', 'response_column', 'ignored_columns']

model_12_actual_params = {k: v['actual'] for k, v in model_12.params.items() if k not in excluded_params}



reprod_model_12 = H2OGradientBoostingEstimator(**model_12_actual_params)

reprod_model_12.train(

    x=feature_cols,

    y='dep_delayed_15min',

    training_frame=X_y_train_h

)

reprod_model_12.auc(xval=True)

# ^ 0.741785, the same as at the leaderboard
from h2o.estimators.glm import H2OGeneralizedLinearEstimator

from h2o.grid.grid_search import H2OGridSearch



model_93 = h2o.load_model(os.path.join(models_path, "GLM_grid_1_AutoML_20190414_112210_model_1"))



excluded_params = ['model_id', 'response_column', 'ignored_columns', 'lambda']

model_93_actual_params = {k: v['actual'] for k, v in model_93.params.items() if k not in excluded_params}



reprod_model_93 = H2OGeneralizedLinearEstimator(**model_93_actual_params)

reprod_model_93.train(

    x=feature_cols,

    y='dep_delayed_15min',

    training_frame=X_y_train_h

)

reprod_model_93.auc(xval=True)

# ^ 0.699418, the same as at the leaderboard
from catboost import Pool, CatBoostClassifier, cv



cb_model = CatBoostClassifier(

    eval_metric='AUC',

    use_best_model=True,

    random_seed=17

)



cv_data = cv(

    Pool(X_train, y_train, cat_features=[4,5,6,7]),

    cb_model.get_params(),

    fold_count=5,

    verbose=False

)



print("CatBoostClassifier: the best cv auc is", np.max(cv_data['test-AUC-mean']))
df_train = pd.read_csv('../input/nyc-taxi-trip-duration/train.csv', index_col=0)

df_test  = pd.read_csv('../input/nyc-taxi-trip-duration/test.csv',  index_col=0)
df_train['pickup_datetime'] = pd.to_datetime(df_train.pickup_datetime)

df_train.loc[:, 'pickup_date'] = df_train['pickup_datetime'].dt.date

df_train['dropoff_datetime'] = pd.to_datetime(df_train.dropoff_datetime)

df_train['store_and_fwd_flag'] = 1 * (df_train.store_and_fwd_flag.values == 'Y')

df_train['check_trip_duration'] = (df_train['dropoff_datetime'] - df_train['pickup_datetime']).map(

    lambda x: x.total_seconds()

)

df_train['log_trip_duration'] = np.log1p(df_train['trip_duration'].values)



cnd = np.abs(df_train['check_trip_duration'].values  - df_train['trip_duration'].values) > 1

duration_difference = df_train[cnd]



if len(duration_difference[['pickup_datetime', 'dropoff_datetime', 'trip_duration', 'check_trip_duration']]) == 0:

    print('Trip_duration and datetimes are ok.')

else:

    print('Ooops.')
common_cols = [

    'vendor_id', 

    'pickup_datetime', 

    'passenger_count', 

    'pickup_longitude', 'pickup_latitude', 

    'dropoff_longitude', 'dropoff_latitude',

    'store_and_fwd_flag',

]



X_y_train_h = h2o.H2OFrame(

    pd.concat(

        [df_train[common_cols], df_train['log_trip_duration']],

        axis='columns'

    )

)



for ft in ['vendor_id', 'store_and_fwd_flag']:

    X_y_train_h[ft] = X_y_train_h[ft].asfactor()

    

X_y_train_h.describe()
# aml = H2OAutoML(

#     max_runtime_secs=(3600 * 8),  # 8 hours

#     max_models=None,  # no limit

#     seed=SEED,

# )



# aml.train(

#     x=common_cols,

#     y='log_trip_duration',

#     training_frame=X_y_train_h

# )



# lb = aml.leaderboard

# model_ids = list(lb['model_id'].as_data_frame().iloc[:,0])

# out_path = "."



# for m_id in model_ids:

#     mdl = h2o.get_model(m_id)

#     h2o.save_model(model=mdl, path=out_path, force=True)



# h2o.export_file(lb, os.path.join(out_path, 'aml_leaderboard.h2o'), force=True)

models_path = "../input/h2o-automl-saved-models-regress/"



lb = h2o.import_file(path=os.path.join(models_path, "aml_leaderboard.h2o"))

lb.head(rows=10)
from catboost import Pool, CatBoostRegressor, cv



cb_model = CatBoostRegressor(

    eval_metric='RMSE',

    use_best_model=True,

    random_seed=17

)



cv_data = cv(

    Pool(df_train[common_cols], df_train['log_trip_duration'], cat_features=[0,7]),

    cb_model.get_params(),

    fold_count=5,

    verbose=False

)
print("CatBoostRegressor: the best cv rmse is", np.min(cv_data['test-RMSE-mean']))