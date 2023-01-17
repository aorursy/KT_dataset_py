import gc, math, pickle, datetime, os, random

import numpy as np 

import pandas as pd 

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_log_error

from sklearn.metrics import mean_squared_log_error

from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb

import matplotlib.pyplot as plt

import seaborn as sns



from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
def reduce_mem_usage(df, verbose=True):

    """

    :param df: Dataframe with columns unprocessed so they use more memory than needed

    

    :returns:

        df -> Dataframe with lower memory use

    """

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

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

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df



def clean_timestamps(df):

    """

    :param df: Dataframe containing a "timestamp" field which will be broken down in hour, year, day,...

    """

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df["year"] = df["timestamp"].dt.year.astype(np.uint16)

    df["month"] = df["timestamp"].dt.month.astype(np.uint8)

    df["day"] = df["timestamp"].dt.day.astype(np.uint8)

    df["hour"] = df["timestamp"].dt.hour.astype(np.uint8)

    df["weekend"] = df["timestamp"].dt.weekday.astype(np.uint8)

    

def drop_cols(df):

    """

    :param df: Dataframe with unnecessary cols

    

    :returns:

            df -> dataframe containing only the desired columns

    """

    #drop_cols = ['timestamp','primary_use', 'site_id', 'floor_count',"precip_depth_1_hr", "sea_level_pressure", "wind_direction", "wind_speed", "building_id"]

    drop_cols = ['timestamp']

    df = df.drop(drop_cols, axis = 1)

    return df



# Predictions lower than zero are turned zero

def fix_predictions(y):

    """

    :param y: Column with predictions

    """

    y[y < 0] = 0



# Predictibility in our predictions

def seed_everything(seed=0):

    """

    :param seed: Value for seeding random functions

    """

    random.seed(seed)

    np.random.seed(seed)



# Fill given categories with their average values

def fill_averages(df):

    """

    :param df: Dataframe containing normal and nan values

    """

    data_ratios = df.count()/len(df)

    cols = data_ratios[data_ratios < 1.0].index

    for col in cols:

        df[col] = df[col].fillna(-1)

        df[col] = df[col].astype(np.int8)

        more_zero = df[col] >= 0

        less_zero = df[col] < 0

        mean = df[more_zero][col].mean()

        df.loc[less_zero, col] = mean
SEED = 5

seed_everything(SEED)
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#PATH = '../input/ashrae-energy-prediction/'

#train_df = reduce_mem_usage(pd.read_csv(PATH + 'train.csv'))

#building = reduce_mem_usage(pd.read_csv(PATH + 'building_metadata.csv'))

#weather_train = reduce_mem_usage(pd.read_csv(PATH + 'weather_train.csv'))



#train_merged = train_df.merge(building, left_on = "building_id", right_on = "building_id", how = "left")

#train = train_merged.merge(weather_train, left_on = ["site_id", "timestamp"], right_on = ["site_id", "timestamp"])
PATH = '/kaggle/input/'

train = reduce_mem_usage(pd.read_csv(PATH + 'lgb-train-test/train.csv'))

test = reduce_mem_usage(pd.read_csv(PATH + 'lgb-train-test/test.csv'))
print(train.shape)

print(test.shape)
clean_timestamps(train)
plt.figure(figsize=(10, 8))

sns.countplot(x="floor_count",data=train, order = train['floor_count'].value_counts().index)

plt.title('Floor count feature column')

plt.tight_layout()

plt.show()
plt.figure(figsize=(10, 8))

ax = sns.countplot(x="primary_use",data=train, order = train['primary_use'].value_counts().index)

plt.title('Buiding type count (primary_use) feature column')

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

plt.show()
# Añadir leyenda con diccionario

building_id = 213

plt.figure(figsize=(14, 8))

ax = sns.lineplot(x="timestamp", y="meter_reading", hue="meter", data=train[train['building_id'] == building_id])

plt.title('Meter readings from building_id {}'.format(building_id))

plt.show()
train = drop_cols(train)



train_y = np.log1p(train['meter_reading'])



train_x = train.drop('meter_reading', axis=1)

train_x['primary_use'] = LabelEncoder().fit_transform(train_x['primary_use'])



train_x.head()
del train

#del train_x

#del train_y 



gc.collect()
# Train and Validation splits

test_size = 0.20

X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, test_size=test_size, random_state=SEED)
 # lgb_params = {

 #                   'objective':'regression',

 #                   'boosting_type':'gbdt',

 #                   'metric':'rmse',

 #                   'n_jobs':-1,

 #                   'learning_rate':0.07,

 #                   'num_leaves': 2**8,

 #                   'max_depth':-1,

 #                   'tree_learner':'serial',

 #                   'colsample_bytree': 0.7,

 #                   'subsample_freq':1,

 #                   'subsample':0.5,

 #                   'n_estimators':8500,

 #                   'max_bin':255,

 #                   'verbose':1,

 #                   'seed': SEED,

 #                   'early_stopping_rounds':3500, 

 #               } 
del train_x

del train_y



gc.collect()
# load model

print('Loading model')

gbm = lgb.Booster(model_file= PATH + 'lbg-model/lgb_classifier_20-10-2019_0.20834363390357943.txt')
#lgb_train = lgb.Dataset(X_train, y_train)

#lgb_eval = lgb.Dataset(X_val, y_val)

#gbm = lgb.train(

#            lgb_params,

#            lgb_train,

#            num_boost_round=5000,

#            valid_sets=(lgb_train, lgb_eval),

#            verbose_eval = 50

#            )
y_pred = gbm.predict(X_val, num_iteration=gbm.best_iteration)
fix_predictions(y_pred)
rmsle = np.sqrt(mean_squared_log_error(y_pred, (y_val)))

print('RMSLE: ', rmsle)
del y_val

del y_train

del X_val

del X_train

#del y_pred



gc.collect()
# save model



#gbm.save_model('lgb_classifier_{}_{}.txt'.format(datetime.datetime.now().strftime("%d-%m-%Y"), rmsle), num_iteration=gbm.best_iteration)
feature_imp = pd.DataFrame(sorted(zip(gbm.feature_importance(), gbm.feature_name()),reverse = True), columns=['Value','Feature'])

plt.figure(figsize=(10, 5))

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))

plt.title('LightGBM Features')

plt.tight_layout()

plt.show()
#building = reduce_mem_usage(pd.read_csv(PATH + 'building_metadata.csv'))

#test = reduce_mem_usage(pd.read_csv(PATH + "test.csv"))

#weather_test = reduce_mem_usage(pd.read_csv(PATH + "weather_test.csv"))



#test = test.merge(building, left_on = "building_id", right_on = "building_id", how = "left")

#test = test.merge(weather_test, left_on = ["site_id", "timestamp"], right_on = ["site_id", "timestamp"], how="left")



#del weather_test

#del building

gc.collect()
clean_timestamps(test)

test = drop_cols(test)

test = test.drop('row_id', axis = 1)

test['primary_use'] = LabelEncoder().fit_transform(test['primary_use'])
submission = pd.read_csv(PATH+'ashrae-energy-prediction/sample_submission.csv')
test_1 = test[:len(test)//3]

y_pred_1 = gbm.predict(test_1, num_iteration=gbm.best_iteration)



del test_1



gc.collect()
test_2 = test[len(test)//3:(len(test)*2)//3]

y_pred_2 = gbm.predict(test_2, num_iteration=gbm.best_iteration)



del test_2



gc.collect()
test_3 = test[(len(test)*2)//3:]

y_pred_3 = gbm.predict(test_3, num_iteration=gbm.best_iteration)



del test_3



gc.collect()
y_pred_test = np.concatenate([y_pred_1, y_pred_2, y_pred_3], axis=0)



del y_pred_1

del y_pred_2

del y_pred_3



gc.collect()
y_pred_test = np.expm1(y_pred_test)

fix_predictions(y_pred_test)

submission['meter_reading'] = y_pred_test

submission
np.log1p(submission['meter_reading']).hist(bins=30)
sns.distplot(y_pred, color="blue", label="train prediction")

sns.distplot(np.log1p(y_pred_test), color="green", label="test prediction")

plt.legend()
submission.to_csv('submission.csv', index=False)