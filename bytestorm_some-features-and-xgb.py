import pandas as pd

import numpy as np

import time

from datetime import datetime

from collections import Counter, defaultdict

import matplotlib.pyplot as plt

import operator
%matplotlib inline
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

df_macro = pd.read_csv('../input/macro.csv')
df_train.head()
# Set index

df_train = df_train.set_index('id')

df_test = df_test.set_index('id')



# <Move> target variable to seperate dataframe

df_y = np.log1p(df_train['price_doc'])

df_train.drop('price_doc', 1, inplace = True)
# Convert string timestamp to integer

to_uTimestamp = lambda s: datetime.strptime(s, '%Y-%m-%d').year

df_train['year'] = df_train['timestamp'].apply(to_uTimestamp, 0).astype(np.int32)

df_test['year'] = df_test['timestamp'].apply(to_uTimestamp, 0).astype(np.int32)



to_uTimestamp = lambda s: datetime.strptime(s, '%Y-%m-%d').month

df_train['month'] = df_train['timestamp'].apply(to_uTimestamp, 0).astype(np.int32)

df_test['month'] = df_test['timestamp'].apply(to_uTimestamp, 0).astype(np.int32)



to_uTimestamp = lambda s: datetime.strptime(s, '%Y-%m-%d').day

df_train['day'] = df_train['timestamp'].apply(to_uTimestamp, 0).astype(np.int32)

df_test['day'] = df_test['timestamp'].apply(to_uTimestamp, 0).astype(np.int32)



to_uTimestamp = lambda s: datetime.strptime(s, '%Y-%m-%d').weekday()

df_train['dow'] = df_train['timestamp'].apply(to_uTimestamp, 0).astype(np.int32)

df_test['dow'] = df_test['timestamp'].apply(to_uTimestamp, 0).astype(np.int32)
to_uTimestamp = lambda s: int(datetime.strptime(s, '%Y-%m-%d').timestamp()/86400)

df_train['timestamp'] = df_train['timestamp'].apply(to_uTimestamp, 0).astype(np.int32)

df_test['timestamp'] = df_test['timestamp'].apply(to_uTimestamp, 0).astype(np.int32)
def auto_id_gen():

    start = 1

    while(True):

        yield start

        start += 1
fields = ['product_type', 'sub_area', 'culture_objects_top_25', 'thermal_power_plant_raion', 'incineration_raion', 'oil_chemistry_raion', 'radiation_raion', 'railroad_terminal_raion', 'big_market_raion', 'nuclear_reactor_raion', 'detention_facility_raion', 'water_1line', 'big_road1_1line', 'railroad_1line', 'ecology']

converter = {}

for f in fields:

    print('Unique Items for: ', f)

    

    item_id = 1

    unique_items = df_train[f].unique()

    gen = auto_id_gen()

    converter[f] = defaultdict(lambda: next(gen))

    for u in unique_items:

        converter[f][u]
for field in fields:

    convert = lambda value: converter[field][value]

    df_test[field] = df_test[field].apply(convert, 0).astype(np.int32)

    df_train[field] = df_train[field].apply(convert, 0).astype(np.int32)
# is top floor, # is ground floor

df_train['is_top2floors'] = (df_train.floor == (df_train.max_floor)) | (df_train.floor == (df_train.max_floor - 1))

df_train['is_top_floor'] = (df_train.floor == (df_train.max_floor))

df_train['is_ground_floor'] = (df_train.floor == 0)



df_test['is_top2floors'] = (df_test.floor == (df_test.max_floor)) | (df_test.floor == (df_test.max_floor - 1))

df_test['is_top_floor'] = (df_test.floor == (df_test.max_floor))

df_test['is_ground_floor'] = (df_test.floor == 0)
# extra_sq = full_sq - life_sq

df_train['extra_sq'] = df_train['full_sq'] - df_train['life_sq']

df_test['extra_sq'] = df_test['full_sq'] - df_test['life_sq']



# male to female pop ratio # gender_ratio

df_train['gender_ratio'] = df_train['male_f']/df_train['female_f']

df_test['gender_ratio'] = df_test['male_f']/df_test['female_f']
def addFeatures(df):

    # district dimension

    df['dimension_sub_area'] = df['area_m'].apply(np.sqrt, 0)



    # high school -> school_quota / 7to14_age_Persons

    df['school_seat_availability'] = df['school_quota']/df['children_school']



    # number of school per area

    df['school_per_area'] = 1e7 * df['school_education_centers_raion'] / df['area_m']



    # how closeby school

    df['school_closeness'] = df['school_km'] / df['dimension_sub_area']



    # Preschool seat per child

    df['preschool_seat_availability'] = df['preschool_quota'] / df['children_preschool']



    # number of preschool per area

    df['preschool_per_area']  = 1e7 * df['preschool_education_centers_raion'] / df['area_m']



    # how close is preschool

    df['preschool_closeness'] = df['preschool_km'] / df['dimension_sub_area']



    # is preschool same as school

    df['diff_school'] = df['preschool_km'] == df['school_km']



    # closeness of offices 

    df['close_office'] = df['office_km'] / df['dimension_sub_area']



    # work_availability

    df['work_avail'] = df['office_raion'] / df['work_all']



    # density of healthcare centres

    df['healthcare_density'] = 1e7 * df['healthcare_centers_raion'] / df['area_m']



    # Pollution coeff - relative dist. to indu_zone

    df['safe_nature'] = df['industrial_km'] / df['green_zone_km']



    # Pollution coeff - relative dist. to water treatment

    df['safe_watre'] = df['industrial_km'] / df['water_treatment_km']



    # extent of higher education

    df['high_ed_extent'] = df['school_km'] / df['kindergarten_km']



    # closeness of public healthcare

    df['close_public_health'] = df['public_healthcare_km'] / df['dimension_sub_area']



    # close to office?

    df['close_office'] = df['office_km'] / df['dimension_sub_area']



    # Density of shopping malls

    df['shop_density'] = 1e7 * df['shopping_centers_raion'] / df['area_m']



    # closeness of shopping malls 

    df['close_shops'] =  df['shopping_centers_km'] / df['dimension_sub_area']



    # New City or Old city

    # df['build_count_after_1995'] / (df['build_count_1971-1995'] + df['build_count_1946-1970'])

    df['new_or_old_city'] =  df['build_count_after_1995'] / (df['build_count_1971-1995'])

    

    return df
df_train = addFeatures(df_train)
df_test = addFeatures(df_test)
import xgboost as xgb
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
d_test = xgb.DMatrix(data=df_test)
X_train, X_valid, y_train, y_valid = train_test_split(df_train, df_y, test_size=0.2, random_state=5468)
# read in data

d_train = xgb.DMatrix(data=X_train, label=y_train)

d_valid = xgb.DMatrix(data=X_valid, label=y_valid)
class xgCallback:

    def __init__(self):

        self.models = []

        self.train_scores = []

        self.valid_scores = []

        self.best_model = None

        self.best_score = None

    def callback(self, a = None):

        self.models.append(a.model.copy())

        self.train_scores.append(a.evaluation_result_list[0][1])

        self.valid_scores.append(a.evaluation_result_list[1][1])
watchlist = [(d_train, 'train'), (d_valid, 'valid')]



# specify parameters via map

params = {}

params['objective'] = 'reg:linear'

params['eval_metric'] = 'rmse'

params['eta'] = 0.03

params['max_depth'] = 5

params['subsample'] = 1

params['base_score'] = 0.5

params['colsample_bytree'] = 0.8

params['tree_method'] = 'hist'
callback1 = xgCallback()

callbacks = [callback1.callback]



# params['updater'] = 'grow_gpu'

bst = xgb.train(params, d_train, 400, watchlist, verbose_eval=50, callbacks=callbacks)
best_model = np.argmin(callback1.valid_scores)

print(best_model)
bst = callback1.models[best_model]
# Validation score of best iteration



np.sqrt(mean_squared_error(y_valid, bst.predict(d_valid)))
# make prediction

preds = bst.predict(d_test)
df_sub = pd.DataFrame({'id': df_test.index, 'price_doc': np.expm1(preds)})
df_sub.head()
df_sub.to_csv('./submission.csv', index=False)