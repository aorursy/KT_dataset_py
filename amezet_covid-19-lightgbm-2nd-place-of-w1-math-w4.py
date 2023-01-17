import os, gc, pickle, copy, datetime, warnings

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb

from sklearn import metrics

pd.set_option('display.max_columns', 100)

warnings.filterwarnings('ignore')
df_train = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")

print(df_train.shape)

df_train.head()
df_test = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")

print(df_test.shape)

df_test.head()
# concat train and test

df_traintest = pd.concat([df_train, df_test])

print(df_train.shape, df_test.shape, df_traintest.shape)
# process date

df_traintest['Date'] = pd.to_datetime(df_traintest['Date'])

df_traintest['day'] = df_traintest['Date'].apply(lambda x: x.dayofyear).astype(np.int16)

df_traintest.head()
day_before_valid = 7+85-7 # 3-18 day  before of validation

day_before_public = 7+85 # 3-25, the day before public LB period

day_before_private = df_traintest['day'][pd.isna(df_traintest['ForecastId'])].max() # last day of train

print(df_traintest['Date'][df_traintest['day']==day_before_valid].values[0])

print(df_traintest['Date'][df_traintest['day']==day_before_public].values[0])

print(df_traintest['Date'][df_traintest['day']==day_before_private].values[0])
# concat Country/Region and Province/State

def func(x):

    try:

        x_new = x['Country_Region'] + "/" + x['Province_State']

    except:

        x_new = x['Country_Region']

    return x_new

        

df_traintest['place_id'] = df_traintest.apply(lambda x: func(x), axis=1)

df_traintest.head()
df_traintest[(df_traintest['day']>=day_before_public-3) & (df_traintest['place_id']=='China/Hubei')].head()
# concat lat and long

df_latlong = pd.read_csv("../input/smokingstats/df_Latlong.csv")

df_latlong.head()
# concat Country/Region and Province/State

def func(x):

    try:

        x_new = x['Country/Region'] + "/" + x['Province/State']

    except:

        x_new = x['Country/Region']

    return x_new

        

df_latlong['place_id'] = df_latlong.apply(lambda x: func(x), axis=1)

df_latlong = df_latlong[df_latlong['place_id'].duplicated()==False]

df_latlong.head()
df_traintest = pd.merge(df_traintest, df_latlong[['place_id', 'Lat', 'Long']], on='place_id', how='left')

df_traintest.head()
# count the places with no Lat and Long.

tmp = np.sort(df_traintest['place_id'][pd.isna(df_traintest['Lat'])].unique())

print(len(tmp)) # count Nan

tmp
# get place list

places = np.sort(df_traintest['place_id'].unique())

print(len(places))
# calc cases, fatalities per day

df_traintest2 = copy.deepcopy(df_traintest)

df_traintest2['cases/day'] = 0

df_traintest2['fatal/day'] = 0

tmp_list = np.zeros(len(df_traintest2))

for place in places:

    tmp = df_traintest2['ConfirmedCases'][df_traintest2['place_id']==place].values

    tmp[1:] -= tmp[:-1]

    df_traintest2['cases/day'][df_traintest2['place_id']==place] = tmp

    tmp = df_traintest2['Fatalities'][df_traintest2['place_id']==place].values

    tmp[1:] -= tmp[:-1]

    df_traintest2['fatal/day'][df_traintest2['place_id']==place] = tmp

print(df_traintest2.shape)

df_traintest2[df_traintest2['place_id']=='China/Hubei'].head()
# aggregate cases and fatalities

def do_aggregation(df, col, mean_range):

    df_new = copy.deepcopy(df)

    col_new = '{}_({}-{})'.format(col, mean_range[0], mean_range[1])

    df_new[col_new] = 0

    tmp = df_new[col].rolling(mean_range[1]-mean_range[0]+1).mean()

    df_new[col_new][mean_range[0]:] = tmp[:-(mean_range[0])]

    df_new[col_new][pd.isna(df_new[col_new])] = 0

    return df_new[[col_new]].reset_index(drop=True)



def do_aggregations(df):

    df = pd.concat([df, do_aggregation(df, 'cases/day', [1,1]).reset_index(drop=True)], axis=1)

    df = pd.concat([df, do_aggregation(df, 'cases/day', [1,7]).reset_index(drop=True)], axis=1)

    df = pd.concat([df, do_aggregation(df, 'cases/day', [8,14]).reset_index(drop=True)], axis=1)

    df = pd.concat([df, do_aggregation(df, 'cases/day', [15,21]).reset_index(drop=True)], axis=1)

    df = pd.concat([df, do_aggregation(df, 'fatal/day', [1,1]).reset_index(drop=True)], axis=1)

    df = pd.concat([df, do_aggregation(df, 'fatal/day', [1,7]).reset_index(drop=True)], axis=1)

    df = pd.concat([df, do_aggregation(df, 'fatal/day', [8,14]).reset_index(drop=True)], axis=1)

    df = pd.concat([df, do_aggregation(df, 'fatal/day', [15,21]).reset_index(drop=True)], axis=1)

    for threshold in [1, 10, 100]:

        days_under_threshold = (df['ConfirmedCases']<threshold).sum()

        tmp = df['day'].values - 22 - days_under_threshold

        tmp[tmp<=0] = 0

        df['days_since_{}cases'.format(threshold)] = tmp

            

    for threshold in [1, 10, 100]:

        days_under_threshold = (df['Fatalities']<threshold).sum()

        tmp = df['day'].values - 22 - days_under_threshold

        tmp[tmp<=0] = 0

        df['days_since_{}fatal'.format(threshold)] = tmp

    

    # process China/Hubei

    if df['place_id'][0]=='China/Hubei':

        df['days_since_1cases'] += 35 # 2019/12/8

        df['days_since_10cases'] += 35-13 # 2019/12/8-2020/1/2 assume 2019/12/8+13

        df['days_since_100cases'] += 4 # 2020/1/18

        df['days_since_1fatal'] += 13 # 2020/1/9

    return df
df_traintest3 = []

for place in places[:]:

    df_tmp = df_traintest2[df_traintest2['place_id']==place].reset_index(drop=True)

    df_tmp = do_aggregations(df_tmp)

    df_traintest3.append(df_tmp)

df_traintest3 = pd.concat(df_traintest3).reset_index(drop=True)

df_traintest3[df_traintest3['place_id']=='China/Hubei'].head()
# add Smoking rate per country

# data of smoking rate is obtained from https://ourworldindata.org/smoking

df_smoking = pd.read_csv("../input/smokingstats/share-of-adults-who-smoke.csv")

print(np.sort(df_smoking['Entity'].unique())[:10])

df_smoking.head()
# extract newest data

df_smoking_recent = df_smoking.sort_values('Year', ascending=False).reset_index(drop=True)

df_smoking_recent = df_smoking_recent[df_smoking_recent['Entity'].duplicated()==False]

df_smoking_recent['Country_Region'] = df_smoking_recent['Entity']

df_smoking_recent['SmokingRate'] = df_smoking_recent['Smoking prevalence, total (ages 15+) (% of adults)']

df_smoking_recent.head()
# merge

df_traintest4 = pd.merge(df_traintest3, df_smoking_recent[['Country_Region', 'SmokingRate']], on='Country_Region', how='left')

print(df_traintest4.shape)

df_traintest4.head()
# fill na with world smoking rate

SmokingRate = df_smoking_recent['SmokingRate'][df_smoking_recent['Entity']=='World'].values[0]

print("Smoking rate of the world: {:.6f}".format(SmokingRate))

df_traintest4['SmokingRate'][pd.isna(df_traintest4['SmokingRate'])] = SmokingRate

df_traintest4.head()
# add data from World Economic Outlook Database

# https://www.imf.org/external/pubs/ft/weo/2017/01/weodata/index.aspx

df_weo = pd.read_csv("../input/smokingstats/WEO.csv")

df_weo.head()
print(df_weo['Subject Descriptor'].unique())
subs  = df_weo['Subject Descriptor'].unique()[:-1]

df_weo_agg = df_weo[['Country']][df_weo['Country'].duplicated()==False].reset_index(drop=True)

for sub in subs[:]:

    df_tmp = df_weo[['Country', '2019']][df_weo['Subject Descriptor']==sub].reset_index(drop=True)

    df_tmp = df_tmp[df_tmp['Country'].duplicated()==False].reset_index(drop=True)

    df_tmp.columns = ['Country', sub]

    df_weo_agg = df_weo_agg.merge(df_tmp, on='Country', how='left')

df_weo_agg.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in df_weo_agg.columns]

df_weo_agg.columns

df_weo_agg['Country_Region'] = df_weo_agg['Country']

df_weo_agg.head()
# merge

df_traintest5 = pd.merge(df_traintest4, df_weo_agg, on='Country_Region', how='left')

print(df_traintest5.shape)

df_traintest5.head()
# add Life expectancy

# Life expectancy at birth obtained from http://hdr.undp.org/en/data

df_life = pd.read_csv("../input/smokingstats/Life expectancy at birth.csv")

tmp = df_life.iloc[:,1].values.tolist()

df_life = df_life[['Country', '2018']]

def func(x):

    x_new = 0

    try:

        x_new = float(x.replace(",", ""))

    except:

#         print(x)

        x_new = np.nan

    return x_new

    

df_life['2018'] = df_life['2018'].apply(lambda x: func(x))

df_life.head()
df_life = df_life[['Country', '2018']]

df_life.columns = ['Country_Region', 'LifeExpectancy']
# merge

df_traintest6 = pd.merge(df_traintest5, df_life, on='Country_Region', how='left')

print(len(df_traintest6))

df_traintest6.head()
# add additional info from countryinfo dataset

df_country = pd.read_csv("../input/countryinfo/covid19countryinfo.csv")

df_country.head()
df_country['Country_Region'] = df_country['country']

df_country = df_country[df_country['country'].duplicated()==False]
print(df_country[df_country['country'].duplicated()].shape)
df_country[df_country['country'].duplicated()]
df_traintest7 = pd.merge(df_traintest6, 

                         df_country.drop(['tests', 'testpop', 'country'], axis=1), 

                         on=['Country_Region',], how='left')

print(df_traintest7.shape)

df_traintest7.head()
def encode_label(df, col, freq_limit=0):

    df[col][pd.isna(df[col])] = 'nan'

    tmp = df[col].value_counts()

    cols = tmp.index.values

    freq = tmp.values

    num_cols = (freq>=freq_limit).sum()

    print("col: {}, num_cat: {}, num_reduced: {}".format(col, len(cols), num_cols))



    col_new = '{}_le'.format(col)

    df_new = pd.DataFrame(np.ones(len(df), np.int16)*(num_cols-1), columns=[col_new])

    for i, item in enumerate(cols[:num_cols]):

        df_new[col_new][df[col]==item] = i



    return df_new



def get_df_le(df, col_index, col_cat):

    df_new = df[[col_index]]

    for col in col_cat:

        df_tmp = encode_label(df, col)

        df_new = pd.concat([df_new, df_tmp], axis=1)

    return df_new



df_traintest7['id'] = np.arange(len(df_traintest7))

df_le = get_df_le(df_traintest7, 'id', ['Country_Region', 'Province_State'])

df_traintest8 = pd.merge(df_traintest7, df_le, on='id', how='left')
df_traintest8['cases/day'] = df_traintest8['cases/day'].astype(np.float)

df_traintest8['fatal/day'] = df_traintest8['fatal/day'].astype(np.float)
# covert object type to float

def func(x):

    x_new = 0

    try:

        x_new = float(x.replace(",", ""))

    except:

#         print(x)

        x_new = np.nan

    return x_new

cols = [

    'Gross_domestic_product__constant_prices', 

    'Gross_domestic_product__current_prices', 

    'Gross_domestic_product__deflator', 

    'Gross_domestic_product_per_capita__constant_prices', 

    'Gross_domestic_product_per_capita__current_prices', 

    'Output_gap_in_percent_of_potential_GDP', 

    'Gross_domestic_product_based_on_purchasing_power_parity__PPP__valuation_of_country_GDP', 

    'Gross_domestic_product_based_on_purchasing_power_parity__PPP__per_capita_GDP', 

    'Gross_domestic_product_based_on_purchasing_power_parity__PPP__share_of_world_total', 

    'Implied_PPP_conversion_rate', 'Total_investment', 

    'Gross_national_savings', 'Inflation__average_consumer_prices', 

    'Inflation__end_of_period_consumer_prices', 

    'Six_month_London_interbank_offered_rate__LIBOR_', 

    'Volume_of_imports_of_goods_and_services', 

    'Volume_of_Imports_of_goods', 

    'Volume_of_exports_of_goods_and_services', 

    'Volume_of_exports_of_goods', 'Unemployment_rate', 'Employment', 'Population', 

    'General_government_revenue', 'General_government_total_expenditure', 

    'General_government_net_lending_borrowing', 'General_government_structural_balance', 

    'General_government_primary_net_lending_borrowing', 'General_government_net_debt', 

    'General_government_gross_debt', 'Gross_domestic_product_corresponding_to_fiscal_year__current_prices', 

    'Current_account_balance', 'pop'

]

for col in cols:

    df_traintest8[col] = df_traintest8[col].apply(lambda x: func(x))  

print(df_traintest8['pop'].dtype)
df_traintest8[df_traintest8['place_id']=='China/Hubei'].head()
def calc_score(y_true, y_pred):

    y_true[y_true<0] = 0

    score = metrics.mean_squared_error(np.log(y_true.clip(0, 1e10)+1), np.log(y_pred[:]+1))**0.5

    return score
# train model to predict fatalities/day

# params

SEED = 42

params = {'num_leaves': 8,

          'min_data_in_leaf': 5,  # 42,

          'objective': 'regression',

          'max_depth': 8,

          'learning_rate': 0.02,

          'boosting': 'gbdt',

          'bagging_freq': 5,  # 5

          'bagging_fraction': 0.8,  # 0.5,

          'feature_fraction': 0.8201,

          'bagging_seed': SEED,

          'reg_alpha': 1,  # 1.728910519108444,

          'reg_lambda': 4.9847051755586085,

          'random_state': SEED,

          'metric': 'mse',

          'verbosity': 100,

          'min_gain_to_split': 0.02,  # 0.01077313523861969,

          'min_child_weight': 5,  # 19.428902804238373,

          'num_threads': 6,

          }

# train model to predict fatalities/day

# features are selected manually based on valid score

col_target = 'fatal/day'

col_var = [

    'Lat', 'Long',

#     'days_since_1cases', 

#     'days_since_10cases', 

#     'days_since_100cases',

#     'days_since_1fatal', 

#     'days_since_10fatal', 'days_since_100fatal',

#     'days_since_1recov',

#     'days_since_10recov', 'days_since_100recov', 

    'cases/day_(1-1)', 

    'cases/day_(1-7)', 

#     'cases/day_(8-14)',  

#     'cases/day_(15-21)', 

    

#     'fatal/day_(1-1)', 

    'fatal/day_(1-7)', 

    'fatal/day_(8-14)', 

    'fatal/day_(15-21)', 

    'SmokingRate',

#     'Gross_domestic_product__constant_prices',

#     'Gross_domestic_product__current_prices',

#     'Gross_domestic_product__deflator',

#     'Gross_domestic_product_per_capita__constant_prices',

#     'Gross_domestic_product_per_capita__current_prices',

#     'Output_gap_in_percent_of_potential_GDP',

#     'Gross_domestic_product_based_on_purchasing_power_parity__PPP__valuation_of_country_GDP',

#     'Gross_domestic_product_based_on_purchasing_power_parity__PPP__per_capita_GDP',

#     'Gross_domestic_product_based_on_purchasing_power_parity__PPP__share_of_world_total',

#     'Implied_PPP_conversion_rate', 'Total_investment',

#     'Gross_national_savings', 'Inflation__average_consumer_prices',

#     'Inflation__end_of_period_consumer_prices',

#     'Six_month_London_interbank_offered_rate__LIBOR_',

#     'Volume_of_imports_of_goods_and_services', 'Volume_of_Imports_of_goods',

#     'Volume_of_exports_of_goods_and_services', 'Volume_of_exports_of_goods',

#     'Unemployment_rate', 

#     'Employment', 'Population',

#     'General_government_revenue', 'General_government_total_expenditure',

#     'General_government_net_lending_borrowing',

#     'General_government_structural_balance',

#     'General_government_primary_net_lending_borrowing',

#     'General_government_net_debt', 'General_government_gross_debt',

#     'Gross_domestic_product_corresponding_to_fiscal_year__current_prices',

#     'Current_account_balance', 

#     'LifeExpectancy',

#     'pop',

    'density', 

#     'medianage', 

#     'urbanpop', 

#     'hospibed', 'smokers', 

]

col_cat = []

df_train = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']<=day_before_valid)]

df_valid = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (day_before_valid<df_traintest8['day']) & (df_traintest8['day']<=day_before_public)]

df_test = df_traintest8[pd.isna(df_traintest8['ForecastId'])==False]

X_train = df_train[col_var]

X_valid = df_valid[col_var]

y_train = np.log(df_train[col_target].values.clip(0, 1e10)+1)

y_valid = np.log(df_valid[col_target].values.clip(0, 1e10)+1)

train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=col_cat)

valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature=col_cat)

num_round = 15000

model = lgb.train(params, train_data, num_round, valid_sets=[train_data, valid_data],

                  verbose_eval=100,

                  early_stopping_rounds=150,)



best_itr = model.best_iteration
y_true = df_valid['fatal/day'].values

y_pred = np.exp(model.predict(X_valid))-1

score = calc_score(y_true, y_pred)

print("{:.6f}".format(score))
# display feature importance

tmp = pd.DataFrame()

tmp["feature"] = col_var

tmp["importance"] = model.feature_importance()

tmp = tmp.sort_values('importance', ascending=False)

tmp
# train with all data before public

df_train = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']<=day_before_public)]

df_valid = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']<=day_before_public)]

df_test = df_traintest8[pd.isna(df_traintest8['ForecastId'])==False]

X_train = df_train[col_var]

X_valid = df_valid[col_var]

y_train = np.log(df_train[col_target].values.clip(0, 1e10)+1)

y_valid = np.log(df_valid[col_target].values.clip(0, 1e10)+1)

train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=col_cat)

valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature=col_cat)

model = lgb.train(params, train_data, best_itr, valid_sets=[train_data, valid_data],

                  verbose_eval=100,

                  early_stopping_rounds=150,)
# train model to predict fatalities/day

col_target2 = 'cases/day'

col_var2 = [

    'Lat', 'Long',

#     'days_since_1cases', 

    'days_since_10cases', #selected

#     'days_since_100cases',

#     'days_since_1fatal', 

#     'days_since_10fatal',

#     'days_since_100fatal',

#     'days_since_1recov',

#     'days_since_10recov', 'days_since_100recov', 

    'cases/day_(1-1)', 

    'cases/day_(1-7)', 

    'cases/day_(8-14)',  

    'cases/day_(15-21)', 

    

#     'fatal/day_(1-1)', 

#     'fatal/day_(1-7)', 

#     'fatal/day_(8-14)', 

#     'fatal/day_(15-21)', 

#     'recov/day_(1-1)', 'recov/day_(1-7)', 

#     'recov/day_(8-14)',  'recov/day_(15-21)',

#     'active_(1-1)', 

#     'active_(1-7)', 

#     'active_(8-14)',  'active_(15-21)', 

#     'SmokingRate',

#     'Gross_domestic_product__constant_prices',

#     'Gross_domestic_product__current_prices',

#     'Gross_domestic_product__deflator',

#     'Gross_domestic_product_per_capita__constant_prices',

#     'Gross_domestic_product_per_capita__current_prices',

#     'Output_gap_in_percent_of_potential_GDP',

#     'Gross_domestic_product_based_on_purchasing_power_parity__PPP__valuation_of_country_GDP',

#     'Gross_domestic_product_based_on_purchasing_power_parity__PPP__per_capita_GDP',

#     'Gross_domestic_product_based_on_purchasing_power_parity__PPP__share_of_world_total',

#     'Implied_PPP_conversion_rate', 'Total_investment',

#     'Gross_national_savings', 'Inflation__average_consumer_prices',

#     'Inflation__end_of_period_consumer_prices',

#     'Six_month_London_interbank_offered_rate__LIBOR_',

#     'Volume_of_imports_of_goods_and_services', 'Volume_of_Imports_of_goods',

#     'Volume_of_exports_of_goods_and_services', 'Volume_of_exports_of_goods',

#     'Unemployment_rate', 

#     'Employment', 

#     'Population',

#     'General_government_revenue', 'General_government_total_expenditure',

#     'General_government_net_lending_borrowing',

#     'General_government_structural_balance',

#     'General_government_primary_net_lending_borrowing',

#     'General_government_net_debt', 'General_government_gross_debt',

#     'Gross_domestic_product_corresponding_to_fiscal_year__current_prices',

#     'Current_account_balance', 

#     'LifeExpectancy',

#     'pop',

#     'density', 

#     'medianage', 

#     'urbanpop', 

#     'hospibed', 'smokers', 

]

col_cat = []

df_train = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']<=day_before_valid)]

df_valid = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (day_before_valid<df_traintest8['day']) & (df_traintest8['day']<=day_before_public)]

X_train = df_train[col_var2]

X_valid = df_valid[col_var2]

y_train = np.log(df_train[col_target2].values.clip(0, 1e10)+1)

y_valid = np.log(df_valid[col_target2].values.clip(0, 1e10)+1)

train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=col_cat)

valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature=col_cat)

model2 = lgb.train(params, train_data, num_round, valid_sets=[train_data, valid_data],

                  verbose_eval=100,

                  early_stopping_rounds=150,)

best_itr2 = model2.best_iteration
y_true = df_valid['cases/day'].values

y_pred = np.exp(model2.predict(X_valid))-1

score = calc_score(y_true, y_pred)

print("{:.6f}".format(score))

# display feature importance

tmp = pd.DataFrame()

tmp["feature"] = col_var2

tmp["importance"] = model2.feature_importance()

tmp = tmp.sort_values('importance', ascending=False)

tmp
df_train = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']<=day_before_public)]

df_valid = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']<=day_before_public)]

X_train = df_train[col_var2]

X_valid = df_valid[col_var2]

y_train = np.log(df_train[col_target2].values.clip(0, 1e10)+1)

y_valid = np.log(df_valid[col_target2].values.clip(0, 1e10)+1)

train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=col_cat)

valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature=col_cat)

model2 = lgb.train(params, train_data, best_itr2, valid_sets=[train_data, valid_data],

                  verbose_eval=100,

                  early_stopping_rounds=150,)
# train model to predict fatalities/day

df_train = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']<=day_before_public)]

df_valid = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (day_before_public<df_traintest8['day'])]

df_test = df_traintest8[pd.isna(df_traintest8['ForecastId'])==False]

X_train = df_train[col_var]

X_valid = df_valid[col_var]

y_train = np.log(df_train[col_target].values.clip(0, 1e10)+1)

y_valid = np.log(df_valid[col_target].values.clip(0, 1e10)+1)

train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=col_cat)

valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature=col_cat)

num_round = 15000

model = lgb.train(params, train_data, num_round, valid_sets=[train_data, valid_data],

                  verbose_eval=100,

                  early_stopping_rounds=150,)



best_itr = model.best_iteration
# train with all data

df_train = df_traintest8[(pd.isna(df_traintest8['ForecastId']))]

df_valid = df_traintest8[(pd.isna(df_traintest8['ForecastId']))]

X_train = df_train[col_var]

X_valid = df_valid[col_var]

y_train = np.log(df_train[col_target].values.clip(0, 1e10)+1)

y_valid = np.log(df_valid[col_target].values.clip(0, 1e10)+1)

train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=col_cat)

valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature=col_cat)

model_pri = lgb.train(params, train_data, best_itr, valid_sets=[train_data, valid_data],

                  verbose_eval=100,

                  early_stopping_rounds=150,)
# train model to predict cases/day

df_train = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']<=day_before_public)]

df_valid = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (day_before_public<df_traintest8['day'])]

X_train = df_train[col_var2]

X_valid = df_valid[col_var2]

y_train = np.log(df_train[col_target2].values.clip(0, 1e10)+1)

y_valid = np.log(df_valid[col_target2].values.clip(0, 1e10)+1)

train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=col_cat)

valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature=col_cat)

model2 = lgb.train(params, train_data, num_round, valid_sets=[train_data, valid_data],

                  verbose_eval=100,

                  early_stopping_rounds=150,)

best_itr2 = model2.best_iteration
# train with all data

df_train = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']<=day_before_public)]

df_valid = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']<=day_before_public)]

X_train = df_train[col_var2]

X_valid = df_valid[col_var2]

y_train = np.log(df_train[col_target2].values.clip(0, 1e10)+1)

y_valid = np.log(df_valid[col_target2].values.clip(0, 1e10)+1)

train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=col_cat)

valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature=col_cat)

model2_pri = lgb.train(params, train_data, best_itr2, valid_sets=[train_data, valid_data],

                  verbose_eval=100,

                  early_stopping_rounds=150,)
# remove overlap for public LB prediction

df_tmp = df_traintest8[

    ((df_traintest8['day']<=day_before_public)  & (pd.isna(df_traintest8['ForecastId'])))

    | ((day_before_public<df_traintest8['day']) & (pd.isna(df_traintest8['ForecastId'])==False))].reset_index(drop=True)

df_tmp = df_tmp.drop([

    'cases/day_(1-1)', 'cases/day_(1-7)', 'cases/day_(8-14)', 'cases/day_(15-21)', 

    'fatal/day_(1-1)', 'fatal/day_(1-7)', 'fatal/day_(8-14)', 'fatal/day_(15-21)',

    'days_since_1cases', 'days_since_10cases', 'days_since_100cases',

    'days_since_1fatal', 'days_since_10fatal', 'days_since_100fatal',

                               ],  axis=1)

df_traintest9 = []

for i, place in enumerate(places[:]):

    df_tmp2 = df_tmp[df_tmp['place_id']==place].reset_index(drop=True)

    df_tmp2 = do_aggregations(df_tmp2)

    df_traintest9.append(df_tmp2)

df_traintest9 = pd.concat(df_traintest9).reset_index(drop=True)

df_traintest9[df_traintest9['day']>day_before_public-2].head()
# remove overlap for private LB prediction

df_tmp = df_traintest8[

    ((df_traintest8['day']<=day_before_private)  & (pd.isna(df_traintest8['ForecastId'])))

    | ((day_before_private<df_traintest8['day']) & (pd.isna(df_traintest8['ForecastId'])==False))].reset_index(drop=True)

df_tmp = df_tmp.drop([

    'cases/day_(1-1)', 'cases/day_(1-7)', 'cases/day_(8-14)', 'cases/day_(15-21)', 

    'fatal/day_(1-1)', 'fatal/day_(1-7)', 'fatal/day_(8-14)', 'fatal/day_(15-21)',

    'days_since_1cases', 'days_since_10cases', 'days_since_100cases',

    'days_since_1fatal', 'days_since_10fatal', 'days_since_100fatal',

                               ],  axis=1)

df_traintest10 = []

for i, place in enumerate(places[:]):

    df_tmp2 = df_tmp[df_tmp['place_id']==place].reset_index(drop=True)

    df_tmp2 = do_aggregations(df_tmp2)

    df_traintest10.append(df_tmp2)

df_traintest10 = pd.concat(df_traintest10).reset_index(drop=True)

df_traintest10[df_traintest10['day']>day_before_private-2].head()
# predict test data in public

# predict the cases and fatatilites one day at a time and use the predicts as next day's feature recursively.

df_preds = []

for i, place in enumerate(places[:]):

    df_interest = copy.deepcopy(df_traintest9[df_traintest9['place_id']==place].reset_index(drop=True))

    df_interest['cases/day'][(pd.isna(df_interest['ForecastId']))==False] = -1

    df_interest['fatal/day'][(pd.isna(df_interest['ForecastId']))==False] = -1

    len_known = (df_interest['day']<=day_before_public).sum()

    len_unknown = (day_before_public<df_interest['day']).sum()

    for j in range(len_unknown): # use predicted cases and fatal for next days' prediction

        X_valid = df_interest[col_var].iloc[j+len_known]

        X_valid2 = df_interest[col_var2].iloc[j+len_known]

        pred_f = model.predict(X_valid)

        pred_c = model2.predict(X_valid2)

        pred_c = (np.exp(pred_c)-1).clip(0, 1e10)

        pred_f = (np.exp(pred_f)-1).clip(0, 1e10)

        df_interest['fatal/day'][j+len_known] = pred_f

        df_interest['cases/day'][j+len_known] = pred_c

        df_interest['Fatalities'][j+len_known] = df_interest['Fatalities'][j+len_known-1] + pred_f

        df_interest['ConfirmedCases'][j+len_known] = df_interest['ConfirmedCases'][j+len_known-1] + pred_c

#         print(df_interest['ConfirmedCases'][j+len_known-1], df_interest['ConfirmedCases'][j+len_known], pred_c)

        df_interest = df_interest.drop([

            'cases/day_(1-1)', 'cases/day_(1-7)', 'cases/day_(8-14)', 'cases/day_(15-21)', 

            'fatal/day_(1-1)', 'fatal/day_(1-7)', 'fatal/day_(8-14)', 'fatal/day_(15-21)',

            'days_since_1cases', 'days_since_10cases', 'days_since_100cases',

            'days_since_1fatal', 'days_since_10fatal', 'days_since_100fatal',



                                       ],  axis=1)

        df_interest = do_aggregations(df_interest)

    if (i+1)%10==0:

        print("{:3d}/{}  {}, len known: {}, len unknown: {}".format(i+1, len(places), place, len_known, len_unknown), df_interest.shape)

    df_interest['fatal_pred'] = np.cumsum(df_interest['fatal/day'].values)

    df_interest['cases_pred'] = np.cumsum(df_interest['cases/day'].values)

    df_preds.append(df_interest)

df_preds = pd.concat(df_preds)
# predict test data in public

df_preds_pri = []

for i, place in enumerate(places[:]):

    df_interest = copy.deepcopy(df_traintest10[df_traintest10['place_id']==place].reset_index(drop=True))

    df_interest['cases/day'][(pd.isna(df_interest['ForecastId']))==False] = -1

    df_interest['fatal/day'][(pd.isna(df_interest['ForecastId']))==False] = -1

    len_known = (df_interest['day']<=day_before_private).sum()

    len_unknown = (day_before_private<df_interest['day']).sum()

    for j in range(len_unknown): # use predicted cases and fatal for next days' prediction

        X_valid = df_interest[col_var].iloc[j+len_known]

        X_valid2 = df_interest[col_var2].iloc[j+len_known]

        pred_f = model_pri.predict(X_valid)

        pred_c = model2_pri.predict(X_valid2)

        pred_c = (np.exp(pred_c)-1).clip(0, 1e10)

        pred_f = (np.exp(pred_f)-1).clip(0, 1e10)

        df_interest['fatal/day'][j+len_known] = pred_f

        df_interest['cases/day'][j+len_known] = pred_c

        df_interest['Fatalities'][j+len_known] = df_interest['Fatalities'][j+len_known-1] + pred_f

        df_interest['ConfirmedCases'][j+len_known] = df_interest['ConfirmedCases'][j+len_known-1] + pred_c

#         print(df_interest['ConfirmedCases'][j+len_known-1], df_interest['ConfirmedCases'][j+len_known], pred_c)

        df_interest = df_interest.drop([

            'cases/day_(1-1)', 'cases/day_(1-7)', 'cases/day_(8-14)', 'cases/day_(15-21)', 

            'fatal/day_(1-1)', 'fatal/day_(1-7)', 'fatal/day_(8-14)', 'fatal/day_(15-21)',

            'days_since_1cases', 'days_since_10cases', 'days_since_100cases',

            'days_since_1fatal', 'days_since_10fatal', 'days_since_100fatal',



                                       ],  axis=1)

        df_interest = do_aggregations(df_interest)

    if (i+1)%10==0:

        print("{:3d}/{}  {}, len known: {}, len unknown: {}".format(i+1, len(places), place, len_known, len_unknown), df_interest.shape)

    df_interest['fatal_pred'] = np.cumsum(df_interest['fatal/day'].values)

    df_interest['cases_pred'] = np.cumsum(df_interest['cases/day'].values)

    df_preds_pri.append(df_interest)

df_preds_pri = pd.concat(df_preds_pri)
places_sort = df_traintest10[['place_id', 'ConfirmedCases']][df_traintest10['day']==day_before_private]

places_sort = places_sort.sort_values('ConfirmedCases', ascending=False).reset_index(drop=True)['place_id'].values

print(len(places_sort))

places_sort[:5]
print("Fatalities / Public")

plt.figure(figsize=(30,30))

for i in range(30):

    plt.subplot(5,6,i+1)

    idx = i * 10

    df_interest = df_preds[df_preds['place_id']==places_sort[idx]].reset_index(drop=True)

    tmp = df_interest['fatal/day'].values

    tmp = np.cumsum(tmp)

    sns.lineplot(x=df_interest['day'], y=tmp, label='pred')

    df_interest2 = df_traintest10[(df_traintest10['place_id']==places_sort[idx]) & (df_traintest10['day']<=day_before_private)].reset_index(drop=True)

    sns.lineplot(x=df_interest2['day'].values, y=df_interest2['Fatalities'].values, label='true')

    plt.title(places_sort[idx])

plt.show()
print("Confirmed Cases / Public")

plt.figure(figsize=(30,30))

for i in range(30):

    plt.subplot(5,6,i+1)

    idx = i * 10

    df_interest = df_preds[df_preds['place_id']==places_sort[idx]].reset_index(drop=True)

    tmp = df_interest['cases/day'].values

    tmp = np.cumsum(tmp)

    sns.lineplot(x=df_interest['day'], y=tmp, label='pred')

    df_interest2 = df_traintest10[(df_traintest10['place_id']==places_sort[idx]) & (df_traintest10['day']<=day_before_private)].reset_index(drop=True)

    sns.lineplot(x=df_interest2['day'].values, y=df_interest2['ConfirmedCases'].values, label='true')

    plt.title(places_sort[idx])

plt.show()
print("Fatalities / Private")

plt.figure(figsize=(30,30))

for i in range(30):

    plt.subplot(5,6,i+1)

    idx = i * 10

    df_interest = df_preds_pri[df_preds_pri['place_id']==places_sort[idx]].reset_index(drop=True)

    tmp = df_interest['fatal/day'].values

    tmp = np.cumsum(tmp)

    sns.lineplot(x=df_interest['day'], y=tmp, label='pred')

    df_interest2 = df_traintest10[(df_traintest10['place_id']==places_sort[idx]) & (df_traintest10['day']<=day_before_private)].reset_index(drop=True)

    sns.lineplot(x=df_interest2['day'].values, y=df_interest2['Fatalities'].values, label='true')

    plt.title(places_sort[idx])

plt.show()
print("ConfirmedCases / Private")

plt.figure(figsize=(30,30))

for i in range(30):

    plt.subplot(5,6,i+1)

    idx = i * 10

    df_interest = df_preds_pri[df_preds_pri['place_id']==places_sort[idx]].reset_index(drop=True)

    tmp = df_interest['cases/day'].values

    tmp = np.cumsum(tmp)

    sns.lineplot(x=df_interest['day'], y=tmp, label='pred')

    df_interest2 = df_traintest10[(df_traintest10['place_id']==places_sort[idx]) & (df_traintest10['day']<=day_before_private)].reset_index(drop=True)

    sns.lineplot(x=df_interest2['day'].values, y=df_interest2['ConfirmedCases'].values, label='true')

    plt.title(places_sort[idx])

plt.show()
# merge 2 preds

df_preds[df_preds['day']>day_before_private] = df_preds_pri[df_preds['day']>day_before_private]
df_preds.to_csv("df_preds.csv", index=None)
# load sample submission

df_sub = pd.read_csv("../input/covid19-global-forecasting-week-4/submission.csv")

print(len(df_sub))

df_sub.head()
# merge prediction with sub

df_sub = pd.merge(df_sub, df_traintest3[['ForecastId', 'place_id', 'day']])

df_sub = pd.merge(df_sub, df_preds[['place_id', 'day', 'cases_pred', 'fatal_pred']], on=['place_id', 'day',], how='left')

df_sub.head(10)
# save

df_sub['ConfirmedCases'] = df_sub['cases_pred']

df_sub['Fatalities'] = df_sub['fatal_pred']

df_sub = df_sub[['ForecastId', 'ConfirmedCases', 'Fatalities']]

#df_sub.to_csv("submission.csv", index=None)

df_sub.head(10)
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



from tqdm.notebook import tqdm

from scipy.optimize import curve_fit

from sklearn.metrics import r2_score

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import PolynomialFeatures

from sklearn import linear_model



%matplotlib inline

dpi = 96

plt.rcParams['figure.figsize'] = (1600/dpi, 600/dpi)

plt.style.use('ggplot')



# grabbing prepared dataset from https://www.kaggle.com/jorijnsmit/population-and-sub-continent-for-every-entity

covid = pd.read_csv('../input/covid19/covid.csv', parse_dates=['date'])



# perform same manipulations from the prepared dataset to the test set

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv', parse_dates=['Date'])

test.columns = ['id', 'province_state', 'country_region', 'date']

test['country_region'].update(test['country_region'].str.replace('Georgia', 'Sakartvelo'))

test['entity'] = test['province_state'].where(~test['province_state'].isna(), test['country_region'])

test = test.set_index('id')[['date', 'entity']]



def logistic(t, k, r, a):

    """k > 0: final epidemic size

    r > 0: infection rate

    a = (k - c_0) / c_0

    """

    

    return k / (1 + a * np.exp(-r * t))



def solve(c):

    """port from https://mathworks.com/matlabcentral/fileexchange/74411-fitvirus"""

    

    n = len(c)

    nmax = max(1, n // 2)



    for i in np.arange(1, nmax+1):

        k1 = i

        k3 = n - 1

        if (n - i) % 2 == 0:

            k3 -= 1



        k2 = (k1 + k3) // 2

        m = k2 - k1 - 1



        if k1 < 1 or k2 < 1 or k3 < 1 or m < 1:

            return None



        k1 -= 1

        k2 -= 1

        k3 -= 1



        # calculate k

        v = c[k1] * c[k2] - 2 * c[k1] * c[k3] + c[k2] * c[k3]

        if v <= 0:

            continue

        w = c[k2]**2 - c[k3] * c[k1]

        if w <= 0:

            continue

        k = c[k2] * v / w

        if k <= 0:

            continue



        # calculate r

        x = c[k3] * (c[k2] - c[k1])

        if x <= 0:

            continue

        y = c[k1] * (c[k3] - c[k2])

        if y <= 0:

            continue

        r = (1 / m) * np.log(x / y)

        if r <= 0:

            continue



        # calculate a

        z = ((c[k3] - c[k2]) * (c[k2] - c[k1])) / w

        if z <= 0:

            continue

        a = z * (x / y) ** ((k3 + 1 - m) / m)

        if a <= 0:

            continue

        

        return k, r, a



def plot_fit(x_train, y_train, x_predict, y_predict, r2):

    fig, ax = plt.subplots()

    ax.set_title(f'{subject} {r2}')

    color = 'green' if r2 > 0.99 else 'red'

    pd.Series(y_train, x_train).plot(subplots=True, style='.', color='black', legend=True, label='train')

    pd.Series(y_predict, x_predict).plot(subplots=True, style=':', color=color, legend=True, label='predict')

    plt.show()



herd_immunity = 0.7

test_ratio = 0.2



for target in ['confirmed', 'fatal']:

    for subject in tqdm(covid['entity'].unique()):

        population = covid[covid['entity'] == subject]['population'].max()



        x_train = covid[covid['entity'] == subject]['date'].dt.dayofyear.values

        y_train = covid[covid['entity'] == subject][target].values



        mask = y_train > 0

        x_train_m = x_train[mask]

        y_train_m = y_train[mask]

        

        # no point in modelling a single point or no ints at all

        if x_train_m.size < 2 or x_train_m.sum() == 0:

            continue



        x_predict = test[test['entity'] == subject]['date'].dt.dayofyear.values

        submission_size = x_predict.size

        # start calculating sigmoid at same point x_train_m starts

        x_predict = np.arange(start=x_train_m[0], stop=x_predict[-1]+1)



        params = solve(y_train_m)



        if params != None:

        #try:

            params = (max(params[0], max(y_train_m)), params[1], params[2])

            lower_bounds = (max(y_train_m), 0, 0)

            upper_bounds = (max(population * herd_immunity * test_ratio, params[0]), np.inf, np.inf)



            params, _ = curve_fit(

                logistic,

                np.arange(x_train_m.size),

                y_train_m,

                p0=params,

                bounds=(lower_bounds, upper_bounds),

                maxfev=100000

            )



            y_eval = logistic(np.arange(x_train_m.size), params[0], params[1], params[2])

            y_predict = logistic(np.arange(x_predict.size), params[0], params[1], params[2])



            r2 = r2_score(y_train_m, y_eval)

            covid.loc[covid['entity'] == subject, f'log_{target}'] = r2



        else:

            # we fit a polynomial instead

            # while forcing cumulative behaviour, i.e. never lower numbers

            # it's ugly

            # i know



            model = linear_model.LinearRegression()

#             model = Pipeline([

#                 ("polynomial_features", PolynomialFeatures(degree=2)), 

#                 ("linear_regression", linear_model.Ridge())

#             ])

            if target == 'fatal':

                # pass more features; including confirmed!

                pass

            model.fit(x_train_m.reshape(-1, 1), y_train_m)



            y_eval = model.predict(x_train_m.reshape(-1, 1))

            y_predict = model.predict(x_predict.reshape(-1, 1))

            y_predict = np.maximum.accumulate(y_predict)



            r2 = r2_score(y_train_m, y_eval)

            covid.loc[covid['entity'] == subject, f'poly_{target}'] = r2



        if target == 'confirmed' and subject in ['Hubei', 'Italy', 'New York']:

            plot_fit(x_train, y_train, x_predict, y_predict, r2)



        # assign the prediction to the test dataframe

        delta = submission_size - y_predict.size

        if delta > 0:

            filler = [100] * delta if target == 'confirmed' else [1] * delta

            y_predict = filler + y_predict.tolist()

        test.loc[test['entity'] == subject, target] = y_predict[-submission_size:]



# resulting R2 scores for logistic approach

for target in ['confirmed', 'fatal']:

    r2s = covid.groupby('entity')[f'log_{target}'].max()

    print(r2s.describe())

    print(r2s[r2s.isna()].index)



# any doubtful maxima due to regression?

for target in ['confirmed', 'fatal']:

    df = []

    for subject in covid.loc[covid[f'poly_{target}'].isna()]['entity'].unique():

        df.append(test[test['entity'] == subject][['entity', target]].max().to_dict())

    df = pd.DataFrame(df).set_index('entity')

    print(df[target].sort_values(ascending=False).fillna(0).astype('int').head(10))

    

# @TODO

# some are way too high; this is a problem!

# what are the parameters they are fitted on?



# sanity check before submitting

submission = test[['entity', 'date']].copy()

submission[['confirmed', 'fatal']] = test[['confirmed', 'fatal']].fillna(0).astype('int')

submission[submission['entity'] == 'Netherlands']



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



from tqdm.notebook import tqdm

from scipy.optimize import curve_fit

from sklearn.metrics import r2_score

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import PolynomialFeatures

from sklearn import linear_model



%matplotlib inline

dpi = 96

plt.rcParams['figure.figsize'] = (1600/dpi, 600/dpi)

plt.style.use('ggplot')



# grabbing prepared dataset from https://www.kaggle.com/jorijnsmit/population-and-sub-continent-for-every-entity

covid = pd.read_csv('../input/covid19/covid.csv', parse_dates=['date'])



# perform same manipulations from the prepared dataset to the test set

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv', parse_dates=['Date'])

test.columns = ['id', 'province_state', 'country_region', 'date']

test['country_region'].update(test['country_region'].str.replace('Georgia', 'Sakartvelo'))

test['entity'] = test['province_state'].where(~test['province_state'].isna(), test['country_region'])

test = test.set_index('id')[['date', 'entity']]



def logistic(t, k, r, a):

    """k > 0: final epidemic size

    r > 0: infection rate

    a = (k - c_0) / c_0

    """

    

    return k / (1 + a * np.exp(-r * t))



def solve(c):

    """port from https://mathworks.com/matlabcentral/fileexchange/74411-fitvirus"""

    

    n = len(c)

    nmax = max(1, n // 2)



    for i in np.arange(1, nmax+1):

        k1 = i

        k3 = n - 1

        if (n - i) % 2 == 0:

            k3 -= 1



        k2 = (k1 + k3) // 2

        m = k2 - k1 - 1



        if k1 < 1 or k2 < 1 or k3 < 1 or m < 1:

            return None



        k1 -= 1

        k2 -= 1

        k3 -= 1



        # calculate k

        v = c[k1] * c[k2] - 2 * c[k1] * c[k3] + c[k2] * c[k3]

        if v <= 0:

            continue

        w = c[k2]**2 - c[k3] * c[k1]

        if w <= 0:

            continue

        k = c[k2] * v / w

        if k <= 0:

            continue



        # calculate r

        x = c[k3] * (c[k2] - c[k1])

        if x <= 0:

            continue

        y = c[k1] * (c[k3] - c[k2])

        if y <= 0:

            continue

        r = (1 / m) * np.log(x / y)

        if r <= 0:

            continue



        # calculate a

        z = ((c[k3] - c[k2]) * (c[k2] - c[k1])) / w

        if z <= 0:

            continue

        a = z * (x / y) ** ((k3 + 1 - m) / m)

        if a <= 0:

            continue

        

        return k, r, a



def plot_fit(x_train, y_train, x_predict, y_predict, r2):

    fig, ax = plt.subplots()

    ax.set_title(f'{subject} {r2}')

    color = 'green' if r2 > 0.99 else 'red'

    pd.Series(y_train, x_train).plot(subplots=True, style='.', color='black', legend=True, label='train')

    pd.Series(y_predict, x_predict).plot(subplots=True, style=':', color=color, legend=True, label='predict')

    plt.show()



herd_immunity = 0.7

test_ratio = 0.2



for target in ['confirmed', 'fatal']:

    for subject in tqdm(covid['entity'].unique()):

        population = covid[covid['entity'] == subject]['population'].max()



        x_train = covid[covid['entity'] == subject]['date'].dt.dayofyear.values

        y_train = covid[covid['entity'] == subject][target].values



        mask = y_train > 0

        x_train_m = x_train[mask]

        y_train_m = y_train[mask]

        

        # no point in modelling a single point or no ints at all

        if x_train_m.size < 2 or x_train_m.sum() == 0:

            continue



        x_predict = test[test['entity'] == subject]['date'].dt.dayofyear.values

        submission_size = x_predict.size

        # start calculating sigmoid at same point x_train_m starts

        x_predict = np.arange(start=x_train_m[0], stop=x_predict[-1]+1)



        params = solve(y_train_m)



        if params != None:

        #try:

            params = (max(params[0], max(y_train_m)), params[1], params[2])

            lower_bounds = (max(y_train_m), 0, 0)

            upper_bounds = (max(population * herd_immunity * test_ratio, params[0]), np.inf, np.inf)



            params, _ = curve_fit(

                logistic,

                np.arange(x_train_m.size),

                y_train_m,

                p0=params,

                bounds=(lower_bounds, upper_bounds),

                maxfev=100000

            )



            y_eval = logistic(np.arange(x_train_m.size), params[0], params[1], params[2])

            y_predict = logistic(np.arange(x_predict.size), params[0], params[1], params[2])



            r2 = r2_score(y_train_m, y_eval)

            covid.loc[covid['entity'] == subject, f'log_{target}'] = r2



        else:

            # we fit a polynomial instead

            # while forcing cumulative behaviour, i.e. never lower numbers

            # it's ugly

            # i know



            model = linear_model.LinearRegression()

#             model = Pipeline([

#                 ("polynomial_features", PolynomialFeatures(degree=2)), 

#                 ("linear_regression", linear_model.Ridge())

#             ])

            if target == 'fatal':

                # pass more features; including confirmed!

                pass

            model.fit(x_train_m.reshape(-1, 1), y_train_m)



            y_eval = model.predict(x_train_m.reshape(-1, 1))

            y_predict = model.predict(x_predict.reshape(-1, 1))

            y_predict = np.maximum.accumulate(y_predict)



            r2 = r2_score(y_train_m, y_eval)

            covid.loc[covid['entity'] == subject, f'poly_{target}'] = r2



        if target == 'confirmed' and subject in ['Hubei', 'Italy', 'New York']:

            plot_fit(x_train, y_train, x_predict, y_predict, r2)



        # assign the prediction to the test dataframe

        delta = submission_size - y_predict.size

        if delta > 0:

            filler = [100] * delta if target == 'confirmed' else [1] * delta

            y_predict = filler + y_predict.tolist()

        test.loc[test['entity'] == subject, target] = y_predict[-submission_size:]



# resulting R2 scores for logistic approach

for target in ['confirmed', 'fatal']:

    r2s = covid.groupby('entity')[f'log_{target}'].max()

    print(r2s.describe())

    print(r2s[r2s.isna()].index)



# any doubtful maxima due to regression?

for target in ['confirmed', 'fatal']:

    df = []

    for subject in covid.loc[covid[f'poly_{target}'].isna()]['entity'].unique():

        df.append(test[test['entity'] == subject][['entity', target]].max().to_dict())

    df = pd.DataFrame(df).set_index('entity')

    print(df[target].sort_values(ascending=False).fillna(0).astype('int').head(10))

    

# @TODO

# some are way too high; this is a problem!

# what are the parameters they are fitted on?



# sanity check before submitting

submission = test[['entity', 'date']].copy()

submission[['confirmed', 'fatal']] = test[['confirmed', 'fatal']].fillna(0).astype('int')

submission[submission['entity'] == 'Netherlands']
submission = submission[['confirmed', 'fatal']]

submission.index.name = 'ForecastId'

submission.columns = ['ConfirmedCases', 'Fatalities']



submission['ConfirmedCases'] = ((df_sub['ConfirmedCases'].values + submission['ConfirmedCases'].values)/2).astype(int)

submission['Fatalities'] = ((df_sub['Fatalities'].values + submission['Fatalities'].values)/2).astype(int)



submission.to_csv('submission.csv')