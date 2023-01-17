import os, gc, pickle, copy, datetime, warnings

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb

from xgboost import XGBRegressor

from sklearn.ensemble import RandomForestRegressor

from catboost import CatBoostRegressor

from sklearn import metrics

pd.set_option('display.max_columns', 100) 

warnings.filterwarnings('ignore')
df_train = pd.read_csv("../input/covid19-global-forecasting-week-3/train.csv")

print(df_train.shape)

df_train.head()
df_test = pd.read_csv("../input/covid19-global-forecasting-week-3/test.csv")

print(df_test.shape)

df_test.head()
# concat train and test

df_traintest = pd.concat([df_train, df_test])

print(df_train.shape, df_test.shape, df_traintest.shape)
# concat Country/Region and Province/State

def func(x):

    try:

        x_new = x['Country_Region'] + "/" + x['Province_State']

    except:

        x_new = x['Country_Region']

    return x_new

        

df_traintest['place_id'] = df_traintest.apply(lambda x: func(x), axis=1)

tmp = np.sort(df_traintest['place_id'].unique())

print("num unique places: {}".format(len(tmp)))

print(tmp[:10])
# get place list

places = np.sort(df_traintest['place_id'].unique())
# process date

df_traintest['Date'] = pd.to_datetime(df_traintest['Date'])

df_traintest['day'] = df_traintest['Date'].apply(lambda x: x.dayofyear).astype(np.int16)

df_traintest.head()
# calc cases, fatalities, recover per day

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

#df_life.columns = ['Country_Region', 'LifeExpectancy']

df_life.rename(columns = {'Country' : 'Country_Region'}, inplace=True)

df_life.rename(columns = {'2018' : 'LifeExpectancy'}, inplace=True)
# merge

df_traintest6 = pd.merge(df_traintest5, df_life, on='Country_Region', how='left')

print(len(df_traintest6))

df_traintest6.head()
# add additional info from countryinfo dataset

df_country = pd.read_csv("../input/countryinfo/covid19countryinfo.csv")

df_country.head()
df_country['Country_Region'] = df_country['country']

df_country = df_country[df_country['country'].duplicated()==False]
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
df_traintest8
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

col_target = 'fatal/day'

col_var = [

#    'Lat', 'Long',

     'days_since_1cases', 

     'days_since_10cases', 

     'days_since_100cases',

     'days_since_1fatal', 

     'days_since_10fatal', 'days_since_100fatal',

#     'days_since_1recov',

#     'days_since_10recov', 'days_since_100recov', 

    'cases/day_(1-1)', 

    'cases/day_(1-7)', 

     'cases/day_(8-14)',  

     'cases/day_(15-21)', 

    

     'fatal/day_(1-1)', 

    'fatal/day_(1-7)', 

    'fatal/day_(8-14)', 

    'fatal/day_(15-21)', 

    'SmokingRate',

     'Gross_domestic_product__constant_prices',

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

     'LifeExpectancy',

     'pop',

    'density', 

#     'medianage', 

#     'urbanpop', 

     'hospibed', 

#    'smokers',

]
# train model to predict fatalities/day

col_target2 = 'cases/day'

col_var2 = [

#    'Lat', 'Long',

     'days_since_1cases', 

    'days_since_10cases', #selected

     'days_since_100cases',

     'days_since_1fatal', 

     'days_since_10fatal',

     'days_since_100fatal',

#     'days_since_1recov',

#     'days_since_10recov', 'days_since_100recov', 

    'cases/day_(1-1)', 

    'cases/day_(1-7)', 

    'cases/day_(8-14)',  

    'cases/day_(15-21)', 

    

     'fatal/day_(1-1)', 

     'fatal/day_(1-7)', 

     'fatal/day_(8-14)', 

     'fatal/day_(15-21)', 

#     'recov/day_(1-1)', 'recov/day_(1-7)', 

#     'recov/day_(8-14)',  'recov/day_(15-21)',

#     'active_(1-1)', 

#     'active_(1-7)', 

#     'active_(8-14)',  'active_(15-21)', 

     'SmokingRate',

     'Gross_domestic_product__constant_prices',

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

     'LifeExpectancy',

     'pop',

     'density', 

#     'medianage', 

#     'urbanpop', 

     'hospibed', #'smokers', 

]
# train model to predict fatalities/day

col_cat = []

df_train = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']<88)]

df_valid = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']>=88)]

df_test = df_traintest8[pd.isna(df_traintest8['ForecastId'])==False]

X_train = df_train[col_var]

X_valid = df_valid[col_var]

y_train = np.log(df_train[col_target].values.clip(0, 1e10)+1)

y_valid = np.log(df_valid[col_target].values.clip(0, 1e10)+1)

train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=col_cat)

valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature=col_cat)

num_round = 15000

model_pri = lgb.train(params, train_data, num_round, valid_sets=[train_data, valid_data],

                  verbose_eval=100,

                  early_stopping_rounds=150,)

best_itr = model_pri.best_iteration
df_train = df_traintest8[(pd.isna(df_traintest8['ForecastId']))]

df_valid = df_traintest8[(pd.isna(df_traintest8['ForecastId']))]

# df_valid = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']>=61)]

df_test = df_traintest8[pd.isna(df_traintest8['ForecastId'])==False]

X_train = df_train[col_var]

X_valid = df_valid[col_var]

y_train = np.log(df_train[col_target].values.clip(0, 1e10)+1)

y_valid = np.log(df_valid[col_target].values.clip(0, 1e10)+1)

train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=col_cat)

valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature=col_cat)

model_pri = lgb.train(params, train_data, best_itr, valid_sets=[train_data, valid_data],

                  verbose_eval=100,

                  early_stopping_rounds=150,)



model_pri_xgb = XGBRegressor(n_estimators=1000, eval_set= [(X_valid,y_valid)], early_stopping_rounds=150)

model_pri_xgb.fit(X_train.as_matrix(),y_train)



model_pri_cat = CatBoostRegressor()

model_pri_cat.fit(X_train.as_matrix(),y_train)
# train model to predict fatalities/day

df_train = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']<88)]

df_valid = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']>=88)]

df_test = df_traintest8[pd.isna(df_traintest8['ForecastId'])==False]

X_train = df_train[col_var2]

X_valid = df_valid[col_var2]

y_train = np.log(df_train[col_target2].values.clip(0, 1e10)+1)

y_valid = np.log(df_valid[col_target2].values.clip(0, 1e10)+1)

train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=col_cat)

valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature=col_cat)

model2_pri = lgb.train(params, train_data, num_round, valid_sets=[train_data, valid_data],

                  verbose_eval=100,

                  early_stopping_rounds=150,)

best_itr = model2_pri.best_iteration
df_train = df_traintest8[(pd.isna(df_traintest8['ForecastId']))]

df_valid = df_traintest8[(pd.isna(df_traintest8['ForecastId']))]

# df_valid = df_traintest8[(pd.isna(df_traintest8['ForecastId'])) & (df_traintest8['day']>=61)]

df_test = df_traintest8[pd.isna(df_traintest8['ForecastId'])==False]

X_train = df_train[col_var2]

X_valid = df_valid[col_var2]

y_train = np.log(df_train[col_target2].values.clip(0, 1e10)+1)

y_valid = np.log(df_valid[col_target2].values.clip(0, 1e10)+1)

train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=col_cat)

valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature=col_cat)

model2_pri = lgb.train(params, train_data, best_itr, valid_sets=[train_data, valid_data],

                  verbose_eval=100,

                  early_stopping_rounds=150,)

best_itr = model2_pri.best_iteration



model2_pri_xgb = XGBRegressor(n_estimators=1000, eval_set= [(X_valid,y_valid)], early_stopping_rounds=150)

model2_pri_xgb.fit(X_train.as_matrix(),y_train)



model2_pri_cat = CatBoostRegressor()

model2_pri_cat.fit(X_train.as_matrix(),y_train)
# predict test data in public

day_before_private = 85

df_preds_pri = []

for i, place in enumerate(places[:]):

#     if place!='Japan' and place!='Afghanistan' :continue

    df_interest = copy.deepcopy(df_traintest8[df_traintest8['place_id']==place].reset_index(drop=True))

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
# concat prediction

df_preds_pri= pd.concat(df_preds_pri)

df_preds_pri = df_preds_pri.sort_values('day')

col_tmp = ['place_id', 'ForecastId', 'Date', 'day', 'cases/day', 'cases_pred', 'fatal/day', 'fatal_pred',]

df_preds_pri[col_tmp][(df_preds_pri['place_id']=='Japan') & (df_preds_pri['day']>79)].head(10)
df_preds_pri
# load sample submission

df_sub = pd.read_csv("../input/covid19-global-forecasting-week-3/submission.csv")

print(len(df_sub))

df_sub.head()
df_traintest3[['ForecastId', 'place_id', 'day']]
# merge prediction with sub

df_sub = pd.merge(df_sub, df_traintest3[['ForecastId', 'place_id', 'day']][pd.isna(df_traintest3['ForecastId'])==False])

df_sub
df_sub = pd.merge(df_sub, df_preds_pri[['place_id', 'day', 'cases_pred', 'fatal_pred']][pd.isna(df_traintest3['ForecastId'])==False], on=['place_id', 'day',], how='left')

df_sub.head(10)
# save

df_sub['ConfirmedCases'] = df_sub['cases_pred']

df_sub['Fatalities'] = df_sub['fatal_pred']

df_sub = df_sub[['ForecastId', 'ConfirmedCases', 'Fatalities']]

df_sub.to_csv("submission.csv", index=None)
df_sub