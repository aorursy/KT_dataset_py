import numpy as np 
import pandas as pd

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm
from sklearn import metrics
pd.set_option('display.max_columns', 100)
current_day = 17
train_test_data = pd.DataFrame()
train_old = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")
train_data = pd.read_csv("../input/newcovidtrain/new_train.csv")
test_data = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")
test_data.at[test_data.index % 43 <=15,'ForecastId'] = None
ban = test_data[(test_data['Country_Region'] == 'Diamond Princess') | (test_data['ForecastId'] == None) | (test_data['Country_Region'] == 'US')].index
ban2 = train_data[ (train_data['Country_Region'] == 'Yemen') | (train_data['Country_Region'] == 'US')].index
train_data = train_data.drop(ban2, axis = 0)
train_data = train_data.reset_index()
test_data = test_data.drop(ban,axis = 0)
test_data = test_data.reset_index()
test_data = test_data[pd.isna(test_data['ForecastId']) == False]
test_data = test_data.reset_index()
train_test_data = pd.concat([train_data, test_data])
train_test_data['Date'] = pd.to_datetime(train_test_data['Date'])
train_test_data['day'] = train_test_data['Date'].apply(lambda x: x.dayofyear).astype(np.int16)
def RegionState(df):
    try:
        tmp = df['Country_Region']+'/'+df['Province_State']
    except:
        tmp = df['Country_Region']
    return tmp
        
train_test_data['place_id'] = train_test_data.apply(lambda x: RegionState(x), axis=1)
#добавляем csv со статистиками курения
smoke_data = pd.read_csv("../input/smokingstats/df_Latlong.csv")
def RegionStateSmoke(df):
    try:
        tmp = df['Country/Region']+'/'+df['Province/State']
    except:
        tmp = df['Country/Region']
    return tmp
smoke_data['place_id'] = smoke_data.apply(lambda x: RegionStateSmoke(x), axis=1)
smoke_data = smoke_data[smoke_data['place_id'].duplicated() == False]
train_test_data = pd.merge(train_test_data, smoke_data[['place_id' , 'Lat' , 'Long']], on = 'place_id', how = 'left')
#здесь начинаю
import copy
place_array = np.sort(train_test_data['place_id'].unique())
train_test_data['cases_per_day'] = 0
train_test_data['fatal_per_day'] = 0
tmp_list = np.zeros(len(train_test_data))
for place in place_array:
    tmp = train_test_data['ConfirmedCases'][train_test_data['place_id'] == place].values
    tmp[1:] -= tmp[:-1]
    train_test_data['cases_per_day'][train_test_data['place_id'] == place] = tmp
    tmp = train_test_data['Fatalities'][train_test_data['place_id'] == place].values
    tmp[1:] -= tmp[:-1]
    train_test_data['fatal_per_day'][train_test_data['place_id'] == place] = tmp
def do_aggregation(df, col, mean_range):
    df_new = copy.deepcopy(df)
    col_new = '{}_({}-{})'.format(col, mean_range[0], mean_range[1])
    df_new[col_new] = 0
    tmp = df_new[col].rolling(mean_range[1]-mean_range[0]+1).mean()
    df_new[col_new][mean_range[0]:] = tmp[:-(mean_range[0])]
    df_new[col_new][pd.isna(df_new[col_new])] = 0
    return df_new[[col_new]].reset_index(drop=True)

def do_aggregations(df):
    df = pd.concat([df, do_aggregation(df, 'cases_per_day', [1,1]).reset_index(drop = True)], axis=1)
    df = pd.concat([df, do_aggregation(df, 'cases_per_day', [1,7]).reset_index(drop = True)], axis=1)
    df = pd.concat([df, do_aggregation(df, 'cases_per_day', [8,14]).reset_index(drop = True)], axis=1)
    df = pd.concat([df, do_aggregation(df, 'cases_per_day', [15,21]).reset_index(drop = True)], axis=1)
    df = pd.concat([df, do_aggregation(df, 'fatal_per_day', [1,1]).reset_index(drop = True)], axis=1)
    df = pd.concat([df, do_aggregation(df, 'fatal_per_day', [1,7]).reset_index(drop = True)], axis=1)
    df = pd.concat([df, do_aggregation(df, 'fatal_per_day', [8,14]).reset_index(drop = True)], axis=1)
    df = pd.concat([df, do_aggregation(df, 'fatal_per_day', [15,21]).reset_index(drop = True)], axis=1)
    for threshold in [1, 10, 100]:
        days_under_threshold = (df['ConfirmedCases'] < threshold).sum()
        tmp = df['day'].values - 22 - days_under_threshold
        tmp[tmp <= 0] = 0
        df['days_since_{}cases'.format(threshold)] = tmp
    
    
    for threshold in [1, 10, 100]:
        days_under_threshold = (df['Fatalities'] < threshold).sum()
        tmp = df['day'].values - 22 - days_under_threshold
        tmp[tmp <= 0] = 0
        df['days_since_{}fatal'.format(threshold)] = tmp
    # process China/Hubei
    if df['place_id'][0]=='China/Hubei':
        df['days_since_1cases'] += 35 # 2019/12/8
        df['days_since_10cases'] += 35-13 # 2019/12/8-2020/1/2 assume 2019/12/8+13
        df['days_since_100cases'] += 4 # 2020/1/18
        df['days_since_1fatal'] += 13 # 2020/1/9
    return df
train_test_data_new = []
for place in place_array[:]:
    data_tmp = train_test_data[train_test_data['place_id'] == place].reset_index(drop = True)
    data_tmp = do_aggregations(data_tmp)
    train_test_data_new.append(data_tmp)
train_test_data_new = pd.concat(train_test_data_new).reset_index(drop = True)
train_test_data = train_test_data_new
smoke_data2 = pd.read_csv("../input/smokingstats/share-of-adults-who-smoke.csv")
less_smoke_data = smoke_data2.sort_values('Year', ascending = False).reset_index(drop = True)
less_smoke_data = less_smoke_data[less_smoke_data['Entity'].duplicated() == False]
less_smoke_data['Country_Region'] = less_smoke_data['Entity']
less_smoke_data['SmokingRate'] = less_smoke_data['Smoking prevalence, total (ages 15+) (% of adults)']
train_test_data = pd.merge(train_test_data, less_smoke_data[['Country_Region', 'SmokingRate']], on = 'Country_Region', how = 'left')
SmokingRate = 20.48
train_test_data['SmokingRate'][pd.isna(train_test_data['SmokingRate'])] = SmokingRate
weo_data = pd.read_csv("../input/smokingstats/WEO.csv")
subs  = weo_data['Subject Descriptor'].unique()[:-1]
df_weo_agg = weo_data[['Country']][weo_data['Country'].duplicated()==False].reset_index(drop=True)
for sub in subs[:]:
    df_tmp = weo_data[['Country', '2019']][weo_data['Subject Descriptor']==sub].reset_index(drop=True)
    df_tmp = df_tmp[df_tmp['Country'].duplicated()==False].reset_index(drop=True)
    df_tmp.columns = ['Country', sub]
    df_weo_agg = df_weo_agg.merge(df_tmp, on='Country', how='left')
df_weo_agg.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in df_weo_agg.columns]
df_weo_agg.columns
df_weo_agg['Country_Region'] = df_weo_agg['Country']
df_weo_agg.head()
train_test_data = pd.merge(train_test_data, df_weo_agg, on='Country_Region', how='left')
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
train_test_data = pd.merge(train_test_data, df_life, on='Country_Region', how='left')
print(len(train_test_data))
train_test_data.head()
# add additional info from countryinfo dataset
df_country = pd.read_csv("../input/countryinfo/covid19countryinfo.csv")


hospital_data = pd.read_csv("../input/global-hospital-beds-capacity-for-covid19/hospital_beds_global_v1.csv")
cols = ['country','type','beds','year']
hospital_data = hospital_data[cols]
del_list = hospital_data[hospital_data['type'] != 'TOTAL'].index
hospital_data.drop(del_list , inplace = True)
cols = ['country','beds']
hospital_data = hospital_data[cols]
hospital_data = hospital_data.reset_index()
hospital_data['alpha2code'] = hospital_data['country']
cols = ['alpha2code','beds']
hospital_data = hospital_data[cols]

df_country = pd.merge(df_country, hospital_data, on = ['alpha2code'], how = 'left')
df_country['Country_Region'] = df_country['country']
df_country = df_country[df_country['country'].duplicated()==False]
train_test_data = pd.merge(train_test_data, 
                         df_country.drop(['tests', 'testpop', 'country'], axis=1), 
                         on=['Country_Region',], how='left')
train_test_data['beds'][pd.isna(train_test_data['beds'])] = 3.05
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

train_test_data['id'] = np.arange(len(train_test_data))
df_le = get_df_le(train_test_data, 'id', ['Country_Region', 'Province_State'])
train_test_data = pd.merge(train_test_data, df_le, on='id', how='left')
train_test_data['cases_per_day'] = train_test_data['cases_per_day'].astype(np.float)
train_test_data['fatal_per_day'] = train_test_data['fatal_per_day'].astype(np.float)

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
    train_test_data[col] = train_test_data[col].apply(lambda x: func(x))  
def calc_score(y_true, y_pred):
    y_true[y_true<0] = 0
    score = metrics.mean_squared_error(np.log(y_true.clip(0, 1e10)+1), np.log(y_pred[:]+1))**0.5
    return score
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
day_before_valid = 85
day_before_public = 78 + 14
day_before_launch = 85 + 14
# train model to predict fatalities/day
# features are selected manually based on valid score
col_target = 'fatal_per_day'
col_var = [
    'Lat', 'Long', 
    'cases_per_day_(1-1)', 
    'cases_per_day_(1-7)',  
    'fatal_per_day_(1-7)', 
    'fatal_per_day_(8-14)', 
    'fatal_per_day_(15-21)', 
    'SmokingRate',
     'Gross_domestic_product_per_capita__constant_prices',
     'Gross_domestic_product_based_on_purchasing_power_parity__PPP__per_capita_GDP', 
    'Population', 
     'LifeExpectancy',
    'density', 
     'medianage',
    'beds'
]
col_cat = []
df_train = train_test_data[(pd.isna(train_test_data['ForecastId'])) & (train_test_data['day']<=day_before_valid)]
df_valid = train_test_data[(pd.isna(train_test_data['ForecastId'])) & (day_before_valid<train_test_data['day']) & (train_test_data['day']<=day_before_public)]
df_test = train_test_data[pd.isna(train_test_data['ForecastId'])==False]
X_train = df_train[col_var]
X_valid = df_valid[col_var]
y_train = np.log(df_train[col_target].values.clip(0, 1e10)+1)
y_valid = np.log(df_valid[col_target].values.clip(0, 1e10)+1)
train = lightgbm.Dataset(X_train, label=y_train, categorical_feature=col_cat)
valid = lightgbm.Dataset(X_valid, label=y_valid, categorical_feature=col_cat)
num_round = 15000
model = lightgbm.train(params, train, num_round, valid_sets=[train, valid],
                  verbose_eval=100,
                  early_stopping_rounds=150,)

best_itr = model.best_iteration
# display feature importance
tmp = pd.DataFrame()
tmp["feature"] = col_var
tmp["importance"] = model.feature_importance()
tmp = tmp.sort_values('importance', ascending=False)
tmp
# train with all data
df_train = train_test_data[(pd.isna(train_test_data['ForecastId']))]
df_valid = train_test_data[(pd.isna(train_test_data['ForecastId']))]
X_train = df_train[col_var]
X_valid = df_valid[col_var]
y_train = np.log(df_train[col_target].values.clip(0, 1e10)+1)
y_valid = np.log(df_valid[col_target].values.clip(0, 1e10)+1)
train = lightgbm.Dataset(X_train, label=y_train, categorical_feature=col_cat)
valid = lightgbm.Dataset(X_valid, label=y_valid, categorical_feature=col_cat)
model_pri = lightgbm.train(params, train, best_itr, valid_sets=[train, valid],
                  verbose_eval=100,
                  early_stopping_rounds=150,)
# train model to predict cases/day
col_target2 = 'cases_per_day'
col_var2 = [
    'Lat', 'Long',
    'days_since_10cases',
    'cases_per_day_(1-1)', 
    'cases_per_day_(1-7)', 
    'cases_per_day_(8-14)',  
    'cases_per_day_(15-21)', 
     'SmokingRate',
     'Gross_domestic_product_per_capita__constant_prices',
     'Gross_domestic_product_based_on_purchasing_power_parity__PPP__per_capita_GDP',
     'Population',
     'LifeExpectancy',
     'density', 
     'medianage'
]
col_cat = []
df_train = train_test_data[(pd.isna(train_test_data['ForecastId'])) & (train_test_data['day']<=day_before_valid)]
df_valid = train_test_data[(pd.isna(train_test_data['ForecastId'])) & (day_before_valid<train_test_data['day']) & (train_test_data['day']<=day_before_public)]
X_train = df_train[col_var2]
X_valid = df_valid[col_var2]
y_train = np.log(df_train[col_target2].values.clip(0, 1e10)+1)
y_valid = np.log(df_valid[col_target2].values.clip(0, 1e10)+1)
train = lightgbm.Dataset(X_train, label=y_train, categorical_feature=col_cat)
valid = lightgbm.Dataset(X_valid, label=y_valid, categorical_feature=col_cat)
model2 = lightgbm.train(params, train, num_round, valid_sets=[train, valid],
                  verbose_eval=100,
                  early_stopping_rounds=150,)
best_itr2 = model2.best_iteration
# display feature importance
tmp = pd.DataFrame()
tmp["feature"] = col_var2
tmp["importance"] = model2.feature_importance()
tmp = tmp.sort_values('importance', ascending=False)
tmp
# train with all data
df_train = train_test_data[(pd.isna(train_test_data['ForecastId']))]
df_valid = train_test_data[(pd.isna(train_test_data['ForecastId']))]
X_train = df_train[col_var2]
X_valid = df_valid[col_var2]
y_train = np.log(df_train[col_target2].values.clip(0, 1e10)+1)
y_valid = np.log(df_valid[col_target2].values.clip(0, 1e10)+1)
train = lightgbm.Dataset(X_train, label=y_train, categorical_feature=col_cat)
valid = lightgbm.Dataset(X_valid, label=y_valid, categorical_feature=col_cat)
model2_pri = lightgbm.train(params, train, best_itr2, valid_sets=[train, valid],
                  verbose_eval=100,
                  early_stopping_rounds=150,)
train_test_data[train_test_data['place_id'] == 'US']
day_before_public = 108
df_tmp = train_test_data[
    ((train_test_data['day'] <= day_before_public)  & (pd.isna(train_test_data['ForecastId'])))
    | ((day_before_public < train_test_data['day']) & (pd.isna(train_test_data['ForecastId'])==False))].reset_index(drop=True)
df_tmp = df_tmp.drop([
    'cases_per_day_(1-1)', 'cases_per_day_(1-7)', 'cases_per_day_(8-14)', 'cases_per_day_(15-21)', 
    'fatal_per_day_(1-1)', 'fatal_per_day_(1-7)', 'fatal_per_day_(8-14)', 'fatal_per_day_(15-21)',
    'days_since_1cases', 'days_since_10cases', 'days_since_100cases',
    'days_since_1fatal', 'days_since_10fatal', 'days_since_100fatal',
                               ],  axis=1)
new_data = []
for i, place in enumerate(place_array[:]):
    df_tmp2 = df_tmp[df_tmp['place_id'] == place].reset_index(drop = True)
    df_tmp2 = do_aggregations(df_tmp2)
    new_data.append(df_tmp2)
new_data = pd.concat(new_data).reset_index(drop = True)
# predict the cases and fatatilites one day at a time and use the predicts as next day's feature recursively.
days = 7
day_before_private = 108 #инк
df_preds_pri = []
for i, place in enumerate(place_array[:]):
    print(place)
    df_interest = copy.deepcopy(new_data[new_data['place_id']==place].reset_index(drop=True))
    df_interest['cases_per_day'][(pd.isna(df_interest['ForecastId']))==False] = -1
    df_interest['fatal_per_day'][(pd.isna(df_interest['ForecastId']))==False] = -1
    len_known = (df_interest['day'] <= day_before_private).sum()
    len_unknown = (day_before_private < df_interest['day']).sum()
    for j in range(len_unknown):# use predicted cases and fatal for next days' prediction
        X_valid = df_interest[col_var].iloc[j+len_known]
        X_valid2 = df_interest[col_var2].iloc[j+len_known]
        pred_f = model_pri.predict(X_valid)
        pred_c = model2_pri.predict(X_valid2)
        pred_c = (np.exp(pred_c)-1).clip(0, 1e10)
        pred_f = (np.exp(pred_f)-1).clip(0, 1e10)
        df_interest['fatal_per_day'][j+len_known] = pred_f
        df_interest['cases_per_day'][j+len_known] = pred_c
        df_interest['Fatalities'][j+len_known] = df_interest['Fatalities'][j+len_known-1] + pred_f
        df_interest['ConfirmedCases'][j+len_known] = df_interest['ConfirmedCases'][j+len_known-1] + pred_c
#         print(df_interest['ConfirmedCases'][j+len_known-1], df_interest['ConfirmedCases'][j+len_known], pred_c)
        df_interest = df_interest.drop([
            'cases_per_day_(1-1)', 'cases_per_day_(1-7)', 'cases_per_day_(8-14)', 'cases_per_day_(15-21)', 
            'fatal_per_day_(1-1)', 'fatal_per_day_(1-7)', 'fatal_per_day_(8-14)', 'fatal_per_day_(15-21)',
            'days_since_1cases', 'days_since_10cases', 'days_since_100cases',
            'days_since_1fatal', 'days_since_10fatal', 'days_since_100fatal',

                                       ],  axis=1)
        df_interest = do_aggregations(df_interest)
    if (i+1)%10==0:
        print("{:3d}/{}  {}, len known: {}, len unknown: {}".format(i+1, len(place_array), place, len_known, len_unknown), df_interest.shape)
    df_interest['fatal_pred'] = np.cumsum(df_interest['fatal_per_day'].values)
    df_interest['cases_pred'] = np.cumsum(df_interest['cases_per_day'].values)
    df_preds_pri.append(df_interest)
df_preds_pri = pd.concat(df_preds_pri)
df_preds_pri.iloc[85:100]
cols = ['ConfirmedCases','place_id','Date','Fatalities','day']
df_sub = df_preds_pri[cols]
df_sub['ConfirmedCases'] = round(df_sub['ConfirmedCases'])
df_sub['Fatalities'] = round(df_sub['Fatalities'])
df_sub.to_csv("predictions.csv",index = None)