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
day_before_valid = 99-7 # 4-8 day  before of validation

day_before_public = 99 # 4-15, the day before public LB period

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
df_tmp = df_traintest[['place_id']][df_traintest['place_id'].duplicated()==False]

df_tmp = pd.merge(df_tmp, df_latlong[['place_id', 'Lat', 'Long']], on='place_id', how='left')

tmp = df_latlong[['Lat', 'Long']].values[df_latlong['place_id']=='United Kingdom/United Kingdom']

df_tmp.loc[df_tmp['place_id']=='United Kingdom', ['Lat', 'Long']] = tmp

tmp = df_latlong[['Lat', 'Long']].values[df_latlong['place_id']=='Cruise Ship/Diamond Princess']

df_tmp.loc[df_tmp['place_id']=='Diamond Princess', ['Lat', 'Long']] = tmp

tmp = df_latlong[['Lat', 'Long']].values[df_latlong['place_id']=='Denmark/Denmark']

df_tmp.loc[df_tmp['place_id']=='Denmark', ['Lat', 'Long']] = tmp

tmp = df_latlong[['Lat', 'Long']].values[df_latlong['place_id']=='France/France']

df_tmp.loc[df_tmp['place_id']=='France', ['Lat', 'Long']] = tmp

tmp = df_latlong[['Lat', 'Long']].values[df_latlong['place_id']=='Gambia, The']

df_tmp.loc[df_tmp['place_id']=='Gambia', ['Lat', 'Long']] = tmp

tmp = df_latlong[['Lat', 'Long']].values[df_latlong['place_id']=='Netherlands/Netherlands']

df_tmp.loc[df_tmp['place_id']=='Netherlands', ['Lat', 'Long']] = tmp

df_tmp.loc[df_tmp['place_id']=='Dominica', ['Lat', 'Long']] = (15.3, -61.383333)

df_tmp.loc[df_tmp['place_id']=='Angola', ['Lat', 'Long']] = (-8.830833, 13.245)

df_tmp.loc[df_tmp['place_id']=='Bahamas', ['Lat', 'Long']] = (25.066667, -77.333333)

df_tmp.loc[df_tmp['place_id']=='Belize', ['Lat', 'Long']] = (17.498611, -88.188611)

df_tmp.loc[df_tmp['place_id']=='Cabo Verde', ['Lat', 'Long']] = (14.916667, -23.516667)

df_tmp.loc[df_tmp['place_id']=='Chad', ['Lat', 'Long']] = (12.134722, 15.055833)

df_tmp.loc[df_tmp['place_id']=='Denmark/Greenland', ['Lat', 'Long']] = (64.181389, -51.694167)

df_tmp.loc[df_tmp['place_id']=='El Salvador', ['Lat', 'Long']] = (13.698889, -89.191389)

df_tmp.loc[df_tmp['place_id']=='Eritrea', ['Lat', 'Long']] = (15.322778, 38.925)

df_tmp.loc[df_tmp['place_id']=='Fiji', ['Lat', 'Long']] = (-18.166667, 178.45)

df_tmp.loc[df_tmp['place_id']=='France/Martinique', ['Lat', 'Long']] = (14.666667, -61)

df_tmp.loc[df_tmp['place_id']=='France/New Caledonia', ['Lat', 'Long']] = (-22.2758, 166.458)

df_tmp.loc[df_tmp['place_id']=='Grenada', ['Lat', 'Long']] = (12.05, -61.75)

df_tmp.loc[df_tmp['place_id']=='Guinea-Bissau', ['Lat', 'Long']] = (11.85, -15.566667)

df_tmp.loc[df_tmp['place_id']=='Haiti', ['Lat', 'Long']] = (18.533333, -72.333333)

df_tmp.loc[df_tmp['place_id']=='Laos', ['Lat', 'Long']] = (17.966667, 102.6)

df_tmp.loc[df_tmp['place_id']=='Libya', ['Lat', 'Long']] = (32.887222, 13.191389)

df_tmp.loc[df_tmp['place_id']=='Madagascar', ['Lat', 'Long']] = (-18.933333, 47.516667)

df_tmp.loc[df_tmp['place_id']=='Mali', ['Lat', 'Long']] = (12.639167, -8.002778)

df_tmp.loc[df_tmp['place_id']=='Mozambique', ['Lat', 'Long']] = (-25.966667, 32.583333)

df_tmp.loc[df_tmp['place_id']=='Netherlands/Sint Maarten', ['Lat', 'Long']] = (18.052778, -63.0425)

df_tmp.loc[df_tmp['place_id']=='Nicaragua', ['Lat', 'Long']] = (12.136389, -86.251389)

df_tmp.loc[df_tmp['place_id']=='Niger', ['Lat', 'Long']] = (13.511667, 2.125278)

df_tmp.loc[df_tmp['place_id']=='Papua New Guinea', ['Lat', 'Long']] = (-9.478889, 147.149444)

df_tmp.loc[df_tmp['place_id']=='Saint Kitts and Nevis', ['Lat', 'Long']] = (17.3, -62.733333)

df_tmp.loc[df_tmp['place_id']=='Syria', ['Lat', 'Long']] = (33.513056, 36.291944)

df_tmp.loc[df_tmp['place_id']=='Timor-Leste', ['Lat', 'Long']] = (-8.566667, 125.566667)

df_tmp.loc[df_tmp['place_id']=='Uganda', ['Lat', 'Long']] = (0.313611, 32.581111)

df_tmp.loc[df_tmp['place_id']=='Zimbabwe', ['Lat', 'Long']] = (-17.829167, 31.052222)

df_tmp.loc[df_tmp['place_id']=='United Kingdom/Bermuda', ['Lat', 'Long']] = (32.293, -64.782)

df_tmp.loc[df_tmp['place_id']=='United Kingdom/Isle of Man', ['Lat', 'Long']] = (54.145, -4.482)



df_tmp.loc[df_tmp['place_id']=='Botswana', ['Lat', 'Long']] = (-24.658056, 25.912222)

df_tmp.loc[df_tmp['place_id']=='Burma', ['Lat', 'Long']] = (16.85, 96.183333)

df_tmp.loc[df_tmp['place_id']=='Burundi', ['Lat', 'Long']] = (-3.383333, 29.366667)

df_tmp.loc[df_tmp['place_id']=='Canada/Northwest Territories', ['Lat', 'Long']] = (62.442222, -114.394722)

df_tmp.loc[df_tmp['place_id']=='Canada/Yukon', ['Lat', 'Long']] = (60.716667, -135.05)

df_tmp.loc[df_tmp['place_id']=='Kosovo', ['Lat', 'Long']] = (42.666667, 21.166667)

df_tmp.loc[df_tmp['place_id']=='MS Zaandam', ['Lat', 'Long']] = (26.086111, -80.115278) # Uncertain

df_tmp.loc[df_tmp['place_id']=='Sierra Leone', ['Lat', 'Long']] = (8.484444, -13.234444)

df_tmp.loc[df_tmp['place_id']=='United Kingdom/Anguilla', ['Lat', 'Long']] = (18.220833, -63.051667)

df_tmp.loc[df_tmp['place_id']=='United Kingdom/British Virgin Islands', ['Lat', 'Long']] = (18.431389, -64.623056)

df_tmp.loc[df_tmp['place_id']=='United Kingdom/Turks and Caicos Islands', ['Lat', 'Long']] = (21.783333, -72.283333)

df_tmp.loc[df_tmp['place_id']=='West Bank and Gaza', ['Lat', 'Long']] = (31.703056, 35.195556)

df_tmp[df_tmp['place_id']=='West Bank and Gaza'][['Lat', 'Long']] 
df_traintest = pd.merge(df_traintest, df_tmp[['place_id', 'Lat', 'Long']], on='place_id', how='left')

df_traintest.head()
# count the places with no Lat and Long.

tmp = np.sort(df_traintest['place_id'][pd.isna(df_traintest['Lat'])].unique())

print(len(tmp)) # count Nan

print(tmp)
# get place list

places = np.sort(df_traintest['place_id'].unique())

for place in places:

    print(place)

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

    

    TARGETS = ["ConfirmedCases", "Fatalities"]

    for col in TARGETS:

        df["prev_{}".format(col)] = df[col].shift()

        df["prev10_{}".format(col)] = df[col].shift(10)

        df["delta10_{}".format(col)] = df["prev10_{}".format(col)]  - df[col].shift(9)

        df[f'delta_{col}'] =  df[col]-df[f'prev_{col}']

        df[f'rolling_{col}'] = df[f'delta_{col}'].rolling(window=7).mean()

        df[f'rolling3_{col}'] = df[f'delta_{col}'].rolling(window=3).mean()

        df[f'div_rolling_{col}'] =  (df[f'rolling_{col}']-df[f'rolling_{col}'].shift())/df[f'rolling_{col}'].shift()

        df[f'div_rolling3_{col}'] =  (df[f'rolling3_{col}']-df[f'rolling3_{col}'].shift())/df[f'rolling3_{col}'].shift()

    df['div_rolling_ConfirmedCases']=df['div_rolling_ConfirmedCases'].replace([np.inf, -np.inf], np.nan).fillna(0)

    df['div_rolling_Fatalities']=df['div_rolling_Fatalities'].replace([np.inf, -np.inf], np.nan).fillna(0) 

    df['div_rolling3_ConfirmedCases']=df['div_rolling_ConfirmedCases'].replace([np.inf, -np.inf], np.nan).fillna(0)

    df['div_rolling3_Fatalities']=df['div_rolling_Fatalities'].replace([np.inf, -np.inf], np.nan).fillna(0) 

    df['prev10_ConfirmedCases']=df['div_rolling_ConfirmedCases'].replace([np.inf, -np.inf], np.nan).fillna(-999)

    df['prev10_Fatalities']=df['div_rolling_Fatalities'].replace([np.inf, -np.inf], np.nan).fillna(-999) 

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

df_country = pd.read_csv("../input/covid19-demographic-predictors/covid19_by_country.csv")

df_country.head()
df_country['Country_Region'] = df_country['Country']

df_country = df_country[df_country['Country'].duplicated()==False]
print(df_country[df_country['Country'].duplicated()].shape)
df_country[df_country['Country'].duplicated()]
df_traintest7 = pd.merge(df_traintest6, 

                         df_country.drop(['Tests', 'Test Pop', 'Country'], axis=1), 

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

    'Current_account_balance'

]

for col in cols:

    df_traintest8[col] = df_traintest8[col].apply(lambda x: func(x))  

#print(df_traintest8['pop'].dtype)


df_traintest9 = df_traintest8
for col in df_traintest9.columns:

    print("'{}',".format(col))
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

    'Density', 

    'prev_ConfirmedCases',

'delta_ConfirmedCases',

'rolling_ConfirmedCases',

'div_rolling_ConfirmedCases',

'prev_Fatalities',

'delta_Fatalities',

'rolling_Fatalities',

'div_rolling_Fatalities',

    'div_rolling3_ConfirmedCases',

    'div_rolling3_Fatalities',

    'prev10_ConfirmedCases',

    'delta10_ConfirmedCases',

     'prev10_Fatalities',

    'delta10_Fatalities'

#     'medianage', 

#     'urbanpop', 

#     'hospibed', 'smokers', 

 #   'temp_mean',

 #   'min_mean',

#     'max_mean',

  #  'stp_mean',

  #  'wdsp_mean',

#     'prcp_mean',

#     'fog_mean',

]

col_cat = []

df_train = df_traintest9[(pd.isna(df_traintest9['ForecastId'])) & (df_traintest9['day']<=day_before_valid) & (df_traintest9['prev10_ConfirmedCases']>-1)]

df_valid = df_traintest9[(pd.isna(df_traintest9['ForecastId'])) & (day_before_valid<df_traintest9['day']) & (df_traintest9['day']<=day_before_public) & (df_traintest9['prev10_ConfirmedCases']>-1)]

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
#df_train['Density'].unique()
y_true = df_valid['fatal/day'].values

y_pred = np.exp(model.predict(X_valid))-1

score = calc_score(y_true, y_pred)

print("{:.6f}".format(score))
"""

0.411872 baseline

0.409379 +all weather

0.413820 + temp

0.413059 + temp+min

0.414248+ temp+min+max

0.410319 + temp+min+stp

0.404198 + temp+min+stp+wdsp

0.404198 + temp+min+stp+wdsp+prcp

0.404198+ temp+min+stp+wdsp+prcp+fog

"""
# display feature importance

tmp = pd.DataFrame()

tmp["feature"] = col_var

tmp["importance"] = model.feature_importance()

tmp = tmp.sort_values('importance', ascending=False)

tmp
# train with all data before public

df_train = df_traintest9[(pd.isna(df_traintest9['ForecastId'])) & (df_traintest9['day']<=day_before_valid) & (df_traintest9['prev10_ConfirmedCases']>-1)]

df_valid = df_traintest9[(pd.isna(df_traintest9['ForecastId'])) & (day_before_valid<df_traintest9['day']) & (df_traintest9['day']<=day_before_public) & (df_traintest9['prev10_ConfirmedCases']>-1)]

X_train = df_train[col_var]

X_valid = df_valid[col_var]

y_train = np.log(df_train[col_target].values.clip(0, 1e10)+1)

y_valid = np.log(df_valid[col_target].values.clip(0, 1e10)+1)

train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=col_cat)

valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature=col_cat)

model_pub = lgb.train(params, train_data, best_itr, valid_sets=[train_data, valid_data],

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

#        'temp_mean',

#    'min_mean',

 #   'max_mean',

 #   'stp_mean',

 #   'wdsp_mean',

#     'prcp_mean',

#     'fog_mean',

    'Density', 

    'prev_ConfirmedCases',

'delta_ConfirmedCases',

'rolling_ConfirmedCases',

'div_rolling_ConfirmedCases',

'prev_Fatalities',

'delta_Fatalities',

'rolling_Fatalities',

'div_rolling_Fatalities',

    'div_rolling3_ConfirmedCases',

    'div_rolling3_Fatalities',

      'prev10_ConfirmedCases',

    'delta10_ConfirmedCases',

     'prev10_Fatalities',

    'delta10_Fatalities'

]

col_cat = []

df_train = df_traintest9[(pd.isna(df_traintest9['ForecastId'])) & (df_traintest9['day']<=day_before_valid) & (df_traintest9['prev10_ConfirmedCases']>-1)]

df_valid = df_traintest9[(pd.isna(df_traintest9['ForecastId'])) & (day_before_valid<df_traintest9['day']) & (df_traintest9['day']<=day_before_public) & (df_traintest9['prev10_ConfirmedCases']>-1)]

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

"""

0.872413 baseline

0.871128 + tmp+min+stp+wdsp

0.866764 + tmp+min+stp+wdsp+max

"""
# display feature importance

tmp = pd.DataFrame()

tmp["feature"] = col_var2

tmp["importance"] = model2.feature_importance()

tmp = tmp.sort_values('importance', ascending=False)

tmp
df_train = df_traintest9[(pd.isna(df_traintest9['ForecastId'])) & (df_traintest9['day']<=day_before_valid) & (df_traintest9['prev10_ConfirmedCases']>-1)]

df_valid = df_traintest9[(pd.isna(df_traintest9['ForecastId'])) & (day_before_valid<df_traintest9['day']) & (df_traintest9['day']<=day_before_public) & (df_traintest9['prev10_ConfirmedCases']>-1)]

X_train = df_train[col_var2]

X_valid = df_valid[col_var2]

y_train = np.log(df_train[col_target2].values.clip(0, 1e10)+1)

y_valid = np.log(df_valid[col_target2].values.clip(0, 1e10)+1)

train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=col_cat)

valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature=col_cat)

model2_pub = lgb.train(params, train_data, best_itr2, valid_sets=[train_data, valid_data],

                  verbose_eval=100,

                  early_stopping_rounds=150,)
# train model to predict fatalities/day

df_train = df_traintest9[(pd.isna(df_traintest9['ForecastId'])) & (df_traintest9['day']<=day_before_public) & (df_traintest9['prev10_ConfirmedCases']>-1)]

df_valid = df_traintest9[(pd.isna(df_traintest9['ForecastId'])) & (day_before_public<df_traintest9['day']) & (df_traintest9['prev10_ConfirmedCases']>-1)]

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

df_train = df_traintest9[(pd.isna(df_traintest9['ForecastId'])) & (df_traintest9['prev10_ConfirmedCases']>-1)]

df_valid = df_traintest9[(pd.isna(df_traintest9['ForecastId'])) & (df_traintest9['prev10_ConfirmedCases']>-1)]

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

df_train = df_traintest9[(pd.isna(df_traintest9['ForecastId'])) & (df_traintest9['day']<=day_before_public) & (df_traintest9['prev10_ConfirmedCases']>-1)]

df_valid = df_traintest9[(pd.isna(df_traintest9['ForecastId'])) & (day_before_public<df_traintest9['day']) & (df_traintest9['prev10_ConfirmedCases']>-1)]

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

df_train = df_traintest9[(pd.isna(df_traintest9['ForecastId'])) & (df_traintest9['prev10_ConfirmedCases']>-1)]

df_valid = df_traintest9[(pd.isna(df_traintest9['ForecastId'])) & (df_traintest9['prev10_ConfirmedCases']>-1)]

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

df_tmp = df_traintest9[

    ((df_traintest9['day']<=day_before_public)  & (pd.isna(df_traintest9['ForecastId'])) )

    | ((day_before_public<df_traintest9['day']) & (pd.isna(df_traintest9['ForecastId'])==False))].reset_index(drop=True)

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

df_traintest10[df_traintest10['day']>day_before_public-2].head()
# remove overlap for private LB prediction

df_tmp = df_traintest9[

    ((df_traintest9['day']<=day_before_private)  & (pd.isna(df_traintest9['ForecastId'])))

    | ((day_before_private<df_traintest9['day']) & (pd.isna(df_traintest9['ForecastId'])==False))].reset_index(drop=True)

df_tmp = df_tmp.drop([

    'cases/day_(1-1)', 'cases/day_(1-7)', 'cases/day_(8-14)', 'cases/day_(15-21)', 

    'fatal/day_(1-1)', 'fatal/day_(1-7)', 'fatal/day_(8-14)', 'fatal/day_(15-21)',

    'days_since_1cases', 'days_since_10cases', 'days_since_100cases',

    'days_since_1fatal', 'days_since_10fatal', 'days_since_100fatal',

                               ],  axis=1)

df_traintest11 = []

for i, place in enumerate(places[:]):

    df_tmp2 = df_tmp[df_tmp['place_id']==place].reset_index(drop=True)

    df_tmp2 = do_aggregations(df_tmp2)

    df_traintest11.append(df_tmp2)

df_traintest11 = pd.concat(df_traintest11).reset_index(drop=True)

df_traintest11[df_traintest11['day']>day_before_private-2].head()
# predict test data in public

# predict the cases and fatatilites one day at a time and use the predicts as next day's feature recursively.

df_preds = []

for i, place in enumerate(places[:]):

    df_interest = copy.deepcopy(df_traintest10[df_traintest10['place_id']==place].reset_index(drop=True))

    df_interest['cases/day'][(pd.isna(df_interest['ForecastId']))==False] = -1

    df_interest['fatal/day'][(pd.isna(df_interest['ForecastId']))==False] = -1

    len_known = (df_interest['day']<=day_before_public).sum()

    len_unknown = (day_before_public<df_interest['day']).sum()

    for j in range(len_unknown): # use predicted cases and fatal for next days' prediction

        X_valid = df_interest[col_var].iloc[j+len_known]

        X_valid2 = df_interest[col_var2].iloc[j+len_known]

        pred_f = model_pub.predict(X_valid)

        pred_c = model2_pub.predict(X_valid2)

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

    df_interest = copy.deepcopy(df_traintest11[df_traintest11['place_id']==place].reset_index(drop=True))

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

df_sub.to_csv("submission.csv", index=None)

df_sub.head(10)