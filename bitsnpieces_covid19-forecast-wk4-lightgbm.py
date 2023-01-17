# https://www.kaggle.com/vinhnguyen/gpu-acceleration-for-lightgbm/notebook

# !rm -r /opt/conda/lib/python3.6/site-packages/lightgbm

# !git clone --recursive https://github.com/Microsoft/LightGBM

# !apt-get install -y -qq libboost-all-dev



# %%bash

# cd LightGBM

# rm -r build

# mkdir build

# cd build

# cmake -DUSE_GPU=1 -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/ ..

# make -j$(nproc)



# !cd LightGBM/python-package/;python3 setup.py install --precompile



# !mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# !rm -r LightGBM
# https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html





# DEBUG = True

DEBUG = False



# shift params

# window = 3

MAX_LAG = 3

shift_offset = 0

halflife = 6



# train params

# log_transform = True

log_transform = False

SEED = 42

impute = True

num_round = 500

# num_round = 1000

# num_round = 2000

# num_round = 15000

# early_stopping_rounds = 300   # the smaller the less chances of overfit

# early_stopping_rounds = 500

early_stopping_rounds = 1000

# early_stopping_rounds = 2000



params = {

          'num_leaves': 10,  # 8,  default 31

          'min_data_in_leaf': 10, #5,  # 42,  default 20

          'objective': 'regression',

          'max_depth': 8, #8,  # default no limit -1

          'max_bin': 50, # default 255

          'learning_rate': 0.02,

#            'device_type': 'gpu',

          'boosting': 'gbdt',   #  traditional Gradient Boosting Decision Tree, 

#             'boosting': 'dart',   # , Dropouts meet Multiple Additive Regression Trees

          'bagging_freq': 5,  # 5

          'bagging_fraction': 0.8,  # 0.5,

          'feature_fraction': 0.8201,

          'bagging_seed': SEED,

#           'reg_alpha': 1,  # 1.728910519108444,

#           'reg_lambda': 4.9847051755586085,

          'random_state': SEED,

#           'metric': 'mse',

            'metric': {'l2','l1'},

          'verbosity': 100,

#           'min_gain_to_split': 0.02,  # 0.01077313523861969,

#           'min_child_weight': 5,  # 19.428902804238373,

#           'num_threads': 4,

#             'extra_trees': True,

          }



col_cat = []

f_list = ['DailyFatalities_%d' % d for d in range(1,MAX_LAG+1)]

c_list = ['DailyConfirmedCases_%d' % d for d in range(1,MAX_LAG+1)]

col_var = f_list + c_list + [

    'FirstFatalitiesDays','FirstConfirmedCasesDays',

    'Days',

#     'Month',

    

    'AirportRestrictionDays',

    

    'first_school_closure_days', 'MeasureImplementDays',

    

    'DailyFCRatio_1',

    'ConfirmedLessFatalities_1',

    

#     'DailyFatalities_1', 'DailyFatalities_2', 'DailyFatalities_3', 'DailyFatalities_4', 'DailyFatalities_5', 'DailyFatalities_6',

#     'DailyConfirmedCases_1', 'DailyConfirmedCases_2', 'DailyConfirmedCases_3', 'DailyConfirmedCases_4', 'DailyConfirmedCases_5', 'DailyConfirmedCases_6',

    

#     'DailyFatalitiesSlope_1_2','DailyFatalitiesSlope_2_3',           # difference eventually becomes really small and ends up becoming zero

#     'DailyConfirmedCasesSlope_1_2','DailyConfirmedCasesSlope_2_3',

    

#     'latitude', 'longitude',

#     'CountryCode', 

#     'RegionCode',

    

#     'age_over_65_years_percent',

#     'Median_age', 

#     'sex_male_to_female_over_65',

    

#     'Population_2020',

    

#     'Density_KM2m',

    

    'Precip', 'Temp',

    

#     'Flu_pneumonia_death_rate_per_100000',

    

#     'A1a', 'A2', 'A2a', 'A3', 'A6', 'A7', 'B', 'B1', 'B2', 'B4',  # strain clades

    

#     'ICU-CCB_beds_per_100000', 

#     'NursesPer1000', 'DrPer1000',

]

# !pip install --upgrade --force-reinstall lightgbm
# imports

import plotly.express as px

import plotly.graph_objects as go

from scipy import stats



import matplotlib.pyplot as plt

from scipy import interpolate

import json

import requests

import io

from tqdm import tqdm



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from scipy.stats import lognorm

from scipy.optimize import curve_fit

import string

from scipy.integrate import quad



from sklearn import metrics

from sklearn import preprocessing

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from sklearn.linear_model import SGDRegressor, LinearRegression, Lasso, Ridge, LogisticRegression

from sklearn.base import clone

from sklearn.pipeline import Pipeline, make_pipeline



# https://images.plot.ly/plotly-documentation/images/python_cheat_sheet.pdf

# https://www.apsnet.org/edcenter/disimpactmngmnt/topc/EpidemiologyTemporal/Pages/ModellingProgress.aspx



# https://www.kaggle.com/c/ashrae-energy-prediction/discussion/114614

# !pip install --upgrade numpy==1.17.3

import numpy as np



# !pip install --upgrade --force-reinstall lightgbm -DUSE_GPU=1

#https://lightgbm.readthedocs.io/en/latest/Python-Intro.html

# https://www.kaggle.com/osciiart/covid-19-lightgbm-with-weather-2/data#Model-training

import lightgbm as lgb  # LightGBM is a gradient boosting framework that uses tree based learning algorithms. 



pd.set_option('display.max_columns', 100)

pd.options.display.float_format = '{:.4f}'.format

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
def country_slice(df, country='China', province=''):

    if province is None or pd.isna(province):

        return df[(df['Country_Region']==country) & (pd.isna(df['Province_State']) == True) ]

    else:

        return df[(df['Country_Region']==country) & (df['Province_State']==province)]



def preprocess_df(df, index_date=pd.to_datetime('2020-01-22')):

    try:

        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

    except:

        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')

    

#     df['ConfirmedCases'] = df['ConfirmedCases'].astype({'ConfirmedCases': 'float32'})

#     df['Fatalities'] = df['Fatalities'].astype({'Fatalities': 'float32'})

        

    df['Days'] = (df['Date'] - index_date).dt.days

    df = df.sort_values(by=['Country_Region','Province_State','Date'], ascending=True)

    df = df.rename(columns={'Country/Region':'Country_Region', 'Province/State':'Province_State'})

    df['Province_State'] = df['Province_State'].apply(lambda x: '' if pd.isna(x) else x)

    if 'ConfirmedCases' in df:

        df['DailyConfirmedCases'] = df['ConfirmedCases'].diff()

        df['DailyConfirmedCases'] = df['DailyConfirmedCases'].clip(0).fillna(0)

    if 'Fatalities' in df:

        df['DailyFatalities'] = df['Fatalities'].diff()

        df['DailyFatalities'] = df['DailyFatalities'].clip(0).fillna(0)

    if 'Recovered' in df:

        df['DailyRecovered'] = df['Recovered'].diff()

        df['DailyRecovered'] = df['DailyRecovered'].clip(0).fillna(0)

    

    df['Country_Province'] = df['Country_Region'] + '/' + df['Province_State']

    

    le = preprocessing.LabelEncoder()

    le.fit(df['Country_Region'])

    df['CountryCode'] = le.transform(df['Country_Region'])

    

    return df



train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')

submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/submission.csv')

train = preprocess_df(train)

test = preprocess_df(test)



# print(list(zip(range(84),train['Date'].drop_duplicates().strftime('%Y-%m-%d').values)))



# TODO

# test = test.query('Date < "2020-04-20"')
# sel_countries = [c for c in countries_lots_cases.tolist() if c != 'China/Hubei']

# sel_countries = ["Italy/","Spain/","US/New York","Germany/","China/Hubei","France/","Iran/","United Kingdom/","US/New Jersey","Switzerland/",

#                  "Turkey/","Belgium/","Netherlands/","Austria/","Korea, South/","US/California","US/Michigan","Portugal/",

#                  "US/Massachusetts","US/Illinois","US/Florida","Brazil/","US/Louisiana","Israel/","US/Pennsylvania","US/Washington",

#                  "Sweden/","Norway/","US/Georgia","Canada/Quebec"]   # countries with lots of cases, model separately

sel_countries = ["Italy/","Spain/","US/New York","Germany/","France/","Iran/","United Kingdom/"]

if DEBUG:

    train = train[train['Country_Province'].isin(sel_countries)]

    test = test[test['Country_Province'].isin(sel_countries)]

    print(train.shape, test.shape)


# group countries by number of cases

max_cases = train.groupby(['Country_Province']).agg({'ConfirmedCases':'max'}).reset_index().sort_values(by='ConfirmedCases')

max_50, min_90 = max_cases['ConfirmedCases'].quantile([.5, .9])





# big countries, model separtely





all_countries   = train['Country_Province'].drop_duplicates().values

big_countries   = all_countries

# big_countries   = max_cases.query(f'ConfirmedCases > {min_90}')['Country_Province'].drop_duplicates().values

# small_countries = max_cases.query(f'ConfirmedCases < {min_90}')['Country_Province'].drop_duplicates().values

# small_countries = max_cases.query(f'ConfirmedCases < {max_50}')['Country_Province'].drop_duplicates().values

# med_countries   = set(train['Country_Province'].values).difference(set(big_countries).union(set(small_countries)))

print(f'\n{len(big_countries)} big_countries {big_countries}')

# print(f'\n{len(med_countries)} med_countries {med_countries}')

# print(f'\n{len(small_countries)} small_countries {small_countries}')



# fetch data covid

# note some Countries have no province data like Canada!

# import requests

# import io



# def get_df_from_url(url):

#     s = requests.get(url).content

#     return pd.read_csv(io.StringIO(s.decode('utf-8')))



# covid_url_prefix = 'https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/'

# # df_covid_confirmed = get_df_from_url(covid_url_prefix + 'time_series_covid19_confirmed_global.csv')

# # df_covid_deaths = get_df_from_url(covid_url_prefix + 'time_series_covid19_deaths_global.csv')

# df_covid_recovered = get_df_from_url(covid_url_prefix + 'time_series_covid19_recovered_global.csv')



# df_covid_recovered.info()



# df_covid_recovered = df_covid_recovered.rename(columns={'Province/State':'Province_State', 'Country/Region':'Country_Region'})

# df_covid_recovered

# df_covid_recovered.columns

# start_date_index = list(df_covid_recovered.columns).index('1/22/20')

# value_vars = df_covid_recovered.columns[start_date_index:].values

# df_recovered = pd.melt(df_covid_recovered, id_vars="Province_State	Country_Region	Lat	Long".split('\t'), value_vars=value_vars)

# df_recovered = df_recovered.rename(columns={'variable':'Date', 'value':'Recovered'})

# df_recovered['Date'] = pd.to_datetime(df_recovered['Date'], format='%m/%d/%y')

# df_recovered = preprocess_df(df_recovered)



# # add Recovered info to training data

# train = pd.merge(train, df_recovered, on=['Province_State','Country_Region','Date','Days'], how='left').reset_index()

# del train['index']

# test = pd.merge(test, df_recovered, on=['Province_State','Country_Region','Date','Days'], how='left').reset_index()

# del test['index']

# first confirmed case / fatality

out = []

min_dates = dict()

for metric in ['Fatalities','ConfirmedCases']:

    tmp = train.query(metric + '>0').groupby(['Country_Region','Province_State']).agg({'Date':'min'}).reset_index()

    tmp = tmp.rename(columns={'Date':'First'+metric+'Date'})

#     print(tmp)

    train = pd.merge(train, tmp, how='left')

    test = pd.merge(test, tmp, how='left')

    train['First'+metric+'Days'] = 0

    test['First'+metric+'Days'] = 0

    train['First'+metric+'Days'] = (train['Date'] - train['First'+metric+'Date']).dt.days

    test['First'+metric+'Days'] = (test['Date'] - test['First'+metric+'Date']).dt.days



# fatality / confirmed ratio

train['DailyFCRatio'] = train['DailyFatalities'] / (train['DailyConfirmedCases'] + 0.00000000001)

test['DailyFCRatio'] = 0



# month

train['Month'] = train['Date'].dt.month

test['Month'] = test['Date'].dt.month
# time lag, t-1, t-2



out = []

for key, tmp in list(train.groupby(['Country_Region','Province_State'])):

    country, province = key[0], key[1]

    

#     tmp['AvgConfirmedCases'] = tmp['DailyConfirmedCases'].rolling(window=window).mean()

#     tmp['AvgFatalities']     = tmp['DailyFatalities'].rolling(window=window).mean()

    tmp['AvgConfirmedCases'] = tmp['DailyConfirmedCases'].ewm(halflife=halflife).mean()  # weighted rolling window with half-life decay

    tmp['AvgFatalities']     = tmp['DailyFatalities'].ewm(halflife=halflife).mean()

    tmp['ConfirmedLessFatalities'] = (tmp['DailyConfirmedCases'] - tmp['DailyFatalities'])/(tmp['DailyConfirmedCases'] + 0.000000000001)

#     https://pandas.pydata.org/pandas-docs/stable/user_guide/computation.html#exponentially-weighted-windows



#     for j in range(1, MAX_LAG+1):

#         tmp['DailyConfirmedCases_'+str(j)] = tmp['DailyConfirmedCases'].shift(j, fill_value=0)

#         tmp['DailyFatalities_'+str(j)] = tmp['DailyFatalities'].shift(j, fill_value=0)

    

    for j in range(1, MAX_LAG+1):

#         tmp['DailyConfirmedCases_'+str(j)] = tmp['AvgConfirmedCases'].shift(j, fill_value=0)

#         tmp['DailyFatalities_'+str(j)] = tmp['AvgFatalities'].shift(j, fill_value=0)

        tmp['DailyConfirmedCases_'+str(j)] = tmp['AvgConfirmedCases'].shift(j+shift_offset, fill_value=0)

        tmp['DailyFatalities_'+str(j)] = tmp['AvgFatalities'].shift(j+shift_offset, fill_value=0)

    

    tmp['DailyFCRatio_1'] = tmp['DailyFCRatio'].shift(1, fill_value=0)

    tmp['ConfirmedLessFatalities_1'] = tmp['ConfirmedLessFatalities'].shift(1, fill_value=0)

  

    out.append(tmp)

    

train = pd.concat(out)

tmp = country_slice(train,'Italy')

tmp[['Country_Region','Province_State', 'Date', 'Fatalities', 'DailyFatalities','DailyFatalities_1','DailyFatalities_2']].tail(10)



test['DailyFatalities_1'] = 0  # t-1 lag

test['DailyFatalities_2'] = 0  # t-2 lag

test['DailyFatalities_3'] = 0  # t-3 lag

test['DailyFatalities_4'] = 0  # t-4 lag

test['DailyFatalities_5'] = 0  # t-5 lag

test['DailyFatalities_6'] = 0  # t-6 lag

test['DailyConfirmedCases_1'] = 0  # t-1 lag

test['DailyConfirmedCases_2'] = 0  # t-2 lag

test['DailyConfirmedCases_3'] = 0  # t-3 lag

test['DailyConfirmedCases_4'] = 0  # t-4 lag

test['DailyConfirmedCases_5'] = 0  # t-5 lag

test['DailyConfirmedCases_6'] = 0  # t-6 lag

test['DailyFCRatio_1'] = 0

test['ConfirmedLessFatalities_1'] = 0

min_test_date = min(test['Date'])

for i, row in list(test.iterrows()):

    country, province, test_date = row['Country_Region'], row['Province_State'], row['Date']

    if test_date == min_test_date:

        test.loc[i,'DailyFCRatio_1'] = train.query(f'Country_Region=="{country}" & Date=="{test_date}"')['DailyFCRatio_1'].values[0]

        test.loc[i,'ConfirmedLessFatalities_1'] = train.query(f'Country_Region=="{country}" & Date=="{test_date}"')['ConfirmedLessFatalities_1'].values[0]

        

        for j in range(1, MAX_LAG+1):

            test.loc[i,'DailyConfirmedCases_%d' % j] = train.query(f'Country_Region=="{country}" & Date=="{test_date}"')['DailyConfirmedCases_%d' % j].values[0]

            test.loc[i,'DailyFatalities_%d' % j] = train.query(f'Country_Region=="{country}" & Date=="{test_date}"')['DailyFatalities_%d' % j].values[0]

    elif test_date == min_test_date + pd.DateOffset(1):

        for j in range(2, MAX_LAG+1):

            test.loc[i,'DailyConfirmedCases_%d' % j] = train.query(f'Country_Region=="{country}" & Date=="{test_date}"')['DailyConfirmedCases_%d' % j].values[0]

            test.loc[i,'DailyFatalities_%d' % j] = train.query(f'Country_Region=="{country}" & Date=="{test_date}"')['DailyFatalities_%d' % j].values[0]

    elif test_date == min_test_date + pd.DateOffset(2):

        for j in range(3, MAX_LAG+1):

            test.loc[i,'DailyConfirmedCases_%d' % j] = train.query(f'Country_Region=="{country}" & Date=="{test_date}"')['DailyConfirmedCases_%d' % j].values[0]

            test.loc[i,'DailyFatalities_%d' % j] = train.query(f'Country_Region=="{country}" & Date=="{test_date}"')['DailyFatalities_%d' % j].values[0]

    elif test_date == min_test_date + pd.DateOffset(3):

        for j in range(4, MAX_LAG+1):

            test.loc[i,'DailyConfirmedCases_%d' % j] = train.query(f'Country_Region=="{country}" & Date=="{test_date}"')['DailyConfirmedCases_%d' % j].values[0]

            test.loc[i,'DailyFatalities_%d' % j] = train.query(f'Country_Region=="{country}" & Date=="{test_date}"')['DailyFatalities_%d' % j].values[0]

    elif test_date == min_test_date + pd.DateOffset(4):

        for j in range(5, MAX_LAG+1):

            test.loc[i,'DailyConfirmedCases_%d' % j] = train.query(f'Country_Region=="{country}" & Date=="{test_date}"')['DailyConfirmedCases_%d' % j].values[0]

            test.loc[i,'DailyFatalities_%d' % j] = train.query(f'Country_Region=="{country}" & Date=="{test_date}"')['DailyFatalities_%d' % j].values[0]

    elif test_date == min_test_date + pd.DateOffset(5):

        for j in range(6, MAX_LAG+1):

            test.loc[i,'DailyConfirmedCases_%d' % j] = train.query(f'Country_Region=="{country}" & Date=="{test_date}"')['DailyConfirmedCases_%d' % j].values[0]

            test.loc[i,'DailyFatalities_%d' % j] = train.query(f'Country_Region=="{country}" & Date=="{test_date}"')['DailyFatalities_%d' % j].values[0]



for metric in ['ConfirmedCases','Fatalities']:

    train[f'Daily{metric}Slope_1_2'] = (train[f'Daily{metric}_1']-train[f'Daily{metric}_2'])/2

    train[f'Daily{metric}Slope_2_3'] = (train[f'Daily{metric}_2']-train[f'Daily{metric}_3'])/2

    test[f'Daily{metric}Slope_1_2'] = (test[f'Daily{metric}_1']-test[f'Daily{metric}_2'])/2

    test[f'Daily{metric}Slope_2_3'] = (test[f'Daily{metric}_2']-test[f'Daily{metric}_3'])/2

    
df_merged = pd.read_csv('/kaggle/input/covid19-country-data/covid19_merged.csv')

del df_merged['Unnamed: 0']

df_merged['country'] = df_merged['country'].apply(lambda x:x.replace('South Korea','Korea, South').replace('Taiwan','Taiwan*').replace('United States','US'))

print('in merged but not train', set(df_merged['country'].values).difference(set(train['Country_Region'].values)))

print('in train but not merged', set(train['Country_Region'].values).difference(set(df_merged['country'].values)))

country_with_missing_values = list(set(train['Country_Region'].values).difference(set(df_merged['country'].values)))

country_with_missing_values    # {'Kosovo', 'Botswana', 'MS Zaandam', 'West Bank and Gaza', 'Burundi', 'Sierra Leone', 'South Sudan', 'Sao Tome and Principe', 'Malawi', 'Western Sahara', 'Burma'}

df_merged = df_merged.rename(columns={'country':'Country_Region'})

df_merged['Density_KM2m'] = df_merged['Density_KM2m'].apply(lambda x:str(x).replace(',','')).astype('float32')

df_merged['Median_age'] = df_merged['Median_age'].replace('N.A.',None).apply(lambda x:str(x).replace(',','')).astype('float32')

df_merged['first_school_closure_date'] = pd.to_datetime(df_merged['first_school_closure_date'])

df_merged['Population_2020'] = np.log(df_merged['Population_2020']+1)



train = pd.merge(train, df_merged, how='left', on='Country_Region').reset_index()

del train['index']

test = pd.merge(test, df_merged, how='left', on='Country_Region').reset_index()

del test['index']



train['first_school_closure_days'] = (train['Date'] - train['first_school_closure_date']).dt.days.clip(0).fillna(0)

test['first_school_closure_days'] = (test['Date'] - test['first_school_closure_date']).dt.days.clip(0).fillna(0)





# temp and precip

temp_cols = [ c for c in df_merged.columns if '_temp' in c]

precip_cols = [c for c in df_merged.columns if '_precip' in c]

df_temp = df_merged[['Country_Region']+temp_cols]

df_precip = df_merged[['Country_Region']+precip_cols]

df_temp

del df_temp['annual_temp']

df_precip

del df_precip['Annual_precip']



months = 'jan	feb	mar	apr	may	jun	july	aug	sept	oct	nov	dec'.replace('\t',' ').split(' ')

months_dict = dict(zip(months,range(1,len(months)+1)))

months_dict



# df_temp.columns = ['Country_Region'] + ['temp_' + c.replace('_temp','') for c in df_merged.columns if '_temp' in c and c != 'annual_temp']

df_temp = pd.melt(df_temp, id_vars='Country_Region').rename(columns={'variable':'Month','value':'Temp'})

df_temp['Month'] = [months_dict[m.replace('_temp','')] for m in df_temp['Month'] ]

df_temp



# df_temp.columns = ['Country_Region'] + ['temp_' + c.replace('_temp','') for c in df_merged.columns if '_temp' in c and c != 'annual_temp']

df_precip = pd.melt(df_precip, id_vars='Country_Region').rename(columns={'variable':'Month','value':'Precip'})

df_precip['Month'] = [months_dict[m.lower().replace('_precip','')] for m in df_precip['Month'] ]

df_precip





train = pd.merge(train, df_temp, how='left', on=['Country_Region','Month']).reset_index()

del train['index']

test = pd.merge(test, df_temp, how='left', on=['Country_Region','Month']).reset_index()

del test['index']



train = pd.merge(train, df_precip, how='left', on=['Country_Region','Month']).reset_index()

del train['index']

test = pd.merge(test, df_precip, how='left', on=['Country_Region','Month']).reset_index()

del test['index']

df_air = pd.read_csv('/kaggle/input/uncover/UNCOVER/un_world_food_programme/world-travel-restrictions.csv')

df_air = df_air.rename(columns={'published':'AirportRestrictionDate','iso3':'code_3digit_x'})[['code_3digit_x','AirportRestrictionDate']].dropna()

df_air['AirportRestrictionDate'] = pd.to_datetime(df_air['AirportRestrictionDate'])

train = pd.merge(train, df_air, how='left', on=['code_3digit_x']).reset_index()

del train['index']

train['AirportRestrictionDays'] = (train['Date'] - train['AirportRestrictionDate']).dt.days.clip(0).fillna(0)

test = pd.merge(test, df_air, how='left', on=['code_3digit_x']).reset_index()

del test['index']

test['AirportRestrictionDays'] = (test['Date'] - test['AirportRestrictionDate']).dt.days.clip(0).fillna(0)
# df_strain = pd.read_csv('/kaggle/input/covid19-country-data/covid19_data - covid19_strains.csv')

# df_strain_clade = df_strain.groupby(['Country','Clade']).count().reset_index()

# df_strain_clade['Count'] = 1

# df_strain_clade = df_strain_clade[['Country','Clade','Count']]



# tmp = df_strain_clade.pivot_table(index='Country',columns=['Clade'], aggfunc=np.sum).fillna(0).reset_index()

# # print(tmp.columns.to_frame()['Clade'].values)

# # print(tmp.columns)

# # print(tmp.shape)

# tmp.columns = tmp.columns.to_frame()['Clade'].values

# tmp['Country_Region'] = tmp['']

# del tmp['']

# df_strain_clade = tmp

# print(df_strain_clade.columns)





# train = pd.merge(train, df_strain_clade, how='left', on=['Country_Region']).reset_index()

# del train['index']

# test = pd.merge(test, df_strain_clade, how='left', on=['Country_Region']).reset_index()

# del test['index']



# del df_strain_clade

# del tmp

# del df_strain

# df_nurses = pd.read_csv('/kaggle/input/doctors-and-nurses-per-1000-people-by-country/Nurses_Per_Capital_By_Country.csv')

# df_dr = pd.read_csv('/kaggle/input/doctors-and-nurses-per-1000-people-by-country/Doctors_Per_Capital_By_Country.csv')



# df_nurses = df_nurses.query('TIME == "2017"')

# df_nurses = df_nurses.rename(columns={'LOCATION':'code_3digit_x', 'Value':'NursesPer1000'})

# df_nurses = df_nurses[['code_3digit_x','NursesPer1000']]

# # df_nurses





# df_dr = df_dr.query('TIME == "2017"')

# df_dr = df_dr.rename(columns={'LOCATION':'code_3digit_x', 'Value':'DrPer1000'})

# df_dr = df_dr[['code_3digit_x','DrPer1000']]

# # df_dr





# train = pd.merge(train, df_nurses, how='left', on=['code_3digit_x']).reset_index()

# del train['index']

# test = pd.merge(test, df_nurses, how='left', on=['code_3digit_x']).reset_index()

# del test['index']



# train = pd.merge(train, df_dr, how='left', on=['code_3digit_x']).reset_index()

# del train['index']

# test = pd.merge(test, df_dr, how='left', on=['code_3digit_x']).reset_index()

# del test['index']
print(test.shape, train.shape)
df_acaps = pd.read_csv('/kaggle/input/demographic-factors-for-explaining-covid19/acaps_covid19_database/acaps_covid19_database.csv')

df_acaps['DATE_IMPLEMENTED'] = pd.to_datetime(df_acaps['DATE_IMPLEMENTED'].replace('3/28/2020','28/3/2020'), format='%d/%m/%Y' )

print(min(df_acaps['DATE_IMPLEMENTED']), max(df_acaps['DATE_IMPLEMENTED']))

df_acaps_agg = df_acaps.groupby(['ISO']).min().reset_index()

df_acaps_agg = df_acaps_agg.rename(columns={'DATE_IMPLEMENTED':'MeasureImplementDate', 'REGION':'Region', 'ISO':'code_3digit_x'})

le = preprocessing.LabelEncoder()

le.fit(df_acaps_agg['Region'])

df_acaps_agg['RegionCode'] = le.transform(df_acaps_agg['Region'])

del df_acaps_agg['COUNTRY']

df_acaps_agg = df_acaps_agg.drop_duplicates()

print(df_acaps_agg.shape)



train = pd.merge(train, df_acaps_agg, how='left', on=['code_3digit_x']).reset_index()

del train['index']

test = pd.merge(test, df_acaps_agg, how='left', on=['code_3digit_x']).reset_index()

del test['index']



train['MeasureImplementDays'] = (train['Date'] - train['MeasureImplementDate']).dt.days.clip(0).fillna(0)

test['MeasureImplementDays'] = (test['Date'] - test['MeasureImplementDate']).dt.days.clip(0).fillna(0)

print(train.columns.values)
def plt_country(df, country, province=''):

    nrows = 2

    ncols = 1

    index = 1



    plt.subplots_adjust(hspace=1.)



    for index, metric in enumerate(['DailyConfirmedCases','DailyFatalities']):

        tmp = country_slice(df, country, province)

        if tmp.shape[0] == 0:

            raise Exception(f'Not found {province} {country}')

#         metric = 'DailyFatalities'

        metric_pred = metric + 'Predicted'

        y = np.array(tmp[metric])

        x = np.array(range(len(y)))

        y_pred = np.array(tmp[metric_pred])

        dates = sorted(tmp['Date'].drop_duplicates().tolist())

        min_test_date = test['Date'].min()

        min_test_date_index = dates.index(min_test_date)

        min_test_date_str = str(min_test_date).replace('00:00:00','')



        ax = plt.subplot(nrows, ncols, index+1)

        plt.title(f"{province} {country}")

        plt.plot(x, y, label='Actual')

        plt.plot(x, y_pred, label='Predicted')

        plt.ylabel(metric)

        plt.xlabel('Time')

        plt.axvline(min_test_date_index, 0, 10000,label=f'Test {min_test_date_str}',linestyle='--')

#         ax = plt.gca()

        ax.legend()

    

# plt_country(train, 'China', 'Hubei')
# from https://www.kaggle.com/osciiart/covid-19-lightgbm-with-weather-2/data#Model-training



def calc_score(y_true, y_pred):

    y_true[y_true<0] = 0

    print(stats.describe(y_true))

    print(stats.describe(y_pred))

    if log_transform:

        score = metrics.mean_squared_error(np.log(y_true.clip(0, 1e10)+1), np.log(y_pred[:]+1))**0.5

    else:

        score = metrics.mean_squared_error(y_true.clip(0, 1e10), y_pred[:].clip(0, 1e10))**0.5

    return score





def train_model(df, min_test_date, col_target, col_var, init_model=None):

    

    # filter training data upto the test date

    df_valid = df[df['Date']>=min_test_date]

    df_train = df[df['Date']<min_test_date]

#     print(f"min_test_date={min(test['Date'])}")

#     print(f"df_valid min_date={min(df_valid['Date'])} max={max(df_valid['Date'])}")

#     print(f'train days={max(train["Days"])}, min_date={min(train["Date"])}, max_date={max(train["Date"])}')

#     print(f'test days={max(test["Days"])}, min_date={min(test["Date"])}, max_date={max(test["Date"])}')



    X_train = df_train[col_var]

    X_valid = df_valid[col_var]



    if impute:

        from sklearn.impute import SimpleImputer

        # Some Countries like Canada don't have Recovery broken down by province! so impute nan's

        # np.argwhere(np.isnan(X_train))

        my_imputer = SimpleImputer()

        X_train = my_imputer.fit_transform(X_train)

        X_valid = my_imputer.fit_transform(X_valid)





    print(f'log transform? {log_transform}')

#     col_target = 'DailyFatalities'

    if log_transform:

        y_train = np.log(df_train[col_target].values.clip(0, 1e10)+1)

        y_valid = np.log(df_valid[col_target].values.clip(0, 1e10)+1)

    else:

        # y_train = df_train[col_target].values.clip(0, 1e10)

        # y_valid = df_valid[col_target].values.clip(0, 1e10)

        y_train = df_train[col_target].values

        y_valid = df_valid[col_target].values

    train_data = lgb.Dataset(X_train, label=y_train)   # categorical_feature=col_cat

    valid_data = lgb.Dataset(X_valid, label=y_valid)

    model = lgb.train(params, train_data, num_round, valid_sets=[train_data, valid_data],

                      verbose_eval = 100,

                      init_model = init_model,

                      early_stopping_rounds=early_stopping_rounds,

    #                   early_stopping_rounds=150,

    #                   early_stopping_rounds=500,

                     )



    best_itr = model.best_iteration

    y_true = df_valid[col_target].values



    if log_transform:

        y_pred = np.exp(model.predict(X_valid))-1

    else:

        y_pred = model.predict(X_valid)



    score = calc_score(y_true, y_pred)

    print(f"{col_target} score {round(score,4)}")

    

    return model









# exclude countries with lots of cases

min_test_date = min(test["Date"])

# df_max_cases = train.query(f'Date < "{min_test_date}"').groupby(['Country_Province']).agg({'ConfirmedCases':'max'}).reset_index().sort_values(by='ConfirmedCases')

# df_max_cases.tail(30).plot.bar('Country_Province','ConfirmedCases')

# min_cases = 3e4

# countries_lots_cases = df_max_cases.query(f'ConfirmedCases > {min_cases}')['Country_Province'].values

# countries_lots_cases

all_countries = train['Country_Region'].drop_duplicates().values

print(len(all_countries))



if big_countries is not None:

    print(f'training {sel_countries}')

#     df_test = test[test['Country_Province'].isin(sel_countries)]

    df_train_large = train[train['Country_Province'].isin(big_countries)]

#     df_train_med   = train[train['Country_Province'].isin(med_countries)]

#     df_train_small = train[train['Country_Province'].isin(small_countries)]

    model_large_fatalities = train_model(df_train_large, min_test_date, 'DailyFatalities', col_var)

    model_large_cc         = train_model(df_train_large, min_test_date, 'DailyConfirmedCases', col_var)

#     model_med_fatalities = train_model(df_train_med, min_test_date, 'DailyFatalities', col_var)

#     model_med_cc         = train_model(df_train_med, min_test_date, 'DailyConfirmedCases', col_var)

#     model_small_fatalities       = train_model(df_train_small, min_test_date, 'DailyFatalities', col_var)

#     model_small_cc               = train_model(df_train_small, min_test_date, 'DailyConfirmedCases', col_var)

else:

    df_train = train

    model_fatalities       = train_model(df_train, min_test_date, 'DailyFatalities', col_var)

    model_cc               = train_model(df_train, min_test_date, 'DailyConfirmedCases', col_var)

def predict(df, model):

    

    if type(df)==pd.core.series.Series:

        # if we predict test

        x = np.array(df[col_var]).reshape(1, -1)

    else:

        x = df[col_var]

    

    y_pred = model.predict(x)

    

    if log_transform:

        y_pred = np.exp(y_pred)-1



    

    if y_pred.shape[0] == 1:

        # if we predict test

        y_pred = y_pred[0]

        

    return y_pred







# calculate predictions

df_train_large.loc[:,'DailyFatalitiesPredicted']             = predict(df_train_large, model_large_fatalities)

df_train_large.loc[:,'DailyConfirmedCasesPredicted']         = predict(df_train_large, model_large_cc)

# df_train_med.loc[:,'DailyFatalitiesPredicted']             = predict(df_train_med, model_med_fatalities)

# df_train_med.loc[:,'DailyConfirmedCasesPredicted']         = predict(df_train_med, model_med_cc)

# df_train_small.loc[:,'DailyFatalitiesPredicted']             = predict(df_train_small, model_small_fatalities)

# df_train_small.loc[:,'DailyConfirmedCasesPredicted']         = predict(df_train_small, model_small_cc)



cols = ['Country_Province', 'Date', 'DailyFatalitiesPredicted','DailyConfirmedCasesPredicted']

if 'DailyFatalitiesPredicted' in train:

    print('DailyFatalitiesPredicted in train')

    del train['DailyFatalitiesPredicted']

if 'DailyConfirmedCasesPredicted' in train:

    del train['DailyConfirmedCasesPredicted']

# train = pd.merge(train, pd.concat([df_train_small[cols], df_train_med[cols], df_train_large[cols]]), how='left', on=['Country_Province','Date'])

train = pd.merge(train, pd.concat([df_train_large[cols]]), how='left', on=['Country_Province','Date'])

print(f'train {tmp.shape}')
# plot

for i, c in list(enumerate(big_countries))[:10]:

    country, province = c.split('/')

    plt.figure(i)

    plt_country(df_train_large, country, province)
# plot

# for i, c in enumerate(['Taiwan*/', 'Japan/', 'Philippines/', 'Canada/British Columbia']):

# for i, c in list(enumerate(small_countries))[:10]:

#     country, province = c.split('/')

#     plt.figure(i)

#     plt_country(df_train_small, country, province)
tmp = pd.DataFrame()

tmp["feature"] = col_var

tmp["importance_ConfirmedCases"]   = model_large_cc.feature_importance()

tmp["importance_Fatalities"]       = model_large_fatalities.feature_importance()

tmp = tmp.sort_values('importance_ConfirmedCases', ascending=False)

tmp.to_csv('var_importance_large.csv',index=False)

tmp


# train model to predict fatalities/day

# train_bak = train



# # exclude countries with lots of cases

# min_test_date = min(test["Date"])

# df_max_cases = train.query(f'Date < "{min_test_date}"').groupby(['Country_Province']).agg({'ConfirmedCases':'max'}).reset_index().sort_values(by='ConfirmedCases')

# df_max_cases.tail(30).plot.bar('Country_Province','ConfirmedCases')

# min_cases = 3e4

# countries_lots_cases = df_max_cases.query(f'ConfirmedCases > {min_cases}')['Country_Province'].values

# countries_lots_cases

# all_countries = train['Country_Region'].drop_duplicates().values

# print(len(all_countries))

# sel_countries = countries_lots_cases

# if sel_countries:

#     print(len(sel_countries))

#     test = test[test['Country_Region'].isin(sel_countries)]

#     train = train[train['Country_Region'].isin(sel_countries)]



# col_target = 'DailyFatalities'

# model_fatalities = train_model(train, min_test_date, 'DailyFatalities', col_var)

# model_cc = train_model(train, min_test_date, 'DailyConfirmedCases', col_var)



# # train.to_csv('train_merged.csv', index=False)



# print(f"Max train date {max(df_train['Date'])}")



# X_train = df_train[col_var]

# X_valid = df_valid[col_var]





# if impute:

#     from sklearn.impute import SimpleImputer

#     # Some Countries like Canada don't have Recovery broken down by province! so impute nan's

#     # np.argwhere(np.isnan(X_train))

#     my_imputer = SimpleImputer()

#     X_train = my_imputer.fit_transform(X_train)

#     X_valid = my_imputer.fit_transform(X_valid)





# col_target = 'DailyFatalities'

# if log_transform:

#     y_train = np.log(df_train[col_target].values.clip(0, 1e10)+1)

#     y_valid = np.log(df_valid[col_target].values.clip(0, 1e10)+1)

# else:

#     # y_train = df_train[col_target].values.clip(0, 1e10)

#     # y_valid = df_valid[col_target].values.clip(0, 1e10)

#     y_train = df_train[col_target].values

#     y_valid = df_valid[col_target].values

# train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=col_cat)

# valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature=col_cat)

# model = lgb.train(params, train_data, num_round, valid_sets=[train_data, valid_data],

#                   verbose_eval=100,

#                   early_stopping_rounds=early_stopping_rounds,

# #                   early_stopping_rounds=150,

# #                   early_stopping_rounds=500,

#                  )



# best_itr = model.best_iteration

# y_true = df_valid[col_target].values



# if log_transform:

#     y_pred = np.exp(model.predict(X_valid))-1

# else:

#     y_pred = model.predict(X_valid)

    

# score = calc_score(y_true, y_pred)

# print(f"{col_target} score {round(score,4)}")

# model_fatalities = model





# col_target = 'DailyConfirmedCases'

# if log_transform:

#     y_train = np.log(df_train[col_target].values.clip(0, 1e10)+1)

#     y_valid = np.log(df_valid[col_target].values.clip(0, 1e10)+1)

# else:

#     # y_train = df_train[col_target].values.clip(0, 1e10)

#     # y_valid = df_valid[col_target].values.clip(0, 1e10)

#     y_train = df_train[col_target].values

#     y_valid = df_valid[col_target].values

# train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=col_cat)

# valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature=col_cat)

# model = lgb.train(params, train_data, num_round, valid_sets=[train_data, valid_data],

#                   verbose_eval=100,

#                   early_stopping_rounds=early_stopping_rounds,

# #                   early_stopping_rounds=150,

# #                   early_stopping_rounds=500,

#                  )



# best_itr = model.best_iteration

# y_true = df_valid[col_target].values



# if log_transform:

#     y_pred = np.exp(model.predict(X_valid))-1

# else:

#     y_pred = model.predict(X_valid)

    

# score = calc_score(y_true, y_pred)

# print(f"{col_target} score {round(score,4)}")

# model_cc = model





# col_target = 'DailyRecovered'  

# y_train = np.log(df_train[col_target].values.clip(0, 1e10)+1)

# y_valid = np.log(df_valid[col_target].values.clip(0, 1e10)+1)

# train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=col_cat)

# valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature=col_cat)

# model = lgb.train(params, train_data, num_round, valid_sets=[train_data, valid_data],

#                   verbose_eval=100,

#                   early_stopping_rounds=150,)



# best_itr = model.best_iteration

# y_true = df_valid[col_target].values

# y_pred = np.exp(model.predict(X_valid))-1

# score = calc_score(y_true, y_pred)

# print(f"{col_target} score {round(score,4)}")

# model_recovered = model
def assign_predictions_as_features(df, idx, country_province, pred, metrics):

    """

    assign predictions as features for future predictions

    """

    for j,m in enumerate(metrics, 1):

        try:

            if(df.loc[idx+j,'Country_Province']==country_province):

#                 print(f'assign {m} {j} {pred}')

                df.loc[idx+j, m] = pred

        except:

            pass

    

    

y_fatalities, y_cases = [], []

cols = ['Date','DailyFatalities','DailyConfirmedCases'] + col_var

X_test = test[col_var]

X_train = train.query(f'Date < "{min_test_date}"')[cols]



train_freq = np.ceil(test.shape[0] / len(all_countries))

print(f'train_freq={train_freq}')



for i in tqdm(list(range(test.shape[0]))):

# for i in list(range(test.shape[0]))[:3]:

    country = test.loc[i, 'Country_Region']

    province = test.loc[i, 'Province_State']

    country_province = test.loc[i, 'Country_Province']

    

#     x = np.array(test.loc[i, col_var]).reshape(1, -1)

    

    # pick the model

    m_f = model_large_fatalities

    m_c = model_large_cc

#     if country_province in big_countries:

#         m_f = model_large_fatalities

#         m_c = model_large_cc

#     elif country_province in med_countries:

#         m_f = model_med_fatalities

#         m_c = model_med_cc

#     else:

#         m_f = model_small_fatalities

#         m_c = model_small_cc

    

    # predict

    y_f = predict(test.loc[i], m_f)

    y_c = predict(test.loc[i], m_c)

    y_fatalities.append(y_f)

    y_cases.append(y_c)

    y_fc_ratio = y_f / (y_c + 0.00000000001)

    y_c_less_f = (y_c - y_f) / (y_c + 0.00000000001)

    

    # assign predictions as features for future predictions

#     print(f'====== test_{i} before {test.loc[i][col_var]} ======')

    assign_predictions_as_features(test, i, country_province, y_f, [f'DailyFatalities_{i}' for i in range(1,MAX_LAG+1)] )

#     print(f'====== test_{i} after {test.loc[i][col_var]} ======')

    assign_predictions_as_features(test, i, country_province, y_c, [f'DailyConfirmedCases_{i}' for i in range(1,MAX_LAG+1)] )

    assign_predictions_as_features(test, i, country_province, y_fc_ratio, ['DailyFCRatio_1'] )

    assign_predictions_as_features(test, i, country_province, y_c_less_f, ['ConfirmedLessFatalities_1'] )

    

    test.loc[i,'DailyFatalities']   = y_f

    test.loc[i,'DailyConfirmedCases']   = y_c

    

        

    # update our models occasionally to speedup

    try:

#         if i % 14 == 0:

        if i % 500 == 0:

#         if test.loc[i,'Date'] in [pd.to_datetime(d) for d in ['2020-04-12']]:

            X_valid_date = test.loc[i, 'Date'] - pd.Timedelta('2 days')

            

            test_slice = test.loc[i, cols]

#             X_train = pd.concat([X_train, test_slice])

            X_train = X_train.append(test_slice)

            

            print('X_train_shape',X_train.shape)

            model_large_fatalities = train_model(X_train, X_valid_date, 'DailyFatalities', col_var, model_large_fatalities)

            model_large_cc         = train_model(X_train, X_valid_date, 'DailyConfirmedCases', col_var, model_large_cc)

#             if country_province in big_countries:

#                 model_large_fatalities = train_model(X_train, X_valid_date, 'DailyFatalities', col_var, model_large_fatalities)

#                 model_large_cc = train_model(X_train, X_valid_date, 'DailyConfirmedCases', col_var, model_large_cc)

#             elif country_province in med_countries:

#                 model_med_fatalities = train_model(X_train, X_valid_date, 'DailyFatalities', col_var, model_med_fatalities)

#                 model_med_cc = train_model(X_train, X_valid_date, 'DailyConfirmedCases', col_var, model_med_cc)

#             else:

#                 model_small_fatalities = train_model(X_train, X_valid_date, 'DailyFatalities', col_var, model_small_fatalities)

#                 model_small_cc = train_model(X_train, X_valid_date, 'DailyConfirmedCases', col_var, model_small_cc)

    except Exception as e:

        print(f'Error {e}')

        pass

        

        

test['DailyFatalitiesPredicted']     = y_fatalities

test['DailyConfirmedCasesPredicted'] = y_cases
# plot predictions in test

cols = ['Country_Region','Province_State', 'Date', 'DailyConfirmedCases','DailyFatalities', 'DailyConfirmedCasesPredicted','DailyFatalitiesPredicted'] 

test['DailyFatalities'] = None

test['DailyConfirmedCases'] = None

train_test = pd.concat( [ train[cols], test[test['Date'] > max(train['Date'])][cols] ] ).sort_values(by=['Country_Region','Province_State', 'Date'])



# plt_dict = {'China':'Hubei', 'France':'', 'Japan':'', 'Italy':'', 'Spain':'', 'Germany':'', 'US':'New York', 'Taiwan*':'', 'Korea, South':'', 'United Kingdom':'', 'Canada':'British Columbia'}

# for country, province in plt_dict.items():

for i, cp in enumerate(['Taiwan*/', 'Japan/', 'Philippines/', 'Canada/British Columbia'], 1):

    country, province = cp.split('/')

    if province is None:

            province = ''

    plt.figure(i)

    try:

        plt_country(train_test, country, province)

    except Exception as e:

        print(f'Exception in plt_country {e}')
for i, cp in enumerate(sel_countries, 1):

    country, province = cp.split('/')

    if province is None:

            province = ''

    plt.figure(i)

    plt_country(train_test, country, province)
# country_slice(train,'Canada','British Columbia').query('Date < "2020-04-02"').tail(5)[['Date', 'DailyFatalities','DailyFatalitiesPredicted','DailyConfirmedCases','DailyConfirmedCasesPredicted']+ col_var]
# country_slice(test,'Canada','British Columbia').head(5)[['Date', 'DailyFatalities','DailyFatalitiesPredicted','DailyConfirmedCases','DailyConfirmedCasesPredicted']+ col_var]
# X_train = train[col_var]



# if log_transform:

#     y_fatalities = np.exp(model_fatalities.predict(X_train))-1

#     y_cases = np.exp(model_cc.predict(X_train))-1

# else:

#     y_fatalities = model_fatalities.predict(X_train)

#     y_cases = model_cc.predict(X_train)

    

# train['DailyFatalitiesPredicted'] = y_fatalities

# train['DailyConfirmedCasesPredicted'] = y_cases

train['diff_fatalities'] = abs(train['DailyFatalitiesPredicted']-train['DailyFatalities'])

train['diff_cases'] = abs(train['DailyConfirmedCasesPredicted']-train['DailyConfirmedCases'])



tmp = train[['Province_State','Country_Region', 'Date', 'DailyFatalities', 'DailyFatalitiesPredicted', 'DailyConfirmedCases', 'DailyConfirmedCasesPredicted']]

# country_slice(tmp,'Spain')

country_slice(tmp,'Italy').tail(5)



train_diff = train.groupby(['Country_Region','Province_State']).agg({'diff_fatalities':'sum','diff_cases':'sum'}).reset_index()

train_diff.sort_values(by='diff_fatalities',ascending=False).head(20)
train_diff.sort_values(by='diff_cases',ascending=False).head(20)
# calculate cumulative sum

# get the last training fatalities count and add the y_pred and calculate cumsum

out = []

# for i, rows in list(train[train['Date']==max(train['Date'])].iterrows())[:2]:

for i, rows in list(train[train['Date']==max(train['Date'])].iterrows()):

    country, province, fatalities, confirmed_cases = rows['Country_Region'], rows['Province_State'], rows['Fatalities'], rows['ConfirmedCases']

    tmp = country_slice(test, country, province).sort_values(by='Date')

#     print(country, province, fatalities, confirmed_cases)

    tmp['Fatalities'] = np.cumsum([fatalities] + tmp['DailyFatalitiesPredicted'].tolist())[1:]

    tmp['ConfirmedCases'] = np.cumsum([confirmed_cases] + tmp['DailyConfirmedCasesPredicted'].tolist())[1:]

    out.append(tmp)



results = pd.concat(out).sort_values(by='ForecastId')



tmp = results[['ForecastId', 'Province_State','Country_Region', 'Date', 'Fatalities', 'DailyFatalitiesPredicted', 'ConfirmedCases', 'DailyConfirmedCasesPredicted'] + col_var]

tmp.head()

# train[['Province_State','Country_Region', 'Date', 'Fatalities']]



tmp.to_csv('results.csv', index=False)
results[submission.columns]
results.shape
test.columns
results[submission.columns].to_csv('submission.csv', index=False)

cols = ['Country_Region','Province_State','Date','DailyFatalitiesPredicted','DailyConfirmedCases'] + col_var

test[cols].to_csv('test_out.csv', index=False)

train[cols].to_csv('train_out.csv', index=False)
print('Done!')