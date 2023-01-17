# Installing the required libs

!pip install -q fastprogress fastai2 fastcore fast_tabnet --upgrade 
from fastai2.basics import *

from fastai2.tabular.all import *

from fast_tabnet.core import *
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
import sys

sys.path.insert(0, "../input/covid19-global-forecasting-week-2/")

import warnings

warnings.filterwarnings(action='once')
df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv') 
df.head()
df_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv') 

df_test.head()
df['key'] = df['Country_Region'] + '#' + df['Province_State'].fillna('')

df['province_flag'] = np.where(df['Province_State'].isnull(),0,1)

df['Province_State'] = df['Province_State'].fillna(df['Country_Region'])
df_test['key'] = df_test['Country_Region'] + '#' + df_test['Province_State'].fillna('')

df_test['province_flag'] = np.where(df_test['Province_State'].isnull(),0,1)

df_test['Province_State'] = df_test['Province_State'].fillna(df_test['Country_Region'])
df.head(600)
df_test.head(600)
#firstconfirmed = df[(df['ConfirmedCases']>0) & (df['Date']<'2020-03-19')].groupby(['Province_State','Country_Region'])['Date'].min().reset_index()

firstconfirmed = df[(df['ConfirmedCases']>0)].groupby(['Province_State','Country_Region'])['Date'].min().reset_index()
firstconfirmed.head()
firstconfirmed.shape
firstfatality = df[(df['Fatalities']>0)].groupby(['Province_State','Country_Region'])['Date'].min().reset_index()

#firstfatality = df[(df['Fatalities']>0) & (df['Date']<'2020-03-19')].groupby(['Province_State','Country_Region'])['Date'].min().reset_index()
firstfatality.head()
firstfatality.shape
firstconfirmed.columns = ['Province_State','Country_Region','FirstCaseDate']

firstfatality.columns = ['Province_State','Country_Region','FirstFatalityDate']
df = df.merge(firstconfirmed, left_on=['Province_State','Country_Region'],right_on=['Province_State','Country_Region'],how='left')

df = df.merge(firstfatality, left_on=['Province_State','Country_Region'],right_on=['Province_State','Country_Region'],how='left')
df.head(40)
df_test = df_test.merge(firstconfirmed, left_on=['Province_State','Country_Region'],right_on=['Province_State','Country_Region'],how='left')

df_test = df_test.merge(firstfatality, left_on=['Province_State','Country_Region'],right_on=['Province_State','Country_Region'],how='left')
df_test.head(40)
df.dtypes
df['Date']=pd.to_datetime(df['Date'], infer_datetime_format=True) 

df['FirstCaseDate']=pd.to_datetime(df['FirstCaseDate'], infer_datetime_format=True) 

df['FirstFatalityDate']=pd.to_datetime(df['FirstFatalityDate'], infer_datetime_format=True) 
df.head()
df_test['Date']=pd.to_datetime(df_test['Date'], infer_datetime_format=True) 

df_test['FirstCaseDate']=pd.to_datetime(df_test['FirstCaseDate'], infer_datetime_format=True) 

df_test['FirstFatalityDate']=pd.to_datetime(df_test['FirstFatalityDate'], infer_datetime_format=True) 
df_test.head()
df['days_first_case']=(df['Date']-df['FirstCaseDate']).dt.days

df['days_first_fatality']=(df['Date']-df['FirstFatalityDate']).dt.days
df['days_first_case']
df_test['days_first_case']=(df_test['Date']-df_test['FirstCaseDate']).dt.days

df_test['days_first_fatality']=(df_test['Date']-df_test['FirstFatalityDate']).dt.days
df_test['days_first_case']
df['days_first_case']=np.where(df['days_first_case']<0,0,df['days_first_case'].fillna(0))

df['days_first_fatality']=np.where(df['days_first_fatality']<0,0,df['days_first_fatality'].fillna(0))
df_test['days_first_case']=np.where(df_test['days_first_case']<0,0,df_test['days_first_case'].fillna(0))

df_test['days_first_fatality']=np.where(df_test['days_first_fatality']<0,0,df_test['days_first_fatality'].fillna(0))
df.tail()
df_test.tail()
df[df['Country_Region']=='Brazil'].tail()
df_test[df_test['Country_Region']=='Brazil'].tail()
add_datepart(df,'Date',drop=False)
add_datepart(df_test,'Date',drop=False)
external = pd.read_csv('/kaggle/input/covid19-week2-external-data/external_data.csv',sep=';',decimal=',')
external.head()
df = df.merge(external, left_on='key',right_on='key',how='left')
df.head()
df.tail()
df_test = df_test.merge(external, left_on='key',right_on='key',how='left')
df_test.head()
df_test.tail()
list(df)
df.pivot_table(index='Country_Region', columns='Date', values='ConfirmedCases', aggfunc=np.sum, fill_value=0)
df['Confirmedlast43'] = df['ConfirmedCases'].shift(43) 

df['Fatalitieslast43'] = df['Fatalities'].shift(43)
df['is_valid'] = np.where(df['Date']<'2020-03-29', False, True)
df.groupby('is_valid').size()
df['ConfirmedLog'] = np.log(df['ConfirmedCases']+1)

df['FatalitiesLog'] = np.log(df['Fatalities']+1)
cat_vars = ['Province_State','Country_Region','province_flag']

cont_vars = ['Elapsed',

             'days_first_case',

             'days_first_fatality',

             'pop_density',

             'population',

             'area',

             'lat_min',

             'lat_max',

             'lon_min',

             'lon_max',

             'centroid_x',

             'centroid_y',

             'wdi_country_population',

             'wdi_country_arrivals',

             'wdi_arrivals_per_capita',

             'wdi_gini',

             'wdi_perc_urban_pop',

             'wdi_perc_handwashing',

             'wdi_uhc_coverage',

             'wdi_hospital_beds_p1000',

             'wdi_smoke_prevalence',

             'wdi_diabetes_prevalence',

             'wdi_gdp_per_capita_ppp',

             'wdi_perc_death_comm_diseases',

             'wdi_perc_death_non_comm_diseases',

             'wdi_death_rate_p1000',

             'wdi_perc_basic_sanitation',

             'wdi_dom_govmt_healt_exped_gdp',

             'wdi_dom_govmt_healt_exped_per_cap',

             'wdi_perc_females',

             'wdi_perc_males',

             'wdi_perc_female_20_29',

             'wdi_perc_female_30_39',

             'wdi_perc_female_40_49',

             'wdi_perc_female_50_59',

             'wdi_perc_female_60_69',

             'wdi_perc_female_70_79',

             'wdi_perc_female_80p',

             'wdi_perc_male_20_29',

             'wdi_perc_male_30_39',

             'wdi_perc_male_40_49',

             'wdi_perc_male_50_59',

             'wdi_perc_male_60_69',

             'wdi_perc_male_70_79',

             'wdi_perc_male_80p',

             'wdi_pop_denisty',

             'wdi_perc_anual_growth_pop','Month','Week','Dayofyear']#,'Confirmedlast43','Fatalitieslast43']

dep_vars = ['ConfirmedLog']#,'Fatalities']
procs = [FillMissing, Categorify, Normalize]

splits = ColSplitter('is_valid')(df)
splits
#?TabularPandas
y_block = TransformBlock()
to = TabularPandas(df, procs, cat_names=cat_vars, cont_names=cont_vars, y_names=dep_vars, y_block=RegressionBlock(), splits=splits)
to
dls = to.dataloaders(bs=512)
dls.show_batch()
cats, conts, y = next(iter(dls.train))
cats.shape
conts.shape
y.shape
dls.c
#max_log_y = np.log(10) + np.max(df['ConfirmedCases'])

max_log_y = np.log(1.2) + np.max(df['ConfirmedLog'])

y_range = (0, max_log_y)

y_range
learn = tabular_learner(dls, layers=[300,250], loss_func=MSELossFlat(),

                        config=tabular_config(ps=[0.005,0.015]), y_range=y_range, 

                        metrics=[exp_rmspe, rmse])#, embed_p=0.02
learn.model
learn.load('/kaggle/input/covid19-week2-external-data/best18_2day')
preds, y = learn.get_preds()
np.exp(preds)-1
np.exp(y)-1
dl = learn.dls.test_dl(df)
raw_test_preds = learn.get_preds(dl=dl)
raw_test_preds[0]
raw_test_preds[1]
preds = np.exp(to_np(raw_test_preds[0]))-1
preds[:40]
df['ConfirmedCases'].head(40)
df.head(40)
preds = pd.DataFrame(preds)

preds.columns = ['pred_confirmed']
preds.tail()
df = pd.concat([df, preds], axis=1)
df.tail()
df_test
dl = learn.dls.test_dl(df_test)
raw_test_preds = learn.get_preds(dl=dl)
raw_test_preds[0]
preds_confirmed_test = np.exp(to_np(raw_test_preds[0]))-1
preds_confirmed_test[:40]
cat_vars = ['Province_State','Country_Region','province_flag']

cont_vars = ['Elapsed',

             'days_first_case',

             'days_first_fatality',

             'pop_density',

             'population',

             'area',

             'lat_min',

             'lat_max',

             'lon_min',

             'lon_max',

             'centroid_x',

             'centroid_y',

             'wdi_country_population',

             'wdi_country_arrivals',

             'wdi_arrivals_per_capita',

             'wdi_gini',

             'wdi_perc_urban_pop',

             'wdi_perc_handwashing',

             'wdi_uhc_coverage',

             'wdi_hospital_beds_p1000',

             'wdi_smoke_prevalence',

             'wdi_diabetes_prevalence',

             'wdi_gdp_per_capita_ppp',

             'wdi_perc_death_comm_diseases',

             'wdi_perc_death_non_comm_diseases',

             'wdi_death_rate_p1000',

             'wdi_perc_basic_sanitation',

             'wdi_dom_govmt_healt_exped_gdp',

             'wdi_dom_govmt_healt_exped_per_cap',

             'wdi_perc_females',

             'wdi_perc_males',

             'wdi_perc_female_20_29',

             'wdi_perc_female_30_39',

             'wdi_perc_female_40_49',

             'wdi_perc_female_50_59',

             'wdi_perc_female_60_69',

             'wdi_perc_female_70_79',

             'wdi_perc_female_80p',

             'wdi_perc_male_20_29',

             'wdi_perc_male_30_39',

             'wdi_perc_male_40_49',

             'wdi_perc_male_50_59',

             'wdi_perc_male_60_69',

             'wdi_perc_male_70_79',

             'wdi_perc_male_80p',

             'wdi_pop_denisty',

             'wdi_perc_anual_growth_pop','Month','Week','Dayofyear','pred_confirmed']

dep_vars = ['FatalitiesLog'] 
procs = [FillMissing, Categorify, Normalize]

splits = ColSplitter('is_valid')(df)
to = TabularPandas(df, procs, cat_names=cat_vars, cont_names=cont_vars, y_names=dep_vars, y_block=RegressionBlock(), splits=splits)
dls = to.dataloaders(bs=512)
max_log_y = np.log(1.2) + np.max(df['FatalitiesLog'])

y_range = (0, max_log_y)

y_range
learn = tabular_learner(dls, layers=[300,250], loss_func=MSELossFlat(),

                        config=tabular_config(ps=[0.005,0.015]), y_range=y_range, 

                        metrics=[exp_rmspe, rmse])#, embed_p=0.02
learn.load('/kaggle/input/covid19-week2-external-data/fat_best2_2day')
preds, y = learn.get_preds()
preds
y
preds_confirmed_test = pd.DataFrame(preds_confirmed_test)
preds_confirmed_test.columns=['ConfirmedCases']
preds_confirmed_test.tail()
preds_confirmed_test.shape
df_test.shape
df_test = pd.concat([df_test, preds_confirmed_test], axis=1)
df_test.tail(40)
df_test['pred_confirmed']=df_test['ConfirmedCases']
dl = learn.dls.test_dl(df_test)
raw_test_preds = learn.get_preds(dl=dl)
raw_test_preds[0]
preds_fatalities_test = np.exp(to_np(raw_test_preds[0]))-1
preds_fatalities_test
preds_fatalities_test = pd.DataFrame(preds_fatalities_test)
preds_fatalities_test.columns=['Fatalities']
preds_fatalities_test.tail()
preds_fatalities_test.shape
df_test = pd.concat([df_test, preds_fatalities_test], axis=1)
df_test[df_test['Country_Region']=='Brazil'].tail()
df_test[df_test['Country_Region']=='Brazil'].head(20)
sub_ex = df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/submission.csv')  
sub_ex.head()
sub = df_test[['ForecastId','ConfirmedCases','Fatalities']]
sub.head()
sub_ex.tail()
sub.tail()
sub.to_csv('submission.csv',index=False)