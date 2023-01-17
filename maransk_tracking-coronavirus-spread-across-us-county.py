# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
majordir='/kaggle/input/covid19-us-county-trend/'

datadir=majordir+'csse_covid_19_daily_reports/'

date_today=29
covid_data_world_daily_0322=pd.read_csv(datadir+'03-22-2020.csv')

covid_data_world_daily_0322.rename(columns={'Confirmed':'Confirmed_0322'},inplace=True)

covid_data_world_daily_0322.head()

covid_data_us_daily_0322=covid_data_world_daily_0322[covid_data_world_daily_0322['Country_Region']=='US'].copy()

covid_data_us_daily_0322.shape


for i in range(23,date_today):

    dataset=datadir+'03-'+str(i)+'-2020.csv'

    col='Confirmed_03'+str(i)

    covid_data_world_daily=pd.read_csv(dataset)

    covid_data_world_daily.rename(columns={'Confirmed':col},inplace=True)

    covid_data_us_daily=covid_data_world_daily[covid_data_world_daily['Country_Region']=='US'].copy()



    if i==23:        

        covid_data_us_dailytrend=covid_data_us_daily_0322[['FIPS','Confirmed_0322']].merge(covid_data_us_daily[['FIPS',col]],on='FIPS').dropna()

    else:

        print(i)

        covid_data_us_dailytrend=covid_data_us_dailytrend.merge(covid_data_us_daily[['FIPS',col]],on='FIPS').dropna()

covid_data_us_dailytrend.shape
covid_data_us_daily_0322.head()
covid_data_us_dailytrend.dropna().shape

covid_data_us_dailytrend.head()
census_df_fips = pd.read_excel(majordir+'PopulationEstimates_us_county_level_2018.xlsx',skiprows=1)

census_df_fips.head()
census_df_fips.FIPS=census_df_fips.FIPS.astype(float)
census_df_fips_covid=census_df_fips.merge(covid_data_us_dailytrend,on='FIPS') 
census_df_fips_covid.head()
for i in range(22,date_today):

    col='Confirmed_03'+str(i)

    col_10000='Confirmed_per10000_03'+str(i)

    census_df_fips_covid[col_10000]=10000*(census_df_fips_covid[col]/census_df_fips_covid['POP_ESTIMATE_2018'])
census_df_fips_covid.head()
census_df_fips_covid_top3=census_df_fips_covid.sort_values(by='Confirmed_per10000_0324',ascending=False).reset_index().head(3)
census_df_fips_covid_top3
# plt.plot(census_df_fips_covid_top3['Area_Name'],census_df_fips_covid_top3[['Confirmed_per10000_0322','Confirmed_per10000_0323','Confirmed_per10000_0324','Confirmed_per10000_0325']],'*')
census_df_fips_covid_50000=census_df_fips_covid[(census_df_fips_covid['POP_ESTIMATE_2018']>50000) & (census_df_fips_covid['Confirmed_0322']>10)]

plt.plot(census_df_fips_covid_50000['POP_ESTIMATE_2018'],census_df_fips_covid_50000[['Confirmed_per10000_0322']],'*')

plt.xscale('log')

plt.yscale('log')

plt.xlabel('POP_ESTIMATE_2018')

plt.ylabel('Confirmed_per10000_0322')
current_confirm='Confirmed_03' + str(date_today)

current_confirm_10000='Confirmed_per10000_03' + str(date_today-1)

census_df_fips_covid_50000=census_df_fips_covid[(census_df_fips_covid['POP_ESTIMATE_2018']>50000) & (census_df_fips_covid['Confirmed_0324']>10)]

plt.plot(census_df_fips_covid_50000['POP_ESTIMATE_2018'],census_df_fips_covid_50000[[current_confirm_10000]],'*')

plt.xscale('log')

plt.yscale('log')

plt.xlabel('POP_ESTIMATE_2018')

plt.ylabel(current_confirm_10000)
census_df_fips_covid_50000=census_df_fips_covid[(census_df_fips_covid['POP_ESTIMATE_2018']>50000) & (census_df_fips_covid['Confirmed_0322']>10)]

plt.plot(census_df_fips_covid_50000['POP_ESTIMATE_2018'],census_df_fips_covid_50000[['Confirmed_per10000_0322']],'*')

plt.xscale('log')

plt.yscale('log')

plt.xlabel('POP_ESTIMATE_2018')

plt.ylabel('Confirmed_per10000_0322')



current_confirm='Confirmed_03' + str(date_today)

current_confirm_10000='Confirmed_per10000_03' + str(date_today-1)

census_df_fips_covid_50000=census_df_fips_covid[(census_df_fips_covid['POP_ESTIMATE_2018']>50000) & (census_df_fips_covid['Confirmed_0324']>10)]

plt.plot(census_df_fips_covid_50000['POP_ESTIMATE_2018'],census_df_fips_covid_50000[[current_confirm_10000]],'*')

plt.xscale('log')

plt.yscale('log')

plt.xlabel('POP_ESTIMATE_2018')

plt.ylabel(current_confirm_10000)

# census_df_fips_covid_100000.sort_values(by='Confirmed_per10000_0324',ascending=False).reset_index(drop=True).head(50)
census_df_fips_covid_50000.sort_values(by=current_confirm_10000,ascending=False).reset_index(drop=True).head(10)
census_df_fips_covid_50000.sort_values(by=current_confirm_10000,ascending=False).reset_index(drop=True).drop_duplicates(['FIPS']).tail(10)
!pip install vega_datasets
import altair as alt

from vega_datasets import data
us_counties = alt.topo_feature(data.us_10m.url, 'counties')
covid_data_us_dailytrend_1=census_df_fips_covid[['FIPS','Confirmed_per10000_0322','Confirmed_per10000_0323','Confirmed_per10000_0324','Confirmed_per10000_0325','Confirmed_per10000_0326','Confirmed_per10000_0327','Confirmed_per10000_0328']]

covid_data_us_dailytrend_1.columns=['fips','22','23','24','25','26','27','28']

covid_data_us_dailytrend_1.loc[:,'fips'] =covid_data_us_dailytrend_1.loc[:,'fips'].astype(int)

covid_data_us_dailytrend_1.sort_values(by='28',ascending=False).head()
slider = alt.binding_range(min=22, max=28, step=1)

select_day = alt.selection_single(name="day", fields=['day'],

                                   bind=slider, init={'day': 22})

columns_1=['22','23','24','25','26','27','28']



alt.Chart(us_counties).mark_geoshape(

    stroke='black',

    strokeWidth=0.05

).project(

    type='albersUsa'

).transform_lookup(

    lookup='id',

    from_=alt.LookupData(covid_data_us_dailytrend_1, 'fips', columns_1)

).transform_fold(

    columns_1, as_=['day', 'confirmed_per_pop']

).transform_calculate(

    day='parseInt(datum.day)',

    confirmed_per_pop='isValid(datum.confirmed_per_pop) ? datum.confirmed_per_pop : -1'  

).encode(

    color = alt.condition(

        'datum.confirmed_per_pop > 0',

        alt.Color('confirmed_per_pop:Q', scale=alt.Scale(scheme='plasma')),

        alt.value('#dbe9f6')

    )).add_selection(

    select_day

).properties(

    width=700,

    height=400

).transform_filter(

    select_day

)
covid_data_us_dailytrend_1=covid_data_us_dailytrend_1.drop(1905).copy()

covid_data_us_dailytrend_1=covid_data_us_dailytrend_1.drop(1934).copy()
max_val=covid_data_us_dailytrend_1[['22','23','24','25','26','27','28']].max().max()

covid_data_us_dailytrend_1.loc[3191,'22']=max_val

covid_data_us_dailytrend_1.loc[3191,'23']=max_val

covid_data_us_dailytrend_1.loc[3191,'24']=max_val

covid_data_us_dailytrend_1.loc[3191,'25']=max_val

covid_data_us_dailytrend_1.loc[3191,'26']=max_val

covid_data_us_dailytrend_1.loc[3191,'27']=max_val

covid_data_us_dailytrend_1.loc[3191,'28']=max_val
covid_data_us_dailytrend_1.tail()
slider = alt.binding_range(min=22, max=28, step=1)

select_day = alt.selection_single(name="day", fields=['day'],

                                   bind=slider, init={'day': 22})

columns_1=['22','23','24','25','26','27','28']



alt.Chart(us_counties).mark_geoshape(

    stroke='black',

    strokeWidth=0.05

).project(

    type='albersUsa'

).transform_lookup(

    lookup='id',

    from_=alt.LookupData(covid_data_us_dailytrend_1, 'fips', columns_1)

).transform_fold(

    columns_1, as_=['day', 'confirmed_per_pop']

).transform_calculate(

    day='parseInt(datum.day)',

    confirmed_per_pop='isValid(datum.confirmed_per_pop) ? datum.confirmed_per_pop : -1'  

).encode(

    color = alt.condition(

        'datum.confirmed_per_pop > 0',

        alt.Color('confirmed_per_pop:Q', scale=alt.Scale(scheme='plasma')),

        alt.value('#dbe9f6')

    )).add_selection(

    select_day

).properties(

    width=700,

    height=400

).transform_filter(

    select_day

)