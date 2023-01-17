# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
majordir='/kaggle/input/covid19-us-county-trend/'

datadir=majordir+'csse_covid_19_daily_reports/'

date_today=32
covid_data_world_daily_0322=pd.read_csv(datadir+'03-22-2020.csv')

covid_data_world_daily_0322.rename(columns={'Confirmed':'Confirmed_0322'},inplace=True)

covid_data_world_daily_0322.rename(columns={'Deaths':'Deaths_0322'},inplace=True)

covid_data_us_daily_0322=covid_data_world_daily_0322[covid_data_world_daily_0322['Country_Region']=='US'].copy()

covid_data_us_daily_0322.shape
covid_data_us_daily_0322= covid_data_us_daily_0322[covid_data_us_daily_0322['FIPS'].notna()]

vc=covid_data_us_daily_0322['FIPS'].value_counts()

vclist=vc[vc > 1].index.tolist()

covid_data_us_daily_0322=covid_data_us_daily_0322[~(covid_data_us_daily_0322['FIPS'].isin(vclist)&(covid_data_us_daily_0322['Confirmed_0322']>0))]

covid_data_us_daily_0322.shape
for i in range(23,date_today):

    dataset=datadir+'03-'+str(i)+'-2020.csv'

    colc='Confirmed_03'+str(i)

    covid_data_world_daily=pd.read_csv(dataset)

    covid_data_world_daily.rename(columns={'Confirmed':colc},inplace=True)

    

    cold='Deaths_03'+str(i)

    covid_data_world_daily.rename(columns={'Deaths':cold},inplace=True)

    

    covid_data_us_daily=covid_data_world_daily[covid_data_world_daily['Country_Region']=='US'].copy()    



    if i==23:

        print(i)

        covid_data_us_dailytrend=covid_data_us_daily_0322[['FIPS','Confirmed_0322','Deaths_0322']].merge(covid_data_us_daily[['FIPS',colc,cold]],on='FIPS').dropna()

    else:

        print(i)

        covid_data_us_dailytrend=covid_data_us_dailytrend.merge(covid_data_us_daily[['FIPS',colc,cold]],on='FIPS').dropna()

covid_data_us_dailytrend= covid_data_us_dailytrend[covid_data_us_dailytrend['FIPS'].notna()]

covid_data_us_dailytrend.shape
covid_data_us_dailytrend=covid_data_us_dailytrend.drop_duplicates(['FIPS'])

vc=covid_data_us_dailytrend['FIPS'].value_counts()

vclist=vc[vc > 1].index.tolist()

vc[vc > 1]

covid_data_us_dailytrend.sort_values(by=colc,ascending=False).head()
census_df_fips = pd.read_excel(majordir+'PopulationEstimates_us_county_level_2018.xlsx',skiprows=1)

census_df_fips.FIPS=census_df_fips.FIPS.astype(float)

census_density_df_fips = pd.read_csv(majordir+'uscounty_populationdesity.csv', encoding = "ISO-8859-1",skiprows=1)

census_density_df_fips.rename(columns={'Target Geo Id2':'FIPS'},inplace=True)

census_pop_density_df_fips=census_df_fips.merge(census_density_df_fips[['FIPS','Density per square mile of land area - Population']],on='FIPS')

census_pop_density_df_fips.shape
census_pop_density_df_fips.head()
census_pop_density_fips_covid=census_pop_density_df_fips.merge(covid_data_us_dailytrend,on='FIPS')

census_pop_density_fips_covid.sort_values(by=colc,ascending=False).head()
for i in range(22,date_today):

    colc='Confirmed_03'+str(i)

    colc_10000='Confirmed_per10000_03'+str(i)

    cold='Deaths_03'+str(i)

    cold_100000='Deaths_per100000_03'+str(i)

    census_pop_density_fips_covid[colc_10000]=10000*(census_pop_density_fips_covid[colc]/census_pop_density_fips_covid['POP_ESTIMATE_2018'])

    census_pop_density_fips_covid[cold_100000]=100000*(census_pop_density_fips_covid[cold]/census_pop_density_fips_covid['POP_ESTIMATE_2018'])

census_pop_density_fips_covid.sort_values(by=colc_10000,ascending=False).head(10).T
census_pop_density_fips_covid[(census_pop_density_fips_covid['Confirmed_0330']>500)&(census_pop_density_fips_covid['State']!='NY')&(census_pop_density_fips_covid['State']!='NJ')].sort_values(by='Confirmed_0330',ascending=False)[['State','Area_Name','POP_ESTIMATE_2018','Confirmed_0330','Confirmed_per10000_0330']]
census_pop_density_fips_covid[(census_pop_density_fips_covid['Confirmed_0331']>500)&(census_pop_density_fips_covid['State']!='NY')].sort_values(by='Confirmed_0322',ascending=False)[['State','Area_Name','POP_ESTIMATE_2018','Confirmed_0322','Confirmed_0323','Confirmed_0324','Confirmed_0325','Confirmed_0326','Confirmed_0327','Confirmed_0328','Confirmed_0329','Confirmed_0330','Confirmed_0331']].T
census_pop_density_fips_covid[(census_pop_density_fips_covid['Confirmed_0331']>1000)&(census_pop_density_fips_covid['Confirmed_0331']<2000)].sort_values(by='POP_ESTIMATE_2018',ascending=False)[['State','Area_Name','POP_ESTIMATE_2018','Confirmed_0322','Confirmed_0323','Confirmed_0324','Confirmed_0325','Confirmed_0326','Confirmed_0327','Confirmed_0328','Confirmed_0329','Confirmed_0330','Confirmed_0331']].T
census_pop_density_fips_covid[(census_pop_density_fips_covid['Confirmed_0331']>1000)&(census_pop_density_fips_covid['Confirmed_0331']<2000)].sort_values(by='Confirmed_per10000_0331',ascending=False)[['State','Area_Name','POP_ESTIMATE_2018','Confirmed_per10000_0322','Confirmed_per10000_0323','Confirmed_per10000_0324','Confirmed_per10000_0325','Confirmed_per10000_0326','Confirmed_per10000_0327','Confirmed_per10000_0328','Confirmed_per10000_0329','Confirmed_per10000_0330','Confirmed_per10000_0331']].T
census_pop_density_fips_covid[(census_pop_density_fips_covid['Confirmed_0331']>500)&(census_pop_density_fips_covid['Confirmed_0331']<1000)].sort_values(by='POP_ESTIMATE_2018',ascending=False)[['State','Area_Name','POP_ESTIMATE_2018','Confirmed_0322','Confirmed_0323','Confirmed_0324','Confirmed_0325','Confirmed_0326','Confirmed_0327','Confirmed_0328','Confirmed_0329','Confirmed_0330','Confirmed_0331']].T
census_pop_density_fips_covid[(census_pop_density_fips_covid['Confirmed_0331']>500)&(census_pop_density_fips_covid['Confirmed_0331']<1000)&(census_pop_density_fips_covid['State']!='NJ')].sort_values(by='POP_ESTIMATE_2018',ascending=False)[['State','Area_Name','POP_ESTIMATE_2018','Confirmed_per10000_0322','Confirmed_per10000_0323','Confirmed_per10000_0324','Confirmed_per10000_0325','Confirmed_per10000_0326','Confirmed_per10000_0327','Confirmed_per10000_0328','Confirmed_per10000_0329','Confirmed_per10000_0330','Confirmed_per10000_0331']].T
census_pop_density_fips_covid[(census_pop_density_fips_covid['Confirmed_0331']>500)&(census_pop_density_fips_covid['State']!='NY')&(census_pop_density_fips_covid['State']!='NJ')].sort_values(by='Confirmed_per10000_0322',ascending=True)[['State','Area_Name','POP_ESTIMATE_2018','Confirmed_per10000_0322','Confirmed_per10000_0323','Confirmed_per10000_0324','Confirmed_per10000_0325','Confirmed_per10000_0326','Confirmed_per10000_0327','Confirmed_per10000_0328','Confirmed_per10000_0329','Confirmed_per10000_0330','Confirmed_per10000_0331']].T
census_pop_density_fips_covid[(census_pop_density_fips_covid['POP_ESTIMATE_2018']>100000)&(census_pop_density_fips_covid['Confirmed_per10000_0330']>10)].sort_values(by='POP_ESTIMATE_2018',ascending=False)[['State','Area_Name','POP_ESTIMATE_2018','Confirmed_0331','Confirmed_per10000_0331']]
census_pop_density_fips_covid[(census_pop_density_fips_covid['POP_ESTIMATE_2018']>100000)&(census_pop_density_fips_covid['Confirmed_per10000_0330']>10)].sort_values(by='Confirmed_per10000_0322',ascending=False)[['State','Area_Name','POP_ESTIMATE_2018','Confirmed_per10000_0322','Confirmed_per10000_0323','Confirmed_per10000_0324','Confirmed_per10000_0325','Confirmed_per10000_0326','Confirmed_per10000_0327','Confirmed_per10000_0328','Confirmed_per10000_0329','Confirmed_per10000_0330']][8:-1].T
census_pop_density_fips_covid[(census_pop_density_fips_covid['POP_ESTIMATE_2018']>100000)&(census_pop_density_fips_covid['Confirmed_per10000_0330']>10)].sort_values(by='Confirmed_per10000_0322',ascending=False)[['State','Area_Name','POP_ESTIMATE_2018','Confirmed_0322','Confirmed_0323','Confirmed_0324','Confirmed_0325','Confirmed_0326','Confirmed_0327','Confirmed_0328','Confirmed_0329','Confirmed_0330']][8:-1].T
census_pop_density_fips_covid.sort_values(by=cold_100000,ascending=False).head(10).T
census_pop_density_fips_covid[census_pop_density_fips_covid.State=='PA'].sort_values(by=colc,ascending=False).head(15).T
census_pop_density_fips_covid.sort_values(by=colc_10000,ascending=False)['Confirmed_per10000_0330'][6:100].plot(marker='*',use_index=False)
# census_pop_density_fips_covid[census_pop_density_fips_covid.FIPS==16013.0].T
!pip install vega_datasets
import altair as alt

from vega_datasets import data
us_counties = alt.topo_feature(data.us_10m.url, 'counties')

us_counties
# us_counties1= alt.topo_feature('/kaggle/input/countyjson/us10m.json', 'counties')

# us_counties1
# jj=census_pop_density_fips_covid.columns[-9:].tolist()

# jj.append('FIPS')

# print(jj)
covid_data_us_dailytrend_1=census_pop_density_fips_covid[['FIPS','Confirmed_per10000_0322','Confirmed_per10000_0323','Confirmed_per10000_0324','Confirmed_per10000_0325','Confirmed_per10000_0326','Confirmed_per10000_0327','Confirmed_per10000_0328','Confirmed_per10000_0329','Confirmed_per10000_0330']]

covid_data_us_dailytrend_1.columns=['fips','22','23','24','25','26','27','28','29','30']

covid_data_us_dailytrend_1.fips =covid_data_us_dailytrend_1.fips.astype(int)

covid_data_us_dailytrend_1.sort_values(by='28',ascending=False).head()
slider = alt.binding_range(min=22, max=30, step=1)

select_day = alt.selection_single(name="day", fields=['day'],

                                   bind=slider, init={'day': 22})

columns_1=['22','23','24','25','26','27','28','29','30']



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
covid_data_us_dailytrend_1=covid_data_us_dailytrend_1[covid_data_us_dailytrend_1.fips != 36061.0]

covid_data_us_dailytrend_1=covid_data_us_dailytrend_1[covid_data_us_dailytrend_1.fips != 36119.0]

max_val=covid_data_us_dailytrend_1[['22','23','24','25','26','27','28','29','30']].max().max()

covid_data_us_dailytrend_1.loc[covid_data_us_dailytrend_1.fips == 56045.0,'22']=max_val

covid_data_us_dailytrend_1.loc[covid_data_us_dailytrend_1.fips == 56045.0,'23']=max_val

covid_data_us_dailytrend_1.loc[covid_data_us_dailytrend_1.fips == 56045.0,'24']=max_val

covid_data_us_dailytrend_1.loc[covid_data_us_dailytrend_1.fips == 56045.0,'25']=max_val

covid_data_us_dailytrend_1.loc[covid_data_us_dailytrend_1.fips == 56045.0,'26']=max_val

covid_data_us_dailytrend_1.loc[covid_data_us_dailytrend_1.fips == 56045.0,'27']=max_val

covid_data_us_dailytrend_1.loc[covid_data_us_dailytrend_1.fips == 56045.0,'28']=max_val

covid_data_us_dailytrend_1.loc[covid_data_us_dailytrend_1.fips == 56045.0,'29']=max_val

covid_data_us_dailytrend_1.loc[covid_data_us_dailytrend_1.fips == 56045.0,'30']=max_val

covid_data_us_dailytrend_1.tail()
slider = alt.binding_range(min=22, max=30, step=1)

select_day = alt.selection_single(name="day", fields=['day'],

                                   bind=slider, init={'day': 22})

columns_1=['22','23','24','25','26','27','28','29','30']



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