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
!pip install vega_datasets
import altair as alt

import pandas as pd

from vega_datasets import data
nydata=pd.read_csv('/kaggle/input/nytimes-covid19-data/us-counties.csv')

nydata_st=pd.read_csv('/kaggle/input/nytimes-covid19-data/us-states.csv')

nydata_pivot=pd.pivot_table(nydata,values='cases',columns='date',index='fips').reset_index()

nydata_pivot.head()
date_today='2020-04-05'
nydata_st[nydata_st.date==date_today].groupby('state')['cases'].sum().sum()
nydata_st[nydata_st.date==date_today].groupby('state')['cases'].sum().sort_values(ascending=False).head()
nydata_pivot[nydata_pivot.columns[3:]].sum().plot()
date_last_report=30
jj=[]

for i in range(10,date_last_report+1):

    jj.append(str(i))

jj.append('fips')

len(jj)-1
# covid_data_us_dailytrend_1=nydata_pivot[['fips','2020-03-16','2020-03-17','2020-03-18','2020-03-19','2020-03-20','2020-03-21','2020-03-22','2020-03-23','2020-03-24','2020-03-25','2020-03-25','2020-03-27']].sort_values(by='2020-03-27',ascending=False)

covid_data_us_dailytrend_1=nydata_pivot[nydata_pivot.columns[-(date_last_report-9):].tolist()]

covid_data_us_dailytrend_1['fips'] =nydata_pivot['fips'].astype(int)

covid_data_us_dailytrend_1.columns=jj

# covid_data_us_dailytrend_1['fips'] =covid_data_us_dailytrend_1['fips'].astype(int)

covid_data_us_dailytrend_1.sort_values(by=covid_data_us_dailytrend_1.columns[-2],ascending=False).head(10)
covid_data_us_dailynewspread=covid_data_us_dailytrend_1.copy()

for i in range(len(jj)-1):

    if i==0:

        covid_data_us_dailynewspread.loc[covid_data_us_dailynewspread[covid_data_us_dailynewspread.columns[i]]>0,covid_data_us_dailynewspread.columns[i]]=i+1

    else:         

        covid_data_us_dailynewspread.loc[(covid_data_us_dailynewspread[covid_data_us_dailynewspread.columns[i]]>0),covid_data_us_dailynewspread.columns[i]]=i+1

        for j in range(i):

            covid_data_us_dailynewspread.loc[(covid_data_us_dailynewspread[covid_data_us_dailynewspread.columns[j]]==j+1)&(covid_data_us_dailynewspread[covid_data_us_dailynewspread.columns[i]]==i+1),covid_data_us_dailynewspread.columns[i]]=j+1

covid_data_us_dailynewspread.sort_values(by=covid_data_us_dailynewspread.columns[-2],ascending=False).head(10)
# covid_data_us_dailynewspread[str(date_last_report)].value_counts()
covid_data_us_dailynewspread[str(date_last_report)].value_counts().plot(Marker='*',linestyle = 'None')

covid_data_us_dailynewspread[str(date_last_report)].value_counts().sum()
slider = alt.binding_range(min=10, max=date_last_report, step=1)

select_day = alt.selection_single(name="day", fields=['day'],

                                   bind=slider, init={'day': 10})

columns_1=jj[0:-1]
us_counties = alt.topo_feature(data.us_10m.url, 'counties')
alt.Chart(us_counties).mark_geoshape(

    stroke='black',

    strokeWidth=0.05

).project(

    type='albersUsa'

).transform_lookup(

    lookup='id',

    from_=alt.LookupData(covid_data_us_dailynewspread, 'fips', columns_1)

).transform_fold(

    columns_1, as_=['day', 'day_appeared']

).transform_calculate(

    day='parseInt(datum.day)',

    day_appeared='isValid(datum.day_appeared) ? datum.day_appeared : -1'  

).encode(

    color = alt.condition(

        'datum.day_appeared > 0',

        alt.Color('day_appeared:Q', scale=alt.Scale(scheme='plasma')),

        alt.value('#dbe9f6')

    )).add_selection(

    select_day

).properties(

    width=700,

    height=400

).transform_filter(

    select_day

)
covid_data_us_dailynewspread_skip=covid_data_us_dailynewspread[['10','30','fips']].copy()

# covid_data_us_dailynewspread_skip.loc[((covid_data_us_dailynewspread_skip['20']!=1)&(covid_data_us_dailynewspread_skip['20']!=9)),'20']=0

# covid_data_us_dailynewspread_skip.loc[((covid_data_us_dailynewspread_skip['28']!=1)&(covid_data_us_dailynewspread_skip['28']!=9)&(covid_data_us_dailynewspread_skip['28']!=17)),'28']=0

covid_data_us_dailynewspread_skip.loc[((covid_data_us_dailynewspread_skip['30']!=1)&(covid_data_us_dailynewspread_skip['30']!=21)),'30']=0

covid_data_us_dailynewspread_skip['10'].value_counts(),covid_data_us_dailynewspread_skip['30'].value_counts()
slider = alt.binding_range(min=10, max=date_last_report, step=20)

select_day = alt.selection_single(name="day", fields=['day'],

                                   bind=slider, init={'day': 10})

columns_1=['10','30']
alt.Chart(us_counties).mark_geoshape(

    stroke='black',

    strokeWidth=0.05

).project(

    type='albersUsa'

).transform_lookup(

    lookup='id',

    from_=alt.LookupData(covid_data_us_dailynewspread_skip, 'fips', columns_1)

).transform_fold(

    columns_1, as_=['day', 'day_appeared']

).transform_calculate(

    day='parseInt(datum.day)',

    day_appeared='isValid(datum.day_appeared) ? datum.day_appeared : -1'  

).encode(

    color = alt.condition(

        'datum.day_appeared > 0',

        alt.Color('day_appeared:Q', scale=alt.Scale(scheme='plasma')),

        alt.value('#dbe9f6')

    )).add_selection(

    select_day

).properties(

    width=700,

    height=400

).transform_filter(

    select_day

)
covid_data_us_dailynewspread_skip=covid_data_us_dailynewspread[['10','20','30','fips']].copy()

covid_data_us_dailynewspread_skip.loc[((covid_data_us_dailynewspread_skip['20']!=1)&(covid_data_us_dailynewspread_skip['20']!=11)),'20']=0

covid_data_us_dailynewspread_skip.loc[((covid_data_us_dailynewspread_skip['30']!=1)&(covid_data_us_dailynewspread_skip['30']!=11)&(covid_data_us_dailynewspread_skip['30']!=21)),'30']=0

# covid_data_us_dailynewspread_skip.loc[((covid_data_us_dailynewspread_skip['20']!=1)&(covid_data_us_dailynewspread_skip['28']!=17)),'28']=0

covid_data_us_dailynewspread_skip['30'].value_counts()
slider = alt.binding_range(min=10, max=date_last_report, step=10)

select_day = alt.selection_single(name="day", fields=['day'],

                                   bind=slider, init={'day': 10})

columns_1=['10','20','30']
alt.Chart(us_counties).mark_geoshape(

    stroke='black',

    strokeWidth=0.05

).project(

    type='albersUsa'

).transform_lookup(

    lookup='id',

    from_=alt.LookupData(covid_data_us_dailynewspread_skip, 'fips', columns_1)

).transform_fold(

    columns_1, as_=['day', 'day_appeared']

).transform_calculate(

    day='parseInt(datum.day)',

    day_appeared='isValid(datum.day_appeared) ? datum.day_appeared : -1'  

).encode(

    color = alt.condition(

        'datum.day_appeared > 0',

        alt.Color('day_appeared:Q', scale=alt.Scale(scheme='plasma')),

        alt.value('#dbe9f6')

    )).add_selection(

    select_day

).properties(

    width=700,

    height=400

).transform_filter(

    select_day

)