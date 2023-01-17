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
start_date='03-22-2020'

date_today=nydata['date'].tail(1).tolist()[0]

days=(pd.to_datetime(date_today)-pd.to_datetime(start_date)).days

date_today,days
jj=[start_date]

jjint=[]

for i in range(days+1):

    jj=(pd.to_datetime(start_date)+pd.DateOffset(1)).date().isoformat()
nydata[(nydata.fips.isna()) & (nydata.date==date_today)].sort_values(by='county').head()
nydata_pivot=pd.pivot_table(nydata,values='cases',columns='date',index='fips').reset_index()

nydata_pivot.head()
nydata_pivot[nydata_pivot.fips==36119.0]
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