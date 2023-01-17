import pandas as pd
import numpy as np
from IPython.display import display, HTML
import datetime
import io
import requests

pd.set_option('display.max_rows', 220)
def print_rank(df, title, subtitle):
    df2 = df.reset_index()
    if 'index' in df2:
        del df2['index']
    df2.insert(0,'#', df2.index + 1)
    display(HTML('<h2>' + title + '</h2>'))
    display(HTML('<h3>' + subtitle + '</h3>'))
    display(HTML(df2.head(20).to_html(index=False))) 

#oid_df = pd.read_csv('https://covid.ourworldindata.org/data/owid-covid-data.csv',usecols=['iso_code','location','date','population','new_deaths','new_deaths_per_million','new_deaths_smoothed_per_million'])
oid_df = pd.read_csv('../input/covid-sem-sept12/owid-covid-data.csv',usecols=['iso_code','location','date','population','new_deaths','new_deaths_per_million','new_deaths_smoothed_per_million'])
oid_df[oid_df['location']=='Argentina'].tail(7)
rdate = oid_df['date'].max()
lastest_idx = oid_df.groupby(['location'])['date'].transform(max) == oid_df['date']
s_df = oid_df[lastest_idx].sort_values(['new_deaths_smoothed_per_million'], ascending=False).reset_index()[['location','new_deaths','new_deaths_smoothed_per_million']]
print_rank(s_df, 'Muertes diarias por millon al ' + rdate, '(Our World in Data)')
from_date = datetime.datetime.strptime(rdate, '%Y-%m-%d') - datetime.timedelta(days=7)
from_date = from_date.strftime("%Y-%m-%d")
from_date
last_week_df = oid_df[oid_df['date']> from_date].groupby(['location','population']).sum().reset_index()
last_week_df['weekly_new_deaths_per_m'] = last_week_df['new_deaths'] * 1E6 / last_week_df['population']
lastweek2_df = last_week_df[['location','new_deaths','weekly_new_deaths_per_m']].sort_values(['weekly_new_deaths_per_m'],ascending=False)

print_rank(lastweek2_df, 'Muertes por semana por millon','(Our World in Data)')
# source https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data
#jhu_df = pd.read_csv('https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
jhu_df = pd.read_csv('../input/covid-sem-sept12/time_series_covid19_deaths_global.csv')
#jhu_iso_df = pd.read_csv('https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/UID_ISO_FIPS_LookUp_Table.csv')
jhu_iso_df = pd.read_csv('../input/covid-sem-sept12/UID_ISO_FIPS_LookUp_Table.csv')


colcount = len(jhu_df.columns)
jdate = jhu_df.columns[colcount-1]
jhus_df = jhu_df.iloc[:,0:2]
jhus_df['last_week_deaths'] = jhu_df[jhu_df.columns[colcount-1]] - jhu_df[jhu_df.columns[colcount-8]]

labels_df = pd.merge(jhus_df, jhu_iso_df, left_on=['Country/Region','Province/State'], right_on=['Country_Region','Province_State'], how='left')[['iso3','Country_Region','Population','last_week_deaths']]
jhuf_df = labels_df.groupby(['Country_Region']).sum()
jhuf_df['deaths_per_M'] = jhuf_df['last_week_deaths'] * 1E6 /jhuf_df['Population']
print_rank(jhuf_df.sort_values(['deaths_per_M'],ascending=False).head(20), 'Muertes por semana por millon al ' + jdate, '(Johns Hopkins University)')