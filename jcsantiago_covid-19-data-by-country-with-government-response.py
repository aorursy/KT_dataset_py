import pandas as pd 
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from datetime import datetime
from datetime import timedelta
from datetime import date
# 2019 Novel Coronavirus COVID-19 (2019-nCoV) Data Repository by Johns Hopkins CSSE
# https://github.com/CSSEGISandData/COVID-19
urlc = 'https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv?raw=true'
urld = 'https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv?raw=true'
urlr = 'https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv?raw=true'

confirmed_df = pd.read_csv(urlc)
deaths_df = pd.read_csv(urld)
recoveries_df = pd.read_csv(urlr)
url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_country.csv" 
cases_country_df = pd.read_csv(url,parse_dates=['Last_Update'])
print(cases_country_df.shape)
print(cases_country_df.dtypes)
cases_country_df.sample(3)

url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_time.csv"
cases_time_df = pd.read_csv(url,parse_dates=['Last_Update'])
print(cases_time_df.shape)
print(cases_time_df.dtypes)
cases_time_df.loc[cases_time_df.iso3=='ESP'].sample(5)


#csse_covid_19_daily_reports/05-01-2020.csv

url = 'https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_daily_reports/05-01-2020.csv?raw=true'

daily_df = pd.read_csv(url)
print(daily_df.shape)
print(daily_df.dtypes)

daily_df.sample(3)
#cases_time_df.loc[daily_df.iso3=='ESP']

pcol = ["Province/State","Country/Region","Lat","Long"]

confirmed_df = pd.melt(confirmed_df, id_vars=pcol, var_name="Date", value_name="confirmed")
deaths_df = pd.melt(deaths_df, id_vars=pcol, var_name="Date", value_name="deaths")
recoveries_df = pd.melt(recoveries_df, id_vars=pcol, var_name="Date", value_name="recoveries")

confirmed_df['Date'] =  pd.to_datetime(confirmed_df['Date'], format='%m/%d/%y')
deaths_df['Date'] =  pd.to_datetime(deaths_df['Date'], format='%m/%d/%y')
recoveries_df['Date'] =  pd.to_datetime(recoveries_df['Date'], format='%m/%d/%y')

pcol.append("Date")
cvirus_df = pd.merge(confirmed_df, deaths_df, on=pcol, how='left')
cvirus_df = pd.merge(cvirus_df, recoveries_df, on=pcol, how='left')
cvirus_df = cvirus_df.rename(columns={'Province/State':'ProvinceState'
                                      ,'Country/Region':'CountryRegion'})
print(cvirus_df.Date.max())
print(cvirus_df.Date.min())
cvirus_df.loc[(cvirus_df.Date=='2020-01-22') & (cvirus_df.confirmed > 0)]
cvirus_df.loc[cvirus_df.ProvinceState=='Hubei']
# ToDo Data prior to 01/22/2020

# https://docs.google.com/spreadsheets/d/e/2PACX-1vTBI8MZx7aNt8EjYqkeojTopZKwSYGWCSKUzyS9xobrS5Tfr9SQZ_4hrp3dv6bRGkHk2dld0wRrJIeV/pub?gid=32379430&single=true&output=csv
cvirus_df.dtypes
# Test
plot_df = cvirus_df.groupby(['CountryRegion', 'Date']).confirmed.sum().reset_index()
plot_df = plot_df[plot_df['CountryRegion'].isin(['China','Spain','US','Italy'])]

pcol = ["CountryRegion","Date","confirmed"]
plot_df = plot_df[pcol]


# Format Legend
yesterday = pd.to_datetime(date.today() -  pd.to_timedelta(1, unit='d'))
legends_df =plot_df[plot_df.Date == yesterday]
hueOrder = legends_df.sort_values('confirmed', ascending=0)['CountryRegion']
legends = legends_df['CountryRegion'].str.cat(
    legends_df['confirmed'].apply(lambda x : '{0:,}'.format(x)),sep=" ")
legends = legends.sort_values(ascending=False).reset_index()

plt.figure(figsize=(10,5))
chart  = sns.lineplot(data=plot_df, x='Date', y='confirmed',hue='CountryRegion', hue_order=hueOrder)

plt.legend(legends['CountryRegion'], title='Countries')

# Define the date format
date_form = DateFormatter("%m-%d")
chart.xaxis.set_major_formatter(date_form)

# Ensure a major tick for each week using (interval=1) 
chart.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))

pcol = ["CountryRegion","Date","confirmed","deaths","recoveries"]
plot_df = cvirus_df[pcol]
plot_df = plot_df[plot_df['CountryRegion'].isin(['China','Spain','US','Italy'])]
plot_df = plot_df.groupby(['CountryRegion', 'Date']).sum().reset_index()

pcol = ["CountryRegion","Date"]
plot_df = pd.melt(plot_df, id_vars=pcol, var_name="type", value_name="cases")

plt.figure(figsize=(20,5))

def qqplot(x, y, **kwargs):
    chart = sns.lineplot(x, y, **kwargs)
    date_form = DateFormatter("%d")
    chart.xaxis.set_major_formatter(date_form)
    # Ensure a major tick for each week using (interval=1) 
    chart.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    chart
    
g = sns.FacetGrid(plot_df, col="type", hue='CountryRegion')
#g = g.map(sns.lineplot,'Date','cases').add_legend()
g = g.map(qqplot,'Date','cases').add_legend()
countries_df = pd.read_csv('../input/country-codes/Country_codes.csv', keep_default_na=False, na_values=[''])
countries_df.sample(3)
countries_df = countries_df.groupby(['Name','Alpha2Code','Alpha3Code']).size().reset_index()[['Name','Alpha2Code','Alpha3Code']]
countries_df = countries_df.rename(columns={'Name':'CountryRegion'})
cvirus_country_df = pd.DataFrame(cvirus_df['CountryRegion'].unique())
cvirus_country_df = cvirus_country_df.rename(columns={0:'CountryRegion'})

cvirus_country_no_def_df = cvirus_country_df.merge(countries_df,how='left',on='CountryRegion')
cvirus_country_no_def_df[cvirus_country_no_def_df.Alpha2Code.isnull()].CountryRegion
cvirus_df.loc[cvirus_df.CountryRegion == 'Taiwan*', 'CountryRegion'] = 'Taiwan'
cvirus_country_no_def_df = cvirus_df.merge(countries_df,how='left',on='CountryRegion')
cvirus_country_no_def_df.loc[cvirus_country_no_def_df.Alpha2Code.isnull(), ['ProvinceState','CountryRegion']].drop_duplicates()

cvirus_df.drop(cvirus_df[cvirus_df.CountryRegion.isnull()].index , inplace=True)

pcol = ["CountryRegion","Date","confirmed","deaths","recoveries"]
cvirus_country_df = cvirus_df[pcol]
cvirus_country_df = cvirus_country_df.merge(countries_df,how='left',on='CountryRegion')

pcol = ["CountryRegion","Alpha3Code","Date","confirmed","deaths","recoveries"]
cvirus_country_df = cvirus_country_df[pcol]
cvirus_country_df = cvirus_country_df.rename(columns={"CountryRegion":'Country'
                                           ,"Alpha3Code":'CountryAlpha3Code'
                                           })

cvirus_country_df = cvirus_country_df.groupby(['Country', 'CountryAlpha3Code', 'Date']).sum().reset_index()

top_country = cvirus_country_df.groupby(['CountryAlpha3Code']).confirmed.max().reset_index()
top_country = top_country.sort_values(by='confirmed', ascending=False)
top_country = top_country['CountryAlpha3Code'].tolist()
plot_df = cvirus_country_df[cvirus_country_df['CountryAlpha3Code'].isin(top_country[0:10])]
pcol = ["Country","Date","confirmed"]
plot_df = plot_df[pcol]

# Format Legend
yesterday = pd.to_datetime(date.today() -  pd.to_timedelta(1, unit='d'))
legends_df = plot_df[plot_df.Date == yesterday]
legends_df = legends_df.sort_values(['confirmed'], ascending=[False]).reset_index()
hueOrder = legends_df.sort_values('confirmed', ascending=0)['Country']
legends = legends_df['Country'].str.cat(
    legends_df['confirmed'].apply(lambda x : '{0:,}'.format(x)),sep=" ")

plt.figure(figsize=(7,5))
chart  = sns.lineplot(data=plot_df, x='Date', y='confirmed',hue='Country', hue_order=hueOrder)
plt.legend(legends, title='Countries')


# Define the date format
date_form = DateFormatter("%m-%d")
chart.xaxis.set_major_formatter(date_form)

# Ensure a major tick for each week using (interval=1) 
chart.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
plot_df = cvirus_country_df[cvirus_country_df['CountryAlpha3Code'].isin(top_country[0:10])]
pcol = ["Country","Date","deaths"]
plot_df = plot_df[pcol]

# Format Legend
yesterday = pd.to_datetime(date.today() -  pd.to_timedelta(1, unit='d'))
legends_df = plot_df[plot_df.Date == yesterday]
hueOrder = legends_df.sort_values('deaths', ascending=0)['Country']
legends_df = legends_df.sort_values(['deaths'], ascending=[False]).reset_index()

legends = legends_df['Country'].str.cat(
    legends_df['deaths'].apply(lambda x : '{0:,}'.format(x)),sep=" ")

plt.figure(figsize=(7,5))
chart  = sns.lineplot(data=plot_df, x='Date', y='deaths',hue='Country', hue_order=hueOrder)
plt.legend(legends, title='Countries')

# Define the date format
date_form = DateFormatter("%m-%d")
chart.xaxis.set_major_formatter(date_form)

# Ensure a major tick for each week using (interval=1) 
chart.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
alldays = pd.Series(data=pd.date_range(start=cvirus_df['Date'].min(), end=cvirus_df['Date'].max(), freq='d'))
allcountries = pd.Series(cvirus_country_df['CountryAlpha3Code'].unique())
indexAll = pd.MultiIndex.from_product([allcountries, alldays], names = ["CountryAlpha3Code", "Date"])
indexAll_df = pd.DataFrame(index = indexAll).reset_index()
indexDf = pd.MultiIndex.from_arrays([cvirus_country_df[col] for col in ['CountryAlpha3Code', 'Date']])

f = indexAll.isin(indexDf)
indexNoExists_df = indexAll_df[~f]
indexNoExists_df

cvirus_country_ant_df = cvirus_country_df.copy()
cvirus_country_ant_df.Date += timedelta(days=1)

cvirus_country_merge_df = pd.merge(cvirus_country_df, cvirus_country_ant_df,  how='left', on=['CountryAlpha3Code', 'Date'])

cvirus_country_merge_df = cvirus_country_merge_df.fillna(0)
cvirus_country_merge_df = cvirus_country_merge_df.reindex()

cvirus_country_merge_df.sample(5)
cvirus_country_inc_df = cvirus_country_merge_df[['CountryAlpha3Code', 'Date','confirmed_x','confirmed_y','deaths_x','deaths_y','recoveries_x','recoveries_y']]

cvirus_country_inc_df['confirmed_inc'] = cvirus_country_inc_df.apply(lambda row: row.confirmed_x-row.confirmed_y, axis=1)
cvirus_country_inc_df['deaths_inc'] = cvirus_country_inc_df.apply(lambda row: row.deaths_x-row.deaths_y, axis=1)
cvirus_country_inc_df['recoveries_inc'] = cvirus_country_inc_df.apply(lambda row: row.recoveries_x-row.recoveries_y, axis=1)

pcols = ['CountryAlpha3Code', 'Date','confirmed_inc','deaths_inc','recoveries_inc']
cvirus_country_inc_df = cvirus_country_inc_df[pcols]

cvirus_country_df = pd.merge(cvirus_country_df, cvirus_country_inc_df,  how='left', on=['CountryAlpha3Code', 'Date'])
cvirus_country_df.sample()
plot_df = cvirus_country_df[cvirus_country_df['CountryAlpha3Code'].isin(top_country[0:5])]
pcol = ["Country","Date","confirmed_inc"]
plot_df = plot_df[pcol]

# Legend
yesterday = pd.to_datetime(date.today() -  pd.to_timedelta(1, unit='d'))
legends_df = plot_df[plot_df.Date == yesterday]
hueOrder = legends_df.sort_values('confirmed_inc', ascending=0)['Country']
legends_df = legends_df.sort_values(['confirmed_inc'], ascending=[False]).reset_index()

legends = legends_df['Country'].str.cat(
    legends_df['confirmed_inc'].apply(lambda x : '{0:,}'.format(x)),sep=" ")

plt.figure(figsize=(10,5))
chart  = sns.lineplot(data=plot_df, x='Date', y='confirmed_inc',hue='Country',hue_order=hueOrder)
plt.legend(legends, title='Countries')


# Define the date format
date_form = DateFormatter("%m-%d")
chart.xaxis.set_major_formatter(date_form)

# Ensure a major tick for each week using (interval=1) 
chart.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
#cvirus_country_df.drop(['ECR'], axis=1, inplace=True)

cvirus_country_sig_df = cvirus_country_df.copy()
cvirus_country_sig_df.Date -= timedelta(days=1)

cvirus_country_merge_df = pd.merge(cvirus_country_df, cvirus_country_sig_df,  how='left', on=['CountryAlpha3Code', 'Date'])

cvirus_country_merge_df = cvirus_country_merge_df.fillna(0)
cvirus_country_merge_df = cvirus_country_merge_df.reindex()

cvirus_country_f_df = cvirus_country_merge_df[['CountryAlpha3Code', 'Date','confirmed_x','confirmed_y']]

# Empirical Contagion Rate
cvirus_country_f_df['ECR'] = cvirus_country_f_df.apply(lambda row: 0 if (row.confirmed_x == 0) or (row.confirmed_y == 0) else (row.confirmed_y/row.confirmed_x)-1, axis=1)

pcols = ['CountryAlpha3Code','Date','ECR']
cvirus_country_f_df = cvirus_country_f_df[pcols]

cvirus_country_df = pd.merge(cvirus_country_df, cvirus_country_f_df,  how='left', on=['CountryAlpha3Code', 'Date'])
plot_df = cvirus_country_df[cvirus_country_df['CountryAlpha3Code'].isin(top_country[0:5])]
pcol = ["Country","Date","ECR"]
plot_df = plot_df[pcol]

plt.figure(figsize=(10,5))
chart  = sns.lineplot(data=plot_df, x='Date', y='ECR',hue='Country')

# Define the date format
date_form = DateFormatter("%m-%d")
chart.xaxis.set_major_formatter(date_form)

# Ensure a major tick for each week using (interval=1) 
chart.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
last15days = date.today() -  pd.to_timedelta(15, unit='d')
lastdays = date.today() -  pd.to_timedelta(1, unit='d')

plot_df = cvirus_country_df[(cvirus_country_df['CountryAlpha3Code'].isin(top_country[0:5]))
                               & (cvirus_country_df['Date'] > last15days)
                               & (cvirus_country_df['Date'] < lastdays)]

pcol = ["Country","Date","ECR"]
plot_df = plot_df[pcol]

plt.figure(figsize=(10,5))
chart  = sns.lineplot(data=plot_df, x='Date', y='ECR',hue='Country')

url = 'https://oxcgrtportal.azurewebsites.net/api/CSVDownload'

# Government Response Tracker
GRT_df = pd.read_csv(url)
GRT_df.tail(5)
GRT_df.dtypes
pcol = ['CountryCode', 'Date', 'StringencyIndexForDisplay']
GRT_df = GRT_df[pcol]
GRT_df = GRT_df.rename(
    columns={'CountryCode':'CountryAlpha3Code'
             ,'StringencyIndexForDisplay':'GRTStringencyIndex'             
             })
def intToDate(x):
    strX = str(x)
    y = strX[0:4]
    m = strX[4:6]
    d = strX[6:8]
    strX = y+'/'+m+'/'+d
    return datetime.strptime(strX, '%Y/%m/%d')

GRT_df['Date'] = GRT_df.apply(lambda row: intToDate(row['Date']), axis=1)

cvirus_country_df = pd.merge(cvirus_country_df, GRT_df,  how='left', on=['CountryAlpha3Code', 'Date'])
cvirus_country_df.dtypes



GRT_df.dtypes

plot_df = pd.concat([cvirus_country_df[cvirus_country_df['CountryAlpha3Code'].isin(top_country[0:5])]
                     ,cvirus_country_df[(cvirus_country_df['CountryAlpha3Code']=='CHN')]])

pcol = ["Country","Date","GRTStringencyIndex"]
plot_df = plot_df[pcol]

plt.figure(figsize=(10,5))
chart  = sns.lineplot(data=plot_df, x='Date', y='GRTStringencyIndex',hue='Country')

# Define the date format
date_form = DateFormatter("%m-%d")
chart.xaxis.set_major_formatter(date_form)

# Ensure a major tick for each week using (interval=1) 
chart.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
url = 'https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv'

# Government Response Tracker
google_df = pd.read_csv(url)
# pip install pandas-profiling

# profile = google_df.profile_report(title='Pandas Profiling Report')
# profile.to_file(output_file="fifa_pandas_profiling.html")
print(google_df.shape)
print(google_df.dtypes)
google_df.sample()
firstday1confirmed_df = cvirus_country_df[cvirus_country_df.confirmed>0].groupby(['CountryAlpha3Code']).Date.min()
firstday100confirmed_df = cvirus_country_df[cvirus_country_df.confirmed>100].groupby(['CountryAlpha3Code']).Date.min()

cvirus_country_df = pd.merge(cvirus_country_df, firstday1confirmed_df,  how='left', on='CountryAlpha3Code', suffixes=('', '_1'))
cvirus_country_df = pd.merge(cvirus_country_df, firstday100confirmed_df,  how='left', on='CountryAlpha3Code', suffixes=('', '_100'))


maxDay = cvirus_country_df.Date.max()+np.timedelta64(1, 'D')

cvirus_country_df.loc[cvirus_country_df.Date_1.isnull(), 'Date_1'] = maxDay
cvirus_country_df.loc[cvirus_country_df.Date_100.isnull(), 'Date_100'] = maxDay

cvirus_country_df['DaysSince1Cases'] = ((cvirus_country_df['Date'] - cvirus_country_df['Date_1']) 
                               / np.timedelta64(1, 'D'))

cvirus_country_df['DaysSince100Cases'] = ((cvirus_country_df['Date'] - cvirus_country_df['Date_100']) 
                                 / np.timedelta64(1, 'D'))
cvirus_country_df.drop(columns=['Date_1', 'Date_100'], inplace=True)
cvirus_country_df.DaysSince1Cases = cvirus_country_df.DaysSince1Cases.astype(int)
cvirus_country_df.DaysSince100Cases = cvirus_country_df.DaysSince100Cases.astype(int)
plot_df = pd.concat([cvirus_country_df[cvirus_country_df['CountryAlpha3Code'].isin(top_country[0:5])]
                     ,cvirus_country_df[(cvirus_country_df['CountryAlpha3Code']=='CHN')]])

# Delete data older than the first 100 cases
plot_df.drop(plot_df[plot_df.DaysSince100Cases<0].index, inplace=True)

pcol = ["Country","DaysSince100Cases","confirmed"]
plot_df = plot_df[pcol]

plt.figure(figsize=(10,5))
sns.lineplot(data=plot_df, x='DaysSince100Cases', y='confirmed',hue='Country')
import urllib.request
import os
import zipfile
import shutil

#os.listdir('..')

# if exists temporary directory is removed
if os.path.exists('../tmp'):
    shutil.rmtree('../tmp')

os.makedirs('../tmp') 

print('Beginning file download with urllib2...')

url = 'http://api.worldbank.org/v2/en/indicator/SP.POP.TOTL?downloadformat=csv'
urllib.request.urlretrieve(url,'../tmp/population.zip')
    
# Unzip the files 
with zipfile.ZipFile('../tmp/population.zip',"r") as z:
    z.extractall("../tmp")

filePopulation = '../tmp/' + [file for file in os.listdir('../tmp') if file.startswith('API_SP.POP')][0]

print(filePopulation)

# The first 4 rows of the file are deleted
with open(filePopulation, "r+") as f:
    d = f.readlines()
    f.seek(0)
    for i in d[4:]:
        f.write(i)
    f.truncate()
    
# Country Population
CountryPopulation_df = pd.read_csv(filePopulation, keep_default_na=False, na_values=[''])

# Temporary directory is removed
shutil.rmtree('../tmp')

CountryPopulation_df.sample()

CountryPopulation_df.drop(columns=["Country Name","Indicator Name","Indicator Code"], inplace=True)
CountryPopulation_df = pd.melt(CountryPopulation_df
                               , id_vars=["Country Code"]
                               , var_name="UpdateYear"
                               , value_name="Population")
CountryPopulation_df = CountryPopulation_df.rename(columns={"Country Code":'CountryAlpha3Code'
                                           })

# Records are deleted without defined data
CountryPopulation_df.drop(CountryPopulation_df[CountryPopulation_df.Population.isnull()].index, inplace=True)
CountryPopulation_df.UpdateYear = CountryPopulation_df.UpdateYear.astype(int)

# The last year is selected with data
maxYear_df = CountryPopulation_df.groupby(["CountryAlpha3Code"]).UpdateYear.max().reset_index()
CountryPopulation_df = pd.merge(maxYear_df, CountryPopulation_df,  how='inner', on=["CountryAlpha3Code","UpdateYear"])
CountryPopulation_df.drop(columns={'UpdateYear'}, inplace=True)

# it is verified that there is only one registry per country
print(CountryPopulation_df.shape)
print(CountryPopulation_df.CountryAlpha3Code.unique().size)

CountryPopulation_df.sample()
CountryPopulation_df.to_csv('covid19_country_population.csv',index=False)
#CountryPopulation_df = pd.read_csv('../input/covid19-by-country-with-government-response/covid19_country_population.csv')
cvirus_population_df = cvirus_country_df[['CountryAlpha3Code','Date','confirmed','deaths','recoveries']]
cvirus_population_df = pd.merge(cvirus_population_df, CountryPopulation_df, on='CountryAlpha3Code', how='left')

cvirus_population_df['confirmed_PopPct'] = (cvirus_population_df['confirmed'] * 100) / cvirus_population_df['Population']
cvirus_population_df['deaths_PopPct'] = (cvirus_population_df['deaths'] * 100) / cvirus_population_df['Population']
cvirus_population_df['recoveries_PopPct'] = (cvirus_population_df['recoveries'] * 100) / cvirus_population_df['Population']

cvirus_population_df.drop(columns={'confirmed','deaths','recoveries','Population'}, inplace=True)

cvirus_population_df.sample()

cvirus_country_df = pd.merge(cvirus_country_df, cvirus_population_df, on=['CountryAlpha3Code','Date'], how='left')
plot_df = cvirus_country_df[cvirus_country_df['CountryAlpha3Code'].isin(top_country[0:10])]
pcol = ["Country","DaysSince100Cases","confirmed_PopPct"]
plot_df = plot_df[pcol]

# Delete data older than the first 100 cases
plot_df.drop(plot_df[plot_df.DaysSince100Cases<0].index, inplace=True)

plt.figure(figsize=(7,5))
chart  = sns.lineplot(data=plot_df, x='DaysSince100Cases', y='confirmed_PopPct',hue='Country')
top_country_percent = cvirus_country_df.groupby(['CountryAlpha3Code']).confirmed_PopPct.max().reset_index()
top_country_percent = top_country_percent.sort_values(by='confirmed_PopPct', ascending=False)
top_country_percent = top_country_percent['CountryAlpha3Code'].tolist()
plot_df = cvirus_country_df[cvirus_country_df['CountryAlpha3Code'].isin(top_country_percent[0:10])]
pcol = ["Country","DaysSince1Cases","confirmed_PopPct","deaths_PopPct"]
plot_df = plot_df[pcol]

plot_df.drop(plot_df[plot_df.DaysSince1Cases<0].index, inplace=True)

fig, axs = plt.subplots(2, figsize=(7,10))

plt.figure(figsize=(7,5))
sns.lineplot(data=plot_df, x='DaysSince1Cases', y='confirmed_PopPct',hue='Country', ax=axs[0])
sns.lineplot(data=plot_df, x='DaysSince1Cases', y='deaths_PopPct',hue='Country', ax=axs[1])
print(cvirus_country_df.shape)
cvirus_country_df.dtypes
cvirus_country_df.sample()
cvirus_country_df.columns[cvirus_country_df.isna().any()]
cvirus_country_df[cvirus_country_df.isnull().any(axis=1)]
cvirus_country_df.to_csv('covid19_by_country.csv',index=False)
print(cvirus_country_df.Date.max())
print(cvirus_df.Date.max())