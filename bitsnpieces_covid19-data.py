from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import seaborn as sns

from pandas.api.types import is_numeric_dtype

import scipy.stats

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import requests



# show full dataframe

pd.set_option('max_colwidth', -1)
def clean_country_names(df):

    df['country'] = [c.strip().replace('*','') for c in df['country']]

    df.loc[df['country']=='Czech Republic','country'] = 'Czechia'

    df.loc[df['country']=='Iran, Islamic Republic of','country'] = 'Iran'

    df.loc[df['country']=='Korea (South)','country'] = 'South Korea'

    df.loc[df['country']=='Korea, South','country'] = 'South Korea'

    df.loc[df['country']=='Russian Federation','country'] = 'Russia'

    df.loc[df['country']=='United States of America','country'] = 'United States'

    df.loc[df['country'] == 'US','country'] = 'United States'

    df.loc[df['country']=='Mainland China','country'] = 'China'

    df.loc[df['country'] == 'Taiwan, Republic of China','country'] = 'Taiwan'

    df.loc[df['country'] == 'Tanzania, United Republic of','country'] = 'Tanzania'

    df.loc[df['country'] == 'Viet Nam','country'] = 'Vietnam'

    df.loc[df['country'] == 'Venezuela (Bolivarian Republic)','country'] = 'Venezuela'

    df.loc[df['country'] == 'Syrian Arab Republic (Syria)','country'] = 'Syria'

    df.loc[df['country'] == 'Saint Vincent and Grenadines','country'] = 'Saint Vincent and the Grenadines'

    df.loc[df['country'] == 'Brunei Darussalam','country'] = 'Brunei'

    df.loc[df['country'] == 'Cape Verde','country'] = 'Cabo Verde'

    df.loc[df['country'] == 'Congo, (Kinshasa)','country'] = 'Congo (Kinshasa)'

    df.loc[df['country'] == "Côte d'Ivoire",'country'] = "Cote d'Ivoire"

    # df.loc[df['country'] == ''] = 'Diamond Princess'  # cruise ship!

    df.loc[df['country'] == 'Swaziland','country'] = 'Eswatini'

    df.loc[df['country'] == 'Holy See (Vatican City State)','country'] = 'Holy See'

    df.loc[df['country'] == 'Lao PDR','country'] = 'Laos'

    df.loc[df['country'] == 'Macedonia, Republic of','country'] = 'North Macedonia'
def scatter_plot(df, x, y, title, show_ranks=True, show_countries=['Qatar','Italy','Spain','United States','South Korea', 'Taiwan', 'Kenya']):

    # basic plot

#     x = "covid_deaths_per_100000"

#     y = "h1n1_Deaths_per_100000"

    df = df.dropna(subset=[x, y])

    p1=sns.regplot(data=df, x=x, y=y, fit_reg=False, marker="o", color="skyblue", scatter_kws={'s':50})

    p1.set_title(title)

    

    # highlighting countries

    df_highlight = df[df['country'].isin(show_countries)]

    sns.regplot(data=df_highlight, x=x, y=y, fit_reg=False, marker="o", color="red", scatter_kws={'s':50})

    # add annotations one by one with a loop

    for i in range(df_highlight.shape[0]):

        p1.text(df_highlight[x].iloc[i]+0.2, df_highlight[y].iloc[i], df_highlight['country'].iloc[i], horizontalalignment='left', size='medium', color='black', weight='semibold')

    

def scatter_plot_ranks(df, x, y, title):

    df_ranks = df[['country',x,y]]

    df_ranks.loc[:,x] = df_ranks[x].rank()

    df_ranks.loc[:,y] = df_ranks[y].rank()

    scatter_plot(df_ranks, x, y, title+'_ranks')

        

# y_col = 'age_over_65_years_percent'

# scatter_plot_ranks(df_corr, x_col, y_col, x_col + ' vs ' + y_col)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# https://www.kaggle.com/covid19

# https://www.kaggle.com/bitsnpieces/covid19-country-data

# COVID-19 datasets challenge - https://www.kaggle.com/data/139140



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

# Any results you write to the current directory are saved as output.
# data sources

df_sources = pd.read_csv('/kaggle/input/covid19-country-data/covid19_data - data_sources.csv')

df_sources = df_sources[df_sources['Name'] != 'airport_traffic']

#df_sources[df_sources['Name'] == 'health']['Source'].tolist()

df_sources


country_names = []

df_names = pd.read_csv('/kaggle/input/covid19-country-data/country_names_covid19_forecast.csv')

df_names.columns = ['country']

clean_country_names(df_names)

# df_names.loc[df_names['country'] == 'Korea, South', 'country'] = 'South Korea'

df_names

df_latlong = pd.read_csv('/kaggle/input/covid19-country-data/covid19_data - lat_long.csv')

df_latlong.columns = [ c.replace('country_name','country') for c in df_latlong.columns ]

df_latlong = df_latlong[['country','latitude', 'longitude']]

df_latlong
# male to female ratio

df_sex = pd.read_csv('/kaggle/input/covid19-country-data/covid19_data - sex.csv')

df_sex.columns = ['sex_male_to_female_' + c.replace('–', '_').replace(' ', '_') for c in df_sex.columns ]

# df_sex.columns = [ c.replace('Country/region', 'country') for c in df_sex.columns ]

df_sex.columns = [c.replace('sex_male_to_female_Country/region','country') for c in df_sex.columns ]

df_sex['country'] = [c.strip() for c in df_sex['country'] ]

df_sex.loc[df_sex['country'] == 'Korea, South', 'country'] = 'South Korea'

df_sex

# age

df_age = pd.read_csv('/kaggle/input/covid19-country-data/covid19_data - age.csv')

df_age.columns = [c.replace('Country','country').replace('_years','_years_percent') for c in df_age.columns]

df_age['country'] = [c.strip() for c in df_age['country']]

df_age['age_0_to_14_years_percent'] = [float(x.replace('\xa0%','').replace('%','').strip()) for x in df_age['age_0_to_14_years_percent'].tolist()]

df_age['age_15_to_64_years_percent'] = [float(x.replace('\xa0%','').replace('%','').strip()) for x in df_age['age_15_to_64_years_percent'].tolist()]

df_age['age_over_65_years_percent'] = [float(x.replace('\xa0%','').replace('%','').strip()) for x in df_age['age_over_65_years_percent'].tolist()]

df_age
# 2009 H1N1 pandemic

df_h1n1 = pd.read_csv('/kaggle/input/covid19-country-data/covid19_data - 2009_flu_pandemic.csv')

df_h1n1.columns = ['h1n1_' + c for c in df_h1n1.columns]

df_h1n1.columns = [c.replace('h1n1_Country','country').replace('_clean','') for c in df_h1n1.columns]

df_h1n1
# convert to int

df_h1n1['h1n1_Cases_confirmed'] = [int(v.replace(',','')) for v in df_h1n1['h1n1_Cases_confirmed'].tolist()]

df_h1n1['h1n1_Deaths_confirmed'] = [int(v.replace(',','')) for v in df_h1n1['h1n1_Deaths_confirmed'].tolist()]

df_h1n1.astype({'h1n1_Cases_confirmed': 'int64', 'h1n1_Deaths_confirmed':'int64'})

df_h1n1
# flu / pneumonia deaths

df_death = pd.read_csv('/kaggle/input/covid19-country-data/covid19_data - flu_pneumonia_death.csv')

df_death.columns = [c.replace('Country','country').replace('Rate','Flu_pneumonia_death_rate') for c in df_death.columns]

df_death = df_death[['country','Flu_pneumonia_death_rate_per_100000']]

df_death
# population

df_population = pd.read_csv('/kaggle/input/covid19-country-data/covid19_data - population.csv')

df_population.columns = [c.replace('Country','country') for c in df_population.columns]

df_population = df_population[['country','Population_2020','Density_KM2m','Fertility_rate','Median_age','Urban_pop_pct']]

df_population['Population_2020'] = [int(v.replace(',','')) for v in df_population['Population_2020']]

df_population
# air_traffic

df_air = pd.read_csv('/kaggle/input/covid19-country-data/covid19_data - airport_traffic_world.csv')

df_air.columns = [c.replace('Country Name','country') for c in df_air.columns]

df_air = df_air[['country','airport_traffic_2018_thousands']]



df_air
# hospital

df_hospital = pd.read_csv('/kaggle/input/covid19-country-data/covid19_data - hospital_beds.csv')

df_hospital.columns = [c.replace('Country/territory','country') for c in df_hospital.columns]

df_hospital = df_hospital[['country','hosp_beds_per_1000_2017','ICU-CCB_beds_per_100000']]

df_hospital['country'] = [c.strip() for c in df_hospital['country']]

df_hospital['ICU-CCB_beds_per_100000'] = [float(s2[:s2.find('[')]) for s2 in [str(s).strip()+'[' for s in df_hospital['ICU-CCB_beds_per_100000']] ]

df_hospital

#df_hospital['country'].tolist()
# health

df_health = pd.read_csv('/kaggle/input/covid19-country-data/covid19_data - health.csv')

df_health.columns = [c.replace('Country','country') for c in df_health.columns]

df_health = df_health[['country','Health_Care_Index']]

df_health
# property_prices

df_property = pd.read_csv('/kaggle/input/covid19-country-data/covid19_data - property_prices.csv')

df_property.columns = [ 'property_' + c for c in df_property ]

df_property.columns = [ c.replace('property_Country','country').replace(' ', '_') for c in df_property ]

df_property = df_property[['country','property_Affordability_Index']]

df_property
# gdp

df_gdp = pd.read_csv('/kaggle/input/covid19-country-data/covid19_data - gdp.csv')

del df_gdp['Rank']

df_gdp.columns = ['country','gdp_usd_million']

df_gdp['gdp_usd_million'] = [int(v.replace(',','').strip()) for v in df_gdp['gdp_usd_million']]

df_gdp
# covid

import requests

import io



def get_df_from_url(url):

    s = requests.get(url).content

    return pd.read_csv(io.StringIO(s.decode('utf-8')))



covid_url_prefix = 'https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/'

df_covid_confirmed = get_df_from_url(covid_url_prefix + 'time_series_covid19_confirmed_global.csv')

df_covid_deaths = get_df_from_url(covid_url_prefix + 'time_series_covid19_deaths_global.csv')

df_covid_recovered = get_df_from_url(covid_url_prefix + 'time_series_covid19_recovered_global.csv')



# for d in [df_covid_confirmed, df_covid_deaths, df_covid_recovered]:

#     d.columns = [c.replace('Country/Region','country') for c in d.columns]

latest_covid_dt = df_covid_recovered.columns[-1]

df_covid_confirmed = df_covid_confirmed[['Country/Region',latest_covid_dt]]

df_covid_confirmed.columns = ['country', 'covid_confirmed_'+latest_covid_dt.replace('/','_')]

df_covid_deaths = df_covid_deaths[['Country/Region',latest_covid_dt]]

df_covid_deaths.columns = ['country', 'covid_deaths_'+latest_covid_dt.replace('/','_')]

df_covid_recovered = df_covid_recovered[['Country/Region',latest_covid_dt]]

df_covid_recovered.columns = ['country', 'covid_recovered_'+latest_covid_dt.replace('/','_')]





df_covid_confirmed = df_covid_confirmed.groupby('country').sum().reset_index()

df_covid_deaths = df_covid_deaths.groupby('country').sum().reset_index()

df_covid_recovered = df_covid_recovered.groupby('country').sum().reset_index()



df_covid_latest = pd.merge(pd.merge(df_covid_confirmed, df_covid_deaths, how='left'), df_covid_recovered, how='left')

df_covid_latest
print(f'COVID-19 was last retrieved on {latest_covid_dt} (MM/dd/yy) from CSSEGISandData')
# covid_tests - NOTE: Different covid_test_dates for each country!!!!

df_covid_tests = pd.read_csv('/kaggle/input/covid19-country-data/covid19_data - covid_tests.csv')

df_covid_tests = df_covid_tests[df_covid_tests['Entity'].str.contains('Australia - ') == False]

df_covid_tests['Date'] = pd.to_datetime(df_covid_tests['Date'])

max_covid_test_dt = max(df_covid_tests['Date'])

df_covid_tests = df_covid_tests[df_covid_tests['Entity'].str.contains('United States - ') == False]

df_covid_tests['country'] = [e.split(' - ')[0] for e in df_covid_tests['Entity'] ]



max_covid_test_dt = df_covid_tests.groupby('country').max().reset_index()[['country','Date']]

sum_covid_test = df_covid_tests.groupby('country').sum().reset_index()



df_covid_tests = pd.merge(max_covid_test_dt, sum_covid_test)



df_covid_tests.columns = ['country', 'covid_test_date', 'covid_tests']

df_covid_tests

# covid19 dataset https://www.kaggle.com/vignesh1694/covid19-coronavirus

from datetime import datetime



# TODO: get data from live feed

# https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv

# time_series_covid19_deaths_global.csv

# time_series_covid19_recovered_global.csv

df_covid = pd.read_csv('/kaggle/input/covid19-coronavirus/2019_nCoV_data.csv')

df_covid['country'] = df_covid['Country']

clean_country_names(df_covid)

df_covid['Date_ts'] = [datetime.strptime(d, '%m/%d/%Y %H:%M') for d in df_covid['Date'].tolist()]

del df_covid['Country']

# df_covid.loc[df_covid['Country'] == 'US','Country'] = 'United States'

# df_covid.loc[df_covid['Country'] == 'Mainland China','Country'] = 'China'

# df_covid.loc[df_covid['Country'] == 'Korea, South','Country'] = 'South Korea'

df_covid



#datetime.strptime(,'%M/%d/%Y %H:%M')



df_covid_first_dt = df_covid.loc[df_covid.groupby(['country'])['Date_ts'].idxmin()]

df_covid_first_dt.columns = [c.replace('Date_ts','covid19_first_date') for c in df_covid_first_dt.columns]

df_covid_first_dt = df_covid_first_dt[['country','covid19_first_date']]

df_covid_first_dt
# school closures

df_school = pd.read_csv('/kaggle/input/covid19-country-data/covid19_data - school_closures.csv')

df_school.columns = [c.replace('Country','country') for c in df_school.columns]

clean_country_names(df_school)

df_school['country'] = [c.split(',')[0] for c in df_school['country']]

df_school['Date'] = pd.to_datetime(df_school['Date'],dayfirst=True)

# df_school = df_school[df_school['Scale']=='National']

df_school = df_school[['country','Date']].groupby('country').min().reset_index()

df_school.columns = ['country','first_school_closure_date']

df_school[df_school['country']=='Italy']

# country codes

df_country_code = pd.read_csv('/kaggle/input/country-code/country_code.csv')

df_country_code.columns = [ c.replace('Country_name','country') for c in df_country_code.columns ]

clean_country_names(df_country_code)

# df_country_code.loc[df_country_code['country']=='Czech Republic','country'] = 'Czechia'

# df_country_code.loc[df_country_code['country']=='Iran, Islamic Republic of','country'] = 'Iran'

# df_country_code.loc[df_country_code['country']=='Korea (South)','country'] = 'South Korea'

# df_country_code.loc[df_country_code['country']=='Russian Federation','country'] = 'Russia'

# df_country_code.loc[df_country_code['country']=='United States of America','country'] = 'United States'



df_country_code[df_country_code['code_3digit'].isin(['CZE','USA', 'IRN', 'KOR', 'RUS', 'SGP', 'CIV'])]

# precipitation

df_precipitation = pd.read_excel('/kaggle/input/world-bank-climate-change-data/historical-data-excel-380-kb-.xls', sheet_name='Country_precipitationCRU')

df_precipitation.columns = [ c.replace('ISO_3DIGIT','code_3digit') for c in df_precipitation.columns ]

df_precipitation = pd.merge(df_country_code, df_precipitation, how='left')

del df_precipitation['Unnamed: 0']

df_precipitation

# temperature data

df_temp = pd.read_excel('/kaggle/input/world-bank-climate-change-data/historical-data-excel-380-kb-.xls', sheet_name='Country_temperatureCRU')

df_temp.columns = [ c.replace('ISO_3DIGIT','code_3digit').lower() for c in df_temp.columns ]

df_temp = pd.merge(df_country_code, df_temp, how='left')

del df_temp['Unnamed: 0']

df_temp
# merge everything

dfs = [df_covid_latest, df_covid_first_dt, df_death, df_h1n1, df_school, df_temp, df_precipitation, df_air, df_property, df_health, df_hospital, df_population, df_gdp, df_age, df_sex, df_latlong, ]

print(f'{len(dfs)} data frames to merge')

df = df_names

clean_country_names(df)

for d in dfs:

    clean_country_names(d)

    df = pd.merge(df, d, on='country', how='left')

df = df.drop_duplicates()

del df['code_2digit_y']

del df['code_3digit_y']

print(f'Resulting shape {df.shape}')

df.to_csv('/kaggle/working/covid19_merged.csv')

df = df.round(4)

df

df.describe()
list(zip(df.columns,df.dtypes))
# countries that do not have sex data

df[pd.isnull(df['sex_male_to_female_0_14_years'])]
# countries that do not have lat_long data

df[pd.isna(df['latitude'])]
# countries that do not have age data

df[pd.isna(df['age_15_to_64_years_percent'])]
# missing

df[pd.isna(df['h1n1_Cases_confirmed'])]
# flu death rate missing data

df[pd.isna(df['Flu_pneumonia_death_rate_per_100000'])]
# missing country in population

df[pd.isna(df['Population_2020'])]
# show hospital beds, too many, just show what data we have!

print(df[pd.isna(df['hosp_beds_per_1000_2017'])].shape)

df[pd.notna(df['hosp_beds_per_1000_2017'])]
# show missing Health_Care_Index

df[pd.isna(df['Health_Care_Index'])]
# show missing property_Affordability_Index

df[pd.isna(df['property_Affordability_Index'])]
df[pd.isna(df['airport_traffic_2018_thousands'])]
# covid check, should return no rows that are na!

df[pd.isna(df['covid_confirmed_' + latest_covid_dt.replace('/','_') ])]
# feature engineering

df['covid_confirmed_per_100000'] = df['covid_confirmed_' + latest_covid_dt.replace('/','_')] / (df['Population_2020'] / 100000)

df['covid_deaths_per_100000'] = df['covid_deaths_' + latest_covid_dt.replace('/','_')] / (df['Population_2020'] / 100000)

df['h1n1_Cases_confirmed_per_100000'] = df['h1n1_Cases_confirmed'] / (df['Population_2020'] / 100000)

df['h1n1_Deaths_per_100000'] = df['h1n1_Deaths_confirmed'] / (df['Population_2020'] / 100000)

df['covid_deaths_per_confirmed_per_100'] = df['covid_deaths_' + latest_covid_dt.replace('/','_')] / (df['covid_confirmed_' + latest_covid_dt.replace('/','_')] / 100)

df['h1n1_deaths_per_confirmed_per_100'] = df['h1n1_Deaths_confirmed'] / (df['h1n1_Cases_confirmed'] / 100)

df['school_closed_days_after_first_case'] = df['first_school_closure_date'] - df['covid19_first_date']

df = df.round(4)

df.head(5)
print(df.shape)

df_corr = df.dropna(subset=['covid_deaths_per_100000'])

df_corr = df_corr.sort_values(by='covid_deaths_' + latest_covid_dt.replace('/','_'), ascending=False)

print(df_corr.shape)



# filters

# min_deaths = 20

topn = 100

quantiles = df_corr['covid_deaths_per_100000'].quantile([.1, .9])

# filter_min_deaths = df['covid_deaths_' + latest_covid_dt.replace('/','_')] >= min_deaths

filter_quantiles = (df_corr['covid_deaths_per_100000'] >= quantiles.iloc[0]) & (df_corr['covid_deaths_per_100000'] <= quantiles.iloc[1])



# apply filter

# df_corr = df[filter_min_deaths & filter_quantiles]

# df_corr = df[filter_min_deaths]

df_corr = df_corr.head(topn)



print(df_corr.shape)

df_corr

# sns.boxplot(df_corr['covid_deaths_per_100000'])
df_corr.tail(10)
from pandas.api.types import is_numeric_dtype

import scipy.stats



corrs = []

x_col = 'covid_deaths_per_100000'

for col in df_corr.columns:

    df_xy = df_corr[[x_col, col]]

    df_xy = df_xy.dropna()

    x = df_xy[x_col]

    y = df_xy[col]

    if not is_numeric_dtype(y):

        continue

    sc = scipy.stats.spearmanr(x, y)

    corrs.append((sc.pvalue, sc.correlation, col, y.shape[0], str(y.tolist()[:5]) + '...'))

df_corr_stats = pd.DataFrame(corrs)

df_corr_stats.columns = ['pvalue','spearman_rho_vs_' + x_col, 'feature_name', 'n_values', 'values']

df_corr_stats = df_corr_stats.sort_values(by=['pvalue'])

df_corr_stats.round(3)
y_col = 'age_over_65_years_percent'

scatter_plot(df_corr, x_col, y_col, x_col + ' vs ' + y_col)
y_col = 'age_over_65_years_percent'

scatter_plot_ranks(df_corr, x_col, y_col, x_col + ' vs ' + y_col)
y_col = 'Flu_pneumonia_death_rate_per_100000'

scatter_plot(df_corr, x_col, y_col, x_col + ' vs ' + y_col)
y_col = 'Flu_pneumonia_death_rate_per_100000'

scatter_plot_ranks(df_corr, x_col, y_col, x_col + ' vs ' + y_col)
y_col = 'apr_temp'

scatter_plot(df_corr, x_col, y_col, x_col + ' vs ' + y_col)
y_col = 'apr_temp'

scatter_plot_ranks(df_corr, x_col, y_col, x_col + ' vs ' + y_col)
y_col = 'latitude'

scatter_plot(df_corr, x_col, y_col, x_col + ' vs ' + y_col)
y_col = 'latitude'

scatter_plot_ranks(df_corr, x_col, y_col, x_col + ' vs ' + y_col)
y_col = 'property_Affordability_Index'

scatter_plot(df_corr, x_col, y_col, x_col + ' vs ' + y_col)
y_col = 'property_Affordability_Index'

scatter_plot_ranks(df_corr, x_col, y_col, x_col + ' vs ' + y_col)
# select countries for spot checks

#c = ['China','United States', 'Italy', 'Canada', 'United Kingdom', 'Australia', 'Japan', 'South Korea', 'Taiwan', 'Iran']

df = df.sort_values(by=['covid_confirmed_' + latest_covid_dt.replace('/','_') ], ascending=False)

topn = 20

df_topn_covid = df.head(topn)

df_topn_covid


# basic plot

scatter_plot(df_topn_covid, 'covid_confirmed_per_100000', 'h1n1_Cases_confirmed_per_100000', 'Cases confirmed per 100,000 population')

# p1=sns.regplot(data=df_topn_covid, x=x, y=y, fit_reg=False, marker="o", color="skyblue", scatter_kws={'s':50})

# p1.set_title('Cases confirmed per 100,000 population')

# # add annotations one by one with a loop

# for i in range(df_topn_covid.shape[0]):

#      p1.text(df_topn_covid[x].iloc[i]+0.2, df_topn_covid[y].iloc[i], df_topn_covid['country'].iloc[i], horizontalalignment='left', size='medium', color='black', weight='semibold')

scatter_plot(df_topn_covid, 'covid_deaths_per_100000', 'h1n1_Deaths_per_100000', 'COVID-19 vs H1N1 deaths per 100,000 population \n As of ' + latest_covid_dt.replace('/','_'))
scatter_plot(df_topn_covid, 'covid_deaths_per_confirmed_per_100', 'h1n1_deaths_per_confirmed_per_100', 'COVID-19 vs H1N1 deaths per 100 confirmed \n As of ' + latest_covid_dt.replace('/','_') )
df[df['country'].isin(['Italy','Germany','Spain','United States', 'United Kingdom','Switzerland','Turkey','France'])][['country','hosp_beds_per_1000_2017','ICU-CCB_beds_per_100000']]
# deaths vs beds

scatter_plot(df_topn_covid, 'covid_deaths_per_100000', 'ICU-CCB_beds_per_100000', 'COVID-19 deaths per 100,000 population vs ICU Beds per 100,000 \n As of ' + latest_covid_dt.replace('/','_') )
import scipy.stats

x = df_topn_covid['covid_deaths_per_100000']

y = df_topn_covid['ICU-CCB_beds_per_100000']

scipy.stats.spearmanr(x,y, nan_policy='omit')
# deaths vs gdp

scatter_plot(df_topn_covid, 'covid_deaths_per_100000', 'gdp_usd_million', 'COVID-19 deaths per 100,000 population vs GDP \n As of ' + latest_covid_dt.replace('/','_') )
x = df_topn_covid['covid_deaths_per_100000']

y = df_topn_covid['gdp_usd_million']

scipy.stats.spearmanr(x,y)
# gdp vs beds

x = df_topn_covid['ICU-CCB_beds_per_100000']

y = df_topn_covid['gdp_usd_million']

scipy.stats.spearmanr(x,y,nan_policy="omit")
scatter_plot(df_topn_covid, 'covid_deaths_per_100000', 'age_over_65_years_percent', 'COVID-19 deaths per 100,000 population vs age over 65 yrs (%) \n As of ' + latest_covid_dt.replace('/','_') )
df_topn_covid.dropna(subset=['first_school_closure_date'])
scatter_plot(df_topn_covid.dropna(subset=['first_school_closure_date']), 'covid_deaths_per_100000', 'first_school_closure_date', 'COVID-19 deaths per 100,000 population vs first day of school closure \n As of ' + latest_covid_dt.replace('/','_') )