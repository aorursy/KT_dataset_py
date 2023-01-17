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
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv')
test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/test.csv')
submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/submission.csv')

print(train.shape, test.shape, submission.shape)
train.head()
train.info()
test.head()
test.info()
train['Date'] = pd.to_datetime(train['Date'])
test['Date'] = pd.to_datetime(test['Date'])
print(train.Country_Region.nunique())
countries = train.Country_Region.unique()
countries
country_df = train.groupby(['Date', 'Country_Region'])[['ConfirmedCases', 'Fatalities']].sum().reset_index()
country_df.tail()
target_date = country_df['Date'].max()
print('Date: ', target_date)
for i in [1, 10, 100, 1000, 10000]:
    n_countries = len(country_df.query('(Date == @target_date) & ConfirmedCases > @i'))
    print(f'{n_countries} countries have more than {i} confirmed cases')
import seaborn as sns
ax = sns.distplot(np.log10(country_df.query('Date == "2020-04-03"')['ConfirmedCases'] + 1))
ax.set_xlim([0, 6])
ax.set_xticks(np.arange(7))
_ = ax.set_xticklabels(['0', '10', '100', '1k', '10k', '100k'])
top_country_df = country_df.query('(Date == @target_date) & (ConfirmedCases > 1000)').sort_values('ConfirmedCases', ascending=False)
top_country_melt_df = pd.melt(top_country_df, id_vars='Country_Region', value_vars=['ConfirmedCases', 'Fatalities'])
import plotly.express as px
fig = px.bar(top_country_melt_df.iloc[::-1],
             x='value', y='Country_Region', color='variable', barmode='group',
             title=f'Confirmed Cases/Deaths on {target_date}', text='value', height=1500, orientation='h')
fig.show()
top30_countries = top_country_df.sort_values('ConfirmedCases', ascending=False).iloc[:30]['Country_Region'].unique()
top30_countries_df = country_df[country_df['Country_Region'].isin(top30_countries)]
fig = px.line(top30_countries_df,
              x='Date', y='ConfirmedCases', color='Country_Region',
              title=f'Confirmed Cases for top 30 country as of {target_date}')
fig.show()
top30_countries = top_country_df.sort_values('Fatalities', ascending=False).iloc[:30]['Country_Region'].unique()
top30_countries_df = country_df[country_df['Country_Region'].isin(top30_countries)]
fig = px.line(top30_countries_df,
              x='Date', y='Fatalities', color='Country_Region',
              title=f'Fatalities for top 30 country as of {target_date}')
fig.show()
countries_with_provinces = train[~train['Province_State'].isna()].Country_Region.unique()
countries_with_provinces
countries_no_province = [i for i in countries if i not in countries_with_provinces]
len(countries_no_province)
most_fatalities = train[train['Country_Region'].isin(countries_no_province)].groupby(['Country_Region']).Fatalities.max().sort_values(ascending=False)
plt.figure(figsize=(20,6))
sns.barplot(most_fatalities[:20].index, most_fatalities[:20].values)
most_confirmedCases = train[train['Country_Region'].isin(countries_no_province)].groupby(['Country_Region']).ConfirmedCases.max().sort_values(ascending=False)
plt.figure(figsize=(20,6))
sns.barplot(most_confirmedCases[:20].index, most_confirmedCases[:20].values)
top_country_df = country_df.query('(Date == @target_date) & (ConfirmedCases > 100)')
top_country_df['mortality_rate'] = top_country_df['Fatalities'] / top_country_df['ConfirmedCases']
top_country_df = top_country_df.sort_values('mortality_rate', ascending=False)
fig = px.bar(top_country_df[:30].iloc[::-1],
             x='mortality_rate', y='Country_Region',
             title=f'Mortality rate HIGH: top 30 countries on {target_date}', text='mortality_rate', height=800, orientation='h')
fig.show()
fig = px.bar(top_country_df[-30:],
             x='mortality_rate', y='Country_Region',
             title=f'Mortality rate LOW: top 30 countries on {target_date}', text='mortality_rate', height=800, orientation='h')
fig.show()
country_df['prev_confirmed'] = country_df.groupby('Country_Region')['ConfirmedCases'].shift(1)
country_df['new_case'] = country_df['ConfirmedCases'] - country_df['prev_confirmed']
country_df['new_case'].fillna(0, inplace=True)
top30_country_df = country_df[country_df['Country_Region'].isin(top30_countries)]

fig = px.line(top30_country_df,
              x='Date', y='new_case', color='Country_Region',
              title=f'DAILY NEW Confirmed cases world wide')
fig.show()
country_df['Date'] = country_df['Date'].apply(str)
country_df['confirmed_log1p'] = np.log1p(country_df['ConfirmedCases'])
country_df['fatalities_log1p'] = np.log1p(country_df['Fatalities'])

fig = px.scatter_geo(country_df, locations="Country_Region", locationmode='country names', 
                     color="ConfirmedCases", size='ConfirmedCases', hover_name="Country_Region", 
                     hover_data=['ConfirmedCases', 'Fatalities'],
                     range_color= [0, country_df['ConfirmedCases'].max()], 
                     projection="natural earth", animation_frame="Date", 
                     title='COVID-19: Confirmed cases spread Over Time', color_continuous_scale="Viridis")
fig.show()
fig = px.scatter_geo(country_df, locations="Country_Region", locationmode='country names', 
                     color="Fatalities", size='Fatalities', hover_name="Country_Region", 
                     hover_data=['ConfirmedCases', 'Fatalities'],
                     range_color= [0, country_df['Fatalities'].max()], 
                     projection="natural earth", animation_frame="Date", 
                     title='COVID-19: Fatalities growth Over Time', color_continuous_scale="Viridis")
fig.show()
country_df.loc[country_df['new_case'] < 0, 'new_case'] = 0.
fig = px.scatter_geo(country_df, locations="Country_Region", locationmode='country names', 
                     color="new_case", size='new_case', hover_name="Country_Region", 
                     hover_data=['ConfirmedCases', 'Fatalities'],
                     range_color= [0, country_df['new_case'].max()], 
                     projection="natural earth", animation_frame="Date", 
                     title='COVID-19: Daily NEW cases over Time', color_continuous_scale="Viridis")
fig.show()
