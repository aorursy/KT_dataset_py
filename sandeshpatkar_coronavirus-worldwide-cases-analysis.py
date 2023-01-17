import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go

import datetime as dt

import folium
link = 'https://raw.githubusercontent.com/RamiKrispin/coronavirus-csv/master/coronavirus_dataset.csv'
df = pd.read_csv(link)

df.head()
df.info()
df.shape
df.isnull().sum()
df.tail()
df = df.rename(columns = {'Province.State': 'state', 'Country.Region': 'country'}) #renaming column names for better usuability
df[df.apply(lambda row: row.astype(str).str.contains('Recovered').any(), axis=1)]
df[df['state'] == 'Recovered']['cases'].sum()
df = df[df['state'] != 'Recovered'] #data cleaning

df.shape
df = df.fillna('Not Available') #filling null values

df.isnull().sum()
df.head()
df[df['cases'] < 0]
df['cases'] = np.abs(df['cases'])

df[df['cases'] < 0]
df['date'] = pd.to_datetime(df['date']) #converting the date column to datetime format

df
df[df['cases'] == df['cases'].max()]
df['date'].max()
date_country_total = df.groupby(['date']).sum()

date_country_total['cum_f'] = date_country_total['cases'].cumsum()

date_country_total.reset_index(inplace = True)

date_country_total.tail()
print("Total number of people affected by COVID-19:", date_country_total['cum_f'].iloc[-1])
df2 = df.copy()

cases = df2.drop(['type', 'Lat', 'Long', 'state'], axis = 1)

cases = df2[df2['type'] != 'recovered']

cases = cases.groupby(['country','date']).sum()

cases.reset_index(inplace = True)

cou = list(cases['country'].unique())



new_df = pd.DataFrame()

for c in cou:

    dfc = cases[cases['country'] == c]

    dfc['cum_f'] = dfc['cases'].cumsum()

    new_df = new_df.append(dfc, ignore_index = True)



new_df_50000 = new_df[(new_df['cum_f'] > 50000) & (new_df['country'] != 'China')]

fig10 = px.line(new_df_50000, x = 'date', y = 'cum_f', color = 'country', title = 'Coronavirus Cases: Individual Countries (>50,000)')

fig10.show() 
new_df_100 = new_df[new_df['cum_f'] > 100]

fig10 = px.line(new_df_100, x = 'date', y = 'cum_f', color = 'country', title = 'Coronavirus Cases: Individual Countries')

fig10.show() 
new_df_100_except_china = new_df[(new_df['cum_f'] > 100) & (new_df['country'] != 'China')]

fig11 = px.line(new_df_100_except_china, x = 'date', y = 'cum_f', color = 'country', title = 'Coronavirus Cases: Individual Countries (Except China)')

fig11.show() 
px.line(date_country_total, x = 'date', y = 'cum_f', title = 'Coronavirus cases: World')
px.line(date_country_total.tail(), x = 'date', y = 'cum_f', title = 'Coronavirus cases: World [Past 5 days]')
country_cases = df.groupby(['country', 'Lat', 'Long','type']).sum()

country_cases.reset_index(inplace = True)

country_cases['type'] = country_cases['type'].str.replace('confirmed','1')

country_cases['type'] = country_cases['type'].str.replace('recovered','2')

country_cases['type'] = country_cases['type'].str.replace('death','0')

country_cases['type'] = pd.to_numeric(country_cases['type'])

country_cases
folium_map = folium.Map(location=[24.7117, 46.7242],

                            zoom_start=1,

                            tiles="CartoDB dark_matter"

                            )



for index, row in country_cases.iterrows():

    radius_len = row['cases']/1400

    

    if row['cases'] == 0:

            color = '#ff0000' #red #death

            radius = radius_len

    elif row['type'] == 1:

        color = '#ffd700' #gold #confirmed

        radius = radius_len

    elif row['type'] == 2:

        color = '#00ff00' #green #recovered

        radius = radius_len



        

    folium.CircleMarker(location = (row['Lat'],

                                   row['Long']),

                       radius = radius_len,

                       color = color,

                       fill = True).add_to(folium_map)

                        

folium_map
#map_f = px.density_mapbox()
total_infection_type = df.groupby(['date', 'type']).sum()

total_infection_type.reset_index(inplace = True)

total_infection_type
#finding the total number of cases for confirmed,recovered and deaths



total_infection_type['cases_confirmed'] = total_infection_type[total_infection_type['type'] == 'confirmed']['cases'].cumsum()

total_infection_type['cases_death'] = total_infection_type[total_infection_type['type'] == 'death']['cases'].cumsum()

total_infection_type['cases_recovered'] = total_infection_type[total_infection_type['type'] == 'recovered']['cases'].cumsum()

total_infection_type = total_infection_type.fillna(0)

total_infection_type
#adding the columns to get total cases for each type in one column

total_infection_type['total_cases'] = total_infection_type['cases_confirmed'] + total_infection_type['cases_death']+ total_infection_type['cases_recovered']
total_infection_type = total_infection_type.drop(['cases_confirmed', 'cases_death', 'cases_recovered'], axis = 1) #removing unneeded columns

total_infection_type
f= px.line(total_infection_type, x = 'date', y = 'total_cases', color = 'type', title = 'COVID-19 cases: Confirmed Vs. Recovered Vs. Fatal')

f.update_layout(hovermode = 'x')

f.show()
print('Total number of confirmed cases:', total_infection_type['total_cases'].iloc[-3])

print('Total number of death cases:', total_infection_type['total_cases'].iloc[-2])

print('Total number of recovered cases:', total_infection_type['total_cases'].iloc[-1])

print('Fatality Rate:', round(((total_infection_type['total_cases'].iloc[-2]/total_infection_type['total_cases'].iloc[-3])*100),2),'%')
# only_confirmed_cases['cases'] = total_infection_type[total_infection_type['type'] == 'confirmed']

# # only_confirmed_cases = only_confirmed_cases.drop()

# only_death_cases = total_infection_type[total_infection_type['type'] == 'death']

# only_death_cases.head()

# rate = only_confirmed_cases['total_cases']/only_death_cases['total_cases']

# only_confirmed_cases
infection_type = df.groupby(['country', 'date', 'type']).sum()
infection_type.reset_index(inplace = True)
infection_type.head()
china_deaths = infection_type[(infection_type['country'] == 'China') & (infection_type['type'] == 'death')]['cases'].sum()
a = infection_type.copy() 

a = a[(a['country'] == 'Italy') & (a['type'] == 'confirmed')]

a['fre'] = a['cases'].cumsum()

a.tail()
infection_type[(infection_type['country'] == 'Italy') & (infection_type['type'] == 'death')]['cases'].sum()
date_country = df.groupby(['country', 'date']).sum()

date_country.reset_index(inplace = True)

date_country
n_deaths = infection_type[infection_type['type'] == 'death']

n_deaths.head()
def summary_stats(country):

    

    a = infection_type[infection_type['country'] == country].copy()

    

    fig2 = px.bar(date_country[(date_country['country'] == country) & date_country['cases'] > 0], x = 'date', y = 'cases', color = 'cases',color_continuous_scale=["blue","yellow","red"], text = 'cases',title = 'Cases per day: {c}'.format(c = country))

    fig2.update_traces(textposition = 'outside')

    #fig2.update_layout(uniformtext_minsize=12)

    

    fig4 = px.bar(n_deaths[n_deaths['country'] == country], x = 'date', y = 'cases',color = 'cases',color_continuous_scale=["blue","yellow","red"], text = 'cases', title = "Number of deaths per day: {c}".format(c = country))

    fig4.update_traces(textposition = 'outside')

                           



#     fig2 = px.line(infection_type[(infection_type['country'] == country) & (infection_type['cases'] > 0)], x = 'date', y = 'cases', color = 'type', title = '{c}: Confirmed Vs. Recovered Vs. Deaths'.format(c = country))

#     fig2.show()

    

    cum_freq = infection_type[(infection_type['country'] == country)].copy()

    cum_freq['total'] = cum_freq['cases'].cumsum()

    fig1 = px.line(cum_freq, x = 'date', y = 'total', title = 'Total number of people affected in {c}'.format(c = country))

    

    print("Quick summary for {c}:".format(c= country))

    print("")

    total = infection_type[(infection_type['country'] == country)]['cases'].sum()

    print('Total number of people affected in {c}'.format(c = country), total)

    

    deaths = infection_type[(infection_type['country'] == country) & (infection_type['type'] == 'death')]['cases'].sum()

    fatality_rate = round((deaths/total)*100,2)

    

    for i in ['confirmed', 'death', 'recovered']:

        total = infection_type[(infection_type['country'] == country) & (infection_type['type'] == i)]['cases'].sum()

        print('Total number of {t} cases in {c}:'.format(t = i, c = country), total)

        a[i] = a[(a['country'] == country) & (a['type'] == i)]['cases'].cumsum()

    



    print('Fatality Rate for {c}:'.format(c = country), fatality_rate, '%')

    

    

        

    a = a.fillna(0)

    a['total'] = a['confirmed'] + a['death'] + a['recovered']

    a = a.drop(['confirmed','death', 'recovered'], axis = 1)

    

    #annotations

    date1 = a[(a['type'] == 'death') & (a['total'] >= china_deaths)]

    check = date1.shape[0]



    

    fig3 = px.line(a, x = 'date', y = 'total', color = 'type', title = '{c}: Confirmed Vs. Recovered Vs. Deaths'.format(c = country))

    fig3.update_layout(hovermode = 'x')

        

    if check > 0:

        date1 = date1.iloc[0,1]

        year = int(date1.strftime('%Y'))

        month = int(date1.strftime('%m'))

        day = int(date1.strftime('%d'))

        fig3.update_layout(annotations = [

            dict(

            x = dt.date(year, month, day),

            y = china_deaths,

            xref = 'x',

            yref = 'y',

            showarrow = True,

            text = "Total deaths in China",

            )

        ])

#     ax.annotate('Test', mdates.date2num(date), china_deaths)

    

    fig1.show()

    fig2.show()

    fig4.show()

    fig3.show()

summary_stats('India')
#india = pd.read_csv('https://raw.githubusercontent.com/sandeshpatkar/coronavirus-csv/master/india.csv')

india_data_link = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSAD0SPvZSXA6TWBih-uaKutfl-m_UewVBqozY-kk3HudlM-23Iput1XiRrzd8VopiQXvK5KoN5_Sl3/pub?output=csv'

india = pd.read_csv(india_data_link)

india = india.dropna(axis = 1, how = 'all') #dropping NaN columns

india = india.dropna(axis = 0, how = 'all') ##dropping NaN rows



india.tail()
#renaming columns

col = india.columns



india = india.rename(columns = {col[1]: 'State', col[2]:'confirmed', col[3]:'cured', col[4]:'death', col[5]:'Latitude', col[6]:'Longitude'})



india = india[:-3]

india.tail()
india.info()
india[['confirmed', 'cured', 'death']] = india[['confirmed', 'cured', 'death']].astype(float) #changing to numeric columns
# india['confirmed'] = india['confirmed_india'] + india['confirmed_other']

# india.head()
map_india = folium.Map(location=[20.5936832, 78.962883],

                            zoom_start=4,

                            tiles="CartoDB dark_matter"

                            )



for index, row in india.iterrows():

    radius_confirmed = row['confirmed']/3

    radius_death = row['death']/3

    radius_cured = row['cured']/3

        

    folium.CircleMarker(location = (row['Latitude'],

                                   row['Longitude']),

                       radius = radius_confirmed,

                       color = '#ffd700', #gold #confirmed

                       fill = True,

                       tooltip = row['State'],

                       popup = 'Confirmed Cases:'+str(row['confirmed'])+'\n'+'Cured:'+str(row['cured'])+'\n'+'Deaths:'+str(row['death'])).add_to(map_india)

    

    folium.CircleMarker(location = (row['Latitude'],

                                   row['Longitude']),

                       radius = radius_cured,

                       color = '#00ff00', #green #recovered

                       fill = True,

                       tooltip = row['State'],

                       popup = 'Confirmed Cases:'+str(row['confirmed'])+'\n'+'Cured:'+str(row['cured'])+'\n'+'Deaths:'+str(row['death'])).add_to(map_india)

    

    folium.CircleMarker(location = (row['Latitude'],

                                   row['Longitude']),

                       radius = radius_death,

                       color = '#ff0000',  #red #death

                       fill = True,

                       tooltip = row['State'],

                       popup = 'Confirmed Cases:'+str(row['confirmed'])+'\n'+'Cured:'+str(row['cured'])+'\n'+'Deaths:'+str(row['death'])).add_to(map_india)



map_india
summary_stats('China')
summary_stats('Italy')
summary_stats('Korea, South')
summary_stats('Iran')
summary_stats('France')
summary_stats('US')
summary_stats('Spain')
summary_stats('Israel')
summary_stats('Pakistan')
summary_stats('Australia')
def compare_country(c1, c2, case_type = 'confirmed'):

    

    df = infection_type.copy()

    df.reset_index(inplace = True)

    

    con1 = df[(df['country'] == c1) & (df['type'] == case_type)].copy() #avoiding SettingWithCopy() warning

    con1['total_{c}'.format(c = c1)] = con1['cases'].cumsum()

    con1 = con1.drop(['index', 'Lat', 'Long', 'type'], axis = 1)

    

    con2 = df[(df['country'] == c2) & (df['type'] == case_type)].copy() #avoiding SettingWithCopy() warning

    con2['total_{c}'.format(c = c2)] = con2['cases'].cumsum()

    con2 = con2.drop(['index', 'Lat', 'Long','type'], axis = 1)

    

    #merging dataset

    merged = con1.merge(con2, on = 'date', suffixes = ('_{}'.format(c1), '_{}'.format(c2)))

    #print(merged.head())

    

    #plotting

    fig = go.Figure(data = [

        go.Bar(name = c1, x = merged['date'], y = merged['total_{}'.format(c1)]),

        go.Bar(name = c2, x = merged['date'], y = merged['total_{}'.format(c2)])

    ])

    fig.update_layout(barmode='group')

    fig.update_layout(hovermode = 'x')

    fig.update_layout(title  = '{con1} vs. {con2} comparison: {type1}'.format(con1 = c1, con2 = c2, type1 = case_type.title()))

    fig.show()
compare_country('Italy', 'China', 'death')
compare_country('Italy', 'China', 'confirmed')
compare_country('Italy', 'Korea, South', 'confirmed')
compare_country('Italy', 'US')
compare_country('China', 'US')
compare_country('US', 'United Kingdom')
compare_country('US', 'United Kingdom', 'death')
compare_country('Spain', 'China', 'death')
compare_country('Spain', 'Germany')
compare_country('Spain', 'Germany', 'death')
compare_country('France', 'Germany')