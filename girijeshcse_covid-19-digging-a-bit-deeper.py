import numpy as np

import pandas as pd



import plotly.graph_objects as go

import plotly.express as px

import plotly.io as pio

pio.templates.default = "plotly_dark"

from plotly.subplots import make_subplots



from pathlib import Path

data_dir = Path('../input/covid19-global-forecasting-week-1')



import os

os.listdir(data_dir)
data = pd.read_csv(data_dir/'train.csv', parse_dates=['Date'])

data.head()
data.info()
print(f"Earliest Entry: {data['Date'].min()}")

print(f"Last Entry:     {data['Date'].max()}")

print(f"Total Days:     {data['Date'].max() - data['Date'].min()}")
data.rename(columns={'Date': 'date', 

                     'Id': 'id',

                     'Province/State':'state',

                     'Country/Region':'country',

                     'Lat':'lat',

                     'Long': 'long',

                     'ConfirmedCases': 'confirmed',

                     'Fatalities':'deaths',

                    }, inplace=True)

data.head()
cleaned_data = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv', parse_dates=['Date'])

cleaned_data.head()
cleaned_data['Country/Region'].value_counts()
cleaned_data.info()
cleaned_data.rename(columns={'ObservationDate': 'date', 

                     'Province/State':'state',

                     'Country/Region':'country',

                     'Last Update':'last_updated',

                     'Confirmed': 'confirmed',

                     'Deaths':'deaths',

                     'Recovered':'recovered'

                    }, inplace=True)



# cases 

cases = ['confirmed', 'deaths', 'recovered', 'active']



# Active Case = confirmed - deaths - recovered

cleaned_data['active'] = cleaned_data['confirmed'] - cleaned_data['deaths'] - cleaned_data['recovered']



# replacing Mainland china with just China

cleaned_data['country'] = cleaned_data['country'].replace('Mainland China', 'China')



# filling missing values 

cleaned_data[['state']] = cleaned_data[['state']].fillna('')

cleaned_data[cases] = cleaned_data[cases].fillna(0)

cleaned_data.rename(columns={'Date':'date'}, inplace=True)



data = cleaned_data



# # cases in the ships

# ship = data[data['state'].str.contains('Grand Princess')|data['country'].str.contains('Cruise Ship')]



# # china and the row

# china = data[data['country']=='China']

# rest = data[data['country']!='China']



# # latest

# full_latest = data[data['date'] == max(data['date'])].reset_index()

# china_latest = data[data['country'] == 'China']

# rest_latest = data[data['country'] != 'China']



# # latest condensed

# full_latest_grouped = full_latest.groupby('country')['confirmed', 'deaths', 'recovered', 'active'].sum().reset_index()

# china_latest_grouped = china_latest.groupby('state')['confirmed', 'deaths', 'recovered', 'active'].sum().reset_index()

# rest_latest_grouped = rest_latest.groupby('country')['confirmed', 'deaths', 'recovered', 'active'].sum().reset_index()
print("External Data")

print(f"Earliest Entry: {data['date'].min()}")

print(f"Last Entry:     {data['date'].max()}")

print(f"Total Days:     {data['date'].max() - data['date'].min()}")
grouped = data.groupby('date')['date', 'confirmed', 'deaths'].sum().reset_index()

grouped.head()
grouped = data.groupby('date')['date', 'confirmed', 'deaths'].sum().reset_index()



fig = px.line(grouped, x="date", y="confirmed", 

              title="Worldwide Confirmed Cases Over Time")

fig.show()



fig = px.line(grouped, x="date", y="confirmed", 

              title="Worldwide Confirmed Cases (Logarithmic Scale) Over Time", 

              log_y=True)

fig.show()
grouped_china = data[data['country'] == "China"].reset_index()

grouped_china_date = grouped_china.groupby('date')['date', 'confirmed', 'deaths'].sum().reset_index()



grouped_italy = data[data['country'] == "Italy"].reset_index()

grouped_italy_date = grouped_italy.groupby('date')['date', 'confirmed', 'deaths'].sum().reset_index()



grouped_us = data[data['country'] == "US"].reset_index()

grouped_us_date = grouped_us.groupby('date')['date', 'confirmed', 'deaths'].sum().reset_index()



grouped_rest = data[~data['country'].isin(['China', 'Italy', 'US'])].reset_index()

grouped_rest_date = grouped_rest.groupby('date')['date', 'confirmed', 'deaths'].sum().reset_index()
plot_titles = ['China', 'Italy', 'USA', 'Rest of the World']



fig = px.line(grouped_china_date, x="date", y="confirmed", 

              title=f"Confirmed Cases in {plot_titles[0].upper()} Over Time", 

              color_discrete_sequence=['#F61067'],

              height=300

             )

fig.show()



fig = px.line(grouped_italy_date, x="date", y="confirmed", 

              title=f"Confirmed Cases in {plot_titles[1].upper()} Over Time", 

              color_discrete_sequence=['#91C4F2'],

              height=300

             )

fig.show()



fig = px.line(grouped_us_date, x="date", y="confirmed", 

              title=f"Confirmed Cases in {plot_titles[2].upper()} Over Time", 

              color_discrete_sequence=['#6F2DBD'],

              height=300

             )

fig.show()



fig = px.line(grouped_rest_date, x="date", y="confirmed", 

              title=f"Confirmed Cases in {plot_titles[3].upper()} Over Time", 

              color_discrete_sequence=['#FFDF64'],

              height=300

             )

fig.show()
data['state'] = data['state'].fillna('')

temp = data[[col for col in data.columns if col != 'state']]



latest = temp[temp['date'] == max(temp['date'])].reset_index()

latest_grouped = latest.groupby('country')['confirmed', 'deaths'].sum().reset_index()
fig = px.choropleth(latest_grouped, locations="country", 

                    locationmode='country names', color="confirmed", 

                    hover_name="country", range_color=[1,5000], 

                    color_continuous_scale="mint", 

                    title='Countries with Confirmed Cases')

# fig.update(layout_coloraxis_showscale=False)

fig.show()
europe = list(['Austria','Belgium','Bulgaria','Croatia','Cyprus','Czech Republic','Denmark','Estonia','Finland','France','Germany','Greece','Hungary','Ireland',

          'Italy', 'Latvia','Luxembourg','Lithuania','Malta','Norway','Netherlands','Poland','Portugal','Romania','Slovakia','Slovenia',

         'Spain', 'Sweden', 'United Kingdom', 'Iceland', 'Russia', 'Switzerland', 'Serbia', 'Ukraine', 'Belarus'])



europe_grouped_latest = latest_grouped[latest_grouped['country'].isin(europe)]
fig = px.choropleth(europe_grouped_latest, locations="country", 

                    locationmode='country names', color="confirmed", 

                    hover_name="country", range_color=[1,2000], 

                    color_continuous_scale='portland', 

                    title='European Countries with Confirmed Cases', scope='europe', height=800)

# fig.update(layout_coloraxis_showscale=False)

fig.show()
fig = px.bar(latest_grouped.sort_values('confirmed', ascending=False)[:20][::-1], 

             x='confirmed', y='country',

             title='Confirmed Cases Worldwide', text='confirmed', height=1000, orientation='h')

fig.show()
fig = px.bar(europe_grouped_latest.sort_values('confirmed', ascending=False)[:10][::-1], 

             x='confirmed', y='country', color_discrete_sequence=['#84DCC6'],

             title='Confirmed Cases in Europe', text='confirmed', orientation='h')

fig.show()
usa = cleaned_data[cleaned_data['country'] == "US"]

usa_latest = usa[usa['date'] == max(usa['date'])]

usa_latest = usa_latest.groupby('state')['confirmed', 'deaths'].max().reset_index()



fig = px.bar(usa_latest.sort_values('confirmed', ascending=False)[:10][::-1], 

             x='confirmed', y='state', color_discrete_sequence=['#D63230'],

             title='Confirmed Cases in USA', text='confirmed', orientation='h')

fig.show()
fig = px.line(grouped, x="date", y="deaths", title="Worldwide Deaths Over Time",

             color_discrete_sequence=['#F42272'])

fig.show()



fig = px.line(grouped, x="date", y="deaths", title="Worldwide Deaths (Logarithmic Scale) Over Time", 

              log_y=True, color_discrete_sequence=['#F42272'])

fig.show()
plot_titles = ['China', 'Italy', 'USA', 'Rest of the World']



fig = px.line(grouped_china_date, x="date", y="deaths", 

              title=f"Deaths in {plot_titles[0].upper()} Over Time", 

              color_discrete_sequence=['#F61067'],

              height=300

             )

fig.show()



fig = px.line(grouped_italy_date, x="date", y="deaths", 

              title=f"Deaths in {plot_titles[1].upper()} Over Time", 

              color_discrete_sequence=['#91C4F2'],

              height=300

             )

fig.show()



fig = px.line(grouped_us_date, x="date", y="deaths", 

              title=f"Deaths in {plot_titles[2].upper()} Over Time", 

              color_discrete_sequence=['#6F2DBD'],

              height=300

             )

fig.show()



fig = px.line(grouped_rest_date, x="date", y="deaths", 

              title=f"Deaths in {plot_titles[3].upper()} Over Time", 

              color_discrete_sequence=['#FFDF64'],

              height=300

             )

fig.show()
fig = px.choropleth(latest_grouped, locations="country", 

                    locationmode='country names', color="deaths", 

                    hover_name="deaths", range_color=[1,100], 

                    color_continuous_scale="peach", 

                    title='Countries with Reported Deaths')

# fig.update(layout_coloraxis_showscale=False)

fig.show()
fig = px.choropleth(europe_grouped_latest, locations="country", 

                    locationmode='country names', color="deaths", 

                    hover_name="country", range_color=[1,100], 

                    color_continuous_scale='portland',

                    title='Reported Deaths in EUROPE', scope='europe', height=800)

# fig.update(layout_coloraxis_showscale=False)

fig.show()
fig = px.bar(latest_grouped.sort_values('deaths', ascending=False)[:10][::-1], 

             x='deaths', y='country',

             title='Confirmed Deaths Worldwide', text='deaths', orientation='h')

fig.show()
fig = px.bar(europe_grouped_latest.sort_values('deaths', ascending=False)[:5][::-1], 

             x='deaths', y='country', color_discrete_sequence=['#84DCC6'],

             title='Deaths in Europe', text='deaths', orientation='h')

fig.show()
fig = px.bar(usa_latest.sort_values('deaths', ascending=False)[:5][::-1], 

             x='deaths', y='state', color_discrete_sequence=['#D63230'],

             title='Deaths in USA', text='deaths', orientation='h')

fig.show()
cleaned_data.rename(columns={'Date':'date'}, inplace=True)



grouped_china = cleaned_data[cleaned_data['country'] == "China"].reset_index()

grouped_china_date = grouped_china.groupby('date')['date', 'confirmed', 'deaths', 'active', 'recovered'].sum().reset_index()



grouped_italy = cleaned_data[cleaned_data['country'] == "Italy"].reset_index()

grouped_italy_date = grouped_italy.groupby('date')['date', 'confirmed', 'deaths', 'active', 'recovered'].sum().reset_index()



grouped_us = cleaned_data[cleaned_data['country'] == "US"].reset_index()

grouped_us_date = grouped_us.groupby('date')['date', 'confirmed', 'deaths', 'active', 'recovered'].sum().reset_index()



grouped_rest = cleaned_data[~cleaned_data['country'].isin(['China', 'Italy', 'US'])].reset_index()

grouped_rest_date = grouped_rest.groupby('date')['date', 'confirmed', 'deaths', 'active', 'recovered'].sum().reset_index()
plot_titles = ['China', 'Italy', 'USA', 'Rest of the World']



fig = px.line(grouped_china_date, x="date", y="active", 

              title=f"Active Cases in {plot_titles[0].upper()} Over Time", 

              color_discrete_sequence=['#F61067'],

              height=500

             )

fig.show()



fig = px.line(grouped_italy_date, x="date", y="active", 

              title=f"Active Cases in {plot_titles[1].upper()} Over Time", 

              color_discrete_sequence=['#91C4F2'],

              height=500

             )

fig.show()



fig = px.line(grouped_us_date, x="date", y="active", 

              title=f"Active Cases in {plot_titles[2].upper()} Over Time", 

              color_discrete_sequence=['#6F2DBD'],

              height=500

             )

fig.show()



fig = px.line(grouped_rest_date, x="date", y="active", 

              title=f"Active Cases in {plot_titles[3].upper()} Over Time", 

              color_discrete_sequence=['#FFDF64'],

              height=500

             )

fig.show()
cleaned_data['state'] = cleaned_data['state'].fillna('')

temp = cleaned_data[[col for col in cleaned_data.columns if col != 'state']]



latest = temp[temp['date'] == max(temp['date'])].reset_index()

latest_grouped = latest.groupby('country')['confirmed', 'deaths', 'active', 'recovered'].sum().reset_index()
fig = px.choropleth(latest_grouped, locations="country", 

                    locationmode='country names', color="active", 

                    hover_name="active", range_color=[1,1000], 

                    color_continuous_scale="peach", 

                    title='Active Cases Worldwide')

# fig.update(layout_coloraxis_showscale=False)

fig.show()
europe = list(['Austria','Belgium','Bulgaria','Croatia','Cyprus','Czech Republic','Denmark','Estonia','Finland','France','Germany','Greece','Hungary','Ireland',

          'Italy', 'Latvia','Luxembourg','Lithuania','Malta','Norway','Netherlands','Poland','Portugal','Romania','Slovakia','Slovenia',

         'Spain', 'Sweden', 'United Kingdom', 'Iceland', 'Russia', 'Switzerland', 'Serbia', 'Ukraine', 'Belarus'])



europe_grouped_latest = latest_grouped[latest_grouped['country'].isin(europe)]
fig = px.choropleth(europe_grouped_latest, locations="country", 

                    locationmode='country names', color="active", 

                    hover_name="country", range_color=[1,2000], 

                    color_continuous_scale='portland',

                    title='Active Cases European Countries', scope='europe', height=800)

# fig.update(layout_coloraxis_showscale=False)

fig.show()
fig = px.bar(latest_grouped.sort_values('active', ascending=False)[:10][::-1], 

             x='active', y='country',

             title='Active Cases Worldwide', text='active', orientation='h')

fig.show()
fig = px.bar(europe_grouped_latest.sort_values('active', ascending=False)[:10][::-1], 

             x='active', y='country',

             title='Active Cases EUROPE', text='active', orientation='h')

fig.show()
usa = cleaned_data[cleaned_data['country'] == "US"]

usa_latest = usa[usa['date'] == max(usa['date'])]

usa_latest = usa_latest.groupby('state')['confirmed', 'deaths', 'active', 'recovered'].max().reset_index()



fig = px.bar(usa_latest.sort_values('active', ascending=False)[:10][::-1], 

             x='active', y='state', color_discrete_sequence=['#D63230'],

             title='Active Cases in USA', text='active', orientation='h')

fig.show()
fig = px.bar(latest_grouped.sort_values('recovered', ascending=False)[:10][::-1], 

             x='recovered', y='country',

             title='Recovered Cases Worldwide', text='recovered', orientation='h')

fig.show()
fig = px.bar(europe_grouped_latest.sort_values('recovered', ascending=False)[:10][::-1], 

             x='recovered', y='country',

             title='Recovered Cases in EUROPE', text='recovered', orientation='h', color_discrete_sequence=['cyan'])

fig.show()
temp = cleaned_data.groupby('date')['recovered', 'deaths', 'active'].sum().reset_index()

temp = temp.melt(id_vars="date", value_vars=['recovered', 'deaths', 'active'],

                 var_name='case', value_name='count')





fig = px.line(temp, x="date", y="count", color='case',

             title='Cases over time: Line Plot', color_discrete_sequence = ['cyan', 'red', 'orange'])

fig.show()





fig = px.area(temp, x="date", y="count", color='case',

             title='Cases over time: Area Plot', color_discrete_sequence = ['cyan', 'red', 'orange'])

fig.show()
rest = cleaned_data[cleaned_data['country'] != 'China']

rest_grouped = rest.groupby('date')['recovered', 'deaths', 'active'].sum().reset_index()



temp = rest_grouped.melt(id_vars="date", value_vars=['recovered', 'deaths', 'active'],

                 var_name='case', value_name='count')





fig = px.line(temp, x="date", y="count", color='case',

             title='Cases - Rest of the World: Line Plot', color_discrete_sequence = ['cyan', 'red', 'orange'])

fig.show()





fig = px.area(temp, x="date", y="count", color='case',

             title='Cases - Rest of the World: Area Plot', color_discrete_sequence = ['cyan', 'red', 'orange'])

fig.show()
cleaned_latest = cleaned_data[cleaned_data['date'] == max(cleaned_data['date'])]

flg = cleaned_latest.groupby('country')['confirmed', 'deaths', 'recovered', 'active'].sum().reset_index()



flg['mortalityRate'] = round((flg['deaths']/flg['confirmed'])*100, 2)

temp = flg[flg['confirmed']>100]

temp = temp.sort_values('mortalityRate', ascending=False)



fig = px.bar(temp.sort_values(by="mortalityRate", ascending=False)[:10][::-1],

             x = 'mortalityRate', y = 'country', 

             title='Deaths per 100 Confirmed Cases', text='mortalityRate', height=800, orientation='h',

             color_discrete_sequence=['darkred']

            )

fig.show()
print("Countries with Lowest Mortality Rates")

temp = flg[flg['confirmed']>100]

temp = temp.sort_values('mortalityRate', ascending=True)[['country', 'confirmed','deaths']][:16]

temp.sort_values('confirmed', ascending=False)[['country', 'confirmed','deaths']][:20].style.background_gradient(cmap='Greens')
flg['recoveryRate'] = round((flg['recovered']/flg['confirmed'])*100, 2)

temp = flg[flg['confirmed']>100]

temp = temp.sort_values('recoveryRate', ascending=False)



fig = px.bar(temp.sort_values(by="recoveryRate", ascending=False)[:10][::-1],

             x = 'recoveryRate', y = 'country', 

             title='Recoveries per 100 Confirmed Cases', text='recoveryRate', height=800, orientation='h',

             color_discrete_sequence=['#2ca02c']

            )

fig.show()
print("Countries with Worst Recovery Rates")

temp = flg[flg['confirmed']>100]

temp = temp.sort_values('recoveryRate', ascending=True)[['country', 'confirmed','recovered']][:20]

temp.sort_values('confirmed', ascending=False)[['country', 'confirmed','recovered']][:20].style.background_gradient(cmap='Reds')
formated_gdf = data.groupby(['date', 'country'])['confirmed', 'deaths'].max()

formated_gdf = formated_gdf.reset_index()

formated_gdf['date'] = pd.to_datetime(formated_gdf['date'])

formated_gdf['date'] = formated_gdf['date'].dt.strftime('%m/%d/%Y')

formated_gdf['size'] = formated_gdf['confirmed'].pow(0.3)



fig = px.scatter_geo(formated_gdf, locations="country", locationmode='country names', 

                     color="confirmed", size='size', hover_name="country", 

                     range_color= [0, 1500], 

                     projection="natural earth", animation_frame="date", 

                     title='COVID-19: Spread Over Time', color_continuous_scale="portland")

# fig.update(layout_coloraxis_showscale=False)

fig.show()
formated_gdf = data.groupby(['date', 'country'])['confirmed', 'deaths'].max()

formated_gdf = formated_gdf.reset_index()

formated_gdf['date'] = pd.to_datetime(formated_gdf['date'])

formated_gdf['date'] = formated_gdf['date'].dt.strftime('%m/%d/%Y')

formated_gdf['size'] = formated_gdf['deaths'].pow(0.3)



fig = px.scatter_geo(formated_gdf, locations="country", locationmode='country names', 

                     color="deaths", size='size', hover_name="country", 

                     range_color= [0, 100], 

                     projection="natural earth", animation_frame="date", 

                     title='COVID-19: Deaths Over Time', color_continuous_scale="peach")

# fig.update(layout_coloraxis_showscale=False)

fig.show()
formated_gdf = cleaned_data.groupby(['date', 'country'])['confirmed', 'deaths', 'active', 'recovered'].max()

formated_gdf = formated_gdf.reset_index()

formated_gdf['date'] = pd.to_datetime(formated_gdf['date'])

formated_gdf['date'] = formated_gdf['date'].dt.strftime('%m/%d/%Y')

formated_gdf['size'] = formated_gdf['active'].pow(0.3)



fig = px.scatter_geo(formated_gdf, locations="country", locationmode='country names', 

                     color="active", size='size', hover_name="country", 

                     range_color= [0, 1000], 

                     projection="natural earth", animation_frame="date", 

                     title='COVID-19: Active Cases Over Time', color_continuous_scale="portland")

fig.update(layout_coloraxis_showscale=False)

fig.show()
formated_gdf = cleaned_data.groupby(['date', 'country'])['confirmed', 'deaths', 'active', 'recovered'].max()

formated_gdf = formated_gdf.reset_index()

formated_gdf['date'] = pd.to_datetime(formated_gdf['date'])

formated_gdf['date'] = formated_gdf['date'].dt.strftime('%m/%d/%Y')

formated_gdf['size'] = formated_gdf['recovered'].pow(0.3)



fig = px.scatter_geo(formated_gdf, locations="country", locationmode='country names', 

                     color="recovered", size='size', hover_name="country", 

                     range_color= [0, 100], 

                     projection="natural earth", animation_frame="date", 

                     title='COVID-19: Recoveries Over Time', color_continuous_scale="greens")

fig.update(layout_coloraxis_showscale=False)

fig.show()
formated_gdf = cleaned_data.groupby(['date', 'country'])['confirmed', 'deaths', 'active', 'recovered'].max()

formated_gdf = formated_gdf.reset_index()

formated_gdf['date'] = pd.to_datetime(formated_gdf['date'])

formated_gdf['date'] = formated_gdf['date'].dt.strftime('%m/%d/%Y')

formated_gdf['size'] = formated_gdf['confirmed'].pow(0.3) * 5



fig = px.scatter_geo(formated_gdf, locations="country", locationmode='country names', 

                     color="confirmed", size='size', hover_name="country", 

                     range_color= [0, 5000], 

                     projection="natural earth", animation_frame="date", scope="europe",

                     title='COVID-19: Spread Over Time in EUROPE', color_continuous_scale="portland", height=800)

# fig.update(layout_coloraxis_showscale=False)

fig.show()
formated_gdf['date'] = pd.to_datetime(formated_gdf['date'])

formated_gdf['date'] = formated_gdf['date'].dt.strftime('%m/%d/%Y')

formated_gdf['size'] = formated_gdf['deaths'].pow(0.3)



fig = px.scatter_geo(formated_gdf, locations="country", locationmode='country names', 

                     color="deaths", size='size', hover_name="country", 

                     range_color= [0, 500], 

                     projection="natural earth", animation_frame="date", scope="europe",

                     title='COVID-19: Deaths Over Time in EUROPE', color_continuous_scale="peach", height=800)

# fig.update(layout_coloraxis_showscale=False)

fig.show()
formated_gdf['date'] = pd.to_datetime(formated_gdf['date'])

formated_gdf['date'] = formated_gdf['date'].dt.strftime('%m/%d/%Y')

formated_gdf['size'] = formated_gdf['active'].pow(0.3) * 3.5



fig = px.scatter_geo(formated_gdf, locations="country", locationmode='country names', 

                     color="active", size='size', hover_name="country", 

                     range_color= [0, 3000], 

                     projection="natural earth", animation_frame="date", scope="europe",

                     title='COVID-19: Active Cases Over Time in EUROPE', color_continuous_scale="portland", height=800)

# fig.update(layout_coloraxis_showscale=False)

fig.show()
formated_gdf['date'] = pd.to_datetime(formated_gdf['date'])

formated_gdf['date'] = formated_gdf['date'].dt.strftime('%m/%d/%Y')

formated_gdf['size'] = formated_gdf['recovered'].pow(0.3) * 3.5



fig = px.scatter_geo(formated_gdf, locations="country", locationmode='country names', 

                     color="recovered", size='size', hover_name="country", 

                     range_color= [0, 100], 

                     projection="natural earth", animation_frame="date", scope="europe",

                     title='COVID-19: Deaths Over Time in EUROPE', color_continuous_scale="greens", height=800)

# fig.update(layout_coloraxis_showscale=False)

fig.show()