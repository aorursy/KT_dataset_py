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
cleaned_data = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv', parse_dates=['Date'])

cleaned_data_ro = cleaned_data.loc[cleaned_data['Country/Region'] == 'Romania']

cleaned_data.head()

cleaned_data_ro.head()
cleaned_data.rename(columns={'ObservationDate': 'date', 

                     'Province/State':'state',

                     'Country/Region':'country',

                     'Last Update':'last_updated',

                     'Confirmed': 'confirmed',

                     'Deaths':'deaths',

                     'Recovered':'recovered'

                    }, inplace=True)



cleaned_data_ro.rename(columns={'ObservationDate': 'date', 

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

cleaned_data_ro['active'] = cleaned_data_ro['confirmed'] - cleaned_data_ro['deaths'] - cleaned_data_ro['recovered']





# replacing Mainland china with just China

cleaned_data['country'] = cleaned_data['country'].replace('Mainland China', 'China')



# filling missing values 

cleaned_data[['state']] = cleaned_data[['state']].fillna('')

cleaned_data[cases] = cleaned_data[cases].fillna(0)

cleaned_data.rename(columns={'Date':'date'}, inplace=True)



cleaned_data_ro[['state']] = cleaned_data_ro[['state']].fillna('')

cleaned_data_ro[cases] = cleaned_data_ro[cases].fillna(0)

cleaned_data_ro.rename(columns={'Date':'date'}, inplace=True)



data = cleaned_data

data_ro = cleaned_data_ro
print("External Data")

print(f"Earliest Entry: {data['date'].min()}")

print(f"Last Entry:     {data['date'].max()}")

print(f"Total Days:     {data['date'].max() - data['date'].min()}")
grouped = data.groupby('date')['date', 'confirmed', 'deaths'].sum().reset_index()

grouped_ro = data_ro.groupby('date')['date', 'confirmed', 'deaths'].sum().reset_index()



fig = px.line(grouped_ro, x="date", y="confirmed", 

              title="Romania Confirmed Cases Over Time")

fig.show()



fig = px.line(grouped_ro, x="date", y="confirmed", 

              title="Romania Confirmed Cases (Logarithmic Scale) Over Time", 

              log_y=True)

fig.show()
grouped_china = data[data['country'] == "China"].reset_index()

grouped_china_date = grouped_china.groupby('date')['date', 'confirmed', 'deaths'].sum().reset_index()



grouped_italy = data[data['country'] == "Italy"].reset_index()

grouped_italy_date = grouped_italy.groupby('date')['date', 'confirmed', 'deaths'].sum().reset_index()



grouped_us = data[data['country'] == "US"].reset_index()

grouped_us_date = grouped_us.groupby('date')['date', 'confirmed', 'deaths'].sum().reset_index()



grouped_ro = data[data['country'] == "Romania"].reset_index()

grouped_ro_date = grouped_ro.groupby('date')['date', 'confirmed', 'deaths'].sum().reset_index()
plot_titles = ['China', 'Italy', 'USA', 'Romania']



fig = px.line(grouped_ro_date, x="date", y="confirmed", 

              title=f"Confirmed Cases in {plot_titles[3].upper()} Over Time", 

              color_discrete_sequence=['#FFDF64'],

              height=500

             )

fig.show()



fig = px.line(grouped_china_date, x="date", y="confirmed", 

              title=f"Confirmed Cases in {plot_titles[0].upper()} Over Time", 

              color_discrete_sequence=['#F61067'],

              height=500

             )

fig.show()



fig = px.line(grouped_italy_date, x="date", y="confirmed", 

              title=f"Confirmed Cases in {plot_titles[1].upper()} Over Time", 

              color_discrete_sequence=['#91C4F2'],

              height=500

             )

fig.show()



fig = px.line(grouped_us_date, x="date", y="confirmed", 

              title=f"Confirmed Cases in {plot_titles[2].upper()} Over Time", 

              color_discrete_sequence=['#6F2DBD'],

              height=500

             )

fig.show()
data['state'] = data['state'].fillna('')

temp = data[[col for col in data.columns if col != 'state']]



latest = temp[temp['date'] == max(temp['date'])].reset_index()

latest_grouped = latest.groupby('country')['confirmed', 'deaths'].sum().reset_index()
data['state'] = data['state'].fillna('')

temp = data[[col for col in data.columns if col != 'state']]



latest = temp[temp['date'] == max(temp['date'])].reset_index()

latest_grouped = latest.groupby('country')['confirmed', 'deaths'].sum().reset_index()



europe = list(['Austria','Belgium','Bulgaria','Croatia','Cyprus','Czechia','Denmark','Estonia','Finland','France','Germany','Greece','Hungary','Ireland',

               'Italy', 'Latvia','Luxembourg','Lithuania','Malta','Norway','Netherlands','Poland','Portugal','Romania','Slovakia','Slovenia',

               'Spain', 'Sweden', 'United Kingdom', 'Iceland', 'Russia', 'Switzerland', 'Serbia', 'Ukraine', 'Belarus',

               'Albania', 'Bosnia and Herzegovina', 'Kosovo', 'Moldova', 'Montenegro', 'North Macedonia'])



europe_grouped_latest = latest_grouped[latest_grouped['country'].isin(europe)]
fig = px.choropleth(europe_grouped_latest, locations="country", 

                    locationmode='country names', color="confirmed", 

                    hover_name="country", range_color=[1,2000], 

                    color_continuous_scale='portland', 

                    title='European Countries with Confirmed Cases', scope='europe', height=800)

# fig.update(layout_coloraxis_showscale=False)

fig.show()
fig = px.bar(europe_grouped_latest.sort_values('confirmed', ascending=False)[:25][::-1], 

             x='confirmed', y='country',

             title='Confirmed Cases Worldwide', text='confirmed', height=1000, orientation='h')

fig.show()
fig = px.line(grouped_ro, x="date", y="deaths", title="Romania Deaths Over Time",

             color_discrete_sequence=['#F42272'])

fig.show()



fig = px.line(grouped_ro, x="date", y="deaths", title="Romania Deaths (Logarithmic Scale) Over Time", 

              log_y=True, color_discrete_sequence=['#F42272'])

fig.show()
fig = px.bar(europe_grouped_latest.sort_values('deaths', ascending=False)[:-1][::-1], 

             x='deaths', y='country', color_discrete_sequence=['#84DCC6'],

             title='Deaths in Europe', text='deaths', orientation='h')

fig.show()
cleaned_data.rename(columns={'Date':'date'}, inplace=True)



grouped_china = cleaned_data[cleaned_data['country'] == "China"].reset_index()

grouped_china_date = grouped_china.groupby('date')['date', 'confirmed', 'deaths', 'active', 'recovered'].sum().reset_index()



grouped_italy = cleaned_data[cleaned_data['country'] == "Italy"].reset_index()

grouped_italy_date = grouped_italy.groupby('date')['date', 'confirmed', 'deaths', 'active', 'recovered'].sum().reset_index()



grouped_us = cleaned_data[cleaned_data['country'] == "US"].reset_index()

grouped_us_date = grouped_us.groupby('date')['date', 'confirmed', 'deaths', 'active', 'recovered'].sum().reset_index()



grouped_ro = cleaned_data[cleaned_data['country'] == "Romania"].reset_index()

grouped_ro_date = grouped_ro.groupby('date')['date', 'confirmed', 'deaths', 'active', 'recovered'].sum().reset_index()
plot_titles = ['China', 'Italy', 'USA', 'Romania']



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



fig = px.line(grouped_ro_date, x="date", y="active", 

              title=f"Active Cases in {plot_titles[3].upper()} Over Time", 

              color_discrete_sequence=['#FFDF64'],

              height=500

             )

fig.show()
cleaned_data['state'] = cleaned_data['state'].fillna('')

temp = cleaned_data[[col for col in cleaned_data.columns if col != 'state']]



latest = temp[temp['date'] == max(temp['date'])].reset_index()

latest_grouped = latest.groupby('country')['confirmed', 'deaths', 'active', 'recovered'].sum().reset_index()
europe = list(['Austria','Belgium','Bulgaria','Croatia','Cyprus','Czechia','Denmark','Estonia','Finland','France','Germany','Greece','Hungary','Ireland',

               'Italy', 'Latvia','Luxembourg','Lithuania','Malta','Norway','Netherlands','Poland','Portugal','Romania','Slovakia','Slovenia',

               'Spain', 'Sweden', 'United Kingdom', 'Iceland', 'Russia', 'Switzerland', 'Serbia', 'Ukraine', 'Belarus',

               'Albania', 'Bosnia and Herzegovina', 'Kosovo', 'Moldova', 'Montenegro', 'North Macedonia'])



europe_grouped_latest = latest_grouped[latest_grouped['country'].isin(europe)]
fig = px.bar(europe_grouped_latest.sort_values('active', ascending=False)[:25][::-1], 

             x='active', y='country',

             title='Active Cases EUROPE', text='active', orientation='h')

fig.show()
fig = px.bar(latest_grouped.sort_values('recovered', ascending=False)[:10][::-1], 

             x='recovered', y='country',

             title='Recovered Cases Worldwide', text='recovered', orientation='h')

fig.show()
fig = px.bar(europe_grouped_latest.sort_values('recovered', ascending=False)[:25][::-1], 

             x='recovered', y='country',

             title='Recovered Cases in EUROPE', text='recovered', orientation='h', color_discrete_sequence=['cyan'])

fig.show()
temp = cleaned_data_ro.groupby('date')['recovered', 'deaths', 'active'].sum().reset_index()

temp = temp.melt(id_vars="date", value_vars=['recovered', 'deaths', 'active'],

                 var_name='case', value_name='count')





fig = px.line(temp, x="date", y="count", color='case',

             title='Cases in Romania over time: Line Plot', color_discrete_sequence = ['cyan', 'red', 'orange'])

fig.show()





fig = px.area(temp, x="date", y="count", color='case',

             title='Cases in Romania over time: Area Plot', color_discrete_sequence = ['cyan', 'red', 'orange'])

fig.show()
cleaned_latest = cleaned_data[cleaned_data['date'] == max(cleaned_data['date'])]

flg = cleaned_latest.groupby('country')['confirmed', 'deaths', 'recovered', 'active'].sum().reset_index()



flg['mortalityRate'] = round((flg['deaths']/flg['confirmed'])*100, 2)

temp = flg[flg['confirmed']>100]

temp = temp.sort_values('mortalityRate', ascending=False)



fig = px.bar(temp.sort_values(by="mortalityRate", ascending=False)[:15][::-1],

             x = 'mortalityRate', y = 'country', 

             title='Deaths per 100 Confirmed Cases', text='mortalityRate', height=800, orientation='h',

             color_discrete_sequence=['darkred']

            )

fig.show()



flg = cleaned_latest.groupby('country')['confirmed', 'deaths', 'recovered', 'active'].sum().reset_index()





flg['recoveryRate'] = round((flg['recovered']/flg['confirmed'])*100, 2)

temp = flg[flg['confirmed']>100]

temp = temp.sort_values('recoveryRate', ascending=False)



fig = px.bar(temp.sort_values(by="recoveryRate", ascending=False)[:15][::-1],

             x = 'recoveryRate', y = 'country', 

             title='Recoveries per 100 Confirmed Cases', text='recoveryRate', height=800, orientation='h',

             color_discrete_sequence=['#2ca02c']

            )

fig.show()
cleaned_latest = cleaned_data_ro[cleaned_data_ro['date'] == max(cleaned_data['date'])]

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



flg = cleaned_latest.groupby('country')['confirmed', 'deaths', 'recovered', 'active'].sum().reset_index()





flg['recoveryRate'] = round((flg['recovered']/flg['confirmed'])*100, 2)

temp = flg[flg['confirmed']>100]

temp = temp.sort_values('recoveryRate', ascending=False)



fig = px.bar(temp.sort_values(by="recoveryRate", ascending=False)[:10][::-1],

             x = 'recoveryRate', y = 'country', 

             title='Recoveries per 100 Confirmed Cases', text='recoveryRate', height=800, orientation='h',

             color_discrete_sequence=['#2ca02c']

            )

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