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
df_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv', parse_dates=['Date'])

df_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv', parse_dates=['Date'])



df_train.head()
df_test.head()
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
cleaned_df = pd.read_csv('/kaggle/input/corona-virus-report/covid_19_clean_complete.csv', parse_dates=['Date'])

#cleaned_df.head()

#cleaned_df.tail()
# rename a few columns

#cleaned_df.columns

cleaned_df.rename(columns={

                          'Province/State': 'state',

                          'Country/Region': 'country',

                          'Lat': 'lat',

                          'Long': 'long',

                          'Date': 'date',

                          'Confirmed': 'confirmed',

                          'Deaths': 'deaths',

                          'Recovered': 'recovered'

                          }, inplace=True)



# Cases

cases = ['confirmed', 'deaths', 'recovered', 'active']



cleaned_df['active'] = cleaned_df['confirmed']- cleaned_df['deaths'] - cleaned_df['recovered']
cleaned_df.sort_values('country', ascending=True).country.unique().tolist()
cleaned_df.info()
# filling missing values

cleaned_df['state'] = cleaned_df['state'].fillna('')

# fill missing count if any with 0

cleaned_df['state'] =cleaned_df[['state']].fillna(0)
cleaned_df.head()
import plotly.express as px

import plotly.io as pio

import plotly.graph_objects as go

pio.templates.default='plotly_dark'

from plotly.subplots import make_subplots
temp = cleaned_df.groupby('date')['confirmed', 'deaths', 'recovered', 'active'].sum().reset_index()



from plotly.subplots import make_subplots

import plotly.graph_objects as go



# Reference: https://plotly.com/python/line-and-scatter/



fig = make_subplots(rows=1, cols=4, subplot_titles=('Confirmed', 'Deaths', 'Recovered', 'Active'))



trace1 = go.Scatter(

                   x = temp['date'],

                   y = temp['confirmed'],

                   name='confirmed',

                   #mode='lines+markers' 

                   )



trace2 = go.Scatter(

                   x = temp['date'],

                   y = temp['deaths'],

                   name='deaths'

                   )



trace3 = go.Scatter(

                   x = temp['date'],

                   y = temp['recovered'],

                   name='recovered'

                   )



trace4 = go.Scatter(

                   x = temp['date'],

                   y = temp['active'],

                   name='active' 

                   )

fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 2)

fig.append_trace(trace3, 1, 3)

fig.append_trace(trace4, 1, 4)



# change the layout

fig.update_layout(template='plotly_dark', title_text= '<b>Global Spread of Coronavirus over time </b>') #<b> for bold text



#fig = px.scatter(temp, x=temp['date'], y=temp['confirmed'])

fig.show()
# Number of confirmed cases across the globe



confirmed_globe = cleaned_df.groupby('date')['date', 'confirmed', 'deaths'].sum().reset_index()



import plotly.express as px

import plotly.io as pio

import plotly.graph_objects as go

pio.templates.default='plotly_dark'

from plotly.subplots import make_subplots



# line chart

fig = px.line(confirmed_globe, x='date', y='confirmed', title= 'Confirmed Cases Wordwide over time')

fig.show()



# line chart - logarithmic

fig = px.line(confirmed_globe, x='date', y = 'confirmed', log_y=True,\

              title='Confirmed Cases World Wide (Logarithmic Scale) over time')

fig.show()
#cleaned_df.head()

confirmed_country = cleaned_df.groupby('country')['country', 'confirmed'].sum().reset_index().sort_values('confirmed', ascending=False)

#confirmed_country
fig = px.bar(confirmed_country, x='country', y='confirmed', title = 'Confirmed case(Log Scale) by country', log_y=True)

fig.show()



fig = px.bar(confirmed_country.head(), x='country', y='confirmed', color='country',title = 'Confirmed cases in top 5 countries')

fig.show()
grouped_CN = cleaned_df[cleaned_df['country'] == 'China'].groupby('date')['date', 'confirmed', 'deaths', 'recovered', 'active'].sum().reset_index()



grouped_IT = cleaned_df[cleaned_df['country'] == 'Italy'].groupby('date')['date', 'confirmed', 'deaths', 'recovered', 'active'].sum().reset_index()



grouped_IR = cleaned_df[cleaned_df['country'] == 'Iran'].groupby('date')['date', 'confirmed', 'deaths', 'recovered', 'active'].sum().reset_index()



grouped_IN = cleaned_df[cleaned_df['country'] == 'India'].groupby('date')['date', 'confirmed', 'deaths', 'recovered', 'active'].sum().reset_index()



cleaned_rest = cleaned_df[~cleaned_df['country'].isin(['China, Italy', 'Iran', 'India'])]

grouped_rest = cleaned_rest.groupby('date')['date', 'confirmed', 'deaths', 'recovered', 'active'].sum().reset_index()
# plots

plot_titles = ['CN', 'IT', 'IR', 'IN', 'Rest of the world']

plot_string = 'Confirmed Cases for :{} '



fig1 = px.line(grouped_CN, x='date', y='confirmed', color_discrete_sequence =['#636EFA'], title= plot_string.format(plot_titles[0]))

fig1.show()



fig2 = px.line(grouped_IT, x='date', y='confirmed', color_discrete_sequence =['#EF553B'], title= plot_string.format(plot_titles[1]))

fig2.show()



fig3 = px.line(grouped_IR, x='date', y='confirmed', color_discrete_sequence =['#00CC96'], title= plot_string.format(plot_titles[2]))

fig3.show()



fig4 = px.line(grouped_IN, x='date', y='confirmed', color_discrete_sequence =['#AB63FA'], title= plot_string.format(plot_titles[3]))

fig4.show()



fig5 = px.line(grouped_rest, x='date', y='confirmed', color_discrete_sequence =['#FFA15A'], title= plot_string.format(plot_titles[4]))

fig5.show()
# get the latest count by countries or take the max count by country



latest_cnt = cleaned_df[cleaned_df['date'] == max(cleaned_df['date'])]

latest_grouped =latest_cnt.groupby('country')['confirmed', 'deaths', 'recovered', 'active'].sum().reset_index()

#latest_grouped

#latest_grouped[latest_grouped['country']=='Australia']
#grouped_country = cleaned_df.groupby('country')['confirmed', 'deaths'].max().reset_index()

#grouped_country[grouped_country['country']=='Australia']
fig = px.choropleth(latest_grouped, \

                    locations='country', locationmode='country names', \

                    hover_name='country', color='confirmed', range_color=[1, 300],\

                    title='Countries with Confirmed Cases',\

                    #scope='europe'

                   )

fig.show()
europe_filter = ['Austria','Belgium','Bulgaria','Croatia','Cyprus','Czech Republic','Denmark','Estonia','Finland','France','Germany','Greece','Hungary','Ireland',

          'Italy', 'Latvia','Luxembourg','Lithuania','Malta','Norway','Netherlands','Poland','Portugal','Romania','Slovakia','Slovenia',

         'Spain', 'Sweden', 'United Kingdom', 'Iceland', 'Russia', 'Switzerland', 'Serbia', 'Ukraine', 'Belarus']



europe_grouped = latest_grouped[latest_grouped['country'].isin(europe_filter)]

europe_grouped.head(5)
fig = px.choropleth(europe_grouped, locations='country', locationmode='country names',\

             color='confirmed', range_color=[europe_grouped['confirmed'].min(), europe_grouped['confirmed'].max()],\

             color_continuous_scale='Viridis',\

             title='European countries with confirmed cases',\

             scope='europe')

fig.show()
# Another way of creating the same plot was to use 'scope'

fig = px.choropleth(latest_grouped, \

                    locations='country', locationmode='country names', \

                    hover_name='country', color='confirmed', range_color=[1, 5000],\

                    color_continuous_scale='portland',\

                    title='Countries with Confirmed Cases',\

                    scope='europe'

                   )

fig.show()
fig = px.choropleth(latest_grouped, \

                    locations='country', locationmode='country names', \

                    hover_name='country', color='confirmed', range_color=[1, 5000],\

                    color_continuous_scale='portland',\

                    title='Asian Countries with Confirmed Cases',\

                    scope='asia'

                   )

fig.show()
#latest_grouped.sort_values('confirmed',ascending=False).head(20)[::-1]
fig = px.bar(latest_grouped.sort_values('confirmed',ascending=False).head(20)[::-1],\

             x = 'confirmed', y = 'country',\

             title='Confirmed Cases worldwide',\

             orientation='h',\

             text='confirmed')

fig.show()
fig = px.bar(europe_grouped.sort_values('confirmed', ascending=False).head(20)[::-1],\

             x='confirmed', y='country',\

             orientation='h',\

             title='Confirmed cases in Europe',\

             text = 'confirmed')

fig.show()
usa = cleaned_df[cleaned_df['country']=='US']

usa_latest = usa[usa['date'] == max(usa['date'])]



usa_latest = usa_latest.groupby('state')['confirmed', 'deaths', 'recovered', 'active'].max().reset_index()



fig = px.bar(usa_latest.sort_values('confirmed', ascending=False).head(20)[::-1],\

             x = 'confirmed', y='state',\

             title='Confirmed cases in USA states',\

             orientation='h', text='confirmed',\

             color_discrete_sequence=['#D62728']) # https://plot.ly/python/discrete-color/

fig.show()
global_cases = cleaned_df.groupby('date')['date', 'confirmed', 'deaths'].sum().reset_index()



# linear scale

fig = px.line(global_cases, x='date', y = 'deaths',\

             title= 'Deaths Global (linear scale)',\

             color_discrete_sequence=['#F6222E'])

fig.show()



#Logarithmic scale

fig = px.line(global_cases, x='date', y = 'deaths',\

             title= 'Deaths Global (Logarithmic Scale)',\

             color_discrete_sequence=['#F6222E'],\

             log_y=True)

fig.show()
# plots

plot_titles = ['CN', 'IT', 'IR', 'IN', 'Rest of the world']

plot_string = 'Deaths for :{} '



fig1 = px.line(grouped_CN, x='date', y='deaths', color_discrete_sequence =['#636EFA'], title= plot_string.format(plot_titles[0]))

fig1.show()



fig2 = px.line(grouped_IT, x='date', y='deaths', color_discrete_sequence =['#EF553B'], title= plot_string.format(plot_titles[1]))

fig2.show()



fig3 = px.line(grouped_IR, x='date', y='deaths', color_discrete_sequence =['#00CC96'], title= plot_string.format(plot_titles[2]))

fig3.show()



fig4 = px.line(grouped_IN, x='date', y='deaths', color_discrete_sequence =['#AB63FA'], title= plot_string.format(plot_titles[3]))

fig4.show()



fig5 = px.line(grouped_rest, x='date', y='deaths', color_discrete_sequence =['#FFA15A'], title= plot_string.format(plot_titles[4]))

fig5.show()
#latest_grouped

fig = px.choropleth(latest_grouped, locations='country', locationmode='country names',\

             hover_name='deaths', color='deaths', range_color=[1, 100],\

             #color_continuous_scale='Viridis',\

             title= 'Countries with reported deaths')



fig.show()
fig = px.choropleth(latest_grouped, locations='country', locationmode='country names',\

             hover_name='deaths', color='deaths', range_color=[1, 100],\

             color_continuous_scale='Viridis',\

             title= 'Countries with reported deaths - Europe',\

             scope='europe')



fig.show()
fig = px.choropleth(latest_grouped, locations='country', locationmode='country names',\

             hover_name='deaths', color='deaths', range_color=[1, 100],\

             color_continuous_scale='portland',\

             title= 'Countries with reported deaths - Asia',\

             scope='asia')



fig.show()
fig = px.bar(latest_grouped.sort_values('deaths', ascending=False).head(20)[::-1],\

             x='deaths', y='country',\

             title= 'Confirmed deaths by Country',\

             orientation='h',

             text='deaths')

fig.show()
#europe_grouped

fig = px.bar(europe_grouped.sort_values('deaths', ascending=False).head(20)[::-1],\

            x= 'deaths', y='country',

            title='Confirmed deaths in Europe',\

            orientation='h',\

            text='deaths',\

            color_discrete_sequence=['#22FFA7']

            )

fig.show()
#usa_latest



fig = px.bar(usa_latest.sort_values('deaths', ascending=False).head(10)[::-1],\

             x='deaths', y='state',\

             title= 'Confirmed deaths by state in USA - Top 10',\

             orientation='h',\

             text='deaths',\

             )

fig.show()
active_cases = cleaned_df.groupby('date')['date', 'confirmed', 'deaths', 'recovered', 'active'].sum().reset_index()

fig = px.line(active_cases, x='date', y='active', title='active cases worlwide')

fig.show()
# plots

plot_titles = ['CN', 'IT', 'IR', 'IN', 'Rest of the world']

plot_string = 'active Cases for :{} '



fig1 = px.line(grouped_CN, x='date', y='active', color_discrete_sequence =['#636EFA'], title= plot_string.format(plot_titles[0]))

fig1.show()



fig2 = px.line(grouped_IT, x='date', y='active', color_discrete_sequence =['#EF553B'], title= plot_string.format(plot_titles[1]))

fig2.show()



fig3 = px.line(grouped_IR, x='date', y='active', color_discrete_sequence =['#00CC96'], title= plot_string.format(plot_titles[2]))

fig3.show()



fig4 = px.line(grouped_IN, x='date', y='active', color_discrete_sequence =['#AB63FA'], title= plot_string.format(plot_titles[3]))

fig4.show()



fig5 = px.line(grouped_rest, x='date', y='active', color_discrete_sequence =['#FFA15A'], title= plot_string.format(plot_titles[4]))

fig5.show()
fig = px.choropleth(latest_grouped, locations='country', locationmode='country names',\

                    hover_name='country', color='active', range_color=[1,2000],\

                    color_continuous_scale='portland',\

                    title='Active cases worldwide')

fig.show()
fig = px.choropleth(latest_grouped, locations='country', locationmode='country names',\

                    hover_name='country', color='active', range_color=[1,2000],\

                    color_continuous_scale='portland',\

                    title='Active cases - Europe',\

                    scope='europe')

fig.show()
fig = px.choropleth(latest_grouped, locations='country', locationmode='country names',\

                    hover_name='country', color='active', range_color=[1,2000],\

                    color_continuous_scale='portland',\

                    title='Active cases Asia',\

                    scope='asia')

fig.show()
fig = px.bar(latest_grouped.sort_values('active', ascending=False).head(20)[::-1],\

             x = 'active', y='country',\

             text='active',\

             title='Active cases worldwide',\

             orientation='h'

            )

fig.show()
fig = px.bar(europe_grouped.sort_values('active', ascending=False).head(20)[::-1],\

             x = 'active', y='country',\

             text='active',\

             title='Active cases - Europe',\

             orientation='h'

            )

fig.show()
fig = px.bar(usa_latest.sort_values('active', ascending=False).head(10)[::-1],\

             x='active', y='state',\

             title= 'Confirmed active cases by state in USA - Top 10',\

             orientation='h',\

             text='active',\

             )

fig.show()

#usa_latest
#active_cases

fig = px.line(active_cases, x='date', y='recovered', title='recovered cases worlwide')

fig.show()
plot_titles = ['CN', 'IT', 'IR', 'IN', 'Rest of the world']

plot_string = 'recovered Cases for :{} '



fig1 = px.line(grouped_CN, x='date', y='recovered', color_discrete_sequence =['#636EFA'], title= plot_string.format(plot_titles[0]))

fig1.show()



fig2 = px.line(grouped_IT, x='date', y='recovered', color_discrete_sequence =['#EF553B'], title= plot_string.format(plot_titles[1]))

fig2.show()



fig3 = px.line(grouped_IR, x='date', y='recovered', color_discrete_sequence =['#00CC96'], title= plot_string.format(plot_titles[2]))

fig3.show()



fig4 = px.line(grouped_IN, x='date', y='recovered', color_discrete_sequence =['#AB63FA'], title= plot_string.format(plot_titles[3]))

fig4.show()



fig5 = px.line(grouped_rest, x='date', y='recovered', color_discrete_sequence =['#FFA15A'], title= plot_string.format(plot_titles[4]))

fig5.show()
fig = px.choropleth(latest_grouped, locations='country', locationmode='country names',\

                    hover_name='country', color='recovered', range_color=[1,2000],\

                    color_continuous_scale='portland',\

                    title='recovered cases worldwide')

fig.show()
fig = px.choropleth(latest_grouped, locations='country', locationmode='country names',\

                    hover_name='country', color='recovered', range_color=[1,2000],\

                    color_continuous_scale='portland',\

                    title='recovered cases - Europe',\

                    scope='europe')

fig.show()
fig = px.choropleth(latest_grouped, locations='country', locationmode='country names',\

                    hover_name='country', color='recovered', range_color=[1,2000],\

                    color_continuous_scale='portland',\

                    title='recovered cases Asia',\

                    scope='asia')

fig.show()
fig = px.bar(latest_grouped.sort_values('recovered', ascending=False).head(20)[::-1],\

             x = 'recovered', y='country',\

             text='recovered',\

             title='recovered cases worldwide',\

             orientation='h'

            )

fig.show()
fig = px.bar(latest_grouped.sort_values('recovered', ascending=False).head(20)[::-1],\

             x = 'recovered', y='country',\

             text='recovered',\

             title='recovered cases worldwide',\

             orientation='h'

            )

fig.show()
fig = px.bar(europe_grouped.sort_values('recovered', ascending=False).head(20)[::-1],\

             x = 'recovered', y='country',\

             text='recovered',\

             title='recovered cases - Europe',\

             orientation='h'

            )

fig.show()
usa_latest['recovered'].unique()
fig = px.bar(usa_latest.sort_values('recovered', ascending=False).head(10)[::-1],\

             x='recovered', y='state',\

             title= 'Recovered cases by state in USA - Top 10',\

             orientation='h',\

             text='recovered',\

             )

fig.show()
cleaned_df.head()
# temp = cleaned_df.melt(id_vars=['date', 'country'], value_vars=['confirmed', 'deaths', 'recovered', 'active'])

# fig = px.line(temp.loc[temp.country == 'China'], x= 'date', y= 'value', color='variable')

# fig.show()



all_cases = cleaned_df.groupby('date')['confirmed', 'deaths', 'recovered', 'active'].sum().reset_index()

#all_cases.head()
all_cases_melt = all_cases.melt(id_vars='date', value_vars=['confirmed', 'deaths', 'recovered', 'active'], var_name='cases', value_name='count')



# line plot

fig1 = px.line(all_cases_melt, x='date', y='count', color='cases',\

             title='Cases over time worldwide - Line Plot')

fig1.show()



# area plot

fig2 = px.area(all_cases_melt, x='date', y='count', color='cases',\

             title='Cases over time worldwide - Area Plot')

fig2.show()
china_cases = cleaned_df.loc[cleaned_df.country=='China'].groupby('date')['confirmed', 'deaths', 'recovered', 'active'].sum().reset_index()

china_cases.head()
china_cases_melt = china_cases.melt(id_vars='date', value_vars=['confirmed', 'deaths', 'recovered', 'active'],\

                                   var_name='cases', value_name='count')

china_cases_melt

fig = px.line(china_cases_melt, x='date', y='count', color='cases', title= 'CN cases over time -Line')

fig.show()



fig = px.area(china_cases_melt, x='date', y='count', color='cases', title= 'CN cases over time - Area')

fig.show()
except_china_cases = cleaned_df.loc[cleaned_df.country!='China'].groupby('date')['confirmed', 'deaths', 'recovered', 'active'].sum().reset_index()

except_china_cases.head()



except_china_cases_melt = except_china_cases.melt(id_vars='date', value_vars=['confirmed', 'deaths', 'recovered', 'active'],\

                                   var_name='cases', value_name='count')

except_china_cases_melt

fig = px.line(except_china_cases_melt, x='date', y='count', color='cases', title= 'Except CN cases over time -Line')

fig.show()



fig = px.area(except_china_cases_melt, x='date', y='count', color='cases', title= 'Except CN cases over time - Area')

fig.show()
# get the latest count

latest = cleaned_df.loc[cleaned_df.date == max(cleaned_df['date'])]

#cleaned_df.groupby('country')

latest_cnt = latest.groupby('country')['confirmed', 'deaths', 'recovered', 'active'].max().reset_index()



# mortality rate = deaths / confirmed

latest_cnt['mortalityRate'] = round(latest_cnt['deaths'] / latest_cnt['confirmed'] *100, 2)

latest_cnt



# plots

fig = px.bar(latest_cnt.sort_values('mortalityRate', ascending=False).head(10)[::-1],\

        x = 'mortalityRate', y= 'country',\

        orientation='h',\

        text = 'mortalityRate',\

        color_discrete_sequence=['red'],\

        title = 'highest mortality rate countries - for every 100 confirmed cases')



fig.show()

#latest_cnt.sort_values('mortalityRate', ascending=True).head(20).sort_values('confirmed', ascending=False)

#latest_cnt.query('deaths == 0 and confirmed > 100').sort_values('confirmed', ascending=False).head(5)
latest_cnt.query('mortalityRate == 0 and confirmed > 100').sort_values('confirmed', ascending=False).head(5).style.background_gradient(cmap='Greens')
# recovery_rate = recovered / confirmed

latest_cnt['recoveryRate'] = round(latest_cnt['recovered'] / latest_cnt['confirmed'] *100, 2)

latest_cnt



fig = px.bar(latest_cnt.sort_values('recoveryRate', ascending=False).head(10)[::-1]\

             , x = 'recoveryRate', y='country', orientation='h',\

            text='recovered',\

            title='Recovery rates for every 100 confirmed cases - wordlwide',\

            color_discrete_sequence=['lightgreen'])

fig.show()
temp = latest_cnt[latest_cnt['confirmed'] > 100]

temp.sort_values('recoveryRate', ascending=True)[:20].sort_values('confirmed', ascending=False)[:10].style.background_gradient(cmap='Reds')
temp = cleaned_df.groupby(['date', 'country'])['confirmed', 'deaths', 'recovered', 'active'].max().reset_index()

# size of the circle

temp['size'] = temp['confirmed'].pow(0.3)

temp['date'] = temp['date'].dt.strftime('%m-%d-%Y')



# plots

fig = px.scatter_geo(temp, locations='country', locationmode= 'country names',\

                    hover_name='country', size = 'size', color='confirmed',\

                    range_color=[1, 2000], color_continuous_scale='portland',\

                    projection='natural earth',\

                    animation_frame='date',\

                    title = 'COVID-19 confirmed cases over time - worldwide')

fig.show()
# plots

fig = px.scatter_geo(temp, locations='country', locationmode= 'country names',\

                    hover_name='country', size = 'size', color='confirmed',\

                    range_color=[1, 2000], color_continuous_scale='portland',\

                    projection='natural earth',\

                    animation_frame='date',\

                    title = 'COVID-19 confirmed cases over time - Europe',\

                    scope='europe')

fig.show()
temp = cleaned_df.groupby(['date', 'country'])['confirmed', 'deaths', 'recovered', 'active'].max().reset_index()

# size of the circle

temp['size'] = temp['deaths'].pow(0.3)

temp['date'] = temp['date'].dt.strftime('%m-%d-%Y')



# plots

fig = px.scatter_geo(temp, locations='country', locationmode= 'country names',\

                    hover_name='country', size = 'size', color='deaths',\

                    range_color=[1, 2000], color_continuous_scale='portland',\

                    projection='natural earth',\

                    animation_frame='date',\

                    title = 'COVID-19 deaths over time - worldwide')

fig.show()
fig = px.scatter_geo(temp, locations='country', locationmode= 'country names',\

                    hover_name='country', size = 'size', color='deaths',\

                    range_color=[1, 2000], color_continuous_scale='portland',\

                    projection='natural earth',\

                    animation_frame='date',\

                    title = 'COVID-19 deaths over time - Europe',\

                    scope='europe')

fig.show()
temp = cleaned_df.groupby(['date', 'country'])['confirmed', 'deaths', 'recovered', 'active'].max().reset_index()

# size of the circle

temp['size'] = temp['recovered'].pow(0.3)

temp['date'] = temp['date'].dt.strftime('%m-%d-%Y')



# plots

fig = px.scatter_geo(temp, locations='country', locationmode= 'country names',\

                    hover_name='country', size = 'size', color='recovered',\

                    range_color=[1, 2000], color_continuous_scale='greens',\

                    projection='natural earth',\

                    animation_frame='date',\

                    title = 'COVID-19 recoveries over time - worldwide')

fig.show()
cleaned_df[cleaned_df['country'] == 'US']