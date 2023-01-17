csv_url = 'https://www.kaggle.com/imdevskp/corona-virus-report/download'
# csv_filename = 'complete_data_new_format.csv' 
csv_filename = 'covid_19_clean_complete.csv'

import pandas as pd
csv_df = pd.read_csv('../input/corona-virus-report/' + csv_filename)
from datetime import datetime,date

# Format date
formatter_string = "%m/%d/%y" 
try:
    # To allow re-run instead of throwing exception
    csv_df['date'] = pd.to_datetime(csv_df.Date, format=formatter_string)
    csv_df['Date'] = csv_df['date'].dt.strftime('%Y-%m-%d')
except:
    pass

# Calculate Active/Recovered
csv_df['Active'] = csv_df['Confirmed'] - csv_df['Deaths'] - csv_df['Recovered']

# Cleanup Province/State which has NaN
csv_df['Province/State'] = csv_df['Province/State'].fillna('-')

# Cleanup Confirmed and Death can be negative
csv_df['Confirmed'] = csv_df['Confirmed'].apply(lambda x: x if x > 0 else 0)
csv_df['Deaths'] = csv_df['Deaths'].apply(lambda x: x if x > 0 else 0)

data_dates = csv_df.date.unique()
max_data_date = max(data_dates)
print("Last date in {csv_filename} is {max_data_date}".format(max_data_date=max_data_date,csv_filename=csv_filename))
title = 'COVID-2019'
# d = pd.to_datetime(str(max_data_date)).strftime('%Y-%m-%d')
chart_title = title + ' as of ' + pd.to_datetime(str(max_data_date)).strftime('%Y-%m-%d')

legend_orientation="v"
chart_height = 900
import plotly.offline as py
import plotly.graph_objs as go

py.offline.init_notebook_mode(connected=True)
# cases_df = csv_df[['date','Confirmed','Deaths','Recovered','Active']].groupby('date').sum() # clean.csv
cases_df = csv_df[['date','Confirmed','Deaths','Active','Recovered']].groupby('date').sum()

data = []
# categories = ['Confirmed', 'Deaths', 'Recovered', 'Active'] # clean.csv
categories = ['Confirmed', 'Deaths','Active','Recovered']

for category in categories:
    trace = go.Scatter(
        x=cases_df.index, 
        y=cases_df[category],
        mode="markers+lines",
        name = category
    )
    data.append(trace)

py.iplot({
    "data": data,
    "layout": go.Layout(title=chart_title)
})
df = csv_df.loc[csv_df.date==max_data_date]
# by_country = df[['Country/Region','Confirmed','Deaths', 'Recovered','Active']].groupby(['Country/Region']).sum() # clean.csv
by_country = df[['Country/Region','Confirmed','Deaths','Active','Recovered']].groupby(['Country/Region']).sum() # clean.csv
by_country.sort_values(by='Confirmed',ascending=False,inplace=True)

countries = by_country.index.to_list()
print('Number of countries with confirmed cases = ',len(countries))

# Looks lot have hit a limit of Sunburst chart
max_countries = 173
countries = countries[:max_countries]
ids = countries
labels = countries
parents = [title] * len(countries)
values = by_country.Confirmed.to_list()[:max_countries]

classifications = by_country.columns.drop('Confirmed')

for cty in countries: 
    for c in classifications:
        ids = ids + [cty + '_' + c]
        parents = parents + [cty]
        labels = labels + [c]
        values = values + [by_country.loc[cty][c]]

trace = go.Sunburst(
    ids=ids,
    labels=labels,
    parents=parents,
    values=values,
    branchvalues="total",
    outsidetextfont={"size": 20, "color": "#377eb8"},
#     leaf={"opacity": 0.4},
    marker={"line": {"width": 2}}
)

layout = go.Layout(
    title = chart_title + "<br>(click on country)",
    margin = go.layout.Margin(t=100, l=0, r=0, b=0),
    sunburstcolorway=["#636efa","#ef553b","#00cc96"]
)

fig = go.Figure([trace], layout)

py.iplot(fig)
import plotly.express as px
import numpy as np

# Log confirmed values to show cases, otherwise countries with smaller number of cases will be a pixel
csv_df['Confirmed_log'] = np.log(1+csv_df['Confirmed'])
csv_df['Confirmed_log'] = csv_df['Confirmed_log'].fillna(0)
csv_df['color'] = 'fuschia'

a_df = csv_df.sort_values(by=['date'])
fig = px.scatter_mapbox(a_df,
                    animation_frame='Date',
                    animation_group="Country/Region",
                    lat="Lat", lon="Long", hover_name="Province/State", 
                    hover_data=["Province/State","Country/Region","Confirmed","Deaths","Recovered"], # clean.csv
                    size="Confirmed_log",
                    color_discrete_sequence=['hsla(360, 100%, 50%, 0.5)'], # lightsalmon, rgb(255,70,0)
                    zoom=0.5
                    )
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(title=chart_title
                 , width = 900, height = 600)
fig.show()
top_country_confirmed = by_country.sort_values(by="Confirmed",ascending=False).index.to_list()
top_country_deaths = by_country.sort_values(by="Deaths",ascending=False).index.to_list()
# top_country_recovered = by_country.sort_values(by="Recovered",ascending=False).index.to_list() # clean.csv
top_country = {
    'Confirmed' : top_country_confirmed,
    'Deaths' : top_country_deaths,
#     'Recovered' : top_country_recovered, # clean.csv
}
top_n = 20
print('Top {} Confirmed countries : {}'.format(top_n,top_country['Confirmed'][:top_n]))

top_n = 6
# country_cases_df = csv_df[['Country/Region','date','Confirmed','Deaths','Recovered']].groupby(['Country/Region','date']).sum() # clean.csv
country_cases_df = csv_df[['Country/Region','date','Confirmed','Deaths','Active','Recovered']].groupby(['Country/Region','date']).sum()

data = []
countries = np.sort(csv_df['Country/Region'].unique())
category = 'Confirmed'
for c in countries:
    country = country_cases_df.loc[c]
    
    visible_flag = 'legendonly'
    if c in top_country[category][:top_n]:
        visible_flag = None
    trace = go.Scatter(
        x=country.index.to_list(), 
        y=country[category],
        mode="markers+lines",
        name = c,
        text = c,
        visible=visible_flag
    )
    data.append(trace)

py.iplot({
    "data": data,
    "layout": go.Layout(title='<B>{}</B><BR>\
        <I>Top {} countries with {} shown. Click legend to show others</I>'.format(chart_title,top_n,category
             , height = chart_height,legend_orientation=legend_orientation))
})
country_cases_df['Confirmed_1st_diff'] = country_cases_df['Confirmed'].diff().fillna(0)

# Cleanup Confirmed_diff negative to 0
country_cases_df['Confirmed_1st_diff'] = country_cases_df['Confirmed_1st_diff'].apply(lambda x: x if x > 0 else 0)

# average over n intervals
n = 7
country_cases_df['Confirmed_ewm'] = pd.Series.ewm(country_cases_df['Confirmed_1st_diff'], span=7).mean()


data = []
countries = np.sort(csv_df['Country/Region'].unique())
for c in countries:
    country = country_cases_df.loc[c]

    visible = 'legendonly'
    if c in top_country['Confirmed'][:top_n]:
        visible = None
    trace = go.Scatter(
        x=country.index.to_list(), 
        y=country['Confirmed_ewm'],
        mode="markers+lines",
        name = c,
        visible=visible
    )
    data.append(trace)

py.iplot({
    "data": data,
    "layout": go.Layout(title='<B>{} - ewm</B><BR><I>\
    Only top {} countriesshown</I>'.format(chart_title,top_n)
                       , legend_orientation=legend_orientation)
})
category_n = 100
category = 'Confirmed'

country_cases_df = csv_df[['Province/State','Country/Region','date','Confirmed','Deaths','Recovered','Active']].groupby(['Country/Region','date']).sum()
country_cases_df_growth = country_cases_df.loc[country_cases_df[category] > category_n]

# Insert a placeholder column
country_cases_df_growth.insert(loc=0, column='ID', value=0)
country_cases_df_growth.reset_index().set_index(['Country/Region'])

# Set ID counter for each country which will be used as index
pd.options.mode.chained_assignment = None
for c in countries:
    try:
        country_cases_df_growth.loc[c,'ID'] = np.arange(len(country_cases_df_growth.loc[c]))
    except:
        pass

country_cases_df_growth = country_cases_df_growth.reset_index().set_index(['Country/Region','ID'])
data = []
for c in countries:
    try:
        country = country_cases_df_growth.loc[c]

        visible_flag = 'legendonly'
        if c in top_country[category][:top_n]:
            visible_flag = None
        trace = go.Scatter(
            x=country.index.to_list(), 
            y=country[category],
            mode="markers+lines",
            name = c,
            text = country.date,
            visible=visible_flag
        )
        data.append(trace)
    except:
        pass

py.iplot({
    "data": data,
    "layout": go.Layout(title='<B>{chart_title}</B><BR><I>\
    Shifted to align increase - {category} > {category_n}<BR>\
    Top {top_n} countries with {category} shown. Click legend to show others</I>'.format(chart_title=chart_title,top_n=top_n,category=category,category_n=category_n)
                       , legend_orientation=legend_orientation)
})
category_n = 100
category = 'Deaths'

country_cases_df = csv_df[['Province/State','Country/Region','date','Confirmed','Deaths','Recovered','Active']].groupby(['Country/Region','date']).sum()
country_cases_df_growth = country_cases_df.loc[country_cases_df[category] > category_n]

# Insert a placeholder column
country_cases_df_growth.insert(loc=0, column='ID', value=0)
country_cases_df_growth.reset_index().set_index(['Country/Region'])

# Set ID counter for each country which will be used as index
pd.options.mode.chained_assignment = None
for c in countries:
    try:
        country_cases_df_growth.loc[c,'ID'] = np.arange(len(country_cases_df_growth.loc[c]))
    except:
        pass

country_cases_df_growth = country_cases_df_growth.reset_index().set_index(['Country/Region','ID'])
data = []
for c in countries:
    try:
        country = country_cases_df_growth.loc[c]

        visible_flag = 'legendonly'
        if c in top_country[category][:top_n]:
            visible_flag = None
        trace = go.Scatter(
            x=country.index.to_list(), 
            y=country[category],
            mode="markers+lines",
            name = c,
            text = country.date,
            visible=visible_flag
        )
        data.append(trace)
    except:
        pass

py.iplot({
    "data": data,
    "layout": go.Layout(title='<B>{chart_title}</B><BR><I>\
    Shifted to align increase - {category} > {category_n}<BR>\
    Top {top_n} countries with {category} shown. Click legend to show others</I>'.format(chart_title=chart_title,top_n=top_n,category=category,category_n=category_n)
                       , legend_orientation=legend_orientation)
})
country_cases_df['Confirmed_pct'] = country_cases_df.pct_change().fillna(0)['Confirmed']
country_cases_df_growth = country_cases_df.loc[country_cases_df['Confirmed_pct'] > 0]

# Insert a placeholder column
country_cases_df_growth.insert(loc=0, column='ID', value=0)
country_cases_df_growth.reset_index().set_index(['Country/Region'])

# Set ID counter for each country which will be used as index
pd.options.mode.chained_assignment = None
for c in countries:
    try:
        country_cases_df_growth.loc[c,'ID'] = np.arange(len(country_cases_df_growth.loc[c]))
    except:
        pass

country_cases_df_growth = country_cases_df_growth.reset_index().set_index(['Country/Region','ID'])
data = []
category = 'Confirmed'
for c in countries:
    try:
        country = country_cases_df_growth.loc[c]

        visible_flag = 'legendonly'
        if c in top_country[category][:top_n]:
            visible_flag = None
        trace = go.Scatter(
            x=country.index.to_list(), 
            y=country[category],
            mode="markers+lines",
            name = c,
            text = country.date,
            visible=visible_flag
        )
        data.append(trace)
    except:
        pass

py.iplot({
    "data": data,
    "layout": go.Layout(title='<B>{}</B><BR>\
        <I>Shifted to align increase<BR>\
        Top {} countries with {} shown. Click legend to show others</I>'.format(chart_title,top_n,category
             , height = chart_height,legend_orientation=legend_orientation))
})
data = []
category = 'Deaths'
for c in countries:
    try:
        country = country_cases_df_growth.loc[c]

        visible_flag = 'legendonly'
        if c in top_country[category][:top_n]:
            visible_flag = None
        trace = go.Scatter(
            x=country.index.to_list(), 
            y=country[category],
            mode="markers+lines",
            name = c,
            text = country.date,
            visible=visible_flag
        )
        data.append(trace)
    except:
        pass

py.iplot({
    "data": data,
    "layout": go.Layout(title='<B>{}</B><BR>\
        <I>Shifted to align increase<BR>\
        Top {} countries with {} shown. Click legend to show others</I>'.format(chart_title,top_n,category
            , height = chart_height,legend_orientation=legend_orientation))
})