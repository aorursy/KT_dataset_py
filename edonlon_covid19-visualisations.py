import pandas as pd

import plotly.express as px

import plotly.graph_objects as go

import matplotlib

from ipywidgets import widgets
train = pd.read_csv("../input/covid19-global-forecasting-week-2/train.csv")

train.drop(columns=['Id'], inplace=True)   # Drop this column as not necessary for data exploration

train.head()
print("Number of countries included in dataset:", train.Country_Region.nunique())

print("Date Range is from ", train.Date.min(), "to", train.Date.max())

total = train.groupby(['Date']).sum().reset_index(drop=False)

total.columns = ['Date', 'Global Confirmed Cases', 'Global Fatalities']

total.head()
px.line(total, x="Date", y="Global Confirmed Cases", title="Global Confirmed Cases")
px.line(total, x="Date", y="Global Fatalities", title="Global Fatalities")
default_country = 'Australia'



country = widgets.Dropdown(

    options=list(train['Country_Region'].unique()),

    value=default_country,

    description='Country:',

)



temp_df = train[train.Country_Region == default_country].groupby("Date").sum().reset_index()



# Assign first figure widget

trace1 = go.Scatter(x=temp_df['Date'], y=temp_df['ConfirmedCases'], name='Confirmed Cases')

trace2 = go.Scatter(x=temp_df['Date'], y=temp_df['Fatalities'], name='Fatalities')

g = go.FigureWidget(data=[trace1, trace2], layout=go.Layout(title=dict(text='COVID-19 by Country')))



def response(change):

    temp_df = train[train.Country_Region == country.value].groupby("Date").sum().reset_index()

    

    x = temp_df['Date']

    y1 = temp_df['ConfirmedCases']

    y2 = temp_df['Fatalities']

    with g.batch_update():

        g.data[0].x = x

        g.data[0].y = y1

        g.data[1].y = y2

        g.layout.xaxis.title = 'Date'

        g.layout.yaxis.title = 'Confirmed Cases'



country.observe(response, names="value")



container = widgets.HBox([country])

widgets.VBox([container, g])
grouped = train.groupby(["Country_Region", "Date"]).sum().reset_index()

latest = grouped.groupby(["Country_Region"]).last().reset_index()

latest.head()
most_cases = latest.sort_values(by="ConfirmedCases", ascending=False).head(10)

most_cases.head()
fig = go.Figure()



fig.add_trace(go.Bar(x=most_cases['Country_Region'],

                y=most_cases['ConfirmedCases'],

                name='Confirmed Cases',

                marker_color='darkblue'))



fig.add_trace(go.Bar(x=most_cases['Country_Region'],

                y=most_cases['Fatalities'],

                name='Fatalities',

                marker_color='darkgrey'))



fig.update_layout(

    title='Top Ten Countries by Confirmed Cases',

    barmode='group',

)



fig.show()
top_countries = most_cases.Country_Region.values

print(top_countries)
pd.options.mode.chained_assignment = None  # Suppress chained assignment warning. default='warn'



top = train[train['Country_Region'].isin(top_countries)]

top['Country_Region'] = top['Country_Region'].astype('category')

top['Country_Region'].cat.set_categories(top_countries, inplace=True)    # Sort this dataframe by highest

top.sort_values(["Country_Region", "Date"], inplace=True)

top = top.groupby(["Country_Region", "Date"]).sum().reset_index()

top.columns = ['Country', 'Date', 'Confirmed Cases', 'Fatalities']

top.head()
fig = px.line(top, x="Date", y="Confirmed Cases", color='Country', title="Countries with most Confirmed Cases")

fig.show()
latest.head()
country_df = px.data.gapminder()[["country", "iso_alpha"]].drop_duplicates().reset_index(drop=True)

country_df.columns = ['Country_Region', 'Country_Code']

country_df.head()
latest_df = latest.merge(country_df, how='left', on='Country_Region')

latest_df.head()
latest_df[latest_df.isnull().any(axis=1)].shape
latest_df.at[162, 'Country_Code'] = 'USA'

latest_df[latest_df.Country_Region == "US"]
latest_df.head()
top_df = latest_df.sort_values(by="ConfirmedCases", ascending=False).head(10)

top_df
fig = px.scatter_geo(top_df, locations="Country_Code", color="ConfirmedCases",

                     hover_name="Country_Region", size="ConfirmedCases")

fig.show()