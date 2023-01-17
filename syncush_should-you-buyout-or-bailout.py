import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import ast
import plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.figure_factory as ff
import plotly.graph_objs as go
from plotly.graph_objs import *
from geopy.geocoders import Nominatim
import os
import re
import random
init_notebook_mode()
print("Buyouts Files:\n")
print('\t\n'.join(['\t' + x for x in os.listdir("../input/sf-buyout-agreements/")]))
print("\n")
print("Crimes Files:\n")
print('\n'.join(['\t' + x for x in os.listdir("../input/sf-police-calls-for-service-and-incidents/")]))
def prep_data(buyout_df):
    buyout_df.set_index('Case Number', inplace=True)
    buyout_df["pre_buyout_disclsure_date"] = pd.to_datetime(buyout_df['Pre Buyout Disclosure Date'], infer_datetime_format=True)
    buyout_df["buyout_agreement_date"] = pd.to_datetime(buyout_df['Buyout Agreement Date'], infer_datetime_format=True)
    buyout_df["latitude"] = buyout_df["the_geom"].apply(lambda x:ast.literal_eval(x)["latitude"])
    buyout_df["longitude"] = buyout_df["the_geom"].apply(lambda x:ast.literal_eval(x)["longitude"])
    buyout_df["delta_dates"] = buyout_df["buyout_agreement_date"] - buyout_df["pre_buyout_disclsure_date"]
    buyout_df.drop(["Pre Buyout Disclosure Date", "Buyout Agreement Date", "the_geom"], inplace=True, axis=1)
    return buyout_df

def prep_crime_data(df):
    df.set_index('Crime Id', inplace=True)
    df["Address"] = df["Address"].apply(lambda x:x.replace("Block Of", "") + " San Francisco California")
    df["Address"] = df["Address"].apply(lambda x:re.sub(r'\d+ ', '', x))
    df["Call_Date"] = pd.to_datetime(df['Call Date'], infer_datetime_format=True)
    df["Offense_Date"] = pd.to_datetime(df['Offense Date'], infer_datetime_format=True)
    df["Report_Date"] = pd.to_datetime(df['Report Date'], infer_datetime_format=True)
    df.drop(columns=["City", "State", 'Call Date', 'Offense Date', 'Report Date'], inplace=True)
    return df

def decode(street):
    geolocator = Nominatim(user_agent="Meow")
    location = geolocator.geocode(street)
    return (location.latitude, location.longitude) if location is not None else (None, None)
buyout = prep_data(pd.read_csv('../input/sf-buyout-agreements/buyout-agreements.csv'))
buyout.head(5)
crimes = prep_crime_data(pd.read_csv('../input/sf-police-calls-for-service-and-incidents/police-department-calls-for-service.csv'))
crimes.head(5)
hist_data = [buyout["Number of Tenants"]]
group_labels = ['Number of Tenants']
fig = ff.create_distplot(hist_data, group_labels)
iplot(fig, filename='Basic Distplot')
lst_tuples = list(zip(buyout["Analysis Neighborhood"].values, list(buyout["Analysis Neighborhood"].groupby(by=buyout["Analysis Neighborhood"].values, axis=0).count())))
lst_tuples.sort(key=lambda tup: tup[1], reverse=True)
streets_names = [x[0] for x in lst_tuples]
count_streets = [x[1] for x in lst_tuples]
trace1 = {
  "y": count_streets, 
  "x": streets_names, 
  "marker": {"color": "rgb(100, 100, 5)"}, 
  "type": "bar"
}
layout = {
  "title": "Neighborhood Buyout Frequency", 
  "xaxis": {
    "tickfont": {"size": 12}, 
    "title": "Street"
  }, 
  "yaxis": {
    "title": "Frequency <br>", 
    "titlefont": {"size": 12}
  }
}
fig = Figure(data=[trace1], layout=layout)
iplot(fig, filename='Street Count Plot')

labels = ['True','False']
values = [buyout[buyout["Other Consideration"] == True].dropna().count().iloc[0], buyout[buyout["Other Consideration"] == False].dropna().count().iloc[0]]

trace = go.Pie(labels=labels, values=values)

iplot([trace], filename='Other Consideration')
data = [
    {
        'x': buyout["Number of Tenants"],
        'y': buyout["Buyout Amount"],
        'name':'kale',
        'marker': {
            'color': '#5647b7'
        },
        'boxmean': False,
        'orientation': 'v',
        "type": "box"
    }
]
layout = {
    'xaxis': {
        'title': 'Number of Tenants',
        'zeroline': False,
    },
    'yaxis': {
        'title': 'Buyout Amount',
        'zeroline': False,
    },
    'boxmode': 'group',
}
fig = go.Figure(data=data, layout=layout)
iplot(fig)
hist_data = [buyout["delta_dates"].apply(lambda x:x.days).dropna()]
group_labels = ['Time Delta Between Pre-Buyout to Buyout(in days)']
fig = ff.create_distplot(hist_data, group_labels, bin_size=7)
iplot(fig, filename='Basic Distplot')
hist_data = [buyout["Buyout Amount"].dropna()]
group_labels = ['Buyout Amount']
fig = ff.create_distplot(hist_data, group_labels, bin_size=10000)
iplot(fig, filename='Meow')
temp = crimes.groupby(by="Address", as_index=False).count().nlargest(25, 'Agency Id')
tuples = temp["Address"].apply(decode)
temp["lan"] = [x[0] for x in tuples]
temp["lon"] = [x[1] for x in tuples]
data = [
    go.Scattermapbox(
        lat=buyout["latitude"],
        lon=buyout["longitude"],
        mode='markers',
        name="Buyouts",
        marker=dict(
            size = buyout['Buyout Amount'].apply(lambda x: int(x/20000) + 4 if not pd.isnull(x) else 0),
            color='rgb(35, 5, 112)',
            opacity=0.7
        ),
        text= buyout["Buyout Amount"].apply(lambda x: "{:,}$".format(x))
    ),
    go.Scattermapbox(
        lat=temp["lan"],
        lon=temp["lon"],
        mode='markers',
        name="Crimes",
        marker=dict(
            size = temp['Agency Id'].apply(lambda x: int(x/500) + 4 if not pd.isnull(x) else 0),
            color='rgb(153, 55, 73)',
            opacity=0.5
        ),
        text= temp.apply(lambda x: "{:,} Crimes Committed \n Street: {}".format(x["Agency Id"], x["Address"]), axis=1)
    )
]

layout = go.Layout(
    autosize=True,
    hovermode='closest',
    title="Buyouts With Top 25 Worst Streets", 
    mapbox=dict(
        accesstoken="pk.eyJ1Ijoic3luY3VzaCIsImEiOiJjam05aTEyNHUwMDNnM3JscjRvODFuMDY1In0.Iw54eGGxr-h70qh86bMFjA",
        bearing=0,
        center=dict(
            lat=37.773972,
            lon=-122.431297
        ),
        pitch=0,
        zoom=10
    ),
)

fig = dict(data=data, layout=layout)
iplot(fig, filename='Multiple Mapbox')
r = lambda: random.randint(0,255)
data = []
lst_tuples = list(zip(buyout["Analysis Neighborhood"].values, list(buyout["Analysis Neighborhood"].groupby(by=buyout["Analysis Neighborhood"].values, axis=0).count())))
lst_tuples.sort(key=lambda tup: tup[1], reverse=True)
streets_names = [x[0] for x in lst_tuples]
for neighborhood in streets_names[:5]:
    temp = buyout[buyout["Analysis Neighborhood"] == neighborhood].sort_values(by="buyout_agreement_date")
    trace_high = go.Scatter(
        x=temp["buyout_agreement_date"],
        y=temp["Buyout Amount"],
        name = neighborhood,
        line = dict(color = '#%02X%02X%02X' % (r(),r(),r())),
        opacity = 0.8)
    data.append(trace_high)

layout = dict(
    title='Top 5 Most Buyouts Neighborhoods',
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label='1m',
                     step='month',
                     stepmode='backward'),
                dict(count=6,
                     label='6m',
                     step='month',
                     stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(
            visible = True
        ),
        type='date'
    )
)

fig = dict(data=data, layout=layout)
iplot(fig, filename = "Time Series with Rangeslider")