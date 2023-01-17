import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import ast
from PIL import Image
import os
import plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.figure_factory as ff
import plotly.graph_objs as go
from plotly.graph_objs import *
import os
import re
import random
import matplotlib.pyplot as plt
import seaborn as sns
init_notebook_mode()
from wordcloud import WordCloud, STOPWORDS

print("Files are:\n")
print('\n'.join(os.listdir('../input/sf-restaurant-scores-lives-standard/')))
# Any results you write to the current directory are saved as output.
res_vio = pd.read_csv('../input/sf-restaurant-scores-lives-standard/restaurant-scores-lives-standard.csv')
res_vio.head(5)
res_vio.tail(5)
num_rows, num_cols = res_vio.shape
print("There are {} Rows and {} Cols".format(num_rows, num_cols))
res_vio.isna().sum()
def prep_data():
    df = pd.read_csv('../input/sf-restaurant-scores-lives-standard/restaurant-scores-lives-standard.csv')
    #df.set_index(['business_name', 'business_address'], inplace=True)
    df["inspection_date"] = pd.to_datetime(df['inspection_date'], infer_datetime_format=True)
    df.drop(columns=['business_city', 'business_state', 'business_postal_code', 'business_location', 'business_phone_number'], inplace=True)
    return df
res_vio = prep_data()
res_vio.head(10)
temp = res_vio[['inspection_type', 'risk_category']]
fig = {
  "data": [
    {
      "values": temp['inspection_type'].value_counts(),
      "labels": list(temp['inspection_type'].value_counts().index),
      "name": "Inspections Type",
      "hoverinfo":"label+percent+name",
      "hole": .1,
      "type": "pie"
    }],
  "layout": {
        "title":"Types of Inspections"
    }
}
iplot(fig, filename='pie')
temp = res_vio[['inspection_type', 'risk_category']]
fig = {
  "data": [
    {
      "values": temp['risk_category'].value_counts(),
      "labels": list(temp['risk_category'].value_counts().index),
      "name": "Inspections Type",
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    }],
  "layout": {
        "title":"Risk Category Distribution"
    }
}
iplot(fig, filename='pie')
d = {'inspection_score': 'mean', 'business_latitude': 'first', 'business_longitude':'first', 'violation_id':'count', 'risk_category':'last'}
res_vio_agg = res_vio.groupby(by=["business_name","business_address", 'inspection_date'], as_index=True).aggregate(d).dropna()
worst_100 = res_vio_agg.nsmallest(100, 'inspection_score').reset_index()
top_100 = res_vio_agg.nlargest(100, 'inspection_score').reset_index()
data = [
    go.Scattermapbox(
        lat=top_100["business_latitude"],
        lon=top_100["business_longitude"],
        mode='markers',
        name="Top 100",
        marker=dict(
            size = top_100["inspection_score"].apply(lambda x: (x / 10) + 10),
            color='rgb(135, 14, 87)',
            opacity=0.5
        ),
        text= top_100.apply(lambda x: "Resturant Name:\t{}</br>Street:\t{}</br>Inspection Score(X/100):\t{:,}</br># Violations:\t{}".format(x["business_name"], x["business_address"], x["inspection_score"], x["violation_id"]), axis=1)
    ),
     go.Scattermapbox(
        lat=worst_100["business_latitude"],
        lon=worst_100["business_longitude"],
        mode='markers',
        name="Worst 100",
        marker=dict(
            size = worst_100["inspection_score"].apply(lambda x: (x / 2) + 5),
            color='rgb(50, 14, 25)',
            opacity=0.5
        ),
        text= worst_100.apply(lambda x: "Resturant Name:\t{}</br>Street:\t{}</br>Inspection Score(X/100):\t{:,}</br># Violations:\t{}".format(x["business_name"], x["business_address"], x["inspection_score"],x["violation_id"]), axis=1)
    ),
    
]

layout = go.Layout(
    autosize=True,
    hovermode='closest',
    title="Top & Worst 100 Resturants on the Map", 
    mapbox=dict(
        accesstoken="pk.eyJ1Ijoic3luY3VzaCIsImEiOiJjam05aTEyNHUwMDNnM3JscjRvODFuMDY1In0.Iw54eGGxr-h70qh86bMFjA",
        bearing=0,
        center=dict(
            lat=37.7749,
            lon=-122.4194
        ),
        pitch=0,
        zoom=10
    ),
)

fig = dict(data=data, layout=layout)
iplot(fig, filename='Multiple Mapbox')
d = {'inspection_score': 'last', 'business_latitude': 'last', 'business_longitude':'last', 'violation_id':'last', 'risk_category':'last', 'inspection_date':'last', 'violation_description':'last'}
last_inspections = res_vio.groupby(by=["business_name","business_address"], as_index=True).aggregate(d).dropna()
low_risk = last_inspections[last_inspections["risk_category"] == "Low Risk"].reset_index()
med_risk = last_inspections[last_inspections["risk_category"] == "Moderate Risk"].reset_index()
high_risk = last_inspections[last_inspections["risk_category"] == "High Risk"].reset_index()
r = lambda: random.randint(0,255)
data = []
colors = ['rgb(0, 255, 114)', 'rgb(229, 247, 113)', 'rgb(255, 38, 106)']
for i,(df, name) in enumerate([(low_risk,"Low"), (med_risk, "Moderate"), (high_risk, "High")]):
    temp = go.Scattermapbox(
            lat=df["business_latitude"],
            lon=df["business_longitude"],
            mode='markers',
            name=name,
            marker=dict(
                size = df["inspection_score"].apply(lambda x: (x / 10) + 10),
                color=colors[i],
                opacity=0.5
            ),
            text= df.apply(lambda x: "Resturant Name:\t{}</br>Street:\t{}</br>Inspection Score(X/100):\t{:,}</br># Violations:\t{}</br>Last Inspection:\t{:%d, %b %Y}</br>Violation:\t{}".format(x["business_name"], x["business_address"], x["inspection_score"], x["violation_id"], x["inspection_date"], x['violation_description']), axis=1)
    )
    data.append(temp)

layout = go.Layout(
    autosize=True,
    hovermode='closest',
    title="Risk Map of Most Recent Inspections", 
    mapbox=dict(
        accesstoken="pk.eyJ1Ijoic3luY3VzaCIsImEiOiJjam05aTEyNHUwMDNnM3JscjRvODFuMDY1In0.Iw54eGGxr-h70qh86bMFjA",
        bearing=0,
        center=dict(
            lat=37.7749,
            lon=-122.4194
        ),
        pitch=0,
        zoom=10
    ),
)

fig = dict(data=data, layout=layout)
iplot(fig, filename='Multiple Mapbox')