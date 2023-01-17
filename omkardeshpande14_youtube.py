import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

print(os.listdir("../input"))
df = pd.read_csv('../input/data.csv')

df=df.rename(index=str, columns={"Channel name": "Channel_name", "Video Uploads": "Video_Uploads","Video views":"Video_views"})

df.sample(15)
df.info()
df_50=df.head(50)

df_50
df_50['Subscribers'] = df_50['Subscribers'].apply(lambda x: str(x).replace('--', '0') if '--' in str(x) else x)

#df_50['Video_Uploads'] = df_50['Video_Uploads'].apply(lambda x: str(x).replace('--', '0') if '--' in str(x) else x)

df_50['Ratio']=df_50['Video_views']/pd.to_numeric(df_50['Subscribers'])

df_50
import plotly

# connected=True means it will download the latest version of plotly javascript library.

plotly.offline.init_notebook_mode(connected=True)

import plotly.graph_objs as go



data = [go.Scatter(

                    x = df_50.Rank,

                    y = df_50.Subscribers,

                    mode = "lines",

                    name = "suscribers",

                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),

                    text= df_50.Channel_name)]



layout = dict(title = 'Suscribers of top 50 youtube channels',

              xaxis= dict(title= 'Channel Rank',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

plotly.offline.iplot(fig)
data = [go.Scatter(

                    x = df_50.Rank,

                    y = df_50.Video_views,

                    mode = "lines+markers",

                    name = "views",

                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),

                    text= df_50.Channel_name)]



layout = dict(title = 'Views of top 50 youtube channels',

              xaxis= dict(title= 'Channel Rank',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

plotly.offline.iplot(fig)
data = [go.Scatter(

                    x = df_50.Rank,

                    y = df_50.Ratio,

                    mode = "lines+markers",

                    name = "views",

                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),

                    text= df_50.Channel_name)]



layout = dict(title = 'View-Sub ratio of top 50 youtube channels',

              xaxis= dict(title= 'Channel Rank',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

plotly.offline.iplot(fig)
x = df_50.Channel_name



trace1 = {

  'x': x,

  'y': df_50.Subscribers,

  'name': 'Subscribers',

  'type': 'bar'

};

trace2 = {

  'x': x,

  'y': df_50.Video_views,

  'name': 'Video_views',

  'type': 'bar'

};

data = [trace1, trace2];

layout = {

  'xaxis': {'title': 'Top 50 channels'},

  'barmode': 'relative',

  'title': 'Suscribers and Views of top 50 youtube channels'

};

fig = go.Figure(data = data, layout = layout)

plotly.offline.iplot(fig)
df_10=df_50.iloc[:10]



pie1 = df_10.Video_views

pie1_list = []

for each in df_10.Video_views:

    pie1_list.append(int(each))

print(pie1_list)

labels = df_10.Channel_name+df_10.Rank

# figure

fig = {

  "data": [

    {

      "values": pie1_list,

      "labels": labels,

      "domain": {"x": [0, .5]},

      "name": "Number Of Views",

      "hoverinfo":"label+percent+name",

      "hole": .3,

      "type": "pie"

    },],

  "layout": {

        "title":"Channels with number of views",

        "annotations": [

            { "font": { "size": 20},

              "showarrow": False,

              "text": "Number of views",

                "x": 0.20,

                "y": 1

            },

        ]

    }

}

plotly.offline.iplot(fig)
# import figure factory

import plotly.figure_factory as ff

# prepare data

data = df_50.loc[:,["Subscribers","Video_views","Video_Uploads"]]

data["index"] = np.arange(1,len(data)+1)

# scatter matrix

fig = ff.create_scatterplotmatrix(data, diag='box', index='index',colormap='Portland',

                                  colormap_type='cat',

                                  height=700, width=700)

plotly.offline.iplot(fig)