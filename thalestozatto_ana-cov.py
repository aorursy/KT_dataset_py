import numpy as np

import pandas as pd

import plotly.graph_objects as go

import plotly.express as px

import json



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

files = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        file = os.path.join(dirname, filename)

        print(file)

        if filename == 'brazil_covid19.csv':

            covid_file = file

        elif filename == 'brazil.json':

            brazil_geo = file # from https://github.com/codeforamerica/click_that_hood/blob/master/public/data/brazil-states.geojson

            

# Any results you write to the current directory are saved as output.
df = pd.read_csv(covid_file)

df.head(100)
def get_nlargest(df, n, feature):

    days = pd.unique(df["date"])

    states = pd.unique(df["state"])

    last = {}

    for s in states:

        last[s] = int(df.loc[(df["date"]==days[-1]) & (df["state"]==s)][feature])

    largest_dict = {}

    for j in range(n):

        largest_number = 0

        for i, s in enumerate(states):

            try:

                if last[s] > largest_number:

                    largest = s

                    largest_number = last[s]



            except:

                pass

        largest_dict[largest] = last[largest]

        del last[largest]

    return largest_dict
def plot_state(df, states, feature, period=None, log=True):

    if not period:

        days = pd.unique(df.loc[df["date"] >= '2020-02-26']["date"])

    else:

        days = pd.unique(df.loc[(df["date"] >= period[0]) & (df["date"] <= period[1])]["date"])

        

    fig = go.Figure()

    

    for st in states:

        ser = np.zeros(len(days))

        for i, d in enumerate(days):

            ser_aux = df.loc[(df["date"] == d) & (df["state"] == st)][feature]

            if len(ser_aux) == 1:

                ser[i] = int(ser_aux)

            elif len(ser_aux) == 0:

                ser[i] = ser[i-1]

            else:

                ser[i] = int(ser_aux.iloc[-1])

        fig.add_trace(go.Scatter(y=ser, mode='lines', name=st))



    if not period:

        xlabel = "Days from the start"

    else:

        xlabel = "Days from {}".format(period[0])



    if log:

        fig.update_layout(yaxis_type="log")

        title = "covid-19 {} by state (in log scale)".format(feature)

    else:

        title = "covid-19 {} by state".format(feature)



    fig.update_layout(title=title, xaxis_title=xlabel)

    fig.show()
plot_state(df, get_nlargest(df, 5, "cases").keys(), "cases")
plot_state(df, get_nlargest(df, 5, "suspects").keys(), "suspects")
def create_dfmap(df, feature):

    states = pd.unique(df["state"])

    df_map = pd.DataFrame(columns=[feature], index=states)



    for st in states:

        df_map.loc[st] = [list(df.loc[df["state"] == st][feature])[-1]]

    return df_map
def plot_map(df, feature):

    df_map = create_dfmap(df, feature)

    # IPython.embed()

    with open(brazil_geo, 'rb') as f:

        geojson = json.load(f, encoding="utf-8")

    

    fig = go.Figure(data=go.Choropleth(z=df_map[feature], 

                                       geojson=geojson,

                                       locations=df_map.index, 

                                       colorbar_title=feature,

                                       featureidkey="properties.name",

                                       colorscale="dense"

                          ))

    fig.update_geos(fitbounds="locations", visible=False)

    fig.update_layout(title_text="covid-19 {} by state in Brazil".format(feature))

    fig.show()
plot_map(df, "cases")
def plot_animated_state(df, feature, frames=None):

    with open(brazil_geo, 'rb') as f:

        geojson = json.load(f, encoding="utf-8")

    df_aux = pd.DataFrame(columns=['date','state',feature])



    df_aux['date']=df.loc[df['date']>='2020-02-26']['date']

    df_aux['state']=df.loc[df['date']>='2020-02-26']['state']

    df_aux[feature]=df.loc[df['date']>='2020-02-26'][feature]



    if frames:

        days = pd.unique(df_aux["date"])

        total_frames = len(days)

        interval = int(total_frames/(frames-1))

        idx = np.arange(0,(frames-1)*interval,interval)

        days_f = list(days[idx])

        days_f.append(days[-1])

        df_aux = df_aux.loc[df_aux['date'].isin(days)]

    

    fig = go.Figure(px.choropleth(df_aux,

                        geojson=geojson,

                        locations='state', 

                        featureidkey="properties.name",

                        color=feature,

                        animation_frame='date',

                        animation_group='state',

                        color_continuous_scale='dense',

                        range_color=(0, max(df_aux[feature]))

                        ))

    fig.update_geos(fitbounds="locations", visible=False, resolution=110)

    fig.update_layout(title_text="covid-19 {} by state in Brazil".format(feature))

    fig.show()
plot_animated_state(df, 'cases')