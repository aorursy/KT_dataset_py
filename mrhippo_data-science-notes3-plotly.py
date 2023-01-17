import numpy as np # linear algebra
from numpy import random
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.graph_objs as go 
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, iplot, plot
init_notebook_mode(connected=True) 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data1 = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
data1.head()
data2 = pd.read_csv("/kaggle/input/videogamesales/vgsales.csv")
data2.head()
data3 = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")
data3.head()
data_last = data3.tail(1)
data_last_day = data3[data3["ObservationDate"] == data_last["ObservationDate"].iloc[0]] 
country_list = list(data_last_day["Country/Region"].unique())
confirmed = []
deaths = []
recovered = []
for i in country_list:
    x = data_last_day[data_last_day["Country/Region"] == i]
    confirmed.append(sum(x["Confirmed"]))
    deaths.append(sum(x["Deaths"]))
    recovered.append(sum(x["Recovered"]))
data3_country = pd.DataFrame(list(zip(country_list,confirmed,deaths,recovered)),columns = ["Country","Confirmed","Deaths","Recovered"])
data3_country.head()
date_list1 = list(data3["ObservationDate"].unique())
confirmed = []
deaths = []
recovered = []
for i in date_list1:
    x = data3[data3["ObservationDate"] == i]
    confirmed.append(sum(x["Confirmed"]))
    deaths.append(sum(x["Deaths"]))
    recovered.append(sum(x["Recovered"]))
data3 = pd.DataFrame(list(zip(date_list1,confirmed,deaths,recovered)),columns = ["Date","Confirmed","Deaths","Recovered"])
data3.head()
from datetime import date, timedelta, datetime
data3["Date"] = pd.to_datetime(data3["Date"])
data3.info()
fig = go.Figure(go.Scatter(x = data3["Date"],
                          y = data3["Confirmed"],
                          mode = "lines"),
               layout = go.Layout(title = "standard lineplot",
                                 xaxis_title = "xlabel",
                                 yaxis_title = "ylabel"))
iplot(fig)
fig = go.Figure(go.Scatter(x = data1["age"],
                          y = data1["oldpeak"],
                          mode = "markers"),
               layout = go.Layout(title = "standard scatterplot"))
iplot(fig)
fig = go.Figure(go.Bar(x = data3.head(10)["Date"],
                          y = data3.head(10)["Confirmed"]),
               layout = go.Layout(title = "standard barplot"))
iplot(fig)
fig = go.Figure(go.Histogram(x = data1["age"]),
               layout = go.Layout(title = "standard histogram"))
iplot(fig)
labels = ["Confirmed","Deaths","Recovered"]
values = [data3.tail(1)["Confirmed"].iloc[0],data3.tail(1)["Deaths"].iloc[0],data3.tail(1)["Recovered"].iloc[0]]

fig = go.Figure(go.Pie(values = values, 
                       labels = labels,
                       insidetextorientation='radial'),
               layout = go.Layout(title = "standard piechart"))
iplot(fig)
fig = go.Figure(go.Box(x = data1["age"]),
               layout = go.Layout(title = "standard boxplot"))
iplot(fig)
fig = go.Figure(data=go.Violin(y=data1['age'],
                               meanline_visible=True, opacity=0.6,
                               ))

fig.update_layout(title = "standard violinplot")
fig.show()
x = ["A","B","C","D","E","F"]
y = [10,20,11,22,14,18]

fig = go.Figure(data=
    go.Scatterpolar(
        r = y,
        theta = x,
        mode = 'lines',
    ))

fig.update_layout(showlegend=False,title = "linepolar")
fig.show()
x = ["A","B","C","D","E","F"]
y = [10,20,11,22,14,18]

fig = go.Figure(data=
    go.Scatterpolar(
        r = y,
        theta = x,
        mode = 'markers',
    ))

fig.update_layout(showlegend=False,title = "scatterpolar")
fig.show()
x = ["A","B","C","D","E","F"]
y = [10,20,11,22,14,18]

fig = go.Figure(data=
    go.Barpolar(
        r = y,
        theta = x,
    ))

fig.update_layout(showlegend=False,title = "barpolar")
fig.show()
fig = go.Figure(go.Scatter3d(
                              x = [1,5,7,9],
                              y = [1000,2580,5673,8979],
                              z = [10,23,26,38],
                              mode = "lines",
                              name = "standard 3D lineplot",
                              ),layout = go.Layout(title_text = "standard 3D lineplot",
                                                    ))
iplot(fig)
trace1 = go.Scatter(
    x=data3["Date"],
    y=data3["Confirmed"],
    name = "Confirmed"
)
# second line plot
trace2 = go.Scatter(
    x=data3["Date"],
    y=data3["Deaths"],
    xaxis='x2',
    yaxis='y2',
    name = "Deaths"
)
data = [trace1, trace2]
layout = go.Layout(
    xaxis2=dict(
        domain=[0.06, 0.5],
        anchor='y2',        
    ),
    yaxis2=dict(
        domain=[0.6, 0.95],
        anchor='x2',
    ),
    title = '2 plots together',
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=[0, 1, 2, 3, 4, 5, 6, 7, 8],
    y=[0, 1, 3, 2, 4, 3, 4, 6, 5]
))

fig.update_layout(
    showlegend=False,
    annotations=[
        dict(
            x=2,
            y=5,
            xref="x",
            yref="y",
            text="this is a ariel Text",
            showarrow=False,
            font = dict(size = 30, 
                        family = "ariel",
                       color = "LightSeaGreen")
        )
    ]
)

fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=[0, 1, 2, 3, 4, 5, 6, 7, 8],
    y=[0, 4, 5, 1, 2, 2, 3, 4, 2]
))

fig.add_annotation(
            x=2,
            y=5,
            text="dict Text",
            font = dict(size = 20))
fig.add_annotation(
            x=4,
            y=4,
            text="dict Text 2")
fig.update_annotations(dict(
            xref="x",
            yref="y",
            showarrow=True,
            arrowhead=7,
            ax=0,
            ay=-40
))

fig.update_layout(showlegend=False)

fig.show()
fig = go.Figure(go.Scatter(x = data3["Date"],
                          y = data3["Confirmed"],
                           name = "confirmed",
                          mode = "lines+markers",
                marker = dict(color = 'rgba(102, 204, 255, 0.5)')),
               layout = go.Layout(title = "styled lineplot",
                                 xaxis_title = "Dates",
                                 yaxis_title = "Confirmed(log scale)",
                                 yaxis_type="log",
                                 template = "plotly_white",
                                 hovermode = "x unified"))

iplot(fig)
fig = go.Figure(go.Scatter(x = data1["age"],
                          y = data1["oldpeak"],
                          mode = "markers",
                              marker=dict(
                                size=data1["age"]/3,
                                color=data1["oldpeak"],
                                colorbar=dict(
                                    title="oldpeak"
                                ),
                                colorscale="BrBG"
                            ),),
               layout = go.Layout(title = "styled scatterplot",
                                 template = "plotly_white",
                                 xaxis_title = "age",
                                 yaxis_title = "oldpeak"))
iplot(fig)
fig = go.Figure(go.Bar(x = data3_country.sort_values(by = ["Confirmed"]).tail(10)["Country"],
                          y = data3_country.sort_values(by = ["Confirmed"]).tail(10)["Confirmed"],
                       text = data3_country.sort_values(by = ["Confirmed"]).tail(10)["Confirmed"],
                              textposition = "outside",
                marker=dict(
                                color=data3_country.sort_values(by = ["Confirmed"]).tail(10)["Confirmed"],
                    line = dict(color = "rgb(0,0,0)", width = 2),
                                colorbar=dict(
                                    title="Colorbar"
                                ),
                                colorscale="Viridis"
                            ),),
               layout = go.Layout(title = "styled barplot",
                                 template = "simple_white",
                                 xaxis_title = "country",
                                 yaxis_title = "confirmed"))
iplot(fig)
labels = ["Confirmed","Deaths","Recovered"]
values = [data3.tail(1)["Confirmed"].iloc[0],data3.tail(1)["Deaths"].iloc[0],data3.tail(1)["Recovered"].iloc[0]]
colors = ["#0099ff","#ff1a1a","#33cc33"]

fig = go.Figure(go.Pie(values = values, 
                       labels = labels,
                       insidetextorientation='radial',
                       hole = .5,
                       pull=[0.05, 0.2, 0.05]),
                layout = go.Layout(title = "styled piechart",
                                annotations=[dict(text='Piechart', x=0.5, y=0.5, 
                                                  font_size=20, 
                                                  showarrow=False)]
))

fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=18,
                  marker=dict(colors = colors, line=dict(color='#000000', width=2)))


iplot(fig)
fig = go.Figure(go.Box(y = data1["age"],
                      boxpoints='all',
                      boxmean = "sd",
                      marker_color='darkblue',
                      name = "age",
                       notched=True),
               layout = go.Layout(title = "styled boxplot",
                                 template = "plotly_white"))
iplot(fig)
fig = go.Figure(data=go.Violin(y=data1['age'], box_visible=True, line_color='black',
                               meanline_visible=True, fillcolor='lightseagreen', opacity=0.6,
                              x0='age',
                              points='all'))

fig.update_layout(title = "styled violinplot",
                 template = "plotly_white")
fig.show()
import plotly.graph_objects as go
from plotly.colors import n_colors

# 12 sets of normal distributed random data, with increasing mean and standard deviation
data = (np.linspace(1, 2, 12)[:, np.newaxis] * np.random.randn(12, 200) +
            (np.arange(12) + 2 * np.random.random(12))[:, np.newaxis])

colors = n_colors('rgb(5, 200, 200)', 'rgb(200, 10, 10)', 12, colortype='rgb')

fig = go.Figure()
for data_line, color in zip(data, colors):
    fig.add_trace(go.Violin(x=data_line, line_color=color))

fig.update_traces(orientation='h', side='positive', width=3, points=False)
fig.update_layout(template = "plotly_white", xaxis_zeroline=False)
fig.show()
fig = go.Figure(go.Scatter3d(
                              x = data3["Confirmed"],
                              y = data3["Deaths"],
                              z = data3["Recovered"],
                              mode = "lines",
                              name = "states and confirmed",
                              line = dict(width = 7),
                              marker = dict(
                                   size = 4,
                                   color = "rgb(255,100,100)"
                              )),layout = go.Layout(title = "styled 3D lineplot",
                                                    template = "plotly_white",
                                                    scene = dict(
                                                    xaxis =dict(
                                                        title = "Confirmed"),
                                                    yaxis =dict(
                                                        title ="Deaths"),
                                                    zaxis =dict(
                                                        title = "Recovered"),)))
iplot(fig)
fig = make_subplots(rows=2,cols=1,
                    subplot_titles = ("recovered percentage last 10 days",
                                      "Recovered Covid-19 last 10 days"))

data3_last10 = data3.tail(10)

recovered_percent = ((data3_last10["Recovered"]*100)/data3_last10["Confirmed"])

fig.append_trace(go.Scatter(x=data3_last10["Date"],
                                  y = recovered_percent,
                                  mode = "lines",
                                  name = "Reocvered percentage",
                                  marker = dict(color = 'rgba(1,108,89, 0.8)')),row = 1, col = 1)

fig.append_trace(go.Bar(x=data3_last10["Date"],
                                  y = data3_last10["Recovered"],
                                  name = "Recovered",
                                  marker = dict(color = 'rgba(1,108,89, 0.8)')),row = 2, col = 1)

fig.update_layout(height = 700,
                  title = "subplots",
                  template="plotly_white",
                  hovermode='x unified')

fig.update_xaxes(title_text="Dates", row=1, col=1)
fig.update_xaxes(title_text="Dates", row=2, col=1)

fig.update_yaxes(title_text="Percentage(%)", row=1, col=1)
fig.update_yaxes(title_text="Percentage(%)", row=2, col=1)

iplot(fig)
fig = make_subplots(
    rows=2, cols=2,
    specs=[[{"colspan": 2}, None],
           [{},{}]],
    subplot_titles=("focused Subplot","other Subplot", "other Subplot"))

fig.add_trace(go.Scatter(x=[1, 2], y=[1, 2]),
                 row=1, col=1)

fig.add_trace(go.Scatter(x=[1, 2], y=[1, 2]),
                 row=2, col=1)
fig.add_trace(go.Scatter(x=[1, 2, 3], y=[2, 1, 2]),
                 row=2, col=2)

fig.update_layout(title_text="subplots, one focused plot",
                 template = "simple_white")
fig.show()
for template in ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"]:
    x = random.randint(100, size=(15))
    y = random.randint(100, size=(15))
    fig = go.Figure(go.Scatter(x=x, 
                               y=y,
                               marker = dict(size = 15,
                                            color = y),
                            mode = "markers"),
                     layout = go.Layout(template=template, 
                                        title=template + " theme"))
    fig.show()