import pandas as pd  #for data analysis
import matplotlib.pyplot as plt   #for plotting graphs
import numpy as np    # for mathematical functions
import seaborn as sns   # for interactive data visualizations.
import plotly.graph_objects as go  #  for interactive data visualizations.
import plotly.express as px   #for interactive data visualizations.
df=pd.read_csv("../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")
df.head()
df.columns
df.shape
df.specialisation.unique()
data=go.Histogram(x=df.specialisation,histnorm="percent",marker=dict(color='LightSkyBlue',
                              line=dict(width=2,
                                        color='DarkSlateGrey')))
layout=go.Layout(title_text = "Specialization statistics", title_font = {"size": 30},
        xaxis = dict(
        title_text = "Specialization",
        title_font = {"size": 30},
        title_standoff = 25),
    yaxis = dict(
        title_text = "Percentage",
        title_standoff = 25,
        title_font = {"size": 30})
        )
figure=go.Figure(data=data,layout=layout)
figure
data=go.Histogram(x=df.status,histnorm="percent",marker=dict(
                              line=dict(width=2,
                                        color='black')))
layout=go.Layout(title="Placement percentage in Campus recruitment",
        xaxis = dict(
        tickangle = 0,
        title_text = "Placement Status",
        title_font = {"size": 20},
        title_standoff = 30),
        yaxis = dict(
        title_text = "Percentage",
        title_standoff = 30,
        title_font = {"size": 20}))
figure=go.Figure(data=data,layout=layout)
figure
data=go.Histogram(x=df[df["status"] =="Placed"].specialisation,histnorm="percent",marker=dict(color='LightSkyBlue',
                              line=dict(width=2,
                                        color='DarkSlateGrey')))
layout=go.Layout(title_text = "Placement statistics", title_font = {"size": 30},
        xaxis = dict(
        title_text = "Specialization",
        title_font = {"size": 30},
        title_standoff = 25),
    yaxis = dict(
        title_text = "Percentage",
        title_standoff = 25,
        title_font = {"size": 30})
        )
figure=go.Figure(data=data,layout=layout)
figure
data=go.Histogram(x=df[df["status"] =="Placed"].workex,histnorm="percent",marker=dict(color='LightSkyBlue',
                              line=dict(width=2,
                                        color='DarkSlateGrey')))
layout=go.Layout(title_text = "Effect of work experience on placement", title_font = {"size": 30},
        xaxis = dict(
        title_text = "Work Experience",
        title_font = {"size": 30},
        title_standoff = 25),
    yaxis = dict(
        title_text = "Percentage",
        title_standoff = 25,
        title_font = {"size": 30})
        )
figure=go.Figure(data=data,layout=layout)
figure
data=go.Histogram(x=df[df["status"] =="Placed"].degree_t,histnorm="percent",marker=dict(color='tomato',
                              line=dict(width=2,
                                        color='DarkSlateGrey')))
layout=go.Layout(title_text = "degree statistics of placed students", title_font = {"size": 30},
        xaxis = dict(
        title_text = "degree background",
        title_font = {"size": 30},
        title_standoff = 25),
    yaxis = dict(
        title_text = "Percentage",
        title_standoff = 25,
        title_font = {"size": 30})
        )
figure=go.Figure(data=data,layout=layout)
figure
plt.figure(figsize = (16.0,12.0))
sns.heatmap(df.corr(), annot = True, vmin=-1, vmax=1, center= 0, cmap= 'YlOrRd', linewidths=3, linecolor='black')
plt.show()

data=go.Histogram(x=df[df["status"] =="Placed"].salary,histnorm="percent",marker=dict(color='tomato',
                              line=dict(width=2,
                                        color='DarkSlateGrey')))
layout=go.Layout(title_text = "Salary range of placed students", title_font = {"size": 30},
        xaxis = dict(
        title_text = "Salary",
        title_font = {"size": 30},
        title_standoff = 25),
    yaxis = dict(
        title_text = "Percentage",
        title_standoff = 25,
        title_font = {"size": 30})
        )
figure=go.Figure(data=data,layout=layout)
figure
fig_dims = (12, 8)
fig, ax = plt.subplots(figsize=fig_dims)
sns.barplot(x="specialisation",y="salary",data=df,ax=ax)
data=go.Histogram(x=df.nlargest(50,"salary").degree_t,histnorm="percent",marker=dict(color='tomato',
                              line=dict(width=2,
                                        color='DarkSlateGrey')))
layout=go.Layout(title_text = "degree statistics of top 50 placed students", title_font = {"size": 30},
        xaxis = dict(
        title_text = "degree background",
        title_font = {"size": 30},
        title_standoff = 25),
    yaxis = dict(
        title_text = "Percentage",
        title_standoff = 25,
        title_font = {"size": 30})
        )
figure=go.Figure(data=data,layout=layout)
figure
data=go.Histogram(x=df.nlargest(50,"salary").specialisation,histnorm="percent",marker=dict(color='tomato',
                              line=dict(width=2,
                                        color='DarkSlateGrey')))
layout=go.Layout(title_text = "mba specialisation of top 50 placed students", title_font = {"size": 30},
        xaxis = dict(
        title_text = "specialisation",
        title_font = {"size": 30},
        title_standoff = 25),
    yaxis = dict(
        title_text = "Percentage",
        title_standoff = 25,
        title_font = {"size": 30})
        )
figure=go.Figure(data=data,layout=layout)
figure
data=go.Histogram(x=df.nlargest(50,"salary").workex,histnorm="percent",marker=dict(color='tomato',
                              line=dict(width=2,
                                        color='DarkSlateGrey')))
layout=go.Layout(title_text = "work experience statistics of top 50 placed students", title_font = {"size": 30},
        xaxis = dict(
        title_text = "work experience background",
        title_font = {"size": 30},
        title_standoff = 25),
    yaxis = dict(
        title_text = "Percentage",
        title_standoff = 25,
        title_font = {"size": 30})
        )
figure=go.Figure(data=data,layout=layout)
figure
data=go.Histogram(x=df.nlargest(50,"salary").mba_p,histnorm="percent",marker=dict(color='tomato',
                              line=dict(width=2,
                                        color='DarkSlateGrey')))
layout=go.Layout(title_text = "mba marks of top 50 placed students", title_font = {"size": 30},
        xaxis = dict(
        title_text = "marks",
        title_font = {"size": 30},
        title_standoff = 25),
    yaxis = dict(
        title_text = "Percentage",
        title_standoff = 25,
        title_font = {"size": 30})
        )
figure=go.Figure(data=data,layout=layout)
figure
plt.figure(figsize = (16.0,12.0))
sns.heatmap(df.nlargest(50,"salary").corr(), annot = True, vmin=-1, vmax=1, center= 0, cmap= 'YlOrRd', linewidths=3, linecolor='black')
plt.show()