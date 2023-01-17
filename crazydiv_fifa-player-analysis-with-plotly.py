# Get all the required libraries 

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.plotly as py

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode,iplot

import plotly.figure_factory as ff

init_notebook_mode(connected=True)

import matplotlib.cm as cm

import re

from sklearn import linear_model

# Reading the Input data

df = pd.read_csv("../input/data.csv")
def pietrace(df, col_name=None):

    tmp = df[col_name].value_counts()

    return go.Pie(labels=list(tmp.index), values=list(tmp.values))



def boxtrace(df=None, col_name=None, boxpoints='outliers', boxmean=True):

    return go.Box(y=df[col_name],name=col_name,boxpoints = boxpoints, boxmean=boxmean)



def violintrace(df=None, x_col=None, y_col=None, name=None):

    if not x_col:

        return go.Violin(y=df[y_col], box={"visible": True}, meanline={"visible": True}, name=name)

    return go.Violin(x=df[x_col], y=df[y_col], box={"visible": True}, meanline={"visible": True}, name=name)



def distplot(df=None, col_names=[], show_hist=False):

    data = [df[x].fillna(-1) for x in col_names]

    return ff.create_distplot(data, col_names, show_hist=show_hist)



def bartrace(df=None, x_col=None, y_col=None, name=None):

    return go.Bar(

        y=df[y_col],

        x=df[x_col],

        name=name

    )



def scattertrace(df=None, x_col=None, y_col=None, hover_col=None):

    return go.Scatter(

        y = df[y_col],

        x = df[x_col],

        hovertext= df[hover_col],

        mode = 'markers'

    )
df = pd.read_csv("../input/data.csv")

df.sample(n=5)
def convert_currency(x):

    x = x.replace('€','')

    if x.endswith("M"):

        return float(x.split("M")[0]) * 1000000

    elif x.endswith("K"):

        return float(x.split("K")[0]) * 1000

df['Value_eur'] = df['Value'].apply(convert_currency) # Numeric value in Euros
def feet_to_inches(x):

    tmp = x.split("'")

    return int(tmp[0]) * 12 + int(tmp[1])

df["Height"].fillna("0'0", inplace=True) # To fill missing values

df["height_inches"] = df["Height"].apply(feet_to_inches)
df["weight_lbs"] = df["Weight"].str.replace("lbs", "")

df["weight_lbs"] = df["weight_lbs"].astype('float')
iplot([pietrace(df=df, col_name="Body Type")])
iplot([boxtrace(df=df, col_name="Dribbling"), boxtrace(df=df, col_name="HeadingAccuracy")])
df[df["Dribbling"] > 96].head()
iplot([violintrace(df=df, y_col="HeadingAccuracy")])
data=[violintrace(df=df[df["Club"] == "FC Barcelona"], y_col="Aggression", name="FC Barcelona"),

      violintrace(df=df[df["Club"] == "Roma"], y_col="Aggression", name="Roma"),

      violintrace(df=df[df["Club"] == "Juventus"], y_col="Aggression", name="Juventus"),

      violintrace(df=df[df["Club"] == "Manchester City"], y_col="Aggression", name="Manchester City")]

layout = {

        "title": "Aggression",

        "yaxis": {

            "zeroline": False,

        },

        "violinmode": "group"

    }

fig = go.Figure(data=data, layout=layout)

iplot(fig)
data=[violintrace(df=df[df["Club"] == "FC Barcelona"], x_col="Preferred Foot", y_col="Aggression", name="FC Barcelona"),

      violintrace(df=df[df["Club"] == "Juventus"], x_col="Preferred Foot", y_col="Aggression", name="Juventus"),

      violintrace(df=df[df["Club"] == "Manchester City"], x_col="Preferred Foot", y_col="Aggression", name="Manchester City"),

      violintrace(df=df[df["Club"] == "Roma"], x_col="Preferred Foot", y_col="Aggression", name="Roma")]

layout = {

        "title": "Aggression",

        "yaxis": {

            "zeroline": False,

        },

        "violinmode": "group"

    }

fig = go.Figure(data=data, layout=layout)

iplot(fig)
iplot(distplot(df=df[(df["Club"] == "FC Barcelona") & (df["Preferred Foot"] == "Left")], col_names=["Aggression"]))
iplot(distplot(df=df[df["Club"] == "Paris Saint-Germain"], col_names=["Value_eur"]))
df["Age"].hist()
# Group by Clubs and get mean of overall ratings

df_group1 = df[["Club","Overall"]].groupby(["Club"])["Overall"].mean().reset_index().sort_values("Overall", ascending=False)
df_group1.head()
iplot([bartrace(df=df_group1, x_col="Club", y_col="Overall")])
# We first group by Preferred Foot and Club and get the maximum value of Value_eur (value in euros). Other options are mean(), median() etc

# To minimize the visualization, we restrict players with Value > 25Million

df_grp1 = df[df["Value_eur"] > 25000000][["Club", "Preferred Foot", "Value_eur"]].groupby(["Club", "Preferred Foot"]).max().reset_index().sort_values(["Club", "Value_eur"])

# Here we combine two barplots into a single one for visualization

data = [

            bartrace(df=df_grp1[df_grp1["Preferred Foot"] == "Right"], x_col="Club", y_col="Value_eur", name="Right footed"),

            bartrace(df=df_grp1[df_grp1["Preferred Foot"] == "Left"], x_col="Club", y_col="Value_eur", name="Left footed")

       ]

layout = go.Layout(

    title='Aggregated player value',

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
# Limiting plot to players with Value > 25million

iplot([scattertrace(df=df[df["Value_eur"] > 25000000], x_col="height_inches", y_col="weight_lbs", hover_col="Name")])
x = df[df["Name"] == "Neymar Jr"]

y = df[df["Name"] == "L. Modrić"]

data = [go.Scatterpolar(

  r = [x['Crossing'].values[0],x['Finishing'].values[0],x['Dribbling'].values[0],x['ShortPassing'].values[0],x['LongPassing'].values[0],x['BallControl'].values[0]],

  theta = ['Crossing', 'Finishing', 'Dribbling', 'ShortPassing', 'LongPassing', 'BallControl'],

  fill = 'toself',

  name=x["Name"].values[0]

),

       go.Scatterpolar(

  r = [y['Crossing'].values[0],y['Finishing'].values[0],y['Dribbling'].values[0],y['ShortPassing'].values[0],y['LongPassing'].values[0],y['BallControl'].values[0]],

  theta = ['Crossing', 'Finishing', 'Dribbling', 'ShortPassing', 'LongPassing', 'BallControl'],

  fill = 'toself',

  name=y["Name"].values[0]

)]



layout = go.Layout(

  polar = dict(

    radialaxis = dict(

      visible = True,

    )

  ),

  showlegend = True,

)

fig = go.Figure(data=data, layout=layout)

iplot(fig, filename = "Player stats")
from collections import Counter

def get_num_vowels(x):

    x = x.lower()

    ctr = Counter(x)

    return sum([ctr[x] for x in ['a','e','i','o','u']]) # Return sum of occurences of a,e,i,o,u

df["name_vowels_count"] = df["Name"].apply(get_num_vowels)
# Now we group by the count(num_vowels) and get average value binned into different counts

df_grp2 = df[["name_vowels_count", "Value_eur"]].groupby(["name_vowels_count"]).agg(["max", "mean"]).reset_index()

df_grp2.columns = ['_'.join(tup).rstrip('_') for tup in df_grp2.columns.values]
df_grp2.head()
data = [

            bartrace(df=df_grp2, x_col="name_vowels_count", y_col="Value_eur_max", name="Value_eur_max"),

            bartrace(df=df_grp2, x_col="name_vowels_count", y_col="Value_eur_mean", name="Value_eur_mean")

       ]

layout = go.Layout(

    title='Vowels player value'

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
df[df["name_vowels_count"] == 0].head()