# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import plotly as py

import plotly.graph_objs as go

import warnings

from plotly.offline import init_notebook_mode, iplot, plot

warnings.filterwarnings("ignore")

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_total = pd.read_csv("/kaggle/input/michael-jordan-kobe-bryant-and-lebron-james-stats/totals_stats.csv")

df_high= pd.read_csv("/kaggle/input/michael-jordan-kobe-bryant-and-lebron-james-stats/game_highs_stats.csv")
from keras.preprocessing.image import load_img

img_jordan = load_img('/kaggle/input/michael-jordan-kobe-bryant-and-lebron-james-stats/jordan.jpg')

print('Micheal Jordan')

plt.imshow(img_jordan)

plt.show()
from keras.preprocessing.image import load_img

img_kobe = load_img('/kaggle/input/michael-jordan-kobe-bryant-and-lebron-james-stats/kobe.jpg')

print('Kobe Bryant')

plt.imshow(img_kobe)

plt.show()
from keras.preprocessing.image import load_img

img_lebron = load_img('/kaggle/input/michael-jordan-kobe-bryant-and-lebron-james-stats/james.jpg')

print('Lebron James')

plt.imshow(img_lebron)

plt.show()
df_total.head(10)
df_total.columns
df_total.info()
df_high.head(10)
df_high.columns
df_high.info()
lebron_index=[]

for i in df_total.index:

    if (df_total["Player"][i]=="Lebron James"):

        lebron_index.append(i)

        

lebron_df= df_total.iloc[lebron_index,:]

a=1

b=1

lebron_df["SeasonNumber"] = lebron_df.Season

for i in lebron_df.index: 

    if lebron_df["RSorPO"][i]=="Regular Season":

        lebron_df["SeasonNumber"][i]="Player Season " + str(a)

        a= a+1

    else:

        lebron_df["SeasonNumber"][i]="Playoff Number-" + str(b)

        b= b+1
lebron_df.head()
kobe_index=[]

for i in df_total.index:

    if (df_total["Player"][i]=="Kobe Bryant"):

        kobe_index.append(i)

        

kobe_df= df_total.iloc[kobe_index,:]

kobe_df["SeasonNumber"] = kobe_df.Season

a=1

b=1

for i in kobe_df.index: 

    if kobe_df["RSorPO"][i]=="Regular Season":

        kobe_df["SeasonNumber"][i]="Player Season " + str(a)

        a= a+1

    else:

        kobe_df["SeasonNumber"][i]="Playoff Number-" + str(b)

        b= b+1
kobe_df.head()
jordan_index=[]

for i in df_total.index:

    if (df_total["Player"][i]== "Michael Jordan"):

        jordan_index.append(i)

        

jordan_df= df_total.iloc[jordan_index,:]

a=1

b=1

jordan_df["SeasonNumber"] = jordan_df.Season

for i in jordan_df.index: 

    if jordan_df["RSorPO"][i]=="Regular Season":

        jordan_df["SeasonNumber"][i]="Player Season " + str(a)

        a= a+1

    else:

        jordan_df["SeasonNumber"][i]="Playoff Number-" + str(b)

        b= b+1
jordan_df.head()
lebron_index=[]

for i in df_high.index:

    if (df_high["Player"][i]=="Lebron James"):

        lebron_index.append(i)

        

lebron_high= df_high.iloc[lebron_index,:]

a=1

b=1

lebron_high["SeasonNumber"] = lebron_high.Season

for i in lebron_high.index: 

    if lebron_high["RSorPO"][i]=="Regular Season":

        lebron_high["SeasonNumber"][i]="Player Season " + str(a)

        a= a+1

    else:

        lebron_high["SeasonNumber"][i]="Playoff Number-" + str(b)

        b= b+1
lebron_high.head()
jordan_index=[]

for i in df_high.index:

    if (df_high["Player"][i]=="Michael Jordan"):

        jordan_index.append(i)

        

jordan_high= df_high.iloc[jordan_index,:]

a=1

b=1

jordan_high["SeasonNumber"] = jordan_high.Season

for i in jordan_high.index: 

    if jordan_high["RSorPO"][i]=="Regular Season":

        jordan_high["SeasonNumber"][i]="Player Season " + str(a)

        a= a+1

    else:

        jordan_high["SeasonNumber"][i]="Playoff Number-" + str(b)

        b= b+1
jordan_high.head()
kobe_index=[]

for i in df_high.index:

    if (df_high["Player"][i]=="Kobe Bryant"):

        kobe_index.append(i)

        

kobe_high= df_high.iloc[kobe_index,:]

a=1

b=1

kobe_high["SeasonNumber"] = kobe_high.Season

for i in kobe_high.index: 

    if kobe_high["RSorPO"][i]=="Regular Season":

        kobe_high["SeasonNumber"][i]="Player Season " + str(a)

        a= a+1

    else:

        kobe_high["SeasonNumber"][i]="Playoff Number-" + str(b)

        b= b+1
kobe_high.head()
lebron_df["PPG"] = lebron_df.PTS/lebron_df.G

jordan_df["PPG"] = jordan_df.PTS/jordan_df.G

kobe_df["PPG"]= kobe_df.PTS/kobe_df.G



trace1 = go.Scatter(

                    x = lebron_df.SeasonNumber[lebron_df["RSorPO"]=="Regular Season"],

                    y = lebron_df.PPG[lebron_df["RSorPO"]=="Regular Season"],

                    mode = "lines",

                    name = "Lebron James",

                    marker = dict(color = 'rgba(5, 200, 5, 0.8)'),

                    text= lebron_df.Player+" Season: "+ lebron_df.Season)

# Creating trace2

trace2 = go.Scatter(

                    x = jordan_df.SeasonNumber[jordan_df["RSorPO"]=="Regular Season"],

                    y = jordan_df.PPG[jordan_df["RSorPO"]=="Regular Season"],

                    mode = "lines",

                    name = "Michael Jordan",

                    marker = dict(color = 'rgba(200, 5, 5, 0.8)'),

                    text= jordan_df.Player+" Season: "+ jordan_df.Season)

# Creating trace2

trace3 = go.Scatter(

                    x = kobe_df.SeasonNumber[kobe_df["RSorPO"]=="Regular Season"],

                    y = kobe_df.PPG[kobe_df["RSorPO"]=="Regular Season"],

                    mode = "lines",

                    name = "Kobe Bryant",

                    marker = dict(color = 'rgba(5, 5, 200, 0.8)'),

                    text= kobe_df.Player+" Season: "+ kobe_df.Season)

data = [trace1, trace2,trace3]

layout = dict(title = 'Points Per Game in Normal Season',

              xaxis= dict(title= 'Season Number',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
trace1 = go.Scatter(

                    x = lebron_df.SeasonNumber[lebron_df["RSorPO"]=="Playoffs"],

                    y = lebron_df.PPG[lebron_df["RSorPO"]=="Playoffs"],

                    mode = "lines",

                    name = "Lebron James",

                    marker = dict(color = 'rgba(5, 200, 5, 0.8)'),

                    text= lebron_df.Player +" Season: "+ lebron_df.Season )

# Creating trace2

trace2 = go.Scatter(

                    x = jordan_df.SeasonNumber[jordan_df["RSorPO"]=="Playoffs"],

                    y = jordan_df.PPG[jordan_df["RSorPO"]=="Playoffs"],

                    mode = "lines",

                    name = "Michael Jordan",

                    marker = dict(color = 'rgba(200, 5, 5, 0.8)'),

                    text= jordan_df.Player+" Season: "+ jordan_df.Season)

# Creating trace2

trace3 = go.Scatter(

                    x = kobe_df.SeasonNumber[kobe_df["RSorPO"]=="Playoffs"],

                    y = kobe_df.PPG[kobe_df["RSorPO"]=="Playoffs"],

                    mode = "lines",

                    name = "Kobe Bryant",

                    marker = dict(color = 'rgba(5, 5, 200, 0.8)'),

                    text= kobe_df.Player +" Season: "+ kobe_df.Season)

data = [trace1, trace2,trace3]

layout = dict(title = 'Points Per Game in Playoffs',

              xaxis= dict(title= 'Playoffs Number',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
lebron_df["PPM"] = lebron_df.PTS/lebron_df.MP

jordan_df["PPM"] = jordan_df.PTS/jordan_df.MP

kobe_df["PPM"]= kobe_df.PTS/kobe_df.MP



trace1 = go.Scatter(

                    x = lebron_df.SeasonNumber[lebron_df["RSorPO"]=="Regular Season"],

                    y = lebron_df.PPM[lebron_df["RSorPO"]=="Regular Season"],

                    mode = "lines",

                    name = "Lebron James",

                    marker = dict(color = 'rgba(5, 200, 5, 0.8)'),

                    text= lebron_df.Player+" Season: "+ lebron_df.Season)

# Creating trace2

trace2 = go.Scatter(

                    x = jordan_df.SeasonNumber[jordan_df["RSorPO"]=="Regular Season"],

                    y = jordan_df.PPM[jordan_df["RSorPO"]=="Regular Season"],

                    mode = "lines",

                    name = "Michael Jordan",

                    marker = dict(color = 'rgba(200, 5, 5, 0.8)'),

                    text= jordan_df.Player+" Season: "+ jordan_df.Season)

# Creating trace2

trace3 = go.Scatter(

                    x = kobe_df.SeasonNumber[kobe_df["RSorPO"]=="Regular Season"],

                    y = kobe_df.PPM[kobe_df["RSorPO"]=="Regular Season"],

                    mode = "lines",

                    name = "Kobe Bryant",

                    marker = dict(color = 'rgba(5, 5, 200, 0.8)'),

                    text= kobe_df.Player+" Season: "+ kobe_df.Season)

data = [trace1, trace2,trace3]

layout = dict(title = 'Points Per Minute in Normal Season',

              xaxis= dict(title= 'Season Number',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
trace1 = go.Scatter(

                    x = lebron_df.SeasonNumber[lebron_df["RSorPO"]=="Playoffs"],

                    y = lebron_df.PPM[lebron_df["RSorPO"]=="Playoffs"],

                    mode = "lines",

                    name = "Lebron James",

                    marker = dict(color = 'rgba(5, 200, 5, 0.8)'),

                    text= lebron_df.Player +" Season: "+ lebron_df.Season )

# Creating trace2

trace2 = go.Scatter(

                    x = jordan_df.SeasonNumber[jordan_df["RSorPO"]=="Playoffs"],

                    y = jordan_df.PPM[jordan_df["RSorPO"]=="Playoffs"],

                    mode = "lines",

                    name = "Michael Jordan",

                    marker = dict(color = 'rgba(200, 5, 5, 0.8)'),

                    text= jordan_df.Player+" Season: "+ jordan_df.Season)

# Creating trace2

trace3 = go.Scatter(

                    x = kobe_df.SeasonNumber[kobe_df["RSorPO"]=="Playoffs"],

                    y = kobe_df.PPM[kobe_df["RSorPO"]=="Playoffs"],

                    mode = "lines",

                    name = "Kobe Bryant",

                    marker = dict(color = 'rgba(5, 5, 200, 0.8)'),

                    text= kobe_df.Player +" Season: "+ kobe_df.Season)

data = [trace1, trace2,trace3]

layout = dict(title = 'Points Per Game in Playoffs',

              xaxis= dict(title= 'Playoffs Number',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
trace1 = go.Bar(

                x = lebron_high.SeasonNumber[lebron_high["RSorPO"]=="Regular Season"],

                y = lebron_high.PTS[lebron_high["RSorPO"]=="Regular Season"],

                name = "Lebron James",

                marker = dict(color = 'rgba(255, 0, 0, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = lebron_high.Player+" Season: "+ lebron_df.Season)

# create trace2 

trace2 = go.Bar(

                x = jordan_high.SeasonNumber[jordan_high["RSorPO"]=="Regular Season"],

                y = jordan_high.PTS[jordan_high["RSorPO"]=="Playoffs"],

                name = "Michael Jordan",

                marker = dict(color = 'rgba(0, 255, 0, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = jordan_high.Player+" Season: "+ jordan_df.Season)

trace3 = go.Bar(

                x = kobe_high.SeasonNumber[kobe_high["RSorPO"]=="Regular Season"],

                y = kobe_high.PTS[kobe_high["RSorPO"]=="Regular Season"],

                name = "Kobe Bryant",

                marker = dict(color = 'rgba(0, 0, 255, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = kobe_high.Player+" Season: "+ kobe_df.Season)

data = [trace1, trace2,trace3]

layout = go.Layout(barmode = "group",title="Most Points in Regular Season")

fig = go.Figure(data = data, layout = layout)

iplot(fig)
trace1 = go.Bar(

                x = lebron_high.SeasonNumber[lebron_high["RSorPO"]=="Playoffs"],

                y = lebron_high.PTS[lebron_high["RSorPO"]=="Playoffs"],

                name = "Lebron James",

                marker = dict(color = 'rgba(255, 0, 0, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = lebron_high.Player+" Season: "+ lebron_df.Season)

# create trace2 

trace2 = go.Bar(

                x = jordan_high.SeasonNumber[jordan_high["RSorPO"]=="Playoffs"],

                y = jordan_high.PTS[jordan_high["RSorPO"]=="Playoffs"],

                name = "Michael Jordan",

                marker = dict(color = 'rgba(0, 255, 0, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = jordan_high.Player+" Season: "+ jordan_df.Season)

trace3 = go.Bar(

                x = kobe_high.SeasonNumber[kobe_high["RSorPO"]=="Playoffs"],

                y = kobe_high.PTS[kobe_high["RSorPO"]=="Playoffs"],

                name = "Kobe Bryant",

                marker = dict(color = 'rgba(0, 0, 255, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = kobe_high.Player+" Season: "+ kobe_df.Season)

data = [trace1, trace2,trace3]

layout = go.Layout(barmode = "group",title="Most Points in Playoffs")

fig = go.Figure(data = data, layout = layout)

iplot(fig)
a = [np.mean(lebron_df.PPG),np.mean(jordan_df.PPG),np.mean(kobe_df.PPG)]

b = ["Lebron Career Average","Jordan Career Average","Kobe Career Average"]

sns.barplot(x=a, y=b)

plt.xlabel('Points Average')

plt.ylabel("Player's Name")

plt.title("Player's Career Points Average")

plt.show()
#Points part's weight is 2

points = {}

points = {"Kobe":2*2,

         "Lebron":1*2,

         "Jordan":3*2}
lebron_df["APG"] = lebron_df.AST/lebron_df.G

jordan_df["APG"] = jordan_df.AST/jordan_df.G

kobe_df["APG"]= kobe_df.AST/kobe_df.G



trace1 = go.Scatter(

                    x = lebron_df.SeasonNumber[lebron_df["RSorPO"]=="Regular Season"],

                    y = lebron_df.APG[lebron_df["RSorPO"]=="Regular Season"],

                    mode = "lines",

                    name = "Lebron James",

                    marker = dict(color = 'rgba(5, 200, 5, 0.8)'),

                    text= lebron_df.Player+" Season: "+ lebron_df.Season)

# Creating trace2

trace2 = go.Scatter(

                    x = jordan_df.SeasonNumber[jordan_df["RSorPO"]=="Regular Season"],

                    y = jordan_df.APG[jordan_df["RSorPO"]=="Regular Season"],

                    mode = "lines",

                    name = "Michael Jordan",

                    marker = dict(color = 'rgba(200, 5, 5, 0.8)'),

                    text= jordan_df.Player+" Season: "+ jordan_df.Season)

# Creating trace2

trace3 = go.Scatter(

                    x = kobe_df.SeasonNumber[kobe_df["RSorPO"]=="Regular Season"],

                    y = kobe_df.APG[kobe_df["RSorPO"]=="Regular Season"],

                    mode = "lines",

                    name = "Kobe Bryant",

                    marker = dict(color = 'rgba(5, 5, 200, 0.8)'),

                    text= kobe_df.Player+" Season: "+ kobe_df.Season)

data = [trace1, trace2,trace3]

layout = dict(title = 'Assists Per Game in Normal Season',

              xaxis= dict(title= 'Season Number',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
trace1 = go.Scatter(

                    x = lebron_df.SeasonNumber[lebron_df["RSorPO"]=="Playoffs"],

                    y = lebron_df.APG[lebron_df["RSorPO"]=="Playoffs"],

                    mode = "lines",

                    name = "Lebron James",

                    marker = dict(color = 'rgba(5, 200, 5, 0.8)'),

                    text= lebron_df.Player +" Season: "+ lebron_df.Season )

# Creating trace2

trace2 = go.Scatter(

                    x = jordan_df.SeasonNumber[jordan_df["RSorPO"]=="Playoffs"],

                    y = jordan_df.APG[jordan_df["RSorPO"]=="Playoffs"],

                    mode = "lines",

                    name = "Michael Jordan",

                    marker = dict(color = 'rgba(200, 5, 5, 0.8)'),

                    text= jordan_df.Player+" Season: "+ jordan_df.Season)

# Creating trace2

trace3 = go.Scatter(

                    x = kobe_df.SeasonNumber[kobe_df["RSorPO"]=="Playoffs"],

                    y = kobe_df.APG[kobe_df["RSorPO"]=="Playoffs"],

                    mode = "lines",

                    name = "Kobe Bryant",

                    marker = dict(color = 'rgba(5, 5, 200, 0.8)'),

                    text= kobe_df.Player +" Season: "+ kobe_df.Season)

data = [trace1, trace2,trace3]

layout = dict(title = 'Assists Per Game in Playoffs',

              xaxis= dict(title= 'Playoffs Number',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
trace1 = go.Bar(

                x = lebron_high.SeasonNumber[lebron_high["RSorPO"]=="Regular Season"],

                y = lebron_high.AST[lebron_high["RSorPO"]=="Regular Season"],

                name = "Lebron James",

                marker = dict(color = 'rgba(255, 0, 0, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = lebron_high.Player+" Season: "+ lebron_df.Season)

# create trace2 

trace2 = go.Bar(

                x = jordan_high.SeasonNumber[jordan_high["RSorPO"]=="Regular Season"],

                y = jordan_high.AST[jordan_high["RSorPO"]=="Playoffs"],

                name = "Michael Jordan",

                marker = dict(color = 'rgba(0, 255, 0, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = jordan_high.Player+" Season: "+ jordan_df.Season)

trace3 = go.Bar(

                x = kobe_high.SeasonNumber[kobe_high["RSorPO"]=="Regular Season"],

                y = kobe_high.AST[kobe_high["RSorPO"]=="Regular Season"],

                name = "Kobe Bryant",

                marker = dict(color = 'rgba(0, 0, 255, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = kobe_high.Player+" Season: "+ kobe_df.Season)

data = [trace1, trace2,trace3]

layout = go.Layout(barmode = "group",title="Most Assists in Regular Season")

fig = go.Figure(data = data, layout = layout)

iplot(fig)
trace1 = go.Bar(

                x = lebron_high.SeasonNumber[lebron_high["RSorPO"]=="Playoffs"],

                y = lebron_high.AST[lebron_high["RSorPO"]=="Playoffs"],

                name = "Lebron James",

                marker = dict(color = 'rgba(255, 0, 0, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = lebron_high.Player+" Season: "+ lebron_df.Season)

# create trace2 

trace2 = go.Bar(

                x = jordan_high.SeasonNumber[jordan_high["RSorPO"]=="Playoffs"],

                y = jordan_high.AST[jordan_high["RSorPO"]=="Playoffs"],

                name = "Michael Jordan",

                marker = dict(color = 'rgba(0, 255, 0, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = jordan_high.Player+" Season: "+ jordan_df.Season)

trace3 = go.Bar(

                x = kobe_high.SeasonNumber[kobe_high["RSorPO"]=="Playoffs"],

                y = kobe_high.AST[kobe_high["RSorPO"]=="Playoffs"],

                name = "Kobe Bryant",

                marker = dict(color = 'rgba(0, 0, 255, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = kobe_high.Player+" Season: "+ kobe_df.Season)

data = [trace1, trace2,trace3]

layout = go.Layout(barmode = "group",title="Most Assists in Playoffs")

fig = go.Figure(data = data, layout = layout)

iplot(fig)
a = [np.mean(lebron_df.APG),np.mean(jordan_df.APG),np.mean(kobe_df.APG)]

b = ["Lebron Career Average","Jordan Career Average","Kobe Career Average"]

sns.barplot(x=a, y=b)

plt.xlabel('Points Average')

plt.ylabel("Player's Name")

plt.title("Player's Career Assists Average")

plt.show()
points["Kobe"]=points["Kobe"]+1

points["Lebron"]=points["Lebron"]+3

points["Jordan"]=points["Jordan"]+2
jordan_df["RP"] = jordan_df["PPG"] + jordan_df["APG"]*2

lebron_df["RP"] = lebron_df["PPG"] + lebron_df["APG"]*2

kobe_df["RP"] = kobe_df["PPG"] + kobe_df["APG"]*2

trace1 = go.Scatter(

                    x = lebron_df.SeasonNumber[lebron_df["RSorPO"]=="Regular Season"],

                    y = lebron_df.RP[lebron_df["RSorPO"]=="Regular Season"],

                    mode = "lines",

                    name = "Lebron James",

                    marker = dict(color = 'rgba(5, 200, 5, 0.8)'),

                    text= lebron_df.Player+" Season: "+ lebron_df.Season)

# Creating trace2

trace2 = go.Scatter(

                    x = jordan_df.SeasonNumber[jordan_df["RSorPO"]=="Regular Season"],

                    y = jordan_df.RP[jordan_df["RSorPO"]=="Regular Season"],

                    mode = "lines",

                    name = "Michael Jordan",

                    marker = dict(color = 'rgba(200, 5, 5, 0.8)'),

                    text= jordan_df.Player+" Season: "+ jordan_df.Season)

# Creating trace2

trace3 = go.Scatter(

                    x = kobe_df.SeasonNumber[kobe_df["RSorPO"]=="Regular Season"],

                    y = kobe_df.RP[kobe_df["RSorPO"]=="Regular Season"],

                    mode = "lines",

                    name = "Kobe Bryant",

                    marker = dict(color = 'rgba(5, 5, 200, 0.8)'),

                    text= kobe_df.Player+" Season: "+ kobe_df.Season)

data = [trace1, trace2,trace3]

layout = dict(title = 'Responsible Points in Normal Season',

              xaxis= dict(title= 'Season Number',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
trace1 = go.Scatter(

                    x = lebron_df.SeasonNumber[lebron_df["RSorPO"]=="Playoffs"],

                    y = lebron_df.RP[lebron_df["RSorPO"]=="Playoffs"],

                    mode = "lines",

                    name = "Lebron James",

                    marker = dict(color = 'rgba(5, 200, 5, 0.8)'),

                    text= lebron_df.Player +" Season: "+ lebron_df.Season )

# Creating trace2

trace2 = go.Scatter(

                    x = jordan_df.SeasonNumber[jordan_df["RSorPO"]=="Playoffs"],

                    y = jordan_df.RP[jordan_df["RSorPO"]=="Playoffs"],

                    mode = "lines",

                    name = "Michael Jordan",

                    marker = dict(color = 'rgba(200, 5, 5, 0.8)'),

                    text= jordan_df.Player+" Season: "+ jordan_df.Season)

# Creating trace2

trace3 = go.Scatter(

                    x = kobe_df.SeasonNumber[kobe_df["RSorPO"]=="Playoffs"],

                    y = kobe_df.RP[kobe_df["RSorPO"]=="Playoffs"],

                    mode = "lines",

                    name = "Kobe Bryant",

                    marker = dict(color = 'rgba(5, 5, 200, 0.8)'),

                    text= kobe_df.Player +" Season: "+ kobe_df.Season)

data = [trace1, trace2,trace3]

layout = dict(title = 'Responsible Points in Playoffs',

              xaxis= dict(title= 'Season Number',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
points["Kobe"]=points["Kobe"]+1

points["Lebron"]=points["Lebron"]+2

points["Jordan"]=points["Jordan"]+3
lebron_df["TRPG"] = lebron_df.TRB/lebron_df.G

jordan_df["TRPG"] = jordan_df.TRB/jordan_df.G

kobe_df["TRPG"]= kobe_df.TRB/kobe_df.G



trace1 = go.Scatter(

                    x = lebron_df.SeasonNumber[lebron_df["RSorPO"]=="Regular Season"],

                    y = lebron_df.TRPG[lebron_df["RSorPO"]=="Regular Season"],

                    mode = "lines",

                    name = "Lebron James",

                    marker = dict(color = 'rgba(5, 200, 5, 0.8)'),

                    text= lebron_df.Player+" Season: "+ lebron_df.Season)

# Creating trace2

trace2 = go.Scatter(

                    x = jordan_df.SeasonNumber[jordan_df["RSorPO"]=="Regular Season"],

                    y = jordan_df.TRPG[jordan_df["RSorPO"]=="Regular Season"],

                    mode = "lines",

                    name = "Michael Jordan",

                    marker = dict(color = 'rgba(200, 5, 5, 0.8)'),

                    text= jordan_df.Player+" Season: "+ jordan_df.Season)

# Creating trace2

trace3 = go.Scatter(

                    x = kobe_df.SeasonNumber[kobe_df["RSorPO"]=="Regular Season"],

                    y = kobe_df.TRPG[kobe_df["RSorPO"]=="Regular Season"],

                    mode = "lines",

                    name = "Kobe Bryant",

                    marker = dict(color = 'rgba(5, 5, 200, 0.8)'),

                    text= kobe_df.Player+" Season: "+ kobe_df.Season)

data = [trace1, trace2,trace3]

layout = dict(title = 'Total Rebounds Per Game in Normal Season',

              xaxis= dict(title= 'Season Number',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
trace1 = go.Scatter(

                    x = lebron_df.SeasonNumber[lebron_df["RSorPO"]=="Playoffs"],

                    y = lebron_df.TRPG[lebron_df["RSorPO"]=="Playoffs"],

                    mode = "lines",

                    name = "Lebron James",

                    marker = dict(color = 'rgba(5, 200, 5, 0.8)'),

                    text= lebron_df.Player +" Season: "+ lebron_df.Season )

# Creating trace2

trace2 = go.Scatter(

                    x = jordan_df.SeasonNumber[jordan_df["RSorPO"]=="Playoffs"],

                    y = jordan_df.TRPG[jordan_df["RSorPO"]=="Playoffs"],

                    mode = "lines",

                    name = "Michael Jordan",

                    marker = dict(color = 'rgba(200, 5, 5, 0.8)'),

                    text= jordan_df.Player+" Season: "+ jordan_df.Season)

# Creating trace2

trace3 = go.Scatter(

                    x = kobe_df.SeasonNumber[kobe_df["RSorPO"]=="Playoffs"],

                    y = kobe_df.TRPG[kobe_df["RSorPO"]=="Playoffs"],

                    mode = "lines",

                    name = "Kobe Bryant",

                    marker = dict(color = 'rgba(5, 5, 200, 0.8)'),

                    text= kobe_df.Player +" Season: "+ kobe_df.Season)

data = [trace1, trace2,trace3]

layout = dict(title = 'Total Rebounds Per Game in Playoffs',

              xaxis= dict(title= 'Playoffs Number',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
points["Kobe"]=points["Kobe"]+1

points["Lebron"]=points["Lebron"]+3

points["Jordan"]=points["Jordan"]+2
lebron_df["SBT"] = (lebron_df.STL+lebron_df.BLK-lebron_df.TOV)/lebron_df.G

jordan_df["SBT"] = (jordan_df.STL+jordan_df.BLK-jordan_df.TOV)/jordan_df.G

kobe_df["SBT"]= (kobe_df.STL+kobe_df.BLK-kobe_df.TOV)/kobe_df.G



trace1 = go.Scatter(

                    x = lebron_df.SeasonNumber[lebron_df["RSorPO"]=="Regular Season"],

                    y = lebron_df.SBT[lebron_df["RSorPO"]=="Regular Season"],

                    mode = "lines",

                    name = "Lebron James",

                    marker = dict(color = 'rgba(5, 200, 5, 0.8)'),

                    text= lebron_df.Player+" Season: "+ lebron_df.Season)

# Creating trace2

trace2 = go.Scatter(

                    x = jordan_df.SeasonNumber[jordan_df["RSorPO"]=="Regular Season"],

                    y = jordan_df.SBT[jordan_df["RSorPO"]=="Regular Season"],

                    mode = "lines",

                    name = "Michael Jordan",

                    marker = dict(color = 'rgba(200, 5, 5, 0.8)'),

                    text= jordan_df.Player+" Season: "+ jordan_df.Season)

# Creating trace2

trace3 = go.Scatter(

                    x = kobe_df.SeasonNumber[kobe_df["RSorPO"]=="Regular Season"],

                    y = kobe_df.SBT[kobe_df["RSorPO"]=="Regular Season"],

                    mode = "lines",

                    name = "Kobe Bryant",

                    marker = dict(color = 'rgba(5, 5, 200, 0.8)'),

                    text= kobe_df.Player+" Season: "+ kobe_df.Season)

data = [trace1, trace2,trace3]

layout = dict(title = 'Total Turnover, Steal and Blocks Per Game in Normal Season',

              xaxis= dict(title= 'Season Number',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
trace1 = go.Scatter(

                    x = lebron_df.SeasonNumber[lebron_df["RSorPO"]=="Playoffs"],

                    y = lebron_df.SBT[lebron_df["RSorPO"]=="Playoffs"],

                    mode = "lines",

                    name = "Lebron James",

                    marker = dict(color = 'rgba(5, 200, 5, 0.8)'),

                    text= lebron_df.Player +" Season: "+ lebron_df.Season )

# Creating trace2

trace2 = go.Scatter(

                    x = jordan_df.SeasonNumber[jordan_df["RSorPO"]=="Playoffs"],

                    y = jordan_df.SBT[jordan_df["RSorPO"]=="Playoffs"],

                    mode = "lines",

                    name = "Michael Jordan",

                    marker = dict(color = 'rgba(200, 5, 5, 0.8)'),

                    text= jordan_df.Player+" Season: "+ jordan_df.Season)

# Creating trace2

trace3 = go.Scatter(

                    x = kobe_df.SeasonNumber[kobe_df["RSorPO"]=="Playoffs"],

                    y = kobe_df.SBT[kobe_df["RSorPO"]=="Playoffs"],

                    mode = "lines",

                    name = "Kobe Bryant",

                    marker = dict(color = 'rgba(5, 5, 200, 0.8)'),

                    text= kobe_df.Player +" Season: "+ kobe_df.Season)

data = [trace1, trace2,trace3]

layout = dict(title = 'Total Turnover, Steal and Blocks Per Game in Playoffs',

              xaxis= dict(title= 'Playoffs Number',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
points["Kobe"]=points["Kobe"]+2

points["Lebron"]=points["Lebron"]+1

points["Jordan"]=points["Jordan"]+3
lebron_df["perc"] = lebron_df["eFG%"]+lebron_df["FG%"]

jordan_df["perc"] = jordan_df["eFG%"]+jordan_df["FG%"]

kobe_df["perc"]= kobe_df["eFG%"]+kobe_df["FG%"]



trace1 = go.Scatter(

                    x = lebron_df.SeasonNumber[lebron_df["RSorPO"]=="Regular Season"],

                    y = lebron_df.perc[lebron_df["RSorPO"]=="Regular Season"],

                    mode = "lines",

                    name = "Lebron James",

                    marker = dict(color = 'rgba(5, 200, 5, 0.8)'),

                    text= lebron_df.Player+" Season: "+ lebron_df.Season)

# Creating trace2

trace2 = go.Scatter(

                    x = jordan_df.SeasonNumber[jordan_df["RSorPO"]=="Regular Season"],

                    y = jordan_df.perc[jordan_df["RSorPO"]=="Regular Season"],

                    mode = "lines",

                    name = "Michael Jordan",

                    marker = dict(color = 'rgba(200, 5, 5, 0.8)'),

                    text= jordan_df.Player+" Season: "+ jordan_df.Season)

# Creating trace2

trace3 = go.Scatter(

                    x = kobe_df.SeasonNumber[kobe_df["RSorPO"]=="Regular Season"],

                    y = kobe_df.perc[kobe_df["RSorPO"]=="Regular Season"],

                    mode = "lines",

                    name = "Kobe Bryant",

                    marker = dict(color = 'rgba(5, 5, 200, 0.8)'),

                    text= kobe_df.Player+" Season: "+ kobe_df.Season)

data = [trace1, trace2,trace3]

layout = dict(title = 'eFG%+FG% in Normal Season',

              xaxis= dict(title= 'Season Number',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
trace1 = go.Scatter(

                    x = lebron_df.SeasonNumber[lebron_df["RSorPO"]=="Playoffs"],

                    y = lebron_df.perc[lebron_df["RSorPO"]=="Playoffs"],

                    mode = "lines",

                    name = "Lebron James",

                    marker = dict(color = 'rgba(5, 200, 5, 0.8)'),

                    text= lebron_df.Player +" Season: "+ lebron_df.Season )

# Creating trace2

trace2 = go.Scatter(

                    x = jordan_df.SeasonNumber[jordan_df["RSorPO"]=="Playoffs"],

                    y = jordan_df.perc[jordan_df["RSorPO"]=="Playoffs"],

                    mode = "lines",

                    name = "Michael Jordan",

                    marker = dict(color = 'rgba(200, 5, 5, 0.8)'),

                    text= jordan_df.Player+" Season: "+ jordan_df.Season)

# Creating trace2

trace3 = go.Scatter(

                    x = kobe_df.SeasonNumber[kobe_df["RSorPO"]=="Playoffs"],

                    y = kobe_df.perc[kobe_df["RSorPO"]=="Playoffs"],

                    mode = "lines",

                    name = "Kobe Bryant",

                    marker = dict(color = 'rgba(5, 5, 200, 0.8)'),

                    text= kobe_df.Player +" Season: "+ kobe_df.Season)

data = [trace1, trace2,trace3]

layout = dict(title = 'eFG% + FG% in Playoffs',

              xaxis= dict(title= 'Playoffs Number',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
points["Kobe"]=points["Kobe"]+1

points["Lebron"]=points["Lebron"]+3

points["Jordan"]=points["Jordan"]+2
x=list(points.keys())

y=list(points.values())

sns.barplot(x=x, y=y)

plt.xlabel("Player Name")

plt.ylabel("Total Points")

plt.title("Total Points")

plt.show()