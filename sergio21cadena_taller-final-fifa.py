import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import plotly.express as px

plotly.offline.init_notebook_mode(connected=True)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

data=pd.read_csv("../input/fifa19/data.csv")
data.head()
data100 = data[0:100]
for i in range(0, len(data100["Height"])):

    data100["Height"][i] = data100["Height"][i].replace("'", ".")
for i in range(0,len(data100["Height"])):

    a = float(data100["Height"][i]) // 1 

    b = round(float(data100["Height"][i]) % 1 , 2)

    if (b == 0.10 or b == 0.11):

        b = b * 10

    data100["Height"][i] = (a * 30.48)+(b * 10 * 2.54)
print(data100["Height"])
data100_1 = data100[["Height","Agility","HeadingAccuracy"]]

data100_1.head()


data100_1 = pd.melt(data100_1, id_vars="Height")

data100_1.head()
plot1_1 = px.scatter(data100_1, x="Height", y="value", color ="variable", symbol = "variable", title = "Fútbol y altura")

plot1_1.show()
data100_2 = data100[data100.Position != "GK"]
data100_2 =  data100_2[["Height","Agility","HeadingAccuracy"]]

data100_2 = pd.melt(data100_2, id_vars="Height")

data100_2.head
plot1_2 = px.scatter(data100_2, x="Height", y="value", color ="variable", symbol = "variable", title = "Fútbol y altura (Sin arqueros)")

plot1_2.show()
data500 = data[0:500]
for i in range(0, len(data500["Wage"])):

    data500["Wage"][i] = data500["Wage"][i].replace("€", "")

    data500["Wage"][i] = data500["Wage"][i].replace("K", "")

data500["Wage"].head()

data_box = data500[(data.Nationality == "Argentina")|(data.Nationality=="France")|(data.Nationality == "Germany")|(data.Nationality == "Brazil")|(data.Nationality == "Italy")]

plot2= px.box(data_box, y = "Wage", color = "Nationality", title = "Análisis salarial por nacionalidad")

plot2.show()
data_vs = data [["Name","Penalties","LongShots","Stamina","Agility","SprintSpeed","Finishing"]]

data_vs = data_vs[(data_vs["Name"]=="L. Messi")| (data_vs["Name"]=="Cristiano Ronaldo")]



print(data_vs)
data_vs = pd.melt(data_vs, id_vars="Name")

data_vs.head()
plot_3 = px.bar(data_vs, x = "variable", y="value", color="Name", barmode= "group", orientation="v", title="Cristiano vs Messi")

plot_3.show()
data_al = data[["Club","Age"]]

data_ajax =data_al[data_al["Club"]=="Ajax"]

data_liv=data_al[data_al["Club"]=="Liverpool"]

rep_liv = pd.DataFrame(data_liv["Age"].value_counts())

liv = pd.DataFrame()

liv["Team"] = np.repeat("Liverpool", len(rep_liv))

liv["Age"] = rep_liv.index

liv["Num_Players"] = rep_liv.reset_index()["Age"]



rep_ajax = pd. DataFrame(data_ajax["Age"].value_counts())

ajax = pd.DataFrame()

ajax["Team"] = np.repeat("Ajax", len(rep_ajax))

ajax["Age"] = rep_ajax.index

ajax["Num_Players"] = rep_ajax.reset_index()["Age"]

print(ajax)

print(liv)

sum(rep_ajax["Age"])

sum(rep_liv["Age"])
data_ajax_liv = liv.append(ajax)

print(data_ajax_liv)
plot_4 =px.area(data_ajax_liv, x="Age",y = "Num_Players", color = "Team", title = "Élite del fútbol, ¿Juventud o Experiencia?" )

plot_4.show()
average_speed = np.mean(data["SprintSpeed"])

print(average_speed)
plot_5 = go.Figure()

 

plot_5.add_trace(go.Indicator(

    value = data["SprintSpeed"][0],

    mode = "gauge+number+delta",

    title = {'text': "Speed " + data["Name"][0] + " (" + data["Club"][0] + ")", 'font': {'size': 18}} ,

    delta = {'reference': average_speed},

    gauge = {'axis': {'range': [None, 100],'visible': False},

            'threshold' : {'line': {'color': "red", 'width': 5}, 'thickness': 1, 'value': average_speed}},

            domain = {'row': 0, 'column': 0}))



plot_5.add_trace(go.Indicator(

    value = data["SprintSpeed"][1],

    mode = "gauge+number+delta",

    title = {'text': "Speed " + data["Name"][1] +  " (" + data["Club"][1] + ")", 'font': {'size': 18}} ,

    delta = {'reference': average_speed},

    gauge = {'axis': {'range': [None, 100],'visible': False},

            'threshold' : {'line': {'color': "red", 'width': 5}, 'thickness': 1, 'value': average_speed}},

            domain = {'row': 0, 'column': 1}))



plot_5.add_trace(go.Indicator(

    value = data["SprintSpeed"][0],

    mode = "gauge+number+delta",

    title = {'text': "Speed " + data["Name"][2] + " (" + data["Club"][2] + ")", 'font': {'size': 18}} ,

    delta = {'reference': average_speed},

    gauge = {'axis': {'range': [None, 100],'visible': False},

            'threshold' : {'line': {'color': "red", 'width': 5}, 'thickness': 1, 'value': average_speed}},

            domain = {'row': 1, 'column': 0}))



plot_5.add_trace(go.Indicator(

    value = data["SprintSpeed"][3],

    mode = "gauge+number+delta",

    title = {'text': "Speed " + data["Name"][3] +  "(" + data["Club"][3] + ")", 'font': {'size': 18}} ,

    delta = {'reference': average_speed},

    gauge = {'axis': {'range': [None, 100],'visible': False},

            'threshold' : {'line': {'color': "red", 'width': 5}, 'thickness': 1, 'value': average_speed}},

            domain = {'row': 1, 'column': 1}))



plot_5.add_trace(go.Indicator(

    value = data["SprintSpeed"][4],

    mode = "gauge+number+delta",

    title = {'text': "Speed " + data["Name"][4] + " (" + data["Club"][4] + ")", 'font': {'size': 18}} ,

    delta = {'reference': average_speed},

    gauge = {'axis': {'range': [None, 100],'visible': False},

            'threshold' : {'line': {'color': "red", 'width': 5}, 'thickness': 1, 'value': average_speed}},

            domain = {'row': 2, 'column': 0}))



plot_5.add_trace(go.Indicator(

    value = data["SprintSpeed"][5],

    mode = "gauge+number+delta",

    title = {'text': "Speed " + data["Name"][5] +  " (" + data["Club"][5] + ")", 'font': {'size': 18}} ,

    delta = {'reference': average_speed},

    gauge = {'axis': {'range': [None, 100],'visible': False},

            'threshold' : {'line': {'color': "red", 'width': 5}, 'thickness': 1, 'value': average_speed}},

            domain = {'row': 2, 'column': 1}))



plot_5.update_layout(

    grid = {'rows': 3, 'columns': 2}, height = 950)



plot_5.show()