import datetime

import requests

import pandas as pd

import json

import matplotlib.pyplot as plt

import ipywidgets as widgets
plt.style.use('ggplot')



plt.rcParams['font.family'] = 'serif'

plt.rcParams['font.serif'] = 'Ubuntu'

plt.rcParams['font.monospace'] = 'Ubuntu Mono'

plt.rcParams['font.size'] = 10

plt.rcParams['axes.labelsize'] = 10

plt.rcParams['axes.labelweight'] = 'bold'

plt.rcParams['axes.titlesize'] = 18

plt.rcParams['xtick.labelsize'] = 8

plt.rcParams['ytick.labelsize'] = 8

plt.rcParams['legend.fontsize'] = 10

plt.rcParams['figure.titlesize'] = 18





color = ('c', 'green', 'red')

cat = ("Active", "Recovered", "Decreased")
RAW_DATA = "https://api.rootnet.in/covid19-in/stats/history"

data = requests.get(RAW_DATA)

jsonData = data.json()
history = [x['day'] for x in jsonData['data']]

historyDataTotal = [x['summary']['total'] for x in jsonData['data']]

historyDataRecovered = [x['summary']['discharged'] for x in jsonData['data']]

historyDataDeaths = [x['summary']['deaths'] for x in jsonData['data']]

# historyData
recovered_and_deaths = [historyDataDeaths[i] + historyDataRecovered[i] for i in range(len(historyDataRecovered))]



width = 0.35

fig, ax = plt.subplots(figsize=(24,8))



ax.bar(history, historyDataTotal, width, bottom=recovered_and_deaths, label="Active", color = color[0])

ax.bar(history, historyDataRecovered, width, bottom=historyDataDeaths, label="Recovered", color = color[1])

ax.bar(history, historyDataDeaths, width, label="Death", color = color[2])



ax.set_xlabel("DATE")

ax.set_ylabel("NUMBER OF PATIENTS")

ax.set_title("Patient Count in different category")

ax.legend()



plt.xticks(rotation=90)

plt.show()
fig, ax = plt.subplots(ncols=3, nrows=1,figsize=(24,8))



ax[0].bar(history, historyDataTotal, width, label="Active", color=color[0])

ax[1].bar(history, historyDataRecovered, width, label="Recovered", color = color[1])

ax[2].bar(history, historyDataDeaths, width, label="Death", color= color[2])



ax[0].set_ylabel("Number of Patients")



for i in range(3):

    ax[i].set_xlabel("DATE")

    ax[i].set_title("Patient Count in {} category".format(cat[i]))

    ax[i].set_xticks("")

    ax[i].legend()

plt.show()
# State based Geo Map yet to develop

# import shapefile as shp

# shp_path = "https://map.igismap.com/share-map/export-layer/India_Boundary/b73ce398c39f506af761d2277d853a92"

# shapeFile = requests.get(shp_path)



# sf = shp.Reader(shapeFile)
stateWise = {}

currentStateWise = {}

totalactive = jsonData['data'][-1]['summary']['total']

totalrecovered = jsonData['data'][-1]['summary']['discharged']

totaldead = jsonData['data'][-1]['summary']['deaths']



for state in jsonData['data'][-1]['regional']:

    currentStateWise[state['loc']] = (state['totalConfirmed']/totalactive,state['discharged']/totalrecovered,state['deaths']/totaldead)

    

for data in jsonData['data']:

    for state in data['regional']:

        stateWise[state['loc']] = (list(), list(), list(), list())



for data in jsonData['data']:

    day = data['day']

    for state in data['regional']:

        stateWise[state['loc']][0].append(state['totalConfirmed'])

        stateWise[state['loc']][1].append(state['discharged'])

        stateWise[state['loc']][2].append(state['deaths'])

        stateWise[state['loc']][3].append(day)

        
cat = ("Active", "Recovered", "Decreased")

recovered_and_deaths = [historyDataDeaths[i] + historyDataRecovered[i] for i in range(len(historyDataRecovered))]

width = 0.35



fig, ax = plt.subplots(ncols=3, nrows=1,figsize=(24,8))

for key in stateWise:

    ax[0].plot([x for counter,x in enumerate(stateWise[key][3]) if stateWise[key][0][counter] > 500], [x for x in stateWise[key][0] if x > 500], width)



for key in stateWise:

    ax[1].plot(stateWise[key][3], stateWise[key][1], width)

    

for key in stateWise:

    ax[2].plot(stateWise[key][3], stateWise[key][2], width)



ax[0].set_ylabel("Number of Patients")



for i in range(3):

    ax[i].set_xlabel("DATE")

    ax[i].set_title("Patient Count in {} category".format(cat[i]))

    ax[i].set_xticks("")

    ax[i].legend()

plt.show()
import random

# c = list(zip(prec, labels, explode))

# random.shuffle(c)

# prec, labels, explode = zip(*c)





def func(pct):

    if pct < 1.8:

        return ""

    else:

        return "{:1.1f}%".format(pct)



fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(24,24))

flataxes = []

for i in range(len(ax)):

    for j in range(len(ax[0])):

        flataxes.append(ax[i][j])

        

for i in range(3):

    explode = [0 if x[i] < 0.075 else 0.1 for x in currentStateWise.values()]

    labels = [x if currentStateWise[x][i] > 0.009 else '' for x in currentStateWise.keys()]

    prec = [x[i] for x in currentStateWise.values()]



    c = list(zip(prec, labels, explode))

    c.sort()

    prec, labels, explode = zip(*c)



    flataxes[i].pie(prec, explode = explode, labels=labels, autopct=lambda pct: func(pct),

        shadow=True, startangle=90)



    flataxes[i].axis("equal")

    flataxes[i].set_title("{} Cases percentage State-Wise".format(cat[i]))



ta = [x[0]*totalactive for x in currentStateWise.values()]

tb = [x[1]*totalrecovered for x in currentStateWise.values()]

tc = [x[2]*totaldead for x in currentStateWise.values()]

td = [tb[i] + tc[i] for i in range(len(tb))]

tx = currentStateWise.keys()



flataxes[-1].bar(tx, ta , width, bottom=td, label="Active", color = color[0])

flataxes[-1].bar(tx, tb, width, bottom=tc, label="Recovered", color = color[1])

flataxes[-1].bar(tx, tc, width, label="Death", color = color[2])

flataxes[-1].set_title("State wise patients in different Category")

flataxes[-1].set_yticks([])

flataxes[-1].grid(False)

flataxes[-1].set_facecolor("white")



plt.xticks(rotation=90)

plt.show()
x = widgets.Dropdown(

    options=stateWise.keys(),

    value='Delhi',

    description='Text:',

    disabled=False,

)
import numpy as np



def plot_func(city):



    width = 0.2

    fig, ax = plt.subplots(figsize=(24,8))

    try:

        ax.bar(np.arange(len(history))-width, stateWise[city][0], width, label="Active", color = color[0])

        ax.bar(np.arange(len(history)), stateWise[city][1], width, label="Recovered", color = color[1])

        ax.bar(np.arange(len(history))+width, stateWise[city][2], width, label="Death", color = color[2])

    except:

        ax.bar(np.arange(len(stateWise[city][0]))-width, stateWise[city][0], width, label="Active", color = color[0])

        ax.bar(np.arange(len(stateWise[city][1])), stateWise[city][1], width, label="Recovered", color = color[1])

        ax.bar(np.arange(len(stateWise[city][2]))+width, stateWise[city][2], width, label="Death", color = color[2])

        

    ax.set_xlabel("DATE")

    ax.set_ylabel("NUMBER OF PATIENTS")

    ax.set_title("Patient Count in different category")

    ax.set_xticks(np.arange(len(history)))

    ax.set_xticklabels(history)

    ax.legend()



    plt.xticks(rotation=90)

    plt.show()
x = widgets.interact(plot_func, city = widgets.Dropdown(

                                            options=stateWise.keys(),

                                            value='Maharashtra',

                                            description='City:',

                                            disabled=False,

                                        ))