import numpy as np

import matplotlib.pyplot as plt

from IPython.display import display, Markdown, Latex

from bs4 import BeautifulSoup

import textwrap

import requests

import re

import ast

import json
def scrap(id):

    try:

        r = requests.get("https://www.worldometers.info/coronavirus/country/"+id)

        soup = BeautifulSoup(r.text)

        s = soup.find(id="graph-active-cases-total").next_element.next_element

        x = re.search("[\n\r].*xAxis:\s*(.*\s*categories:\s*\[)([^]]*)", s.text)

        x = ast.literal_eval('['+x.group(2)+']')

        y = re.search("[\n\r].*data:\s*([\[])([^]]*)", s.text)

        y = ast.literal_eval('['+y.group(2)+']')

        

        s = soup.find(id="coronavirus-deaths-log").next_element.next_element.next_element.next_element.next_element.next_element

        p = re.search("[\n\r].*xAxis:\s*(.*\s*categories:\s*\[)([^]]*)", s.text)

        p = ast.literal_eval('['+p.group(2)+']')

        q = re.search("[\n\r].*data:\s*([\[])([^]]*)", s.text)

        q = ast.literal_eval('['+q.group(2)+']')

        return x, y, p, q

    except:

        print("Error {}".format(id))
countries = [

    {"id" : "china", "name" : "Tiongkok", "active" : {}, "deaths" :{}},

    {"id" : "south-korea", "name" : "Korea Selatan", "active" : {}, "deaths" :{}},

    {"id" : "italy", "name" : "Italia", "active" : {}, "deaths" :{}},

    {"id" : "iran", "name" : "Iran", "active" : {}, "deaths" :{}},

    {"id" : "spain", "name" : "Spanyol", "active" : {}, "deaths" :{}},

    {"id" : "germany", "name" : "Jerman", "active" : {}, "deaths" :{}},

    {"id" : "australia", "name" : "Australia", "active" : {}, "deaths" :{}},

    {"id" : "us", "name" : "Amerika Serikat", "active" : {}, "deaths" :{}},

    {"id" : "indonesia", "name" : "Indonesia", "active" : {}, "deaths" :{}}

]

jsonCountries = [

    {"id" : "china", "name" : "Tiongkok", "active" : {}, "deaths" :{}},

    {"id" : "south-korea", "name" : "Korea Selatan", "active" : {}, "deaths" :{}},

    {"id" : "italy", "name" : "Italia", "active" : {}, "deaths" :{}},

    {"id" : "iran", "name" : "Iran", "active" : {}, "deaths" :{}},

    {"id" : "spain", "name" : "Spanyol", "active" : {}, "deaths" :{}},

    {"id" : "germany", "name" : "Jerman", "active" : {}, "deaths" :{}},

    {"id" : "australia", "name" : "Australia", "active" : {}, "deaths" :{}},

    {"id" : "us", "name" : "Amerika Serikat", "active" : {}, "deaths" :{}},

    {"id" : "indonesia", "name" : "Indonesia", "active" : {}, "deaths" :{}}

]

for i,country in enumerate(countries):

    x, y, p, q = scrap(country["id"])

    jsonCountries[i]["active"]["x"] = x

    jsonCountries[i]["active"]["y"] = y

    jsonCountries[i]["deaths"]["x"] = p

    jsonCountries[i]["deaths"]["y"] = q

    

    countries[i]["active"]["x"] = np.array(x)

    countries[i]["active"]["y"] = np.array(y)

    countries[i]["deaths"]["x"] = np.array(p)

    countries[i]["deaths"]["y"] = np.array(q)



with open('countries.json', 'w') as f:

    json.dump(jsonCountries, f)
for i, country in enumerate(countries):

    x = country["active"]['x']

    y = country["active"]['y']

    spacing = 2

    fig, ax = plt.subplots(figsize=(16, 12))

    ax.plot(y, 'b-', label="Active Cases")

    plt.legend(loc="lower right")

    plt.grid()

    plt.title(country["name"])

    plt.xlabel('Date')

    plt.ylabel('Active Cases')

    ax.set_xticks(np.arange(len(x)))

    ax.set_xticklabels(x)

    plt.xticks(rotation=60)

    for i,label in enumerate(ax.xaxis.get_ticklabels()[1::spacing]):

        label.set_visible(False)

    plt.show()
for i, country in enumerate(countries):

    x = country["deaths"]['x']

    y = country["deaths"]['y']

    spacing = 2

    fig, ax = plt.subplots(figsize=(16,12))

    ax.plot(y, 'r-', label="Death Cases")

    plt.legend(loc="lower right")

    plt.grid()

    plt.title(country["name"])

    plt.xlabel('Date')

    plt.ylabel('Death Cases')

    ax.set_xticks(np.arange(len(x)))

    ax.set_xticklabels(x)

    plt.xticks(rotation=60)

    for i,label in enumerate(ax.xaxis.get_ticklabels()[1::spacing]):

        label.set_visible(False)

    plt.show()