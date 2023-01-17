# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os.path as osp # relative paths

#!pip install folium

#!pip install -I pygeocoder==1.2.5

#!pip install colour

import folium

from operator import itemgetter

#from pygeocoder import Geocoder

#from colour import Color







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
conversion_rates = pd.read_csv(osp.join("..","input","conversionRates.csv"))

# convert to dictionary for easy access

conversion_rates = dict(zip(list(conversion_rates.originCountry),list(conversion_rates.exchangeRate))) 

print(conversion_rates)

multiple_choice = pd.read_csv(osp.join("..", "input", "multipleChoiceResponses.csv"),  encoding="ISO-8859-1")

#dic = pd.read_csv(osp.join("..", "input", "multipleChoiceResponses.csv"), names=list(multiple_choice), encoding="ISO-8859-1", header=None)

age_country = multiple_choice[['Country', 'Age']]

# drop null values

age_country = age_country.dropna()

average = age_country.groupby('Country', as_index=False).Age.mean()

print(average)

count = pd.value_counts(age_country['Country'])

count = pd.DataFrame({'Country':count.index, 'Age':count.values})



#print(age_country)

#print("average", average)

#print("count", count)

#k = Geocoder.geocode("USA").valid_address

#print("valid",k)

#print("len", len(count))

#print(count)

average = dict(zip(list(average.Country),list(average.Age))) 

count = dict(zip(list(count.Country), list(count.Age)))

print(average)

print(count)



# place this into separate lists so we can sort this based on one array

averages_n = []

counts_n = []

countries_n = []

for country, avg in average.items():

    #print(country)

    countries_n.append(country)

    #print("average", avg, "count", count[country])

    counts_n.append(count[country])

    averages_n.append(avg)



# dictionary with tuple(lat, lng) for each country

country_placement = {'India': (20.593684, 78.96288), 'Antarctica': (-82.862752, 135), 'Russia': (61.52401, 105.318756), 'United Kingdom': (55.378051, -3.435973), 'Brazil': (-14.235004, -51.92528), "People 's Republic of China": (35.86166, 104.195397), 'Germany': (51.165691, 10.451526), 'France': (46.227638, 2.213749), 'Canada': (56.130366, -106.346771), 'Australia': (-25.274398, 133.775136), 'Spain': (40.46366700000001, -3.74922), 'Japan': (36.204824, 138.252924), 'Taiwan': (23.69781, 120.960515), 'Italy': (41.87194, 12.56738), 'Netherlands': (52.132633, 5.291265999999999), 'Ukraine': (48.379433, 31.1655799), 'South Korea': (35.907757, 127.766922), 'Poland': (51.919438, 19.145136), 'Singapore': (1.352083, 103.819836), 'Pakistan': (30.375321, 69.34511599999999), 'Turkey': (38.963745, 35.243322), 'Indonesia': (-0.789275, 113.921327), 'South Africa': (-30.559482, 22.937506), 'Switzerland': (46.818188, 8.227511999999999), 'Mexico': (23.634501, -102.552784), 'Colombia': (4.570868, -74.297333), 'Iran': (32.427908, 53.688046), 'Israel': (31.046051, 34.851612), 'Argentina': (-38.416097, -63.61667199999999), 'Portugal': (39.39987199999999, -8.224454), 'Ireland': (53.1423672, -7.692053599999999), 'Belgium': (50.503887, 4.469936), 'Sweden': (60.12816100000001, 18.643501), 'Philippines': (12.879721, 121.774017), 'Greece': (39.074208, 21.824312), 'Malaysia': (4.210484, 101.975766), 'Denmark': (56.26392, 9.501785), 'Nigeria': (9.081999, 8.675277), 'New Zealand': (-40.900557, 174.885971), 'Vietnam': (14.058324, 108.277199), 'Republic of China': (23.69781, 120.960515), 'Finland': (61.92410999999999, 25.7481511), 'Hungary': (47.162494, 19.5033041), 'Egypt': (26.820553, 30.802498), 'Hong Kong': (22.396428, 114.109497), 'Kenya': (-0.023559, 37.906193), 'Romania': (45.943161, 24.96676), 'Belarus': (53.709807, 27.953389), 'Czech Republic': (49.81749199999999, 15.472962), 'Norway': (60.47202399999999, 8.468945999999999), 'Chile': (-35.675147, -71.542969), "United States": (37.090240, -95.712891)}



###

### Creatin map to show average age according to color. 

### The size of the circle show how many answered in each contry.

### Antarctica is here used as others

###



m = folium.Map(location=[0, 0], zoom_start=1)

#sort on average age

temp = [x for x in sorted(list(zip(averages_n, countries_n, counts_n)), key=lambda pair: pair[0])]

average_n , countries, counts_n = zip(*temp)

    #average_n , countries, count_n = [list(x) for x in zip(*sorted(zip(averages_n,countries, counts_n), key=itemgetter(0)))]

import math

#ctzzxzz

# white -> black

hexes = ['#fff', '#fafafa', '#f5f5f5', '#f0f0f0', '#ebebeb', '#e6e6e6', '#e1e1e1', '#dcdcdc', '#d7d7d7', '#d2d2d2', '#cdcdcd', '#c8c8c8', '#c3c3c3', '#bebebe', '#b9b9b9', '#b4b4b4', '#afafaf', '#aaa', '#a5a5a5', '#a0a0a0', '#9b9b9b', '#969696', '#919191', '#8c8c8c', '#878787', '#828282', '#7d7d7d', '#787878', '#737373', '#6e6e6e', '#696969', '#646464', '#5f5f5f', '#5a5a5a', '#555', '#505050', '#4b4b4b', '#464646', '#414141', '#3c3c3c', '#373737', '#323232', '#2d2d2d', '#282828', '#232323', '#1e1e1e', '#191919', '#141414', '#0f0f0f', '#0a0a0a', '#050505', '#000']

# blue -> red

hexes = ['#00f', '#0014ff', '#0028ff', '#003cff', '#0050ff', '#0064ff', '#0078ff', '#008cff', '#00a0ff', '#00b4ff', '#00c8ff', '#00dcff', '#00f0ff', '#00fffa', '#00ffe6', '#00ffd2', '#00ffbe', '#0fa', '#00ff96', '#00ff82', '#00ff6e', '#00ff5a', '#00ff46', '#00ff32', '#00ff1e', '#00ff0a', '#0aff00', '#1eff00', '#32ff00', '#46ff00', '#5aff00', '#6eff00', '#82ff00', '#96ff00', '#af0', '#beff00', '#d2ff00', '#e6ff00', '#faff00', '#fff000', '#ffdc00', '#ffc800', '#ffb400', '#ffa000', '#ff8c00', '#ff7800', '#ff6400', '#ff5000', '#ff3c00', '#ff2800', '#ff1400', '#f00']

for ctry_count, ctry,the_hex,age  in zip(counts_n,countries, hexes,average_n):

    if ctry == "Other":

        ctry = "Antarctica"

    x = country_placement[ctry]

    # needed to fix People 's Republic of China

    country = ctry.replace(" '", "")

    if country == "Antarctica":

        country = "Other"

    the_count = math.log(ctry_count / 41.03)*2

    info = country + " Age: " + str(round(age,1))

    folium.CircleMarker([x[0], x[1]],

                    radius=the_count,

                    popup=info,

                    color=the_hex,

                    fill=True,

                    fill_color=the_hex,

                    opacity=1,

                    fill_opacity=1

                   ).add_to(m)

m
country_salary = multiple_choice[['Country', 'CompensationAmount', "CompensationCurrency"]]

country_salary = country_salary.dropna()

# replace "," with empty string

country_salary["CompensationAmount"] = country_salary["CompensationAmount"].str.replace(",","")

# convert form string to float

country_salary["CompensationAmount"] = pd.to_numeric(country_salary["CompensationAmount"], errors="coerce")

print(country_salary)







# CompensationAmount

# map country and money

# Note that people from same country can use different currency when it is put into the fields.

WANTED = "EUR"

total = []

for index, row in country_salary.iterrows():

    Country = row["Country"]

    if row["CompensationCurrency"] in conversion_rates:

        USD = row["CompensationAmount"] * conversion_rates[row["CompensationCurrency"]]

        WANTED_CURR = USD / conversion_rates[WANTED]

        total.append((Country, USD, WANTED_CURR))
import math



# prepare to calculate average. 

# Add up and count occurences for each contry. 

dictionary_count ={}

dictionary_amount = {}

for row in total:

    amount = row[2]

    country = row[0]

    if country not in dictionary_count:

        dictionary_count[country] = 0

    if country not in dictionary_amount:

        dictionary_amount[country] = 0

    dictionary_amount[country] += amount

    dictionary_count[country] +=1 

    

# average for each country. Remove nans.

average_country = {}

for country, amount in dictionary_amount.items():

    if math.isnan(amount): continue

    average_country[country] = amount/ dictionary_count[country]

    

print(average_country)

print(len(average_country.keys()))

import operator

# red to green

fol = folium.Map(location=[0, 0], zoom_start=1)

sorted_avg = sorted(average_country.items(), key=operator.itemgetter(1))



hexes = iter(['#f00', '#fc0b00', '#fa1500', '#f71f00', '#f42900', '#f23200', '#ef3c00', '#ec4500', '#ea4e00', '#e75700', '#e55f00', '#e26800', '#df7000', '#d70', '#da7f00', '#d78700', '#d58e00', '#d29500', '#cf9c00', '#cda200', '#caa800', '#c7af00', '#c5b400', '#c2ba00', '#bfbf00', '#b5bd00', '#abba00', '#a1b800', '#97b500', '#8db200', '#84b000', '#7bad00', '#72aa00', '#69a800', '#60a500', '#58a200', '#50a000', '#489d00', '#409a00', '#399800', '#329500', '#2b9300', '#249000', '#1d8d00', '#178b00', '#180', '#0b8500', '#058300', '#008000'])

for country_, avg_amount in sorted_avg:

    # move other to antarctica

    if country_ == "Other":

        country_ = "Antarctica"

        

    x = country_placement[country_]

    country_ = country_.replace(" '", "")

    info = country_ + " "+WANTED+":" + str(round(avg_amount,2))

    thehex = next(hexes)

    # change the naming of the country to other, after we got the lat,lot for antarctica

    if country == "Antarctica": country = "Other"

    folium.CircleMarker([x[0], x[1]],

                    radius=3,

                    popup=info,

                    color=thehex,

                    fill=True,

                    fill_color=thehex,

                    opacity=1,

                    fill_opacity=1

                   ).add_to(fol)

fol

        