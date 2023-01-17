### importing everything that is necessary for data parsing

import requests

from bs4 import BeautifulSoup

import json

import pandas as pd
## getting the raw html from the web site

url = "https://www.worldometers.info/coronavirus/"

r = requests.get(url)

soup = BeautifulSoup(r.content, 'lxml')

ans2= soup.findAll('div', attrs={'class':'main_table_countries_div'})
data = []

table = soup.findAll('table', attrs={'id':'main_table_countries_today'})

for row in table:

    cells = row.findAll('td')

    for cell in cells:

        data.append(cell)
dd=[]

for x,y in enumerate(data):

    dd.append((list(y)))

    

if ["Bangladesh"] in dd:

    index = dd.index(["Bangladesh"])
country = list(data[index])[0]

total_case = int(list(data[index+1])[0])

new_case = int(list(data[index+2])[0])

total_death = int(list(data[index+3])[0])

try:

    new_death = int(list(data[index+4])[0])

except:

    new_death = 0 

    

total_recovered = int(list(data[index+5])[0])

active_case = int(list(data[index+6])[0])

try:

     critical_case = int(list(data[index+7][0]))

except:

     critical_case = 0



df = {"country" : country,

"total_case" : total_case,

"new_case" : new_case,

"total_death" : total_death,

"new_death" : new_death,

"total_recovered" : total_recovered,

"active_case" : active_case,

"critical_case" : critical_case}
df
df_new = {"total_case" : total_case,

"new_case" : new_case,

"total_death" : total_death,

"new_death" : new_death,

"total_recovered" : total_recovered,

"active_case" : active_case,

"critical_case" : critical_case}
df_new
from pylab import rcParams

rcParams['figure.figsize'] = 10, 20
import matplotlib.pyplot as plt



# Pie chart, where the slices will be ordered and plotted counter-clockwise:

labels = df_new.keys()



sizes = df_new.values()

explode = (0, 0.1, 0, 0,.1,.1,.1)  # only "explode" the 2nd slice (i.e. 'Hogs')



fig1, ax1 = plt.subplots()

ax1.pie(sizes,explode=explode, labels=labels,

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
import matplotlib.pyplot as plt



# Pie chart, where the slices will be ordered and plotted counter-clockwise:

labels = df_new.keys()

sizes = df_new.values()

explode = (.1, .1, .1, .1,.1,.1,.1)  # only "explode" the 2nd slice (i.e. 'Hogs')



fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
plt.bar(df_new.keys(),df_new.values())