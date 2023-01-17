import numpy as np 

import pandas as pd 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        path = os.path.join(dirname, filename)

with open(path) as file: 

    chat = file.read().split("\n") # Splits the txt file to an array of lines
#Ensuring that every line is a new message



lines = []

for i in chat:

    try:

        if i[2]=='/' and i[5]=='/' and i[10]==',':

            lines.append(i)

        else:

            lines[-1] += i

    except IndexError:

        pass

        #Blank new lines are discarded

lines[20:40]
#Changeing from this form: (+countryCode xx xxx xxxx) to this form: (countrycodexxxxxxxxx)

for i, line in enumerate(lines):

    try:

        index = line.index("+")

        lines[i] = line[:index] + line[index:index+12].replace(" ", "") + line[index+12:]

    except ValueError:

        pass

        #What passes through here are lines with people's names rather than numbers,usually gotten from whatsapp

lines[20:40]
#Getting rid of non text messages lines

lines = list(filter(lambda lin: lin.count(":")!=1,lines)) 

lines = list(filter(lambda lin: not ("<Media omitted>" in lin),lines))

lines[:10]
#Splitting lines into three different sections, dates, numbers, messages

dates = list(map(lambda line: line[:line.find("m")+1], lines)) #Takes everything before m in PM\AM

numbers = list(map(lambda line: line[line.find("-")+2 : line.replace(":", "X", 1).find(":")], lines)) #Takes everything between - and the second :

messages = list(map(lambda line: line[-(len(line) - line.replace(":", "X", 1).find(":"))+2 : ], lines)) #Takes everything after the second :
#Creating the DataFrame

data = pd.DataFrame(list(zip(numbers, messages, dates)), columns=['Number', 'Message', 'Time'])

pd.set_option('display.max_rows', 5000) #Restricts rows view to 5000 rows, you dont want to over clutter your notebook

data['Time'] = pd.to_datetime(data['Time'], format='%d/%m/%Y, %I:%M %p')




data.groupby('Message').Message.count().sort_values()[-100:]
data.groupby('Number').Message.count().sort_values()
import matplotlib.pyplot as plt

from matplotlib import style

import matplotlib.dates as mdates

import datetime

graph = []

people = []



numberOfPeople = int(input())

for loop in range(numberOfPeople):

    people.append(input())



fig, ax = plt.subplots(len(people))

fig.set_size_inches(15, 10)

style.use('seaborn-pastel')

fig.tight_layout()



if numberOfPeople == 1:

    ax = [ax]

    

for i, axis in enumerate(ax):

    axis.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    axis.xaxis.set_major_locator(plt.LinearLocator())

    axis.set_title(people[i])

    axis.set_xlim((data.Time.values[0], data.Time.values[-1]))

    

for person in people:

    if person == 'Everyone':

        graph.append(data.Time.values)

    else:

        graph.append(data[data['Number']==person].Time.values)



for i, axis in enumerate(ax):

    axis.hist(graph[i], 70, color='c', label='Dates Histogram', histtype='barstacked')

    

plt.show()
graph = []

people = []



numberOfPeople = int(input())

for loop in range(numberOfPeople):

    people.append(input())

    

fig, ax = plt.subplots(len(people))

fig.set_size_inches(15, 10)

style.use('seaborn-pastel')

fig.tight_layout()



if numberOfPeople == 1:

    ax = [ax]





for i, axis in enumerate(ax):

    axis.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    axis.xaxis.set_major_locator(plt.LinearLocator())

    axis.set_title(people[i])

    axis.set_xlim((730120, 730121))

    axis.axes.get_yaxis().set_visible(False)



for person in people:

    if person == 'Everyone':

        timeOnly = data.Time.map(lambda t: t.replace(year=2000, month=1, day=1)).values

        graph.append(timeOnly)

    else:

        timeOnly = data[data['Number']==person].Time.map(lambda t: t.replace(year=2000, month=1, day=1)).values

        graph.append(timeOnly)



for i, axis in enumerate(ax):

    axis.hist(graph[i], 85, color='g', label='Times Histogram', density=True)

    

plt.show()


