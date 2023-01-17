# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import csv

from matplotlib import pyplot

import matplotlib.dates as mdates

from dateutil import parser



def isfloat(value):

    try:

        float(value)

        return True

    except:

        return False



days = []

minTemps = []

maxTemps = []

events = []

header = None

f=open("../input/weather_madrid_LEMD_1997_2015.csv")

for row in csv.reader(f):

    if header == None:

        header = row

        print (header[0], header[1], header[3], header[21])



    if isfloat(row[1]):

        days.append(parser.parse(row[0]))

        minTemps.append(float(row[3]))

        maxTemps.append(float(row[1]))

    else:

        print (row[1])

            





minTemps = minTemps[0:len(days)]

maxTemps = maxTemps[0:len(days)]

    

pyplot.plot(days, minTemps)

pyplot.plot(days, maxTemps)

pyplot.title(' Min Temperature Madrid Airport')

pyplot.ylabel('Temperature')

pyplot.xlabel('Days')

pyplot.show()



        



# Any results you write to the current directory are saved as output.
            



days = days[0:365]

minTemps = minTemps[0:365]

maxTemps = maxTemps[0:365]

    

pyplot.plot(days, minTemps)

pyplot.plot(days, maxTemps)

pyplot.title(' Min Temperature Madrid Airport')

pyplot.ylabel('Min Temperature')

pyplot.xlabel('Days')

pyplot.show()



        



# Any results you write to the current directory are saved as output.