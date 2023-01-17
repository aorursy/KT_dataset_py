 # This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

data = pd.read_csv("../input/Airplane_Crashes_and_Fatalities_Since_1908.csv",usecols = [0,2,9,10])

data['years'] = [int(data.Date[i][-4:]) for i in range(0,len(data.Date))]

#print(data.years)

def take_years(year1,year2):

    new_data = np.array( [[data.years[i],data.Aboard[i].astype(int),data.Fatalities[i].astype(int)] for i in range (0,len(data.Date)) if (year1<data.years[i]<year2)])

    return(new_data)

ndata = take_years(2000,2009)









# Any results you write to the current directory are saved as output.
totalAb  =  np.sum(ndata[:,1])

totalFat = np.sum(ndata[:,2])

sizes =[totalAb-totalFat,totalFat]

labels = ["Yaşayan sayısı","Ölen sayısı"]

explode = [0.1,0]

fig1,ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')

plt.show()
totalcrashes = len(ndata)

sizes =[365000000-totalcrashes,totalcrashes]

labels = ["Kazasız gerçekleşen uçuş sayısı","Kaza Sayısı"]

explode = [0.1,0]

fig1,ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')

plt.show()
sizes = [totalAb-totalFat,totalFat]

enlabels = ["Alive","Fatalities"]

explode = [0.1,0]

fig1,ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=enlabels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')

plt.show()
sizes =[365000000-totalcrashes,totalcrashes]

enlabels = ["Number of flights without an accident","Number of accidents"]

explode = [0.1,0]

fig1,ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=enlabels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')

plt.show()