import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

data = pd.read_csv('/kaggle/input/videogames-sales-dataset/XboxOne_GameSales.csv',encoding = 'windows-1252')
data.columns
data.info()
data.head(15)
data.tail(15)
data.Publisher.unique()
data.Genre.unique()
plt.subplots(figsize = (15,8))

plt.hist(data.Year,bins = 30)

plt.show()
rockstar = data[data.Publisher == 'Rockstar Games']

activision = data[data.Publisher == 'Activision']

mic = data[data.Publisher == 'Microsoft Studios']

bethesda = data[data.Publisher == 'Bethesda Softworks']
rockstar.head()
plt.subplots(figsize = (10,5))

plt.hist(rockstar.Game)

plt.show()
plt.scatter(rockstar.Year,rockstar['Global'],color = 'red',alpha = 0.5)

plt.xlabel('Year')

plt.ylabel('Global')

plt.show()
activision.head(20)
activision.tail()
plt.subplots(figsize = (10,5))

plt.hist(activision.Game,bins = 40)

plt.xticks(rotation = 90)

plt.show()
plt.subplots(figsize = (10,5))

plt.scatter(activision.Year,activision['North America'],color = 'red',alpha = 0.5)

plt.xlabel('Year')

plt.ylabel('North America Sale')

plt.show()
plt.subplots(figsize = (10,5))

plt.scatter(activision.Year,activision['Europe'],color = 'red',alpha = 0.5)

plt.xlabel('Year')

plt.ylabel('Europe Sale')

plt.show()
plt.subplots(figsize = (10,5))

plt.scatter(activision.Year,activision.Japan,color = 'red',alpha = 0.5)

plt.xlabel('Year')

plt.ylabel('Japan Sale')

plt.show()
plt.subplots(figsize = (10,5))

plt.scatter(activision.Year,activision.Global,color = 'red',alpha = 0.5)

plt.xlabel('Year')

plt.ylabel('Global Sale')

plt.show()
mic.head(10)
mic.tail(10)
plt.subplots(figsize = (10,5))

plt.hist(mic.Game,bins = 30)

plt.xticks(rotation = 90)

plt.show()
plt.subplots(figsize = (10,5))

plt.scatter(mic.Year,mic['North America'],alpha = 0.5,color = 'red')

plt.xlabel('Year')

plt.ylabel('North America Sale')

plt.show()
plt.subplots(figsize = (10,5))

plt.scatter(mic.Year,mic['Europe'],alpha = 0.5,color = 'red')

plt.xlabel('Year')

plt.ylabel('Europe Sale')

plt.show()
plt.subplots(figsize = (10,5))

plt.scatter(mic.Year,mic['Japan'],alpha = 0.5,color = 'red')

plt.xlabel('Year')

plt.ylabel('Japan Sale')

plt.show()
plt.subplots(figsize = (10,5))

plt.scatter(mic.Year,mic['Global'],alpha = 0.5,color = 'red')

plt.xlabel('Year')

plt.ylabel('Global Sale')

plt.show()
bethesda.head(10)
bethesda.tail(10)
plt.subplots(figsize = (10,5))

plt.hist(bethesda.Game,bins = 6)

plt.xticks(rotation = 90)

plt.show()
plt.subplots(figsize = (10,5))

plt.scatter(bethesda.Year,bethesda['North America'],alpha = 0.5,color = 'red')

plt.xlabel('Year')

plt.ylabel('North America Sale')

plt.show()
plt.subplots(figsize = (10,5))

plt.scatter(bethesda.Year,bethesda['Europe'],alpha = 0.5,color = 'red')

plt.xlabel('Year')

plt.ylabel('Europe Sale')

plt.show()
plt.subplots(figsize = (10,5))

plt.scatter(bethesda.Year,bethesda['Japan'],alpha = 0.5,color = 'red')

plt.xlabel('Year')

plt.ylabel('Japan Sale')

plt.show()
plt.subplots(figsize = (10,5))

plt.scatter(bethesda.Year,bethesda['Global'],alpha = 0.5,color = 'red')

plt.xlabel('Year')

plt.ylabel('Global Sale')

plt.show()
plt.subplots(figsize = (10,5))

plt.plot(rockstar.Global,rockstar['North America'],color = 'red',label = 'North America')

plt.plot(rockstar.Global,rockstar['Europe'],color = 'green',label = 'Europe')

plt.plot(rockstar.Global,rockstar['Japan'],color = 'blue',label = 'Japan')

plt.legend()

plt.xlabel('Global Sale')

plt.show()
plt.subplots(figsize = (10,5))

plt.plot(activision.Global,activision['North America'],color = 'red',label = 'North America')

plt.plot(activision.Global,activision['Europe'],color = 'green',label = 'Europe')

plt.plot(activision.Global,activision['Japan'],color = 'blue',label = 'Japan')

plt.legend()

plt.xlabel('Global Sale')

plt.show()
plt.subplots(figsize = (10,5))

plt.plot(mic.Global,mic['North America'],color = 'red',label = 'North America')

plt.plot(mic.Global,mic['Europe'],color = 'green',label = 'Europe')

plt.plot(mic.Global,mic['Japan'],color = 'blue',label = 'Japan')

plt.legend()

plt.xlabel('Global Sale')

plt.show()
plt.subplots(figsize = (10,5))

plt.plot(bethesda.Global,bethesda['North America'],color = 'red',label = 'North America')

plt.plot(bethesda.Global,bethesda['Europe'],color = 'green',label = 'Europe')

plt.plot(bethesda.Global,bethesda['Japan'],color = 'blue',label = 'Japan')

plt.legend()

plt.xlabel('Global Sale')

plt.show()