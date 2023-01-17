# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import urllib.request

import json

import requests

import array 

import math

import matplotlib.pyplot as plt 





arrY = array.array('i', [0])  

arrX = array.array('i', [0])

arrZ = array.array('i', [0])



accumArrX = array.array('i', [0])  

accumArrY = array.array('i', [0])  

accumArrZ = array.array('i', [0])  



# JSON Retreiving see https://covid19.ddc.moph.go.th/th/api

url = 'https://covid19.th-stat.com/api/open/timeline'

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}

result = requests.get(url, headers=headers)



url2 = 'https://covid19.th-stat.com/api/open/today'

headers2 = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}

result2 = requests.get(url2, headers=headers2)

dailyToday = json.loads(result2.content.decode())

safetyMargin = 10

logitMax = dailyToday['Confirmed'] + safetyMargin





# Matrix loading & logistic model calculation

cont = json.loads(result.content.decode())

counter = 0

currentInfect = 1

dailyInfect = 0



currentProjected = 1

dailyProjected = 0



logitProjected = 0



for item in cont['Data']:

    if item['Confirmed'] > currentInfect:

        dailyInfect = item['Confirmed'] - currentInfect

    elif item['Confirmed'] == currentInfect:

        dailyInfect = 0



    #print(item['Confirmed'])     



    counter = counter + 1

    

    #calculate logistic function

    

    logitProjected = logitMax / (1+(math.exp(0.158*(counter-90)*-1)))

    if int(logitProjected) > currentProjected:

        dailyProjected = int(logitProjected) - currentProjected

 

    currentProjected = logitProjected

    currentInfect = item['Confirmed']



    arrX.append(counter)

    arrY.append(dailyInfect)

    arrZ.append(int(dailyProjected))



    accumArrX.append(counter)

    accumArrY.append(item['Confirmed'])

    accumArrZ.append(int(logitProjected))

    

    



    

# Visualization

  

    

plt.plot(arrX, arrY, arrZ)



# naming the x axis 

plt.xlabel('Days (Starting from the first record by DDC at Jan 1, 2020)') 

# naming the y axis 

plt.ylabel('New Infections') 

  

# giving a title to my graph 

plt.title('Daily new Covid-19 infections in Thailand') 

  

# function to show the plot 

plt.show() 



plt.plot(accumArrX, accumArrY, accumArrZ)

  

# naming the x axis 

plt.xlabel('Days (Starting from the first record by DDC at Jan 1, 2020)') 

# naming the y axis 

plt.ylabel('Accumulated Infections') 

  

# giving a title to my graph 

plt.title('Accumulated Covid-19 infections in Thailand') 

  

# function to show the plot 

plt.show() 



print("L = ", logitMax, "or Logistic curve's maximum value.")
