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



arrY = array.array('i', [0])  

arrX = array.array('i', [0])

arrZ = array.array('i', [0])



accumArrX = array.array('i', [0])  

accumArrY = array.array('i', [0])  

accumArrZ = array.array('i', [0])  





url = 'https://api.jsonbin.io/b/5e770612b325b3162e3c2e52/latest'

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}

result = requests.get(url, headers=headers)

#print(result.content.decode())







##parsing response

#r = urllib.request.urlopen(req).read()

#cont = json.loads(r.decode('utf-8'))



cont = json.loads(result.content.decode())

counter = 0

currentInfect = 1

dailyInfect = 0



currentProjectInf = 1

dailyProjectInf = 0



#print(cont)

#print("Infected:", string(cont['data']['TotalInfected']))





#users = json.loads(response.text)

for item in cont:

    if item['TotalInfected'] > currentInfect:

        dailyInfect = item['TotalInfected'] - currentInfect

    elif item['TotalInfected'] == currentInfect:

        dailyInfect = 0

        

    if int(item['ExpInc']) > currentProjectInf:

            dailyProjectInf = int(item['ExpInc']) - currentProjectInf

        #dailyProjectInf = 0



    counter = counter + 1

    arrY.append(dailyInfect)

    arrX.append(counter)

    arrZ.append(dailyProjectInf)

    currentInfect = item['TotalInfected']

    currentProjectInf = int(item['ExpInc'])

    #print(counter, "Total Infected:" , item['TotalInfected'], "Daily Infected:", arrY[counter])

    

    accumArrY.append(item['TotalInfected'])

    accumArrZ.append(int(item['ExpInc']))

    accumArrX.append(counter)



    

#print("Array item:", arrY[:])





# importing the required module 

import matplotlib.pyplot as plt 

  

# x axis values 

# x = [1,2,3] 

# corresponding y axis values 

# y = [2,4,1] 

  

# plotting the points  

# plt.plot(x, y) 



#i = 0

#for j in arr:

#    i+= 1

#    print (i, "---", j)

#    #plt.plot(i, j)

    

plt.plot(arrX, arrY, arrZ)

  

# naming the x axis 

plt.xlabel('Days (Starting from the first record by DDC at Jan 13, 2020)') 

# naming the y axis 

plt.ylabel('New Infections') 

  

# giving a title to my graph 

plt.title('Daily new Covid-19 infections in Thailand') 

  

# function to show the plot 

plt.show() 





plt.plot(accumArrX, accumArrY, accumArrZ)

  

# naming the x axis 

plt.xlabel('Days (Starting from the first record by DDC at Jan 13, 2020)') 

# naming the y axis 

plt.ylabel('Accumulated Infections') 

  

# giving a title to my graph 

plt.title('Accumulated Covid-19 infections in Thailand') 

  

# function to show the plot 

plt.show() 




