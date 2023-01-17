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
import pandas as pd

covid19 = pd.read_csv("../input/covid19-coronavirus/2019_nCoV_data.csv")
covid19.head()
covid19["Deaths"]
covid19["Country"]
countriesList = []

for country in covid19["Country"]:

    for unique in country.split(","):

        if unique not in countriesList:

            countriesList.append(unique)

print(countriesList)
countryTotal = {}    #Total rating

countryCount = {}    #Genre file count



for country in countriesList:

    countryTotal[country] = 0

    countryCount[country] = 0

print(countryTotal)

print(countryCount)
for idx, countries in enumerate(covid19["Country"]):

    

    for country in countries.split(","):

        countryTotal[country] += covid19["Deaths"][idx] 

        countryCount[country] += 1

print(countryTotal)

print("******")

print(countryCount)
countryAverage = {}



for country in countryTotal.keys():

    countryAverage[country]= countryTotal[country] / countryCount[country]

print(countryAverage)
resultCountry = "temp"

resultDeaths = 0

for country in countryAverage.keys():

    if countryAverage[country] > resultDeaths:

        resultCountry = country

        resultDeaths = countryAverage[country]

        

print( resultCountry  + " "  +  "HAS THE MOST DEATHS WHICH IS" + " "  +   str(round(resultDeaths,2)) + " "  +  "BECAUSE OF COVID19")