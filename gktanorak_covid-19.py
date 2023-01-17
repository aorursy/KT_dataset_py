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
from pandas import ExcelFile
from pandas import ExcelWriter
Corona = pd.read_excel('/kaggle/input/COVID-19.xlsx')
Corona.head()
Countrylist = []

for Country in Corona["Country, Other"]:

    Countrylist.append(Country)

print(Countrylist)
Countrycases = {}

Countrydeaths = {}



for Country in Countrylist:

    Countrycases[Country] = 0

    Countrydeaths[Country] = 0

    

print(Countrycases)

print("*******")

print(Countrydeaths)
for idx, countries in enumerate(Corona["Country, Other"]):

    for Country in countries.split(","):

        Countrycases[Country] = Corona["Total Cases"][idx]

        Countrydeaths[Country] = Corona["Total Deaths"][idx]

        

print(Countrycases)

print("**********")

print(Countrydeaths)
resultCountry = "temp"

resultdeaths = 0

for Country in Countrydeaths.keys():

    if Countrydeaths[Country] > resultdeaths:

        resultCountry = Country

        resultdeaths = Countrydeaths[Country]

        

print(resultCountry + " is the country that most people died from COVID-19, with a number of: " + str(resultdeaths) + ".")
DeathRatio = {}

for Country in Countrydeaths.keys():

    DeathRatio[Country] = Countrydeaths[Country] / Countrycases[Country]

print(DeathRatio)
ratiodeaths = 0

for Country in DeathRatio.keys():

    if DeathRatio[Country] > ratiodeaths:

        ratioCountry = Country

        ratiodeaths = DeathRatio[Country]

        

print(ratioCountry + " is the country that has the highest death ratio from COVID-19, with a raito of: " + str(ratiodeaths) + "." )