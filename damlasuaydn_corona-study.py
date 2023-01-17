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
print("Important note: This study is based on an old dataset.")
cov = pd.read_csv("../input/coronavirus-2019ncov/covid-19-all.csv")

cov.head()
#Handling Missing Values (NaN)

#Source: https://machinelearningmastery.com/handle-missing-data-python/

print(cov.isnull().sum())
cov.dropna(inplace=True)

print(cov.isnull().sum())

print("Shape:")

print(cov.shape)
CountryList = []

for i in cov["Country/Region"]:

    if i not in CountryList:

        CountryList.append(i)

print(CountryList)
CntryTotalCase = {} #Because my intention was to calculate death rate

CntryTotalDeath = {}

CntryDeathRate



for j in CountryList:

    CntryTotalCase[j] = 0

    CntryTotalDeath[j] = 0

    CntryDeathRate[j] = 0

       

print(CntryTotalCase)

print("*****")

print(CntryTotalDeath)
for idx, k in enumerate(cov["Country/Region"]):

    for k in CountryList:

        CntryTotalCase[k] += cov["Confirmed"][idx]

        

### BURDA ERROR ALDIM VE İLERLEYEMEDİM =((((

print(CntryTotalCase)
for idx, k in enumerate(cov["Country/Region"]):

    CntryTotalDeath[k] += cov["Confirmed"][idx]

print(CntryTotalDeath)
for l in CntryTotalCase.keys():

    CntryDeathRate[l] = (CntryTotalDeath[l] * 100) / CntryTotalCase[l]
THEcountry = "blabla"

THEdeathRate = 0

for m in CntryDeathRate.keys():

    if CntryDeathRate[m] > THEdeathRate:

        THEcountry = m

        THEdeathRate = CntryDeathRate[m]

        

print(THEcountry + "has the highest death rate due to COVID-19, which is: " + str(round(THEdeathRate,2)) + "!!!")