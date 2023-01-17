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

COVID19 = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv")

COVID19_open_line_list = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv")

covid_19_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

time_series_covid_19_confirmed = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

time_series_covid_19_deaths = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")

time_series_covid_19_recovered = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")
COVID19.head()
countryList = []

for country in COVID19["country"]:

    if country not in countryList:

        countryList.append(country)

print(countryList)
countryTotal = {}



for country in countryList:

    countryTotal[country] = 0

print(countryTotal)
for index in range(len(COVID19["id"])):

    for country in countryList:

        if country == COVID19["country"][index]:

            if COVID19["death"][index] == "1":

                countryTotal[country] += 1

print(countryTotal)
resultCountry = "temp"

resultDeath = 0

for countries in countryTotal.keys():

    if countryTotal[countries] > resultDeath:

        resultCountry = countries

        resultDeath = countryTotal[countries]

        

print(str(resultDeath) + " people died from Covid-19 in " + resultCountry)