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
covid19 = pd.read_csv("../input/corona-virus-report/covid_19_clean_complete.csv")
covid19.head()
covid19["Country/Region"]
covid19["Deaths"]
CountryList = []

for country in covid19["Country/Region"]:

    if country not in CountryList:

        CountryList.append(country)

    

print(CountryList)
DeathCount = {}

for country in CountryList:

    DeathCount[country] = 0

    

print(DeathCount)
a = 0

deathliest = 0

for idx, country in enumerate(covid19["Country/Region"]):

    DeathCount[country] += covid19["Deaths"][idx]

    if DeathCount[country] > a:

        a = DeathCount[country]

        deathliest = country

        

print("Highest death count belongs to " + deathliest + " with " + str(a))