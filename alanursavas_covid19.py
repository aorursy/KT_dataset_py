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
deathrates = pd.read_csv("../input/coronavirus-covid19-mortality-rate-by-country/global_covid19_mortality_rates.csv")

usa_covid19_mortality_rates = pd.read_csv("../input/coronavirus-covid19-mortality-rate-by-country/usa_covid19_mortality_rates.csv")
deathrates.head()
deathrates["Deaths"]
deaths=[]

for deaths in deathrates["Deaths"]:

    print(deaths)
country={}

deathnum={}

country[deaths]=0

deathnum[deaths]=0

print(country)

print(deathnum)
for deaths in (deathrates["Deaths"]):

    country[deaths] += deathrates["Country"]

    deathnum[deaths] += deathrates["Deaths"]

print(country)

print(deathnum)