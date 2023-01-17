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
covid_19_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")
covid_19_data.head()
covid_19_data.columns[-1]
death_dict = {}

for i in covid_19_data["Country/Region"]:

    death_dict[i] = 0

for count,city in enumerate(covid_19_data["Country/Region"].values.tolist()):

    death_dict[city] = death_dict[city] + int(covid_19_data[covid_19_data.columns[-1]][count])



import operator

sorted_death_dict = sorted(death_dict.items(), key=operator.itemgetter(1))    



death_cities = []

death_count = []



for element in sorted_death_dict:

    death_cities.append(element[0])

    death_count.append(element[1])

    

print(sorted_death_dict)    

print(death_cities)

print(death_count)
import matplotlib.pyplot as plt



plt.rcdefaults()

fig, ax = plt.subplots(figsize=(10,8))



# Example data

countries = death_cities

y_pos = np.arange(len(countries))

death = list(death_count)



ax.tick_params(axis='y', which='major', labelsize=3)

ax.tick_params(axis='y', which='minor', labelsize=3)



ax.barh(y_pos, death, align='center')

ax.set_yticks(y_pos)

ax.set_yticklabels(countries)

#ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('Confirmed')

ax.set_title('Confirmed patient count')



plt.show()
print(confirmed_dict)