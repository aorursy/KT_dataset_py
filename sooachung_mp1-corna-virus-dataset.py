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

import pandas_datareader.data as web

import matplotlib.pyplot as plt

import datetime as dt
import pandas as pd

data = pd.read_csv("../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv")

data_conf = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_2019_ncov_confirmed.csv")

data_deaths = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_2019_ncov_deaths.csv")

data_recov = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_2019_ncov_recovered.csv")
data.head()
data_conf.head()
data_recov.head()
data[data.Country == 'Mainland China']
data.info()
#Total numbers of countires that are effected



countries = data['Country'].unique().tolist()

print(countries)

print("\nTotal countries affected by virus: ",len(countries))
#Combining China and Mainland China cases



data['Country'].replace({'Mainland China':'China'},inplace=True)

countries = data['Country'].unique().tolist()

print(countries)

print("\nTotal countries affected by virus: ",len(countries))
#Mainland China

China = data_latest[data_latest['Country']=='China']

China
data_latest.groupby(['Country','Province/State']).sum()
f, ax = plt.subplots(figsize=(12, 8))



sns.set_color_codes("pastel")

sns.barplot(x="Confirmed", y="Province/State", data=China[1:],

            label="Confirmed", color="r")



sns.set_color_codes("muted")

sns.barplot(x="Recovered", y="Province/State", data=China[1:],

            label="Recovered", color="g")



# Add a legend and informative axis label

ax.legend(ncol=2, loc="lower right", frameon=True)

ax.set(xlim=(0, 400), ylabel="",

       xlabel="Stats")

sns.despine(left=True, bottom=True)
#To see the comparison between confirmed and death cases.

x = data[['Deaths']]

y = data[['Confirmed']]

data.plot(kind = 'scatter', x ='Deaths', y = 'Confirmed')
data.plot(y='Confirmed')
data[['Confirmed','Deaths','Recovered']].plot()