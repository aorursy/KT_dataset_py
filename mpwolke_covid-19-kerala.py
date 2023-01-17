#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSKR0ho0LMe7e4hH2udAWHCrKv8zt67g14vaSweJPisTVbii6aF',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/covid19-kerala/Covid19_Kerala.csv")

df.head().style.background_gradient(cmap='Reds')
df.dtypes
df = df.rename(columns={'Cumulative No.of Persons Under Home Isolation':'Isolated', 'Cumulative No.of Persons Hospitalized': 'Hospitalized', 'Tested Negative Till Date': 'Negative'})
#Let's visualise the evolution of results

evolution = df.groupby('Date').sum()[['ConfirmedIndianNational','Deaths','Cured', 'ConfirmedForeignNational']]

evolution['Deaths Rate'] = (evolution['Deaths'] / evolution['ConfirmedIndianNational']) * 100

evolution['Cured Rate'] = (evolution['Cured'] / evolution['ConfirmedIndianNational']) * 100

evolution['ConfirmedForeignNational Rate'] = (evolution['ConfirmedForeignNational'] / evolution['ConfirmedIndianNational']) * 100



evolution.head()
plt.figure(figsize=(20,7))

plt.plot(evolution['ConfirmedIndianNational'], label='ConfirmedIndianNational')

plt.plot(evolution['Deaths'], label='Deaths')

plt.plot(evolution['Cured'], label='Cured')

plt.legend()

#plt.grid()

plt.title('Evolution of COVID-19 Results')

plt.xticks(evolution.index,rotation=45)

plt.xlabel('Date')

plt.ylabel('Number of Patients')

plt.show()
#What about the evolution of ConfirmedIndianNational rate ?

plt.figure(figsize=(20,7))

plt.plot(evolution['ConfirmedIndianNational'], label='Confirmed Kerala Rate')

plt.legend()

#plt.grid()

plt.title('Evolution of COVID-19 Confirmed Kerala Rate')

plt.xticks(evolution.index,rotation=45)

plt.ylabel('Rate %')

plt.show()
#This is another way of visualizing the evolution: plotting the increase evolution (difference from day to day)

diff_evolution = evolution.diff().iloc[1:]

plt.figure(figsize=(20,7))

plt.plot(diff_evolution['ConfirmedIndianNational'], label='Confirmed Kerala Increase Evolution')

plt.legend()

plt.grid()

plt.title('Evolution of COVID-19 Kerala Patients')

plt.xticks(evolution.index,rotation=45)

plt.ylabel('Rate %')

plt.show()
print('Statistics About New Patients Evolutions')

#Here, "Cured Rate" represents the difference of this rate day to day

diff_evolution.describe()
#Last update

last_date = df['Date'].iloc[-1]

last_df = df[df['Date'] == last_date].groupby('Sno').sum()[['ConfirmedIndianNational', 'Deaths','Cured', 'ConfirmedForeignNational']]
last_df = last_df.sort_values(by='ConfirmedIndianNational', ascending=False)

print('Kerala Results by Sno')

#We can find different camp options here: https://matplotlib.org/3.2.0/tutorials/colors/colormaps.html

last_df.style.background_gradient(cmap='Reds')
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSfQHvOYwpaOqirslVhj-8l0DSwYcIwW0k6SrH0ZEeCygKryyVf',width=400,height=400)

#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRZfU3G-Vfrj_9V84mGdY-mL6aOI484VVcjvCC-MPy3Ok2fAdFK',width=400,height=400)