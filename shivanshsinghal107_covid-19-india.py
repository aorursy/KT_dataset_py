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
df = pd.read_csv('/kaggle/input/covid19-india-dataset/covid_india.csv')

df.head()
df.drop(columns = ['Unnamed: 0', 'S. No.'], inplace = True)

df.head()
df.columns = ['State/UT', 'Confirmed', 'Recovered', 'Deaths']

df['Active'] = df['Confirmed'] - df['Deaths'] - df['Recovered']

df = df[['State/UT', 'Confirmed', 'Deaths', 'Recovered', 'Active']]

df.head()
# Ordered by Active cases

df_ordered = df.sort_values(by = 'Active', ascending = False).reset_index(drop = True)

df_ordered.style.background_gradient(cmap = 'Reds')
grp = df.groupby('State/UT')

active = grp['Active'].agg(np.sum)

deaths = grp['Deaths'].agg(np.sum)

recovered = grp['Recovered'].agg(np.sum)
import matplotlib.pyplot as plt
plt.figure(figsize = (15,10))

plt.xlabel('STATE/UNION TERRITORY')

plt.ylabel('NUMBER OF CASES')

plt.xticks(rotation = 90)

plt.title('COVID-19 IMAPCT ON INDIA (STATE-WISE) BY DATE 1/4/2020')



plt.plot(active, label = 'ACTIVE')

plt.plot(deaths, label = 'DEATHS')

plt.plot(recovered, label = 'RECOVERED')



plt.legend(loc = 'best', fontsize = 'large')

plt.grid()
plt.figure(figsize = (10, 10))



plt.xlabel('STATE/UNION TERRITORY')

plt.ylabel('NUMBER OF CASES ACTIVE')

plt.title('STATE-WISE COVID-19 REPORTS BY DATE 1/4/2020')

plt.xticks(rotation = 90)



plt.plot(active, label = 'ACTIVE')



plt.grid()
plt.figure(figsize = (10, 10))



plt.xlabel('STATE/UNION TERRITORY')

plt.ylabel('NUMBER OF CASES DEAD')

plt.title('STATE-WISE COVID-19 REPORTS BY DATE 1/4/2020')

plt.xticks(rotation = 90)



plt.plot(deaths, label = 'ACTIVE')



plt.grid()
plt.figure(figsize = (10, 10))



plt.xlabel('STATE/UNION TERRITORY')

plt.ylabel('NUMBER OF CASES RECOVERED')

plt.title('STATE-WISE COVID-19 REPORTS BY DATE 1/4/2020')

plt.xticks(rotation = 90)



plt.plot(recovered, label = 'ACTIVE')



plt.grid()
df_new = df_ordered[0:5]

plt.pie(df_new['Active'], labels = df_new['State/UT'], autopct='%1.1f%%')

plt.title('Active Cases')
df_new = df_ordered[0:5]

plt.pie(df_new['Deaths'], labels = df_new['State/UT'], autopct='%1.1f%%')

plt.title('Deaths Cases')
df_new = df_ordered[0:5]

plt.pie(df_new['Recovered'], labels = df_new['State/UT'], autopct='%1.1f%%')

plt.title('Recovered Cases')
overall = df[['Confirmed', 'Deaths', 'Recovered', 'Active']].sum().reset_index()

overall.style.background_gradient('Pastel1')