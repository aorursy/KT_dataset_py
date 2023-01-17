#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRJDhgNbIG2OsPgp2aMZP6F7CelGdA9DQn831l61y2UkFwYTFNh',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time

from datetime import datetime, timedelta

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_excel("../input/corona-virus-pakistan-dataset-2020/COVID_FINAL_DATA.xlsx")
df.head().style.background_gradient(cmap='Greens')
df.dtypes
# List of unique Date

dates = df['Date'].unique()

dates
# Sub-dataframe: day

day = df[df['Date'] == '15-Mar-2020']



# Sub-dataframe: subdf1

subdf = day.groupby(['Region'

                      ]).sum().sort_values(by='Cumulative',

                                           ascending=False).copy()



subdf = subdf.iloc[:10]



# Initialize the matplotlib figure

f, ax = plt.subplots(figsize=(15,6))



#plot the confirmed cases

sns.set_color_codes('pastel')

sns.barplot(x='Cumulative',

            y=subdf.index,

            data=subdf,

            label='Cumulative',

            color='g'

           )



# plot the cases that lead to death

sns.set_color_codes('pastel')

sns.barplot(x='Expired',

            y=subdf.index,

            data=subdf,

            label='Expired',

            color='r'

           )



# plot the cases that lead to recovery

sns.set_color_codes('pastel')

sns.barplot(x='Discharged',

            y=subdf.index,

            data=subdf,

            label='Discharged',

            color='b'

           )



# Add a legend and informative axis label

ax.legend(ncol=3, loc='lower right', frameon=True)

ax.set(xlabel='COVID-19 Patients')

sns.despine(left=True, bottom=True)
#Let's visualise the evolution of results

evolution = df.groupby('Date').sum()[['Cumulative','Expired','Discharged']]

evolution['Expiration Rate'] = (evolution['Expired'] / evolution['Cumulative']) * 100

evolution['Discharging Rate'] = (evolution['Discharged'] / evolution['Cumulative']) * 100

evolution.head()
plt.figure(figsize=(20,7))

plt.plot(evolution['Cumulative'], label='Cumulative')

plt.plot(evolution['Expired'], label='Expired')

plt.plot(evolution['Discharged'], label='Discharged')

plt.legend()

#plt.grid()

plt.title('Evolution of COVID-19 Results')

plt.xticks(evolution.index,rotation=45)

plt.xlabel('Date')

plt.ylabel('Number of Patients')

plt.show()
#What about the evolution of Cumulative rate ?

plt.figure(figsize=(20,7))

plt.plot(evolution['Cumulative'], label='Cumulative Rate')

plt.legend()

#plt.grid()

plt.title('Evolution of COVID-19 Cumulative Rate')

plt.xticks(evolution.index,rotation=45)

plt.ylabel('Rate %')

plt.show()
#This is another way of visualizing the evolution: plotting the increase evolution (difference from day to day)

diff_evolution = evolution.diff().iloc[1:]

plt.figure(figsize=(20,7))

plt.plot(diff_evolution['Cumulative'], label='Cumulative Increase Evolution')

plt.legend()

plt.grid()

plt.title('Evolution of COVID-19 Cumulative Patients')

plt.xticks(evolution.index,rotation=45)

plt.ylabel('Rate %')

plt.show()
diff_evolution = evolution.diff().iloc[1:]

plt.figure(figsize=(20,7))

plt.plot(diff_evolution['Discharged'], label='Discharged Increase Evolution')

plt.legend()

plt.grid()

plt.title('Evolution of COVID-19 New Discharged Patients')

plt.xticks(evolution.index,rotation=45)

plt.ylabel('Rate %')

plt.show()
print('Statistics About New Patients Evolutions')

#Here, "Discharged Rate" represents the difference of this rate day to day

diff_evolution.describe()
#Last update

last_date = df['Date'].iloc[-1]

last_df = df[df['Date'] == last_date].groupby('Region').sum()[['Cumulative', 'Expired','Discharged']]
last_df = last_df.sort_values(by='Cumulative', ascending=False)

print('Pakistan Results by Region')

#We can find different camp options here: https://matplotlib.org/3.2.0/tutorials/colors/colormaps.html

last_df.style.background_gradient(cmap='Greens')
#Cumulative Partition

c = last_df

conf_max = c['Cumulative'][:4] 

conf_max.loc['Other'] = c['Cumulative'][4:].sum()

plt.figure(figsize=(11,6))

plt.pie(conf_max, labels=conf_max.index, autopct='%1.1f%%', explode=(0,0,0,0,1), shadow=True)

plt.title('COVID-19 Cumulative Patients Partition')

plt.show()
import plotly.express as px
#NB: 8 is the index of "others" in last_df region, we want to show Regions by names so we'll drop it.

bar_df_conf = last_df.reset_index().drop(8)[1:11] 

px.bar(bar_df_conf.sort_values('Cumulative', ascending=True),

       y="Cumulative", 

       x="Region", 

       title="COVID-19 Affected Regions",

       hover_data=['Discharged'],

       color='Discharged',

       orientation='v')
plt.figure(figsize=(15,5))

fctgrid = sns.FacetGrid(data=df,

                        col='Region',

                        hue='Region',

                        col_wrap=4,

                        sharey=False)

fctgrid.map(plt.plot, 'Date', 'Cumulative')

fctgrid.set(xticks=df['Date'].unique()[10::30])

#fctgrid.set_xticklabels(rotation=90)

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSyyxchiaodTgKTl875dsFIFjxpHlpnSWES81GvJL3-xJ9vsvNc',width=400,height=400)