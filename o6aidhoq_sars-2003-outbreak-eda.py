from IPython.display import Image

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

from datetime import datetime



import os

print(os.listdir("../input"))
Image("../input/sars-img/SARS.jpg")
sars = pd.read_csv('../input/sars-outbreak-2003-complete-dataset/sars_2003_complete_dataset_clean.csv')

sars = sars.rename(columns={'Cumulative number of case(s)': 'total_cases', 'Number of deaths': 'deaths', 

                            'Number recovered':'recovered'}) 

#changing column names cuz too many spaces and I cant be bothered to find out how to do it with spaces, 

#semi colons etc.

for col in sars.columns: 

    print(col) 
count_row = sars.shape[0]

print(str(count_row) + " instances of the dataset is available")
sars.head()
grouped = sars.groupby(['Date']).sum()

grouped['total_cases'].plot(color = 'blue')

grouped['deaths'].plot(color = 'red')

grouped['recovered'].plot(color = 'green')

#also create a plot with cases, deaths and recovered by top 5 countries
x = sars[['Country', 'total_cases', 'deaths', 'recovered']]

y = x.set_index('Country')

z = y.groupby('Country').sum().sort_values("total_cases", ascending=False).head(7)

z.plot.bar(stacked = True)
sars['Month'] = pd.DatetimeIndex(sars['Date']).month

sars = sars.set_index('Date') 

#I have learned that using date as an index can make plotting easy and visually pleasing

sars.index = pd.to_datetime(sars.index)

sars.index
x = sars[['Month', 'total_cases', 'deaths', 'recovered']]

y = x.set_index('Month')

z = y.groupby('Month').sum()

z.plot.bar(stacked = True)
grouped_month = sars.groupby(['Month']).sum()

grouped_month
xx = np.arange(5)

fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

ax.bar(xx + 0.00, grouped_month['total_cases'], color = 'b')

ax.bar(xx + 0.25, grouped_month['recovered'], color = 'g')

ax.bar(xx + 0.50, grouped_month['deaths'], color = 'r')