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


#Import Required library

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.dates as mdates

import matplotlib.cbook as cbook
#read Amazon Forest Fire Library

forest_fire = pd.read_csv('/kaggle/input/forest-fires-in-brazil/amazon.csv',encoding = "ISO-8859-1")
#Display 5 data for observation 

forest_fire.head(5)
#Description of numeric attriburesattributes

forest_fire['number'].describe()
#check if missing values

forest_fire.isna().values.any()
#Forest fire frequency per state

fire_month = forest_fire[['state','number']].groupby('state')['number'].sum()

sns.set(style="darkgrid", rc={'figure.figsize':(20,10)})

sns.barplot(fire_month.index, fire_month.values, alpha=0.9)

plt.title('Frequency Distribution of Forest Fire per state')

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('State')

plt.xticks(fontsize=14, rotation=90)

plt.show()
#Forest Fire legal Amazon state

amazon_state = ['Acre','Amapa','Amazonas','Maranhao','Mato Grosso',

                                                 'Pará','Rondonia','Roraima',

                                                 'Tocantins']

forest_fire_amazon = forest_fire[forest_fire['state'].isin(amazon_state)]

#forest fire for amazon state

#Forest fire frequency per state

fire_month = forest_fire_amazon[['state','number']].groupby('state')['number'].sum()

sns.set(style="darkgrid", rc={'figure.figsize':(20,10)})

sns.barplot(fire_month.index, fire_month.values, alpha=0.9)

plt.title('Frequency Distribution of Forest Fire per legal Amazon state')

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('State')

plt.xticks(fontsize=14, rotation=90)

plt.show()
#Forest fire frequency for particular year and month in all state

ff_year_month_num = forest_fire_amazon[['year','month','number']].groupby(['year','month'])['number'].sum().unstack()

fig, ax = plt.subplots(figsize=(20,10))

#to avoid repeative color

colors = sns.color_palette("hls", 12)

ax.set_prop_cycle('color', colors)

ax.xaxis.set_ticks(np.arange(min(forest_fire_amazon['year']), max(forest_fire_amazon['year'] +1), 1))

ff_year_month_num.plot(ax=ax)

#ff_year_month_num['numbner']#[ff_year_month_num['number'==ff_year_month_num.min()]]