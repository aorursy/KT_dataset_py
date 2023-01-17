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
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# Encoding latin1 is working on this database

data = pd.read_csv('/kaggle/input/forest-fires-in-brazil/amazon.csv', encoding='latin1')
data.head()
data[50:55]
month_number={'Janeiro': 1, 'Fevereiro': 2, 'Março': 3, 'Abril': 4, 'Maio': 5,

               'Junho': 6, 'Julho': 7, 'Agosto': 8, 'Setembro': 9, 'Outubro': 10,

               'Novembro': 11, 'Dezembro': 12}



month_english={'Janeiro': 'January', 'Fevereiro': 'February', 'Março': 'March', 'Abril': 'April', 'Maio': 'May',

               'Junho': 'June', 'Julho': 'July', 'Agosto': 'August', 'Setembro': 'September', 'Outubro': 'October',

               'Novembro': 'November', 'Dezembro': 'December'}

data['Month'] = data.month.map(month_english)

data['Month No.'] = data.month.map(month_number)

data.drop(['date','month'],axis = 1, inplace = True)
# Just check if everything went right

data.tail()
# Wow no NAs

data.isna().sum()
fig = plt.figure(figsize = (20,4))

sns.set_style('white') # sets background style as white, other 4 options are whitegrid, dark, darkgrid,ticks 

sns.set_context('talk', font_scale = 0.9) # sets the scale/size of the chart. Other 4 options are paper << notebook << talk <<  poster

yearly_chart = sns.barplot(x = 'year', y = 'number', data = data, color = 'red')

yearly_chart.set(xlabel = 'Year', ylabel = 'Count of Fires', Title = 'Number of Fires in Brazil, 1998-2007')

sns.despine()
fig = plt.figure(figsize = (15,6))

sns.set_style('whitegrid') # see the horizontal lines

sns.set_context('poster', font_scale = 0.6) # see how the font has to be reduced and how all things appear bigger

monthly_chart = sns.barplot(x = 'Month No.',y='number', data = data, color = 'orange')

monthly_chart.set(title = "Total Number of Fires, distributed by Month of Year", xlabel = 'Month Number', ylabel = 'Count of Fires')

sns.despine(offset = 20, left = True)
# gathered avg temperature in Brazil from web and setting it for each month

avg_temp={1:-3.1, 2:-0.8, 3:4.9, 4:11.4, 5:17,

               6:22, 7:24, 8:22.8, 9:19.1, 10:12.7,

               11:5.9,12:0.3}

data['Temp'] = data['Month No.'].map(avg_temp)
fig = plt.figure(figsize = (15,5))

sns.set_style('white')

sns.set_context('notebook', font_scale = 1.2)

month_chart = sns.lineplot(x = 'Month No.', y = 'number', color = 'orange',data = data, legend = 'full')

ax2 = month_chart.twinx()

sns.lineplot(x = 'Month No.', y = 'Temp',ax = ax2, color = 'red',  data = data,  legend = 'full')

month_chart.set(title = 'Relation between the Temperature and Number of Fires in a Month', xlabel = 'Month', ylabel = 'Count of Fires')

sns.despine(left = True)
fig = plt.figure(figsize = (15,8))

sns.set_style('dark')

sns.set_context('talk', font_scale = 0.9)

year_month_matrix = sns.heatmap(data.pivot_table(index = 'Month No.', columns = 'year',values = 'number',aggfunc='sum'), cmap = 'Reds')

year_month_matrix.set(title = 'Fire Matrix - Year vs. Month')

sns.despine()
# Lets make a tree map

import squarify

fig = plt.figure(figsize = (20,10))

sns.set_style('dark')

sns.set_context('talk', font_scale = 0.7)

states_tree_chart = squarify.plot(sizes=a['Pct Fires'], label=a['state'],color = 'red', alpha=0.8, linewidth = 5)

states_tree_chart.set(title = 'Treemap Showing Shareof Different States in Total Fires')