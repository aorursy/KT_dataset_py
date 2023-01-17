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





import matplotlib.pyplot as plt## plotting used matlab

import seaborn as sns ##statitstics 



import plotly.graph_objects as go



df = pd.read_csv("../input/forest-fires-in-brazil/amazon.csv", encoding = 'latin1')

df.head()

df.tail()
df.isnull().sum()## checking if any column is null
df.month.unique()
plt.figure(figsize=(20,10))

sns.swarmplot(x = 'month', y = 'number',data = df)

plt.xticks(rotation = 90)

plt.show()

## The beeplot can see that June and August have had the maximum number of fires
plt.figure(figsize = (20,5))

sns.violinplot(x='month',y='number',data = df)

plt.show()

## Violin plot shows maximum density in which month
df.boxplot(column="number", by="year")

plt.xticks(rotation=75)

plt.show()
mnth = {'Janeiro': 'January', 'Fevereiro': 'February', 'Março': 'March', 'Abril': 'April', 'Maio': 'May',

          'Junho': 'June', 'Julho': 'July', 'Agosto': 'August', 'Setembro': 'September', 'Outubro': 'October',

          'Novembro': 'November', 'Dezembro': 'December'}

df['month']=df['month'].map(mnth)

## Changing the months  name to English
df.drop(columns = ['date','year'],axis=1)
df['Year'] = pd.DatetimeIndex(df['date']).year
df = df[['state','number','month','Year']]

df
years=list(df.Year.unique())

no_of_fires_each_year = []

for i in years:

    each = df.loc[df['Year'] == i].number.sum()

    no_of_fires_each_year.append(each)

fire_dict = {'Year':years, 'Total_Fires':no_of_fires_each_year}
time_plot_dataframe = pd.DataFrame(fire_dict)

time_plot_dataframe.head()

time_plot_dataframe

## I have sorted according to the total number of fires each  year that has happened
time_plot_1 = go.Figure(go.Scatter(x = time_plot_dataframe.Year, y = time_plot_dataframe.Total_Fires, mode = 'lines+markers', line = {'color':'red'}))

time_plot_1.update_layout(title='Brazil Fires from 1998-2017 Years',

                   xaxis_title='Year',

                   yaxis_title='Fires')

time_plot_1.show()

## Shows that the total number of fires keeps increasing each year

## This has been plotted with help of taken from other resources
