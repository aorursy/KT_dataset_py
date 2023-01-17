#%matplotlib notebook

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



plt.style.use('fivethirtyeight')
# Load data set

forest_fire = pd.read_csv('../input/forest-fires-in-brazil/amazon.csv', encoding='ISO-8859–1')

forest_fire.head()
forest_fire.info()

#forest_fire.describe()
# change date column to pandas date time

forest_fire.date = pd.to_datetime(forest_fire.date)

forest_fire.info()
# total states

print('Total states \n', forest_fire.state.unique())
# get state wise data for each year

df = forest_fire.groupby(['date', 'state'])['number'].sum().to_frame()

df.reset_index(inplace=True)

df
#get top 5 states based on number of fires

s = df.groupby('state')['number'].sum()



s.sort_values(ascending=False, inplace=True)



top5state = s[:5].index.values
# keep only top 5 states

df = df[df['state'].isin(top5state)]
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



# Plot each state graph

plt.figure(figsize=(20,8))



grouped = df.groupby('state')



for name, frame in grouped:

    plt.plot(frame.date, frame.number, label=name)



plt.xlabel('Year')

plt.ylabel('Number of fire')

plt.title('Top 5 states with most number of forest fire')

plt.legend(loc=1)

plt.show()
# Plot each state graph

plt.figure(figsize=(20,10))

#sns.set()



cp = sns.color_palette(['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00'])

bars = sns.barplot(x='date', y='number', hue='state', data=df, palette=cp)



# get current axis

ax = plt.gca()

# get current xtick labels

xticks = ax.get_xticks()

ax.set_xlabel('Year', fontsize=18)

ax.set_ylabel('Number of fire', fontsize=18)

ax.set_title('Top 5 states with most number of forest fire', fontsize=18)

ax.set_xticklabels(df.date.dt.strftime('%Y').unique());

ax.legend().set_title('States')

#legend = ax.legend()

#legend.texts[0].set_text("Whatever else")

# fire in 2017

data = df[df.date.dt.year == 2017]

data
plt.figure(figsize=(12,8))

#plt.pie(x=data.number, labels=data.state);

explode = (0, .1, 0, 0, 0)  

plt.pie(data.number, explode=explode, labels=data.state, autopct='%1.1f%%',

        shadow=True, startangle=90);

plt.title('Pie chart - Forest fire in 2017');
monthwisefire = forest_fire.groupby('month')['number'].mean()



## month mapping to date time. Year doesnt matter. We will use month to print it on plot

month_mapping = {

                 'Janeiro' : '1/1/2017', 'Fevereiro' : '1/2/2017', 'Março' : '1/3/2017',

                 'Abril' : '1/4/2017', 'Maio' : '1/5/2017', 'Junho' : '1/6/2017',

                 'Julho' : '1/7/2017', 'Agosto' : '1/8/2017', 'Setembro' : '1/9/2017',

                 'Outubro' : '1/10/2017', 'Novembro' : '1/11/2017', 'Dezembro' : '1/12/2017'

                 }



monthwisefire = monthwisefire.reset_index()

monthwisefire.month.replace(month_mapping, inplace=True)

monthwisefire.month = pd.to_datetime(monthwisefire.month, format='%d/%m/%Y')

monthwisefire.sort_values(by='month', inplace=True)

import matplotlib.dates as mdates

import datetime



plt.figure(figsize=(12, 8))

pal = sns.color_palette("Reds_r", len(monthwisefire))

rank = monthwisefire.number.argsort().argsort()

sns.barplot(monthwisefire.month.dt.strftime('%b'), monthwisefire.number, palette=np.array(pal[::-1])[rank], 

            linewidth=1, edgecolor=".2");



ax = plt.gca()

ax.set_xlabel('Months')

ax.set_ylabel('Average number of Fire')

ax.set_title('Month wise average number of fire');