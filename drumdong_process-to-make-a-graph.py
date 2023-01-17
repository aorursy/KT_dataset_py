# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
fire_amazon = pd.read_csv('../input/forest-fires-in-brazil/amazon.csv', encoding='latin1')

fire_amazon.head()
fire_amazon.isnull().any() # There is not null
fire_amazon.dtypes
fire_amazon.describe
with sns.axes_style('darkgrid'):

  bars = sns.barplot(x='number',y='month',

                     data=fire_amazon.sort_values('number',ascending=False))

  bars.set_title('Bar plot')

plt.show() 
month_graph=fire_amazon['number'].groupby(fire_amazon['month']).sum().reset_index(name='number')
month_graph
with sns.axes_style('darkgrid'):

  bars = sns.barplot(x='number',y='month',

                     data=month_graph.sort_values('number',ascending=False))

  bars.set_title('Bar plot')
with sns.axes_style('darkgrid'):

  bars = sns.barplot(x='number',y='state',

                     data=fire_amazon.sort_values('number',ascending=False))

  bars.set_title('Bar plot')

plt.show()  
state_graph=fire_amazon['number'].groupby(fire_amazon['state']).sum().reset_index(name='number')

state_graph
with sns.axes_style('darkgrid'):

  bars = sns.barplot(x='number',y='state',

                     data=state_graph.sort_values('number',ascending=False))

  bars.set_title('Bar plot')

plt.show() 
total_number=fire_amazon['number'].groupby(fire_amazon['year']).sum().reset_index(name='total_number')

total_number
plt.figure(figsize=(10,4),facecolor='w')

with sns.axes_style('darkgrid'):

  lines = sns.lineplot(x='year',y='total_number',data=total_number)

  lines.set_title('Total Fire in Amazon')



plt.xticks(ticks=

           [1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017])  

plt.show()
# Watch top 5 number of fire in every state

fire_number_state=fire_amazon['number'].groupby(fire_amazon['state']).sum().reset_index()

fire_number_state.sort_values(by='number',ascending=False).head()
total_group_number=fire_amazon['number'].groupby([fire_amazon['year'],fire_amazon['state']]).sum().reset_index(name='total_number')

total_group_number
# total fire of top 5 state



plt.figure(figsize=(12,4),facecolor='w')

with sns.axes_style('darkgrid'):

  for kk in ['Mato Grosso','Paraiba','Sao Paulo','Rio','Bahia']:

    sns.lineplot(x='year',y='total_number',

                 label=kk,

                 data=total_group_number[total_group_number['state']==kk])

  

plt.xlabel("year")      #x축 이름 설정

plt.ylabel("total_fire")  #y축 이름 설정

plt.xticks(ticks=

           [1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017])  

plt.show()
fire_amazon.columns
fire_amazon.head()
new_dataframe=fire_amazon['number'].groupby([fire_amazon['month'],fire_amazon['state']]).sum().reset_index(name='number')

new_dataframe.head()
new_dataframe['month']
new_dataframe['month']= new_dataframe['month'].map({'Janeiro':1,'Fevereiro':2,'Março':3,'Abril':4,

                                                    'Maio':5,'Junho':6,'Julho':7,'Agosto':8,

                                                    'Setembro':9,'Outubro':10,'Novembro':11,'Dezembro':12})



new_dataframe
data_bargraph=new_dataframe['number'].groupby(new_dataframe['month']).max().reset_index()

data_bargraph
# insert columns -> state name



data_bargraph['state'] = 'non'

for i in range(0,12):

  #new_dataframe[new_dataframe['number']==data_bargraph['number'][i]]

  kk=new_dataframe[new_dataframe['number']==data_bargraph['number'][i]]['state'].reset_index()

  data_bargraph['state'][i]=kk['state'][0]



data_bargraph
#total highest fire number on each month of every years

plt.figure(figsize=(10,4),facecolor='w')

with sns.axes_style('darkgrid'):

  sns.barplot(x='month',

              y='number',

              hue='state',data=data_bargraph)



plt.title('total fire on each month')

plt.xlabel("month")      #x축 이름 설정

plt.ylabel("total_fire_number")  #y축 이름 설정

plt.show()