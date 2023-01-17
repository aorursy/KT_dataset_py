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
data = pd.read_csv('../input/world-happiness-report-2019.csv')
data.columns
data.columns = ['country', 'ladder', 'ladder_sd', 'positive_affect', 'negative_affect', 

              'social_support', 'freedom', 'corruption', 'generosity', 'gdp_per_capita', 'healthy_life_expectancy']



data.columns
data.info()
#learn the size

data.shape
data.describe()
#return first five row



data.head(10)
#return last five row



data.tail(10)
environment=(data.ladder_sd.sum()/len(data.ladder_sd))

data['affect']=['High'if i >environment else 'Low' for i in data.positive_affect]

data.loc[:13,['affect']]

#correlation map



f,ax = plt.subplots(figsize=(20, 20))

sns.heatmap(data.corr(), annot=True,linewidths=.5,fmt='.2f',ax=ax)

plt.show()
# line plot



data.healthy_life_expectancy.plot(kind='line',color='b',label='healthy_life_expectancy',linewidth=1,alpha=.5,grid=True,linestyle='-.')

data.freedom.plot(kind='line',color='r',label='freedom',linewidth=1,alpha=.5,grid=True,linestyle='--')



plt.legend(loc='upper right')

plt.xlabel('Ladder')

plt.ylabel('Values')

plt.title('Healthy Life Expectancy vs Freedom')

plt.show()
#cumulative hist 

data.plot(kind='hist',y='generosity',bins=500,normed=True,cumulative=True)
#Scatter Plot



data.plot(kind='scatter', x='generosity',y='positive_affect',alpha=.5,color='r')

plt.xlabel('Generosity')

plt.ylabel('Positive Affect')

plt.title('Generosity vs Positive Affect')

plt.show()
# Bar Plot

plt.figure(figsize=(13,10))

plt.bar(data.country[:10],data.ladder_sd[:10])

plt.title('Top Ten Standard Deviation Of The Ladder.')

plt.xlabel('Country')

plt.ylabel('Ladder SD')



# Side Bar Plot



plt.figure(figsize=(12,12))

plt.barh(data.country[78:100],data.positive_affect[78:100])

plt.title('Positive Emotion Measurement From Countries Between 78 and 100')

plt.xlabel('Positive Affect')

plt.ylabel('Country')
#Sub plot

#positive and negative emotion measurements of ten countries in reverse order



plt.figure(figsize=(17,8))

plt.subplot(2,1,1)

plt.plot(data.country.tail(10),data.negative_affect.tail(10),color='b',label='Negative Affect')

plt.ylabel('Negative Affect')

plt.subplot(2,1,2)

plt.plot(data.country.tail(10),data.positive_affect.tail(10),color='r',label='Positive Affect')

plt.ylabel('Positive Affect')


