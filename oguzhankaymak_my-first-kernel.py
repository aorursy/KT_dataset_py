# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import os





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
file_2017 = '../input/world-happiness/2017.csv'

data_2017 = pd.read_csv(file_2017)
data_2017.columns
data_2017.info()

data_2017.head(10)
data_2017.corr()
#Corelation Map

f,ax = plt.subplots(figsize=(15,15))

sns.heatmap(data_2017.corr(),annot = True,linewidths = 5,fmt='.1f',ax=ax)

plt.show()
#Column name edit 

data_2017= data_2017.rename(columns={'Happiness.Rank': 'Happiness_Rank','Happiness.Score':'Happiness_Score','Whisker.high':'Whisker_high','Whisker.low':'Whisker_low','Economy..GDP.per.Capita.':'Economy_GDP_per_Capita','Health..Life.Expectancy.':'Health_Life_Expectancy','Trust..Government.Corruption.':'Trust_Government_Corruption','Dystopia.Residual':'Dystopia_Residual'})

print(data_2017.Country.unique())
plt.plot(data_2017.Happiness_Score,data_2017.Freedom,color='b',label='Happiness_Score',linewidth=1.3,alpha=0.7,linestyle=':')

plt.xlabel('Happiness Score')

plt.ylabel('Freedom Score')

plt.title('Happiness-Freedom-2017')
data_2017.plot(kind='scatter',x='Happiness_Score',y='Trust_Government_Corruption',alpha=0.5,color='r')

plt.xlabel('Happiness_Score')

plt.ylabel('Trust_Government_Corruption')

plt.title('Happiness Score - Trust Government Corruption - 2017')


data_2017.Generosity.plot(kind='hist',bins=50,figsize = (15,15))

plt.show()
data_2017.describe()


good_score = data_2017['Happiness_Score'] > 5.354019

data_2017[good_score]

data_2017[np.logical_and(data_2017['Happiness_Score'] > 5.354019 ,data_2017['Freedom'] < 0.408786)]
x = data_2017.iloc[:,-1]

print(x)
y = data_2017.iloc[0:51,8:9]

print(y)