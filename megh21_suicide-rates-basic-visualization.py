# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

sns.set_style('darkgrid')

plt.rcParams['figure.figsize'] = 15,8

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

df1 = pd.read_csv('../input/master.csv')

# Any results you write to the current directory are saved as output.
df1.info()
df1.columns
df1.columns = ['country', 'year', 'sex', 'age', 'suicides', 'population',

       'suicides/100k', 'country-year', 'HDI',

       ' gdp ', 'gdp_pc', 'generation']
df1.columns
df1.year = df1.year.astype('category')

df1.age = df1.age.astype('category')

df1.sex = df1.sex.astype('category')

df1.generation = df1.generation.astype('category')

df1.country = df1.country.astype('category')
df1.describe()
#############Suicide Rates Over the Years around the world########

viz1 = sns.barplot(data = df1, x = 'year', y= 'suicides' , ci =None , color = 'green')

plt.rcParams['figure.figsize'] = 15,8
#############Suicide Rates Over the Years around the world based on gender########

viz2 = sns.lineplot(data = df1, x = 'year', y= 'suicides' , ci =None , hue = 'sex')

plt.rcParams['figure.figsize'] = 15,8
#############Suicide Rates Over the Years around the world based on generation########

viz3 = sns.barplot(data = df1, x = 'year', y= 'suicides' , ci =None , hue = 'generation')

plt.rcParams['figure.figsize'] = 15,10
#TO FIND COUNTRY WITH HIGHEST SUICIDES

print(max(df1.suicides))

df1[df1['suicides'] == max(df1.suicides)]
############EXTRACTING ALL DATA FOR Russian Federation#############

df2 = df1[df1.country == 'Russian Federation']

df2.head()
##############NUMBER OF SUICIDES PER YEAR IN Russian Federation##############
viz4 = sns.barplot(data = df2, x= 'year', y = 'suicides', estimator = sum , ci =None)

plt.yticks(np.arange(0,70000 ,10000))

plt.show()
##################Breaking down Suicides Gender Wise in Russian Federation##########

viz5 = sns.barplot(data = df2, x= 'year', y = 'suicides', estimator = sum , ci =None ,hue = 'sex')

plt.yticks(np.arange(0,70000 ,10000))

plt.show()
viz6 = sns.lmplot(data = df2, x= 'suicides', y = 'gdp_pc', ci=None,fit_reg=False,size=8,hue='generation',aspect=2)