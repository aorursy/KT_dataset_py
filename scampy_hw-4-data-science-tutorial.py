# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/kc_house_data.csv')
#First 5 rows 
df.head()
# Summary of dataframe
df.info()
# Manuplate date object with date column
df['dateofdata'] = pd.to_datetime(df.date.apply(lambda x: x[0:8]))
df.dateofdata.head()
#Create a year column to filter data by year
df['year'] = df.dateofdata.apply(lambda x: x.year)
#Filter the date according to date
df_2014 = df[df.year == 2014]
df_2015 = df[df.year == 2015]
# Statistical details of datas from 2014 
df_2014.describe()
# Statistical details of datas from 2015 
df_2015.describe()
#Filtering data
df_2014_cond_morethanthree = df_2014[df_2014.condition > 3]
df_2015_cond_morethanthree = df_2015[df_2015.condition > 3]
# Exploratory Statistics
df_2014_cond_morethanthree.describe()
# Exploratory Statistics
df_2015_cond_morethanthree.describe()
plt.figure(figsize=(10, 8))
plt.subplot(211)
plt.hist(df_2014_cond_morethanthree.price,bins = 150, range = [0,7700000])
plt.title('2014 House Prices')
plt.annotate('Medium Price ' + str(round(df_2015_cond_morethanthree.price.mean()/1e6,2)) + ' million',
             xy=(df_2014_cond_morethanthree.price.mean(),0),
             xytext = (df_2014_cond_morethanthree.price.mean()+1000000,100),
             arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc,angleA=0,armA=50,rad=10"))
plt.annotate('Maximum Price '+ str(df_2014_cond_morethanthree.price.max()/1e6) + ' million', 
             xy=(df_2014_cond_morethanthree.price.max(),0),
             xytext = (df_2014_cond_morethanthree.price.max(),400),
             arrowprops=dict(arrowstyle="wedge,tail_width=0.5", alpha=0.3))
plt.subplot(212)
plt.hist(df_2015_cond_morethanthree.price,bins = 150, range = [0,5500000])
plt.title('2015 House Prices')
plt.annotate('Medium Price ' + str(round(df_2015_cond_morethanthree.price.mean()/1e6,2)) + ' million',
             xy=(df_2015_cond_morethanthree.price.mean(),0),
             xytext = (df_2015_cond_morethanthree.price.mean()+1000000,100),
             arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc,angleA=0,armA=50,rad=10"))
plt.annotate('Maximum Price ' + str(df_2015_cond_morethanthree.price.max()/1e6) + ' million' , 
             xy=(df_2015_cond_morethanthree.price.max(),0),
             xytext = (df_2015_cond_morethanthree.price.max(),100),
             arrowprops=dict(arrowstyle="wedge,tail_width=0.5", alpha=0.3))
plt.show()
# Do visualization in a different way
df[df.condition > 3].boxplot(column='price', by='year', figsize=(8,5))
plt.show()