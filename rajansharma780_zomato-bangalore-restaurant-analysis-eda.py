# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#read the csv file and make the data frame 

zomato_df = pd.read_csv('../input/zomato.csv')
#display the data frame

zomato_df
#display the first 5 rows

zomato_df.head()
#display the last five rows

zomato_df.tail()
#display the how many rows and columns are there

print("the data frame has {} rows and {} columns".format(zomato_df.shape[0],zomato_df.shape[1]))
#display the data types of columns

zomato_df.dtypes
#display the size of data frame

print("the size of data frame is {}".format(zomato_df.size))
#dispaly the information of data frame

zomato_df.info()
#display null values for each column

zomato_df.apply(lambda x:sum(x.isnull()))
#display null values for each column graphically

sns.heatmap(zomato_df.isnull(),cbar=0)

plt.show()
#make the copy of data frame and apply changes to the new data frame

new_zomato_df = zomato_df.copy()
#so approx_cost is object type we have to make it integer type and remove ','

import re

def align_proper(x):

    if (re.search(',',x)):

        return(x.replace(',',''))

    else:

        return x

new_zomato_df['approx_cost(for two people)'] = new_zomato_df[new_zomato_df['approx_cost(for two people)'].notnull()]['approx_cost(for two people)'].apply(align_proper).astype('int')
#fill null values of approx_cost with mean value of approx_cost

new_zomato_df['approx_cost(for two people)'] = new_zomato_df['approx_cost(for two people)'].fillna(value=np.mean(new_zomato_df['approx_cost(for two people)']))
new_zomato_df['approx_cost(for two people)'] = new_zomato_df['approx_cost(for two people)'].astype('int')
#display the list of restaurant whose approx_cost(for two people) is minimum

new_zomato_df[new_zomato_df['approx_cost(for two people)']==new_zomato_df['approx_cost(for two people)'].min()]
#display the list of restaurant whose approx_cost(for two people) is maximum

new_zomato_df[new_zomato_df['approx_cost(for two people)']==new_zomato_df['approx_cost(for two people)'].max()]
#display 5 number summary of approx_cost(for two people)

new_zomato_df['approx_cost(for two people)'].describe()
#display graphically how approx_cost(for two people) distributed

sns.distplot(new_zomato_df['approx_cost(for two people)'],kde=0)

plt.show()
#display which location has more restaurants

new_zomato_df['listed_in(city)'].value_counts()
#display which type of restaurants are more

new_zomato_df['listed_in(type)'].value_counts()
chart =sns.countplot(new_zomato_df['listed_in(type)'])

chart.set_xticklabels(chart.get_xticklabels(),rotation=30)
#display how many restaurants with online order 

new_zomato_df['online_order'].value_counts()
#display which type of restaurant having online order option or not

chart =sns.countplot(new_zomato_df['listed_in(type)'],hue=new_zomato_df['online_order'])

chart.set_xticklabels(chart.get_xticklabels(),rotation=30)
pd.crosstab(new_zomato_df['listed_in(type)'],new_zomato_df['online_order'])