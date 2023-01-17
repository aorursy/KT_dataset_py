import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder

import seaborn as sns

sns.set_style('whitegrid')
#insert dataset

df=pd.read_csv("../input/forest-fires-in-brazil/amazon.csv",encoding='ISO-8859-1')

df.head(10)
month=df["month"].unique().tolist()

month
bulan=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']

for i in range(0, len(bulan)):

    df['month'][df['month']==month[i]]=bulan[i]

df.head()
#Check Missing Values

df.isnull().sum()
df.describe()
df.dtypes
print(df.groupby(['state']).count())

state_plot=sns.countplot(x='state',data=df)

state_plot.set_xticklabels(state_plot.get_xticklabels(), rotation = 70)
essential_data=df.groupby(by=['year','state','month']).sum().reset_index()

essential_data
from matplotlib.pyplot import MaxNLocator,FuncFormatter

plt.figure(figsize=(10,5))

plot=sns.lineplot(data=essential_data,x='year',y='number',markers=True)

plot.xaxis.set_major_locator(plt.MaxNLocator(19))

plot.set_xlim(1998,2017)
year_number_data=essential_data[['year','number']]

year_number_data[year_number_data['year']==1998]
increasing_list = [1998, 1999, 2000, 2001, 2002, 2008, 2011, 2013, 2014, 2015]

decreasing_list = [2003, 2004, 2005, 2006, 2007, 2009, 2010, 2012, 2016]



increasing_dataframe = pd.DataFrame()

for i in increasing_list:

    df = year_number_data[year_number_data['year'] == i]

    increasing_dataframe = increasing_dataframe.append([df])

increasing_dataframe.head()
decreasing_dataframe = pd.DataFrame()

for i in decreasing_list:

    df1 = year_number_data[year_number_data['year'] == i]

    decreasing_dataframe = decreasing_dataframe.append([df1])

decreasing_dataframe.head()
plt.figure(figsize=(10,5))

plot=sns.lineplot(data=increasing_dataframe,

                 x='year',

                 y='number',

                 lw=1,

                 err_style="bars",

                 ci=100)

plot=sns.lineplot(data=decreasing_dataframe,

                 x='year',

                 y='number',

                 lw=1,

                 err_style="bars",

                 ci=100)

plot.xaxis.set_major_locator(plt.MaxNLocator(19))

plot.set_xlim(1998,2017)