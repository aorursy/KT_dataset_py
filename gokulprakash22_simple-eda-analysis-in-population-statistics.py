import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("darkgrid")

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("../input/cities-in-tamil-nadu-population-statistics/Cities and Towns in Tamil Nadu - Population statistics.csv")

df.head()
df.info()
df.describe()
df.shape
df.replace(',','', regex=True, inplace=True)

df.head()
df = df.rename(columns = {'Population\nCensus\n1991-03-01': '1991'}, inplace = False)

df = df.rename(columns = {'Population\nCensus\n2001-03-01': '2001'}, inplace = False)

df = df.rename(columns = {'Population\nCensus\n2011-03-01': '2011'}, inplace = False)

df.head()
df = df[(df['1991'] != '...')]

df.head()
df['1991'] = pd.to_numeric(df['1991'])

df['2001'] = pd.to_numeric(df['2001'])

df['2011'] = pd.to_numeric(df['2011'])

df.info()
Status = df['Status'].unique()

Status
df_sum = df.groupby('Status').sum()

df_sum
N = len(Status)

sum_1991 = df_sum['1991']

sum_2001 = df_sum['2001']

sum_2011 = df_sum['2011']



ind = np.arange(N) 

width = 0.35       

plt.bar(ind, sum_1991, width, label='1991')

plt.bar(ind + width, sum_2001, width, label='2001')

plt.bar(ind + 2*width, sum_2011, width, label='2011')



plt.xlabel('Status of city/town in Tamil Nadu')

plt.ylabel('Population')

plt.title('Population growth over years in Tamil Nadu')



plt.xticks(ind + width / 2, Status)

plt.xticks(rotation=90)

plt.legend(loc='best')

plt.show()