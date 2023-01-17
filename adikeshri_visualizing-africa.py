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
data=pd.read_csv('../input/africa-economic-banking-and-systemic-crisis-data/african_crises.csv')

data.sample(5)

unique_countries=data.country.unique()

unique_countries
import matplotlib.pyplot as plt

import seaborn as sns

import random

sns.set(style='whitegrid')

plt.figure(figsize=(20,20))

count=1

for country in unique_countries:

    plt.subplot(5,3,count)

    count+=1

    col="#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])

    sns.lineplot(data[data.country==country]['year'],

                 data[data.country==country]['exch_usd'],

                 label=country,

                 color=col)

    plt.scatter(data[data.country==country]['year'],

                data[data.country==country]['exch_usd'],

                color=col,

                s=28)

    plt.plot([np.min(data[np.logical_and(data.country==country,data.independence==1)]['year']),

              np.min(data[np.logical_and(data.country==country,data.independence==1)]['year'])],

             [0,

              np.max(data[data.country==country]['exch_usd'])],

             color='black',

             linestyle='dotted',

             alpha=0.8)

    plt.text(np.min(data[np.logical_and(data.country==country,data.independence==1)]['year']),

             np.max(data[data.country==country]['exch_usd'])/2,

             'Independence',

             rotation=-90)

    plt.scatter(x=np.min(data[np.logical_and(data.country==country,data.independence==1)]['year']),

                y=0,

                s=50)

    plt.title(country)

plt.tight_layout()

plt.show()

data.info()
sns.set(style='darkgrid')

cols=['systemic_crisis','domestic_debt_in_default','sovereign_external_debt_default','currency_crises','inflation_crises','banking_crisis']

plt.figure(figsize=(20,20))

count=1

for col in cols:

    plt.subplot(3,2,count)

    count+=1

    sns.countplot(y=data.country,hue=data[col],palette='rocket')

    plt.legend(loc=0)

    plt.title(col)

plt.tight_layout()

plt.show()

sns.set(style='whitegrid')

plt.figure(figsize=(20,20))

count=1

for country in unique_countries:

    plt.subplot(5,3,count)

    count+=1

    col="#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])

    sns.lineplot(data[data.country==country]['year'],

                 data[data.country==country]['inflation_annual_cpi'],

                 label=country,

                 color=col)

    plt.scatter(data[data.country==country]['year'],

                data[data.country==country]['inflation_annual_cpi'],

                color=col,

                s=28)

    plt.plot([np.min(data[np.logical_and(data.country==country,data.independence==1)]['year']),

              np.min(data[np.logical_and(data.country==country,data.independence==1)]['year'])],

             [np.min(data[data.country==country]['inflation_annual_cpi']),

              np.max(data[data.country==country]['inflation_annual_cpi'])],

             color='black',

             linestyle='dotted',

             alpha=0.8)

    plt.text(np.min(data[np.logical_and(data.country==country,data.independence==1)]['year']),

             np.max(data[data.country==country]['inflation_annual_cpi'])/2,

             'Independence',

             rotation=-90)

    plt.scatter(x=np.min(data[np.logical_and(data.country==country,data.independence==1)]['year']),

                y=np.min(data[data.country==country]['inflation_annual_cpi']),

                s=50)

    plt.title(country)

plt.tight_layout()

plt.show()

sns.heatmap(data.corr())