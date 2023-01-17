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
data = pd.read_csv("/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")

data.head()
import matplotlib as mlp

import matplotlib.pyplot as plt
import seaborn as sns
data['neighbourhood_group'].unique()
manhattan = data[data['neighbourhood_group'] == 'Manhattan']

brooklyn = data[data['neighbourhood_group'] == 'Brooklyn']

staten_island = data[data['neighbourhood_group'] == 'Staten Island']

queens = data[data['neighbourhood_group'] == 'Queens']

bronx = data[data['neighbourhood_group'] == 'Bronx']
manhattan['neighbourhood'].value_counts().head()
manhattan_5 = manhattan[manhattan['neighbourhood'].isin(['Harlem','Upper West Side',"Hell's Kitchen",'East Village','Upper East Side'])]
m_plot = sns.boxplot(data = manhattan_5, x = 'neighbourhood', y = 'price')

m_plot.set_title("Price distribution of the top 5 neighbourhoods in Manhattan")
brooklyn['neighbourhood'].value_counts().head()
brooklyn_5 = brooklyn[brooklyn['neighbourhood'].isin(['Williamsburg','Bedford-Stuyvesant','Bushwick','Crown Heights','Greenpoint'])]

b_plot = sns.boxplot(data = brooklyn_5, x = 'neighbourhood', y = 'price')

b_plot.set_title("Price distribution of the top 5 neighbourhoods in Brooklyn")
staten_island['neighbourhood'].value_counts().head()
staten_5 = staten_island[staten_island['neighbourhood'].isin(['St. George', 'Tompkinsville','Stapleton','Concord','Arrochar'])]

b_plot = sns.boxplot(data = staten_5, x = 'neighbourhood', y = 'price')

b_plot.set_title('Price distribution of the top 5 neighbourhoods in Staten Island')
queens['neighbourhood'].value_counts().head()
queens_5 = queens[queens['neighbourhood'].isin(['Astoria','Long Island City','Flushing','Ridgewood','Sunnyside'])]

b_plot = sns.boxplot(data = queens_5, x = 'neighbourhood', y = 'price')

b_plot.set_title('Price distribution of the top 5 neighbourhoods in Queens')
bronx['neighbourhood'].value_counts().head()
bronx_5 = bronx[bronx['neighbourhood'].isin(['Kingsbridge','Fordham','Longwood','Mott Haven','Concourse'])]

b_plot = sns.boxplot(data = bronx_5, x = 'neighbourhood', y = 'price')

b_plot.set_title('Price distribution of the top 5 neighbourhoods in Bronx')
data['price'].groupby(data['neighbourhood_group']).describe().round(2)
sns.pairplot(data)
plt.figure(figsize = (10,10))



sns.distplot(manhattan['price'], color="skyblue", label="Manhattan")

sns.distplot(bronx['price'], color = 'pink', label = "Bronx")

sns.distplot(staten_island['price'], color = 'green', label = 'Staten Island')

sns.distplot(queens['price'], color = 'orange', label = 'Queens')

sns.distplot(brooklyn['price'], color = 'red', label = 'Brooklyn')



plt.legend()


