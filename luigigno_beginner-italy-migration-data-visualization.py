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
import numpy as np  

import pandas as pd 

import matplotlib.pyplot as plt 

import seaborn as sns 

print('setup complete') 
italy_imm_filepath = '/kaggle/input/italy-immigration-data-by-the-un/Italy.xlsx'

italy_imm_data = pd.read_excel(italy_imm_filepath, 

                              sheet_name='Italy by Citizenship', 

                              skiprows = range(20), 

                              skipfooter = 2)

italy_imm_data.sample(40) 
italy_imm_data.replace(['..'],0, inplace = True)
italy_imm_data = italy_imm_data[italy_imm_data.Type == 'Immigrants'] 

italy_imm_data 
italy_imm_data.rename(columns={'OdName':'Country'}, inplace=True) 
italy_imm_data = italy_imm_data[italy_imm_data.Country != 'Italy'] 

italy_imm_data 
italy_imm_data = italy_imm_data.drop(['AREA', 'REG', 'DEV'], axis=1) 
italy_imm_data['Total'] = italy_imm_data.sum(axis=1) 
italy_imm_data.columns
df1 = italy_imm_data.groupby(["Total"]) 
df2= df1.apply(lambda x: x.sort_values(['Country']))
df2 
plt.figure(figsize=(12,8)) 

sns.set(style="white") 

sns.barplot(x=df2.Total.tail(10), y=df2.Country.tail(10), 

            palette="BuGn_r", edgecolor=".2");
plt.figure(figsize=(12,8))

sns.set(style="white") 

splot = sns.barplot(y=df2.Total.head(5), x=df2.Country.head(5), 

            palette="cubehelix", edgecolor=".2");



for p in splot.patches:

  splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'baseline', xytext = (0, 10), textcoords = 'offset points')
italy_imm_data.set_index('Country', inplace=True) 
italy_imm_data.head(10) 
italy_imm_data.columns = list(map(str, italy_imm_data.columns))

years = list(map(str, range(1990, 2014)))
argentina = italy_imm_data.loc['Argentina', years]

plt.style.use(['fivethirtyeight'])



argentina.index = argentina.index.map(int)

argentina = argentina.astype(int)



fig = plt.figure(figsize=(13, 8))

ax = fig.add_axes([1,1,1,1])

argentina.plot(kind='line', ax=ax)

plt.text(1997.8, 5590, 'What happened at this time?')



ax.set_title('Immigration from Argentina')

ax.set_ylabel('Number of Immigrants')

ax.set_xlabel('Years')



plt.show()
romania_morocco = italy_imm_data.loc[['Romania', 'Morocco'], years]

romania_morocco.head()
romania_morocco = romania_morocco.T

romania_morocco.head()
romania_morocco.index = romania_morocco.index.map(int)

romania_morocco = romania_morocco.astype(int)



fig = plt.figure(figsize=(13, 8))

ax = fig.add_axes([1,1,1,1])

romania_morocco.plot(kind='line', ax=ax)



ax.set_title('Immigrants from Romania and Morocco')

ax.set_ylabel('Number of immigrants')

ax.set_xlabel('Years')



plt.show()
count, bin_edges = np.histogram(italy_imm_data['2013'])



print(count)

print(bin_edges)
fig = plt.figure(figsize=(13,8))

ax = fig.add_axes([1,1,1,1])

italy_imm_data['2013'].plot(kind='hist', ax=ax)



ax.set_title('Histogram of Immigration from 179 Countries to Italy in 2013') 

ax.set_ylabel('Number of Countries') 

ax.set_xlabel('Number of Immigrants') 



plt.show()
italy_imm_data.loc[['Germany', 'Tunisia', 'Nigeria'], years].T.columns.tolist()
italy_imm_data_t = italy_imm_data.loc[['Germany', 'Tunisia', 'Nigeria'], years].T



fig = plt.figure(figsize=(13,8))

ax = fig.add_axes([1,1,1,1])

italy_imm_data_t.plot(kind='hist', ax=ax)



ax.set_title('Immigration from Germany, Tunisia, and Nigeria from 1990 - 2013')

ax.set_ylabel('Number of Years')

ax.set_xlabel('Number of Immigrants') 



plt.show()
egypt = italy_imm_data.loc['Egypt', years]

egypt = egypt.astype(int)



fig = plt.figure(figsize=(13,8))

ax = fig.add_axes([1,1,1,1])

egypt.plot(kind='bar', ax=ax)



ax.set_xlabel('Year') 

ax.set_ylabel('Number of immigrants') 

ax.set_title('Egyptian immigrants to Italy from 1990 to 2013')



plt.show()
continents = italy_imm_data.groupby('AreaName', axis=0).sum()

print(type(italy_imm_data.groupby('AreaName', axis=0)))

continents.head()
colors_list = ['green', 'red', 'yellow', 'blue', 'orange', 'black']

explode_list = [0.1, 0.1, 0, 0.1, 0.1, 0] 



continents['Total'].plot(kind='pie',

                           figsize=(15, 8),

                           autopct='%1.1f%%', 

                           startangle=90,    

                           shadow=True,       

                           labels=None,         

                           pctdistance=1.14,     

                           colors=colors_list,  

                           explode=explode_list 

                           )



plt.title('Immigration to Italy by Continent [1990 - 2013]', y=1.14) 

plt.axis('equal') 



plt.legend(labels=continents.index, loc='upper left') 



plt.show()
tot = pd.DataFrame(italy_imm_data[years].sum(axis=0))

tot.index = map(int, tot.index)

tot.reset_index(inplace = True)

tot.columns = ['year', 'total']

tot
fig = plt.figure(figsize=(13,8))

ax = fig.add_axes([1,1,1,1])

tot.plot(kind='scatter', x='year', y='total', ax=ax)



ax.set_title('Total Immigration to Italy from 1990 - 2013')

ax.set_xlabel('Year')

ax.set_ylabel('Number of Immigrants')



plt.show()
x = tot['year']      

y = tot['total']   

fit = np.polyfit(x, y, deg=1)

fit
fig = plt.figure(figsize=(13,8))

ax = fig.add_axes([1,1,1,1])

tot.plot(kind='scatter', x='year', y='total', ax=ax)



ax.set_title('Total Immigration to Italy from 1990 - 2013')

ax.set_xlabel('Year')

ax.set_ylabel('Number of Immigrants')



ax.plot(x, fit[0] * x + fit[1], color='red')
italy_imm_data_t = italy_imm_data[years].T

italy_imm_data_t.index = map(int, italy_imm_data_t.index)

italy_imm_data_t.index.name = 'Year'

italy_imm_data_t.reset_index(inplace=True)

italy_imm_data_t
from sklearn.preprocessing import MinMaxScaler



scale_bra = MinMaxScaler()

scale_arg = MinMaxScaler()

norm_brazil = scale_bra.fit_transform(italy_imm_data_t['Brazil'].values.reshape(-1, 1))

norm_arg = scale_arg.fit_transform(italy_imm_data_t['Argentina'].values.reshape(-1, 1))
italy_imm_data_t['weight_arg'] = norm_arg

italy_imm_data_t['weight_brazil'] = norm_brazil



fig = plt.figure(figsize=(13,9))

ax = fig.add_axes([1,1,1,1])





italy_imm_data_t.plot(kind='scatter', x='Year', y='Brazil',

            alpha=0.5,                  # transparency

            s=norm_brazil * 2000 + 10,  # pass in weights 

            ax=ax)





italy_imm_data_t.plot(kind='scatter', x='Year', y='Argentina',

            alpha=0.5,

            color="blue",

            s=norm_arg * 2000 + 10,

            ax=ax)



ax.set_ylabel('Number of Immigrants')

ax.set_title('Immigration from Brazil and Argentina from 1990 - 2013')

ax.legend(['Brazil', 'Argentina'], loc='upper left', fontsize='x-large')



plt.show()