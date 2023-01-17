# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt # this is used for the plot the graph 

from matplotlib import rcParams

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/BRAZIL_CITIES.csv", sep=";", decimal=",")

data.head()
data.shape
data.UBER = data.UBER.replace(np.nan,0)

data.POST_OFFICES = data.POST_OFFICES.replace(np.nan,1)

data.CAPITAL = data.CAPITAL.replace(0,'NO')

data.CAPITAL = data.CAPITAL.replace(1,'YES')
original_data = data.copy(True)
columns = ['CITY', 'STATE', 'CAPITAL', 'IBGE_RES_POP','AREA',

           'IDHM','LONG','LAT','ALT','ESTIMATED_POP','GDP','GDP_CAPITA','COMP_TOT',

           'Cars','Motorcycles','UBER','Wheeled_tractor','POST_OFFICES']

df = data[columns]
df.head()
print("Percentage null or na values in df")

((df.isnull() | df.isna()).sum() * 100 / df.index.size).round(2)
df.dropna(how ='any', inplace = True)

print("Percentage null or na values in df")

((df.isnull() | df.isna()).sum() * 100 / df.index.size).round(2)
df.rename(columns={'IBGE_RES_POP': 'Population(2010)', 

                    'IDHM':'Human Development Index Ranking',

                    'ESTIMATED_POP':'Estimated Population(2018)',

                    'COMP_TOT':'Total companies',

                    'UBER':'Uber',

                    'POST_OFFICES':'Post Offices'}, inplace=True)
df= df.drop_duplicates(subset='CITY',keep='first')

df['Population Growth %']=((df['Estimated Population(2018)']-df['Population(2010)'])/(df['Population(2010)']))*100
correlation= df.corr()

sns.heatmap(correlation)
capital_hdi=sns.violinplot(x = 'CAPITAL', y = 'Human Development Index Ranking', data = df, palette = "Set3")

capital_hdi.set_xlabel(xlabel = 'Capital', fontsize = 9)

capital_hdi.set_ylabel(ylabel = 'Human Development Index Ranking', fontsize = 9)

capital_hdi.set_title(label = 'Capitals vs HDI', fontsize = 20)

plt.show()
# df['Human Development Index Ranking'].mean()

fig, ax = plt.subplots(figsize=[16,4])

category_plot = sns.distplot(df['Human Development Index Ranking'],ax=ax)

ax.set_title( 'HDI Distrubution for all cities')

cmap = sns.cubehelix_palette(dark=.1, light=.3, as_cmap=True)





f, ax = plt.subplots(figsize=(8, 8))

sns.scatterplot(x=df[df['AREA']>= 1500000].LONG,

                y=df[df['AREA']>= 1500000].LAT ,

                palette =cmap,

                hue=df['AREA'],

                size=df['AREA'])

plt.title("Location of Cities with large Area")





f, ax = plt.subplots(figsize=(8, 8))

sns.scatterplot(x=df[df['Total companies']>1000].LONG,

                y=df[df['Total companies']>1000].LAT,

                palette =cmap,

                hue=df['Total companies'],

                size=df['Total companies'])

plt.title("Location of Cities with large number of Companies")







f, ax = plt.subplots(figsize=(8, 8))

sns.scatterplot(x=df[df['Population Growth %'] > 15].LONG,

                y=df[df['Population Growth %'] > 15].LAT,

                palette =cmap,

                hue=df['Population Growth %'],

                size=df['Population Growth %'])

plt.title("Location of Cities with large Population Growth %")
# df['GDP_CAPITA'].max()

newdf=df[['CITY','GDP_CAPITA','CAPITAL','STATE']].groupby(['GDP_CAPITA'])

newdf=newdf.filter(lambda x: x.mean() >= 80000)

newdf=newdf.sort_values(by=['GDP_CAPITA'])

newdf
plt.rcParams['figure.figsize'] = (15, 9)

newdf.STATE.value_counts().sort_index().plot.bar()
# df['Population Growth %'].max()

newdf_pop=df[['CITY','Population Growth %']].groupby(['Population Growth %'])

newdf_pop=newdf_pop.filter(lambda x: x.mean() <= -10)

newdf_pop=newdf_pop.sort_values(by=['Population Growth %'])



newdf_pop_grow=df[['CITY','Population Growth %']].groupby(['Population Growth %'])

newdf_pop_grow=newdf_pop_grow.filter(lambda x: x.mean() >= 10)

newdf_pop_grow=newdf_pop_grow.sort_values(by=['Population Growth %'])
pop_decrease = pd.merge(newdf, newdf_pop, how='inner', on=['CITY'])

print('Cities with Estimated Population decrease with high GDP_CAPITA \n')

pop_decrease
pop_increase=pd.merge(newdf, newdf_pop_grow, how='inner', on=['CITY'])

print('Cities with Estimated Population increase with high GDP_CAPITA \n')

pop_increase
plt.rcParams['figure.figsize'] = (15, 9)

pop_increase.STATE.value_counts().sort_index().plot.bar()

plt.title('States with high Population growth due to high GDP per capita',size=20)
# df.AREA.mean()

newdf_area=df[['CITY','AREA']].groupby(['AREA'])

newdf_area=newdf_area.filter(lambda x: x.mean() >= 1254967)

newdf_area=newdf_area.sort_values(by=['AREA'])
pop_area=pd.merge(pop_increase, newdf_area, how='inner', on=['CITY'])

pop_area
newdf_companies=df[['CITY','Total companies']].groupby(['Total companies'])

newdf_companies=newdf_companies.filter(lambda x: x.mean() >= 1000)

newdf_companies=newdf_companies.sort_values(by=['Total companies'])

newdf_companies
pop_companies=pd.merge(newdf_pop_grow, newdf_companies, how='inner', on=['CITY'])

pop_companies


df['Cars distribution']=((df['Cars'])/(df['Estimated Population(2018)']))

# df.head()

# df['Cars distribution'].describe()

f, ax = plt.subplots(figsize=(8, 8))

sns.scatterplot(x=df[df['Cars distribution'] >= 0.20].LONG,

                y=df[df['Cars distribution'] >= 0.20].LAT,

                palette =cmap,

                hue=df['Cars distribution'],

                size=df['Cars distribution'])
correlation= df.corr()

sns.heatmap(correlation)