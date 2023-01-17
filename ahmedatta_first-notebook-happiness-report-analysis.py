# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
data_2015 = pd.read_csv('/kaggle/input/world-happiness/2015.csv')

data_2016 = pd.read_csv('/kaggle/input/world-happiness/2016.csv')

data_2017 = pd.read_csv('/kaggle/input/world-happiness/2017.csv')

data_2018 = pd.read_csv('/kaggle/input/world-happiness/2018.csv')

data_2019 = pd.read_csv('/kaggle/input/world-happiness/2019.csv')

data_2015.head()
data_2015.info()
data_2015.rename(columns={'Health (Life Expectancy)' : 'Health','Economy (GDP per Capita)' : 'Economy' 

                          , 'Trust (Government Corruption)' :'Trust' },inplace = True)

data_2015.groupby('Region')['Happiness Score'].describe()
sns.set()

fig = plt.figure(figsize = (16,8))

data_2015.groupby('Country')['Happiness Score'].mean().nlargest(10).plot(kind = 'bar',color =sns.color_palette())

plt.ylabel('Happiness Score',fontsize =15)

plt.xlabel('Country' ,Fontsize = 15)

plt.xticks(fontsize = 15)

plt.title('Top 10 Happiest Countries',Fontsize =16,weight ='bold')

plt.ylim(6,8)
sns.set()

fig = plt.figure(figsize = (16,8))

data_2015.groupby('Country')['Happiness Score'].mean().nsmallest(10).plot(kind = 'bar',color =sns.color_palette())

plt.ylabel('Happiness Score',fontsize =15)

plt.xlabel('Country' ,Fontsize = 15)

plt.xticks(fontsize = 15)

plt.title('Least 10 Happiest Countries',Fontsize =16,weight ='bold')

plt.ylim(2,4)
grouped_data = data_2015.groupby('Region')[['Family','Economy','Health','Freedom','Generosity','Trust']].mean()

grouped_data.plot(kind ='bar',figsize=(16,8))

plt.ylabel('Average parameters',fontsize = 16)

plt.xlabel('Region', fontsize= 16)

plt.xticks(fontsize = 14)

plt.legend(bbox_to_anchor=(0.9,0.7))

plt.title('Average parameters for different regions',fontsize = 18)

plt.show()
sns.set()

data_2015.groupby('Region')['Happiness Score'].mean().sort_values(ascending = False).plot(kind = 'bar'

                                                                                          ,figsize = (15,8),grid=True)

plt.ylim(3)

plt.xlabel('Region' , fontsize =15)

plt.xticks(rotation=80 , fontsize =12)

plt.title('Average Happiness Score by Region',fontsize = 18,weight = 'bold')
sns.set()

pd.pivot_table(data_2015,index = 'Region', values = 'Economy'

               ,aggfunc ='sum').sort_values(by = 'Economy',ascending = False).plot(kind = 'bar',figsize= (15,8))

plt.xlabel('Region' , fontsize='16')

plt.ylabel('GDP per Capita',fontsize='16')

plt.xticks(fontsize = '14')

plt.show()
f,ax = plt.subplots(figsize = (9,5))

for i in np.arange(len(grouped_data.columns)):

    sns.barplot(x = grouped_data.values.T[i] , y=grouped_data.index

                ,label =grouped_data.columns[i],color = sns.color_palette('Set2')[i]) 

ax.legend(bbox_to_anchor=(0.8,0.7))

plt.xlabel('Average values of different Factors',fontsize = 15)

plt.ylabel('Region',fontsize= 15)

plt.title('Factors affecting happiness score',fontsize = 16)

plt.show()
f,ax = plt.subplots(figsize=(20,10))

last_p = len(grouped_data.index)-1

happiness_ratio= data_2015.groupby('Region')['Happiness Score'].mean()/data_2015.groupby('Region')['Happiness Score'].max()

sns.pointplot(x=happiness_ratio.index,y=happiness_ratio.values,color='black'

              ,linestyles ='--').annotate('Happiness ratio',xy = (last_p,happiness_ratio.values[last_p]),

                                         xytext=(last_p +0.4 , happiness_ratio.values[last_p]-0.1),fontsize = 15,arrowprops=dict(facecolor="r",arrowstyle="fancy"))

for i in np.arange(len(grouped_data.columns)):

    

    ax = sns.pointplot( x=grouped_data.index ,y = grouped_data.values.T[i],color = sns.color_palette('Set2')[i])

    ax.annotate(grouped_data.T.index[i],xy = (last_p,grouped_data.values.T[i][last_p]),xytext=(last_p +0.4,grouped_data.values.T[i][last_p])

                ,fontsize = 15,arrowprops=dict(facecolor="r",arrowstyle="fancy"))

            

plt.xlabel('Region',fontsize=18)

plt.ylabel('Factors',fontsize= 18)

plt.xticks(rotation=90,fontsize = 16)

plt.title('Factors contributing to happiness',fontsize =20)

plt.show()
data_2016.info()
data_2016.rename(columns={'Health (Life Expectancy)' : 'Health','Economy (GDP per Capita)' : 'Economy' 

                          , 'Trust (Government Corruption)' :'Trust' },inplace = True)

data_2016.info()
regions_2015 = data_2015.groupby('Region')['Happiness Score'].mean().sort_values()

regions_2016 = data_2016.groupby('Region')['Happiness Score'].mean().sort_values()

f,ax = plt.subplots(figsize=(20,10))

sns.lineplot(x= regions_2015.index ,y=regions_2015.values,sort=False,label='2015')

sns.lineplot(x= regions_2016.index ,y=regions_2016.values,sort=False,label = '2016')

plt.xticks(rotation=90,fontsize = 14)

plt.ylabel('Happiness Score' , fontsize =16)

plt.xlabel('Region' , fontsize = 16)

plt.title('Happines Score between 2015 and 2016 for all regions',fontsize = 18 , weight = 'bold')

plt.legend(prop={'size': 20})

plt.show()
data_2017.columns
data_2018.columns
data_2019.columns
f,ax = plt.subplots(figsize=(16,10))

plt.plot(np.sort(data_2015['Happiness Score'].values),label='2015 Score')

plt.plot(np.sort(data_2016['Happiness Score'].values),label='2016 Score')

plt.plot(np.sort(data_2017['Happiness.Score'].values),label='2017 Score')

plt.plot(np.sort(data_2018['Score'].values),label='2018 Score')

plt.plot(np.sort(data_2019['Score'].values),label='2019 Score')

plt.tick_params(labelbottom = False)

plt.xlabel('Countries',fontsize = 15)

plt.ylabel('Happiness Score',fontsize = 15)

plt.legend(fontsize = 'large')

plt.title('Happiness Score difference on different years for all countries' , fontsize = 18,weight = 'bold')

plt.show()