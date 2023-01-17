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

%matplotlib inline

df = pd.read_csv('/kaggle/input/videogamesales/vgsales.csv')

df.head()
df.info()
platforms = df["Platform"].value_counts().index[:15]

games = df['Platform'].value_counts().values[:15]

with plt.style.context('ggplot'):

    plt.figure(figsize = (10, 6))

    plt.barh(platforms[::-1], games[::-1])

    plt.xticks(fontweight = 'bold')

    plt.yticks(fontweight = 'bold')

    plt.title('Distribution of Games on Various Platform')

    plt.xlabel('Number of Games Avilable')

    plt.show()
game_name = df.iloc[0:10]['Name']

global_sale = df.iloc[0:10]['Global_Sales']

with plt.style.context('seaborn'):

    plt.figure(figsize = (8, 5))

    rects = plt.barh(game_name[::-1], global_sale[::-1])

    plt.xlabel('In Millions')

    plt.title('Top 10 Ranked Games In The World')

    plt.show()
fig, axs = plt.subplots(2, 2, figsize = (10, 8), sharey = True, sharex = True)

plt.style.use('fivethirtyeight')



#North America Sales

Top_Five_games = df.loc[0:4, 'Name']

sales_NA = df.loc[0:4, 'NA_Sales']

axs[0, 0].barh(Top_Five_games[::-1], sales_NA[::-1])

axs[0, 0].set(title = 'Sales In North-America')



#Europian Union Sales

sales_EU = df.loc[0:4, 'EU_Sales']

axs[0, 1].barh(Top_Five_games[::-1], sales_EU[::-1])

axs[0, 1].set_title('Sales In Europian Union')



#Japan Sales

sales_JP = df.loc[0:4, 'JP_Sales']

axs[1, 0].barh(Top_Five_games[::-1], sales_JP[::-1])

axs[1, 0].set_title('Sales In Japan')

axs[1, 0].set_xlabel('In Millions', fontsize = 11)



#Other Places

sales_Other = df.loc[0:4, 'Other_Sales']

axs[1, 1].barh(Top_Five_games[::-1], sales_Other[::-1])

axs[1, 1].set_title('Sales In Other Region')

axs[1, 1].set_xlabel('In Millions', fontsize = 11)



fig.suptitle('Performance of Top 5 Ranked Games In Diffrent Regions', fontweight = 'bold',

            fontsize = 22)



fig.show()

Total_sales = df['Global_Sales'].sum()

EU_Sales = (df['EU_Sales'].sum() / Total_sales)*100

NA_Sales = (df['NA_Sales'].sum() / Total_sales)*100

JP_Sales = (df['JP_Sales'].sum() / Total_sales)*100

Other_Sales = (df['Other_Sales'].sum() / Total_sales)*100



plt.pie([EU_Sales, NA_Sales, JP_Sales, Other_Sales], labels = ['EU', 'NA', 'JP', 'Other'],

       autopct = '%1.1f%%', shadow = True, pctdistance=0.6, startangle=90,

        wedgeprops={'edgecolor': 'black'})

plt.show()
Group_by_object = df.groupby('Publisher')

Publs= Group_by_object['Global_Sales'].sum().sort_values(ascending = False).head(10).index

Sales = Group_by_object['Global_Sales'].sum().sort_values(ascending = False).head(10).values  #or nlargest(10)

with plt.style.context('seaborn-whitegrid'):

    plt.barh(Publs[::-1], Sales[::-1], color = '#555555')

    plt.xlabel('In Millions', fontsize = '12')

    plt.title('Top 10 Publisher In The World', fontweight = 'bold')

    plt.show()
df1 = df['Year'].copy()

df1.dropna(inplace = True)

years = df1.astype('int64')

with plt.style.context('ggplot'):

    plt.figure(figsize = (9, 4))

    plt.hist(years,bins = range(1980, 2021),  edgecolor = 'white')

    plt.xticks(range(1980, 2021), rotation = 90, fontweight = 'bold')

    plt.yticks(fontweight = 'bold')

    plt.ylabel('Number Of Games Relesed')

    plt.title('Number Of Games Relesed Every Years')

    plt.show()
groupby_genre = df.groupby('Genre')

genre = groupby_genre['Global_Sales'].sum().nlargest(12).index

value = groupby_genre['Global_Sales'].sum().nlargest(12).values



with plt.style.context('seaborn'):

    plt.barh(genre[::-1], value[::-1], color = '#555555')

    plt.xlabel('In Millions')

    plt.title('Sales Based On Genre', fontweight = 'bold')

    plt.show()