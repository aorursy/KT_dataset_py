# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import DataFrame, Series



# These are the plotting modules adn libraries we'll use:

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as stats



# Command so that plots appear in the iPython Notebook

%matplotlib inline

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
codes=pd.read_excel('/kaggle/input/summer-olympics-medals/Summer-Olympic-medals-1976-to-2008.xlsx',sheet_name='Country Codes')

df=pd.read_excel('/kaggle/input/summer-olympics-medals/Summer-Olympic-medals-1976-to-2008.xlsx',sheet_name='Medalists')

# Total number of Each Medal Awarded

check=df.groupby('Medal').agg('count')

check[['City']]



#The number of medals are way off, could be discrepancies in team sizes, medals not accepted, ect...
co=df['Country'].value_counts().sort_values(ascending=False)

top15=co[:15]

top15.plot(kind='bar',figsize=(10,4))

plt.xticks(rotation=45)

plt.title('All Time Medals (Top 15)')

plt.show()
fil1=df['Country']=='United States'

fil2=df['Country']=='Australia'

fil3=df['Country']=='Soviet Union'

fil4=df['Country']=='Germany'

fil5=df['Country']=='China'

fil6=df['Country']=='Russia'

fil7=df['Country']=='East Germany'

df1=df.where(fil1 | fil2| fil3 | fil4 | fil5 | fil6 | fil7).dropna()

df1=df1[['Year','Country','Medal']]

df2=df1.groupby(['Country','Year']).agg('count').groupby('Country').cumsum().reset_index()

sns.lineplot(data=df2,x='Year',y='Medal',hue='Country')

plt.title('Top 7 Countries With The Most Medals')

plt.show()
my=df.groupby('Year')

my1=my.agg('count')

my1

my1.plot(y='Medal',kind='bar',legend=False,title='Total Medals Awarded per Year')
# USA medals by year

usa=df[df.Country_Code=='USA']

sns.countplot(x="Year", data=usa)

plt.title("USA Total Medals per Year")

plt.show
# USA medals by Type

sns.countplot(x='Year',hue='Medal',data=usa,palette=('gold','gray','silver'))

plt.title('USA Medals by Year and Medal Type')

plt.show()
usa1=usa['Medal'].value_counts()

usa1.plot.pie(colors=('gold','silver','gray'),title='USA Medals by Type')
# Number of Men vs Women Represented by the Year



sns.countplot(data=df,x='Year',hue='Gender',palette=('blue','orange'))

plt.title('Total Medals Awarded: Men vs. Women')

plt.show
# USA medalists Men vs Women



sns.countplot(data=usa,x='Year',hue='Gender', palette=('orange','blue'))

plt.title("USA Medals: Men vs. Women")

plt.show()
fil1=df['Sport']=='Basketball'

fil2=df['Sport']=='Handball'

fil3=df['Sport']=='Baseball'

fil4=df['Sport']=='Softball'

fil5=df['Sport']=='Football'

fil6=df['Sport']=='Hockey'

teamsports=df.where(fil1 | fil2 | fil3 | fil4 | fil5 | fil6).dropna()

sns.countplot(data=teamsports,x='Country')

plt.xticks(rotation=90)

plt.title("Total Medals in Team Sports")

plt.show()


