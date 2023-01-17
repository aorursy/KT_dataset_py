# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/daily-temperature-of-major-cities/city_temperature.csv')

df.head()
df.isna().sum()
df = df.drop('State',axis=1)

df.head()
df.describe()
df['Year'].unique()
df.loc[df['Year']==200,'Year']=2000

df.loc[df['Year']==201,'Year']=2010

df.head()
a= df.groupby(['Year','Region'])['AvgTemperature'].mean().reset_index().sort_values(by='AvgTemperature',ascending=False)

a.head(20).style.background_gradient(cmap='Blues')
plt.figure(figsize=(15,8))

sns.lineplot(x='Year',y='AvgTemperature',hue='Region',data=a,palette='hsv')

plt.grid()

plt.title('YEAR-WISE AVERAGE MEAN TEMPERATURE OF DIFFERENT REGIONS')

plt.show()
b= df.groupby(['Region','Month'])['AvgTemperature'].max().reset_index().sort_values(by='AvgTemperature',ascending=False)

b.head(20).style.background_gradient(cmap='Oranges')
plt.figure(figsize=(15,8))

sns.barplot(x='Month', y= 'AvgTemperature',data=b,hue='Region',palette='hsv',saturation=.80)

plt.title('VARIATION OF MAXIMUM TEMPERATURE OVER THE MONTHS')

plt.show()
c= df.groupby(['Region','Year'])['AvgTemperature'].max().reset_index().sort_values(by='AvgTemperature',ascending=False)

c.head(20).style.background_gradient(cmap='Greens')
plt.figure(figsize=(15,8))

sns.scatterplot(x='Year',y='AvgTemperature',data=c,hue='Region',palette='hsv_r',style='Region')

plt.title(' VARIATION OF MAXIMUM TEMPERATURE OVER THE YEARS')

plt.show()
c= df.groupby(['Country','City'])['AvgTemperature'].mean().reset_index().sort_values(by='AvgTemperature',ascending=False).head(20)

c.style.background_gradient(cmap='Reds')
plt.figure(figsize=(8,10))

sns.barplot(x='AvgTemperature',y='City',data=c,palette='hsv_r')

plt.title('VARIATION OF MEAN TEMPERATURE FOR TOP 20 COUNTRIES')

plt.show()
plt.figure(figsize=(15,8))

sns.lineplot(x='Year',y='AvgTemperature',data=x,color='r')

plt.grid()

plt.title('Mean Temp. Variation in India') 

plt.show()
ind=df[df['Country']=='India']

x= ind.groupby(['Year'])['AvgTemperature'].mean().reset_index().sort_values(by='AvgTemperature',ascending=False)

x.style.background_gradient(cmap='hsv')
ind=df[df['Country']=='India']

x= ind.groupby(['City','Year'])['AvgTemperature'].max().reset_index().sort_values(by='AvgTemperature',ascending=False)

x.head(20).style.background_gradient(cmap='Blues')
plt.figure(figsize=(15,8))

sns.lineplot(x='Year',y='AvgTemperature',data=x,hue='City',style='City',markers=['o','*','^','>'])

plt.grid()

plt.title('Mean Temp. Variation Of Cities of India')

plt.show()
mask1=df['Country']=='India'

mask2=df['City']=='Delhi'



ind=df[mask1 & mask2 ]





y= ind.groupby(['Year','City','Month'])['AvgTemperature'].mean().reset_index().sort_values(by='AvgTemperature',ascending=False)

y.head(20).style.background_gradient(cmap='PiYG')
plt.figure(figsize=(15,12))

plt.subplot(2,1,1)

sns.barplot(x='Year',y='AvgTemperature',data=y,palette='hsv_r')

plt.title('Mean Temp. Variation Of Delhi(Yearly)')



plt.subplot(2,1,2)

sns.barplot(x='Month',y='AvgTemperature',data=y,palette='hsv')

plt.title('Mean Temp. Variation Of Delhi(Monthly)')



plt.show()
plt.figure(figsize=(15,8))

sns.lineplot(x='Month',y='AvgTemperature',data=k,hue='City',style='City',markers=['*','o','<','>'])

plt.grid()

plt.title('Mean Temp. Variation Of Cities of India in 2020')

plt.show()
mask1=df['Country']=='India'

mask2=df['Year']==2020



ind=df[mask1 & mask2 ]





k= ind.groupby(['Year','City','Month'])['AvgTemperature'].mean().reset_index().sort_values(by='AvgTemperature',ascending=False)

k.style.background_gradient(cmap='Greens')