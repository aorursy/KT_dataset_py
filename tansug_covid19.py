# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
covid19countries = pd.read_csv('/kaggle/input/baslik/covid19ut.csv')

yearmillionusd = pd.read_csv('/kaggle/input/yearlymil/yearlymilusd.csv')
yearmillionusd.fillna(0, inplace = True) 

fill=yearmillionusd.drop(columns="2019 Current")
fill.head(122)

ulkeler=fill['Country']
##print(ulkeler)
col=fill.loc[:,"2000":"2019"]

fill['MeanIncome'] = col.mean(axis=1)

df = fill[['Country','MeanIncome']].sort_values('MeanIncome', ascending = True)


df1=covid19countries.groupby("countriesAndTerritories").deaths.sum().reset_index().sort_values("deaths", ascending=False)
df1=df1[df1['deaths'] > 10]

##df1=df1.head(55)
##df=df.head(55)

plt.figure(figsize=(20,10))

plt.subplot(2,2,1)   
plt.plot(df.Country,df.MeanIncome,color="r") 
plt.xlabel("Countries")
plt.ylabel("Incomes Million USD")
plt.title("Countries incomes")

plt.subplot(2,2,2)
plt.plot(df1.countriesAndTerritories,df1.deaths,color="blue")
plt.xlabel("Countries")
plt.ylabel("Deaths")
plt.title("Countries Covid19 Deaths")

plt.show()


ax = df.tail(20).plot.barh(x='Country',y='MeanIncome', rot=0)
ax2 = df1.head(25).plot.barh(x='countriesAndTerritories', y='deaths', rot=0)

ax.plot
ax2.plot

df['situation'] = ["rich" if i>4000 else "poor" for i in df['MeanIncome']]
df.head(100)

df1.boxplot(column='deaths')
df1 = df1.sort_values(by=['Country'])
df = df.sort_values(by=['Country'])
                        
ulkeler=list(df1.Country)
olumler=list(df1.deaths)
silahulke=list(df.Country)
silahparasi=list(df.MeanIncome)


veriler=[ulkeler,olumler,silahulke,silahparasi]
etiketler=['Ülkeler','ölümler','silaha para harcayan ülkeler','silahlar']


df3=df1.merge(df1,on='Country').merge(df,on='Country')
df3 = df3.sort_values(by=['deaths_x'])
df3

