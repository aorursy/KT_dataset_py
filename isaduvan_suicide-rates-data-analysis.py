# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/master.csv")

data.head(10)

data.describe()

data.info()
data.corr()
#correlation map

f,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.columns

data.rename(columns={'HDI for year':'HDIforyear'},inplace=True)

data.rename(columns={' gdp_for_year ($) ':'gdp_for_year'},inplace=True)

data.rename(columns={'gdp_per_capita ($)':'gdp_per_capita'},inplace=True)

data.rename(columns={'suicides/100k pop':'suicides/100k_pop'},inplace=True)

data.columns



years = sorted(data.year.unique())

population = []

suicides =[]

for year in sorted(years):

    population.append(data[data['year']==year]['population'].sum())

    suicides.append(data[data['year']==year]['suicides_no'].sum())



plt.subplot(2,1,1)

plt.plot(years,population)

plt.xlabel("Years")

plt.ylabel("Global Population (Billion)")

plt.grid()

plt.subplot(2,1,2)

plt.plot(years,suicides)

plt.xlabel("Years")

plt.ylabel("Suicides per Year")

plt.grid()





plt.plot(years,np.array(suicides)/np.array(population)*100000,'-o')
age_groups = data['age'].unique()

age_groups = sorted(age_groups,key=lambda x: float(x[0:1]))

age_groups.insert(0, age_groups.pop(4))

suicides_age_groups = [[data[data['age']==age]['suicides_no'].sum()] for age in age_groups]

plt.plot(age_groups,suicides_age_groups,"-o")

plt.xlabel("Age Groups")

plt.ylabel("Suicides Number")

plt.grid()



generations = pd.unique(data['generation']) # take all the ex

y_pos = np.arange(len(generations))

gen_suic = [data[data['generation']== gen]['suicides_no'].sum() for gen in generations] # sum over the generations

plt.barh(y_pos,gen_suic, align='center',color="r")

plt.yticks(y_pos,generations)

plt.grid()
male=data[(data['sex']=="male")]

female=data[(data['sex']=="female")]

m_suic=[]

m_pop=[]

fm_suic=[]

fm_pop=[]

for year in years:

    m_suic.append(male[male['year']==year]['suicides_no'].sum())

    fm_suic.append(female[female['year']==year]['suicides_no'].sum())

    m_pop.append(male[male['year']==year]['population'].sum())

    fm_pop.append(female[female['year']==year]['population'].sum())

plt.subplot(2,1,1)

plt.plot(years,m_pop,color="r",label="Male")

plt.plot(years,fm_pop,color="g",label="Female")

plt.ylabel("Population")

plt.legend()

plt.grid()

plt.subplot(2,1,2)

plt.plot(years,m_suic,color="r",label="Male")

plt.plot(years,fm_suic,color="g",label="Female")

plt.ylabel("Suicides Number")

plt.legend()

plt.grid()