# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/master.csv')
df.head()
df.info()
df.groupby('sex').agg({'population':'sum'})
df.groupby('sex').agg({'suicides_no':'sum'})
d = df.groupby(['sex','year']).agg({'suicides/100k pop':'mean'})
d = d.reset_index()
d.head()
female = d.loc[d['sex']=='female',:]
female.head()
male = d.loc[d['sex']=='male',:]
male.head()
import matplotlib.pyplot as plt
female.plot(x='year',y='suicides/100k pop',kind='bar')

plt.xlabel('Year')

plt.ylabel('Suicides/100k pop')

plt.title('Female Sucide rates/100k population')
male.plot(x='year',y='suicides/100k pop',kind='bar')

plt.xlabel('Year')

plt.ylabel('Suicides/100k pop')

plt.title('Male Sucide rates/100k population')
countrywise = df.groupby('country').agg({'suicides/100k pop':'mean'}).sort_values(by='suicides/100k pop')
countrywise.head()
countrywise = countrywise.reset_index()
countrywise.plot(kind='barh')
#Case study of south africa



sa = df[df['country']=='South Africa'].groupby('year').agg({'suicides_no':'sum','population':'mean'})
sa = sa.reset_index()
sa.head()
sa = sa.astype(int)
sa.head()
year1 = sa.iloc[:,0].values
pop1 = sa.iloc[:,2].values
plt.plot(year1,pop1,color='green')
suicide_no1 = sa.iloc[:,1].values
plt.plot(year1,suicide_no1)
plt.subplot(1,2,1)

plt.plot(year1,suicide_no1)

plt.xlabel('Year')

plt.ylabel('Suicides Numbers')

plt.title('Year vs Sucide number in South Africa')



plt.subplot(1,2,2)

plt.plot(pop1,suicide_no1)

plt.xlabel('Population')

plt.ylabel('Suicides Numbers')

plt.title('Population vs Sucide number in South Africa')
globaldata = df.groupby('year').agg({'suicides_no':'sum'})
globaldata.head()
globaldata = globaldata.reset_index()
plt.plot(globaldata.year,globaldata.suicides_no)

plt.title('Year vs Suicides Globally')

plt.xlabel('Year')

plt.ylabel('Suicides')