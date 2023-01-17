import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import os

print(os.listdir("../input"))

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Any results you write to the current directory are saved as output.
#Reading CSV file into dataframe

df = pd.read_csv('../input/master.csv')
#Exploring how the data is in the dataset

df.head(10)
#Exploring Dataset

df.describe()
#Exploring Dataset

df.info()
s_sum = pd.DataFrame(df['suicides_no'])

s_sum = s_sum.groupby(df['year']).sum()

s_sum = s_sum.reset_index().sort_values(by='suicides_no',ascending=False)

fig = plt.figure(figsize=(40,15))

sns.barplot(x='year',y='suicides_no',data=s_sum,color='cyan')

plt.title('Count Of Suicides Yearly',fontsize=30)

plt.xlabel("Year",fontsize=30)

plt.ylabel("No. of Suicides",fontsize=30)

plt.tight_layout()
plt.rcParams['figure.figsize'] = (15,5)

sns.distplot(df.suicides_no,kde=False,hist_kws={'log':True})

plt.xlabel("No. of suicides")

plt.show()

sns.distplot(df['suicides/100k pop'],kde=False,hist_kws={'log':True})

plt.xlabel("Suicides per 100k Population")

plt.show()

sns.distplot(df.population,kde=False,hist_kws={'log':True})

plt.xlabel("Population")

plt.show()

sns.distplot(df['gdp_per_capita ($)'],kde=False,hist_kws={'log':True})

plt.xlabel("GDP per capita")

plt.show()
unique_country = list(df.country.unique())

x_plot = df[df['sex'] == 'male']

x_plot = x_plot[x_plot['age'] == '15-24 years']

x_plot = x_plot[x_plot['country'] == 'Albania']

x_plot = x_plot.drop('HDI for year',axis = 1)

x_plot

y_plot = df[df['sex'] == 'female']

y_plot = y_plot[y_plot['age'] == '15-24 years']

y_plot = y_plot[y_plot['country'] == 'Albania']
plt.plot(x_plot.year,x_plot.population,'blue')

plt.plot(y_plot.year,y_plot.population,'red')

plt.xlabel('Year')

plt.legend(['Male','Female'])

plt.ylabel('Population')

plt.title('Male and Female Population of age 15-24 Year in Albania')

plt.show()
plt.plot(x_plot.year,x_plot.suicides_no,'red')

plt.plot(y_plot.year,y_plot.suicides_no,'blue')

plt.xlabel('Year')

plt.legend(['Male','Female'])

plt.yticks(np.arange(0,35,2))

plt.ylabel('Population')

plt.title('Suicide number of males and females of age 15-24 year in Albania')

plt.show()
# x_plot.sort_values(' gdp_for_year ($) ')

plt.plot(x_plot.year,x_plot[' gdp_for_year ($) '],'red')

plt.xlabel('Year')

# plt.xlim(1985,2015)

plt.xticks(x_plot.year.unique())

plt.ylabel('GDP')

plt.title('GDP of Albania ')

plt.show()