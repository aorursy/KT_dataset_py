%matplotlib inline

import pandas as pd

from pandas import *

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

import plotly.offline as py

import plotly.figure_factory as ff

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

warnings.filterwarnings('ignore')

from IPython.display import display

death = pd.read_csv('../input/celebrity_deaths_4.csv',sep=None,engine='python')

death.head(5)
death.describe()
death.std()#standart deviation
print(death.isnull().any()) #we have some null's values
print(death.isnull().sum())

print(death.info())#as u can see we dont know cause of death of 12484 -its pretty much myst
death['cause_of_death'].fillna('i_dont_know',inplace=True) #lets reaplace null values for 'i_dont_know' value!

death['famous_for'].fillna('i_dont_know',inplace=True)

death.head(5)
sns.countplot(death.death_year)

plt.title("Number of Celebrities death every year")
a=death[(death['cause_of_death'].str.contains("NaN")==False) & (death['cause_of_death'].str.contains("murdered")==True)]#here is a list of celebrity who was murdered.Not too much

display(a) #i want to follow dzen of python,so for those who didnt replace null's - we just False them.
b=death[(death['cause_of_death'].str.contains("NaN")==False) & (death['cause_of_death'].str.contains("natural causes")==True)]#here is a list of celebrity who was murdered.Not too much

b.sort_values(['age','birth_year'], ascending=[False,True]).head(5)
f=death[(death['cause_of_death'].str.contains("NaN")==False) & (death['cause_of_death'].str.contains("accident")==True)]#here is a list of celebrity who was murdered.Not too much

f.head(10)
type_death=death[death['death_month']!='i_dont_know']

type_death=type_death['death_month'].value_counts()[:10]

data = [go.Bar(

            x=type_death.index,

            y=type_death.values,

        marker = dict(

        color = 'rgba(110, 120, 30,1)',)

            

)]



py.iplot(data, filename='horizontal-bar')# a lot of deaths in January
type_death=death[death['cause_of_death']!='i_dont_know']

type_death=type_death['cause_of_death'].value_counts()[:10]

data = [go.Bar(

            x=type_death.index,

            y=type_death.values,

        marker = dict(

        color = 'rgba(110, 120, 30,1)',)

            

)]



py.iplot(data, filename='horizontal-bar')
##i forgot abt fame score.We dont need it

death.drop('fame_score',axis=1,inplace=True)
death_cancer=death[death['cause_of_death'].str.contains('cancer')]

death_cancer=death_cancer.groupby(['cause_of_death'])['death_year'].count().reset_index()

death_cancer=death_cancer.sort_values(by='death_year',ascending=False)[1:15]

sns.barplot(x='death_year',y='cause_of_death',data=death_cancer,palette='RdYlGn').set_title('Top Types Of Death Causing Cancer')

plt.xlabel('Total Deaths')

plt.ylabel('Type of cancer')

fig=plt.gcf()

fig.set_size_inches(8,6)

plt.show()