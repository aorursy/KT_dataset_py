# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px # data visualization tool

from matplotlib import pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/data-police-shootings/fatal-police-shootings-data.csv')

df.head()
df.info()
df.shape
df = df.drop('id', 1)

df['month'] = pd.to_datetime(df['date']).dt.month

df['year'] = pd.to_datetime(df['date']).dt.year
uncomplete_data = df.isna().sum()*100/df.shape[0]

print(uncomplete_data)

df.dropna(inplace=True)
cardinality ={}

for col in df.columns:

    cardinality[col] = df[col].nunique()

#print(cardinality)

d = {'A':'Asian','B':'Black','H':'Hispanic', 'N':'Native','O':'Other','W':'White'}

df['race'] = df['race'].replace(d)
yearly_shootouts = df['year'].value_counts()

yearly_shootouts = pd.DataFrame(yearly_shootouts)

yearly_shootouts= yearly_shootouts.reset_index()

yearly_shootouts=yearly_shootouts.rename(columns={'index':'year','year':'Shootouts'})

fig = px.bar(yearly_shootouts, y='Shootouts', x='year', barmode='group')

fig.show()
#yearly_population = {'2015':320.7,'2016':323.1,'2017':325.1,'2018':327.2 ,'2019':328.2,'2020':331.0}



yearly_population_array = [320.7,323.1,325.1,327.2 ,328.2,331.0]

yearly_shootouts['Shootouts_normalized'] = yearly_shootouts['Shootouts'].div(yearly_population_array).multiply(yearly_population_array[-1])

fig = px.bar(yearly_shootouts, y='Shootouts_normalized', x='year', barmode='group')

fig.show()
import plotly.figure_factory as ff

for etn in df['race'].unique():

    x = df[df['race']==etn]['age']

    hist_data = [x]

    group_labels = ['Age_'+str(etn)]

    fig = ff.create_distplot(hist_data, group_labels)

    fig.show()
bc_shootout = df['body_camera']

print(bc_shootout.value_counts())

print(bc_shootout.value_counts()[0]/bc_shootout.value_counts()[1])

fig = px.histogram(bc_shootout,x='body_camera',color='body_camera')

fig.show()
for etn in df['race'].unique():

    gender_per_etn = df[df['race']==etn]['gender']

    fig = px.histogram(gender_per_etn,x='gender',color='gender',labels={'gender':'gender_'+str(etn)})

    fig.show()   

gender_shootout = df['gender']

print(gender_shootout.value_counts())

print(gender_shootout.value_counts()[0]/gender_shootout.value_counts()[1])
etn_shootout = df['race']

print(etn_shootout.value_counts())

fig = px.histogram(etn_shootout,x='race',color='race')

fig.show()
etn_population_array = [61.5,12.7,17.3, 5.3,0.9,2.5] # unelegant - to correct later

etn_shootouts_norm = pd.DataFrame()

etn_shootouts_norm['etn_shootouts_norm'] = df['race'].value_counts().divide(etn_population_array)



print(etn_shootouts_norm.head())

fig = px.bar(etn_shootouts_norm,y='etn_shootouts_norm',color='etn_shootouts_norm', barmode='group')

fig.show()
innocent_people = df[(df.signs_of_mental_illness==False) & (df.armed =='unarmed') & (df.flee=='Not fleeing')]

lista= pd.DataFrame()

lista['count'] = innocent_people['race'].value_counts().divide(df['race'].value_counts())



fig = px.bar(lista, y='count', barmode='group')

fig.show()