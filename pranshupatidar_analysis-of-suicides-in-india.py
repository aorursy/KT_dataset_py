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
data = pd.read_csv('../input/suicides-in-india/Suicides in India 2001-2012.csv')
data.head(10)
data.info()
data2 = data.copy()
grp = data2.groupby('State')['Total'].sum()

total_suicides = pd.DataFrame(grp).reset_index().sort_values('Total',ascending=False)

total_suicides = total_suicides[2:]
import matplotlib.pyplot as plt

import seaborn as sns

fig , ax = plt.subplots(figsize=(18,6))

g=sns.barplot(x='State',y='Total',data=total_suicides)

g.set_xticklabels(g.get_xticklabels(),rotation=45)
x = data.groupby(['State','Year'])['Total'].sum()

y = pd.DataFrame(x).reset_index()

y = y.pivot(index='State',columns='Year')

y['sum'] = y.sum(axis=1)

yearly_total = y.sum(axis=0)

y = y.sort_values('sum',ascending=False)

y = y[2:14]

y = y/10

y = y.drop('sum',axis=1)

y
plt.figure(figsize=(8,8))

sns.heatmap(y,linewidth=1,cmap='OrRd',square=True)
data['Type_code'].value_counts()
data3 = data[data['Type_code']=='Causes']

reasons = data3.groupby('Type')['Total'].sum()

suicide_reasons = pd.DataFrame(reasons).reset_index().sort_values('Total',ascending=False)

suicide_reasons = suicide_reasons[:15]

plt.figure(figsize=(18,6))

g2 = sns.barplot(y='Type',x='Total',data=suicide_reasons)
data3 = data[data['Type_code']=='Means_adopted']

reasons = data3.groupby('Type')['Total'].sum()

suicide_reasons = pd.DataFrame(reasons).reset_index().sort_values('Total',ascending=False)

suicide_reasons = suicide_reasons[:15]

plt.figure(figsize=(18,6))

g2 = sns.barplot(y='Type',x='Total',data=suicide_reasons)

data3 = data[data['Type_code']=='Professional_Profile']

reasons = data3.groupby('Type')['Total'].sum()

suicide_reasons = pd.DataFrame(reasons).reset_index().sort_values('Total',ascending=False)

suicide_reasons = suicide_reasons[:15]

plt.figure(figsize=(18,6))

g2 = sns.barplot(y='Type',x='Total',data=suicide_reasons)
data3 = data[data['Type_code']=='Education_Status']

reasons = data3.groupby('Type')['Total'].sum()

suicide_reasons = pd.DataFrame(reasons).reset_index().sort_values('Total',ascending=False)

#suicide_reasons = suicide_reasons[:15]

plt.figure(figsize=(18,6))

g2 = sns.barplot(y='Type',x='Total',data=suicide_reasons)
data3 = data[data['Type_code']=='Social_Status']

reasons = data3.groupby('Type')['Total'].sum()

suicide_reasons = pd.DataFrame(reasons).reset_index().sort_values('Total',ascending=False)

#suicide_reasons = suicide_reasons[:15]

plt.figure(figsize=(18,6))

g2 = sns.barplot(y='Type',x='Total',data=suicide_reasons)
age_grp = data.groupby('Age_group')['Total'].sum()

age = pd.DataFrame(age_grp).reset_index()

age = age[1:]

age
plt.subplots(figsize=(5,5))

g = sns.barplot(x='Age_group',y='Total',data=age)
yearly = pd.DataFrame(yearly_total).reset_index()[:-1].drop('level_0',axis=1)

yearly.columns = ['Year','No of suicides']

plt.figure(figsize=(10,5))

sns.lineplot(x='Year',y='No of suicides',data=yearly)

student = data[data['Type']=='Student']

student
age_grp = student.groupby('Age_group')['Total'].sum()

age = pd.DataFrame(age_grp).reset_index()

age
plt.subplots(figsize=(5,5))

g = sns.barplot(x='Age_group',y='Total',data=age)
std_year = student.groupby('Year')['Total'].sum()

stdyr = pd.DataFrame(std_year).reset_index()

plt.figure(figsize=(12,6))

sns.lineplot(x='Year',y='Total',data=stdyr)
grp = student.groupby('State')['Total'].sum()

total_suicides = pd.DataFrame(grp).reset_index().sort_values('Total',ascending=False)[:10]

plt.figure(figsize=(15,6))

sns.barplot(y='State',x='Total',data=total_suicides)
gen = student.groupby('Gender')['Total'].sum()

gender = pd.DataFrame(gen).reset_index()

gender
piex = ['Female','Male']

piey = pd.Series(gender['Total'])

fig1, ax1 = plt.subplots()

ax1.pie(piey,labels=piex)

plt.show()