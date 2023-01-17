# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # data visualization

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



# reading the terrorism database and selecting only columns of interest 

terror_data = pd.read_csv('../input/globalterrorismdb_0616dist.csv',encoding='ISO-8859-1',usecols=[1,8,10,11,12,27,29,35,58,84,100,103])

terror_data.columns

ax=plt.figure(figsize=(10,22))

sns.set(style="darkgrid")

ax = sns.countplot(x= "iyear", data=terror_data)

locs, labels = plt.xticks()

plt.setp(labels, rotation=80)

sns.plt.show()
ax=plt.figure(figsize=(10,22))

kws = dict(width=.8, color=sns.color_palette('pastel'))

ax = terror_data['country_txt'].value_counts().head(30).plot.barh(**kws)

ax.set_xlabel('No of attacks')

ax.set_ylabel('Countries')

ax.grid(False, axis='x')

plt.show()
ax=plt.figure(figsize=(8,8))

kws = dict(width=.8, color=sns.color_palette('pastel'))

ax = terror_data['attacktype1_txt'].value_counts().plot.barh(**kws)

ax.set_xlabel('No of attacks')

ax.set_ylabel('Types of attacks')

ax.grid(False, axis='x')

plt.show()
pak_terror = terror_data.query('country_txt == "Pakistan" ').reset_index()

pak_terror.columns
ax=plt.figure(figsize=(10,22))

sns.set(style="darkgrid")

ax = sns.countplot(x= "iyear", data=pak_terror)

#ax.set_xticklabels(rotation=30)

locs, labels = plt.xticks()

plt.setp(labels, rotation=80)

sns.plt.show()


pak_terror = terror_data.query('iyear >= 1990 and iyear < 2016 ').reset_index()
ax=plt.figure(figsize=(8,8))

kws = dict(width=.8, color=sns.color_palette('pastel'))

ax = pak_terror['attacktype1_txt'].value_counts().head(10).plot.barh(**kws)

ax.set_xlabel('No of attacks')

ax.set_ylabel('Attack types')

ax.grid(False, axis='x')

#locs, labels = plt.xticks()

#plt.setp(labels, rotation=80)

plt.show()
year_list = pak_terror.iyear.unique()

no_of_kill = []

no_of_wound = []

for ind,year in enumerate(pak_terror.iyear.unique()):

    no_of_kill.append(pak_terror.loc[pak_terror['iyear'] == year, 'nkill'].sum())

    no_of_wound.append(pak_terror.loc[pak_terror['iyear'] == year, 'nwound'].sum())





df = pd.DataFrame({'killed':no_of_kill, 'wounded': no_of_wound})



ax = plt.plot(year_list, df.killed,'bo',linestyle="-",alpha = 0.8,lw = 3.0)

ax = plt.plot(year_list, df.wounded,"g^", linestyle="--",alpha = 0.5,lw = 2.0 )

plt.legend(['Killed', 'Wounded'], loc='upper left')

locs, labels = plt.xticks()

plt.setp(labels, rotation=80)

plt.show()
ax=plt.figure(figsize=(8,10))

kws = dict(width=.8, color=sns.color_palette('pastel'))

ax = pak_terror['gname'].value_counts().head(10).plot.barh(**kws)

ax.set_xlabel('No of attacks')

ax.set_ylabel('gname')

ax.grid(False, axis='x')

plt.show()
sns.set_color_codes("dark")

sns.barplot(x="iyear", y="suicide", data=pak_terror,

            label="Total")

locs, labels = plt.xticks()

plt.setp(labels, rotation=80)

sns.plt.show()