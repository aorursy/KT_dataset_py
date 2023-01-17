# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("/kaggle/input/fifa-20-complete-player-dataset/fifa20_data.csv")
data.columns
data[:10]
counts_preferred_foot = data["foot"].value_counts()

counts_preferred_foot = counts_preferred_foot.reset_index()

counts_preferred_foot.columns = ["PreferredFoot","Count"]

print(counts_preferred_foot)
pt=sns.catplot(y="Count",x="PreferredFoot",data=counts_preferred_foot,palette="RdBu",height=7,aspect=2,kind="bar")
sns.catplot(x="foot", y="Overall", data=data,aspect=2, kind="bar");
counts_Nationality = data["Country"].value_counts()

counts_Nationality = counts_Nationality.reset_index()

counts_Nationality.columns= ["Nations","Counts"]
counts_Nationality
data.Club.unique().shape
data.Age.value_counts()
avgwageoverall = data.groupby("Club", as_index=False)["Potential"].mean()

avgwageoverall.sort_values(by="Potential",inplace=True,ascending=False)

avgwageoverall.head()
top10byoverall = avgwageoverall.iloc[0:10,:]

top10byoverall
sns.catplot(y="Club",x="Potential",data=top10byoverall,height=6,kind="bar",aspect=2,palette="RdBu")
sns.catplot(y="Nations",x="Counts",data=counts_Nationality,height=20,kind="bar")
data.BP
sns.catplot(y="BP",x="PAC",data=data,height=6,kind="bar",aspect=2)
data.Age.value_counts()
sns.lmplot(x="Age", y="PHY",data=data,markers="*",order=2, ci=None, scatter_kws={"color": "green"},line_kws={"linewidth":3,"color":"red"},aspect=2);
data.Penalties.value_counts()
sns.lmplot(x="Finishing", y="Penalties",data=data,markers="*",order=2, ci=None, scatter_kws={"color": "green"},line_kws={"linewidth":3,"color":"red"},aspect=2);
data.isnull().sum()
data.Position

data.Position.unique()
data.Weight
def extract_value_from(value):

  out = value.replace('lbs', '')

  return float(out)

data['Weight'] = data['Weight'].apply(lambda x : extract_value_from(x))

data['Weight'].head()
data.Skill
plt.figure(figsize = (10, 8))

ax = sns.countplot(x = 'Skill', data = data, palette = 'pastel')

ax.set_title(label = 'Count of players on Basis of their skill moves', fontsize = 20)

ax.set_xlabel(xlabel = 'Number of Skill Moves', fontsize = 16)

ax.set_ylabel(ylabel = 'Count', fontsize = 16)

plt.show()
plt.figure(figsize = (13, 8))

ax = sns.countplot(x = 'Height', data = data, palette = 'dark')

ax.set_title(label = 'Count of players on Basis of Height', fontsize = 20)

ax.set_xlabel(xlabel = 'Height in Foot per inch', fontsize = 16)

ax.set_ylabel(ylabel = 'Count', fontsize = 16)

plt.show()
data.iloc[data.groupby(data['Position'])['Potential'].idxmax()][['Position', 'Name', 'Age', 'Club', 'Country']]
youngest = data.sort_values('Age', ascending = True)[['Name', 'Age', 'Club', 'Country']].head(15)

print(youngest)
eldest = data.sort_values('Age', ascending = False)[['Name', 'Age', 'Club', 'Country']].head(10)

print(eldest)
data.groupby(data['Club'])['Country'].nunique().sort_values(ascending = False).head(10)