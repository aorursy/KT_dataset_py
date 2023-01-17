# load modules

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



import warnings

warnings.filterwarnings("ignore")
# load data

train = pd.read_csv('../input/train.csv')

train.head(3)
# missing values

train.isnull().sum()
# rename a column

train.rename(columns={"Event type": "Event_type"}, inplace=True)
train.head(2)
print('County with Highest Terrorist Attacks:',train['County'].value_counts().index[0])

print('Region1 with Highest Terrorist Attacks:',train['Area1'].value_counts().index[0])

print('Region2 with Highest Terrorist Attacks:',train['Area2'].value_counts().index[0])

print('Region3 with Highest Terrorist Attacks:',train['Area3'].value_counts().index[0])

print('Maximum people killed in an attack are:',train['fatalities'].max(),'that took place in',train.loc[train['fatalities'].idxmax()].County)
# Terror attacks over the years

plt.subplots(figsize=(15,6))

sns.countplot('year',data=train,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))

plt.xticks(rotation=90)

plt.title('Number of Terrorist Activities Each Year')

plt.show()
# the perpetrators of terror attacks (ACTOR1)

plt.subplots(figsize=(15,6))

sns.countplot('Actor1',data=train,palette='inferno',order=train['Actor1'].value_counts().index)

plt.xticks(rotation=90)

plt.title('Actor1 - The Perpetrators')

plt.show()
# the casualities of terror attacks (Actor2)

plt.subplots(figsize=(15,6))

sns.countplot('Actor2', data=train,palette='inferno',order=train['Actor2'].value_counts().index)

plt.xticks(rotation=90)

plt.title('Actor2 - The Targets')

plt.show()
# the Event type

plt.subplots(figsize=(15,6))

sns.countplot(train['Event_type'],palette='inferno',order=train['Event_type'].value_counts().index)

plt.xticks(rotation=90)

plt.title('Event type')

plt.show()
# terrorist attacks by Area1

plt.subplots(figsize=(15,6))

sns.countplot('Area1',data=train,palette='RdYlGn',edgecolor=sns.color_palette('dark',7),order=train['Area1'].value_counts().index)

plt.xticks(rotation=90)

plt.title('Number of Terrorist Activities by Area1')

plt.show()
# terrorist attacks by Area2

plt.subplots(figsize=(15,6))

sns.countplot('Area2',data=train,palette='RdYlGn',edgecolor=sns.color_palette('dark',7),order=train['Area2'].value_counts().index)

plt.xticks(rotation=90)

plt.title('Number of terrorist activities by Area 2')

plt.show()
# # terrorist attacks by Area3

plt.subplots(figsize=(15,6))

sns.countplot('Area3', data=train,palette='RdYlGn',edgecolor=sns.color_palette('dark',7),order=train['Area3'].value_counts().index)

plt.xticks(rotation=90)

plt.title('Number of Terrorist Activities By Area3')

plt.show()
# trend in terror activites

train_county = pd.crosstab(train.year,train.County)

train_county.plot(color=sns.color_palette('Set1',12))

fig = plt.gcf()

fig.set_size_inches(18,6)

plt.show()
# Attack Type vs Region

pd.crosstab(train.Area1,train['Event_type']).plot.barh(stacked=True,width=1,color=sns.color_palette('RdYlGn',9))

fig = plt.gcf()

fig.set_size_inches(12,8)

plt.show()
# trend in terror activites

train_Event_type = pd.crosstab(train.year,train.Event_type)

train_Event_type.plot(color=sns.color_palette('Set3',12))

fig = plt.gcf()

fig.set_size_inches(18,6)

plt.show()