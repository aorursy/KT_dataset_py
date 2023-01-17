import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



import os

files  = os.listdir("../input")

print(files)



# Any results you write to the current directory are saved as output.
data = []

for i, file in enumerate(files):

    data.append(pd.read_csv('../input/'+file))
data[0].head()
data[1].head()
data[1].head()
production = pd.read_csv('../input/'+'datafile (2).csv')

production.head()
production.columns
k = production[['Crop             ','Production 2006-07', 'Production 2007-08',

       'Production 2008-09', 'Production 2009-10', 'Production 2010-11']].groupby('Crop             ')

index = list(k.indices.keys())

index[-8:-2]
k.sum()[-9:-2].plot(figsize=(20,8), kind='bar');

plt.title('Year wise production different agricultural products')

plt.ylabel('Production in Quintal')
k.sum()[:-9].plot(figsize=(20,8), kind='bar');

plt.title('Year wise production of agricultural crop')

plt.ylabel('Production in Quintal');
# k.mean().plot(figsize=(12,6), kind='bar');

# plt.figure(figsize=(12,6))

l = len(k['Crop             '])

fig, arraxes = plt.subplots(1,4, figsize=(12,12), sharey=True)

plt.setp(arraxes, yticks=range(len(index)), yticklabels = index)



for axes, p in zip(arraxes.flat,['Production 2006-07', 'Production 2007-08','Production 2008-09', 'Production 2009-10', 'Production 2010-11']):

    axes.barh(range(l), k[p].head())

    axes.set_title(p)

#     axes.tick_params(axis='x',  rotation=90)

#     axes.set_xticklabels(index)

fig.set_figwidth(20)
kc = production[['Crop             ','Area 2006-07', 'Area 2007-08', 'Area 2008-09', 'Area 2009-10',

       'Area 2010-11']].groupby('Crop             ')

kc.sum().plot(figsize=(12,8), kind='barh', stacked= True);



# k.head()['Crop             '].values
data[2].describe()
data[3].groupby('State').sum()
cols = data[3].columns
data[3].groupby('Crop')[cols[:-1]].sum().plot(kind='bar', figsize=(12,6));
data[3].groupby('State')[cols[:-1]].sum().plot(kind='bar', figsize=(12,6));
plt.title('State-wise '+cols[-1], color='red', fontsize=20)

data[3].groupby('State')[cols[-1]].sum().plot(kind='pie', figsize=(12,12));
plt.title('Crop-wise '+cols[-1], color='red', fontsize=20)

data[3].groupby('Crop')[cols[-1]].sum().plot(kind='pie', figsize=(12,12));
data[4].head()
plt.title('No of varieties per crop in India', fontsize=40, color='orange')

data[4]['Crop'].value_counts().plot(kind='bar', figsize= (12,12));