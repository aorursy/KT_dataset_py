# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
zomato_data = pd.read_csv('../input/zomato.csv',encoding="ISO-8859-1")
country = pd.read_excel('../input/Country-Code.xlsx')
zomato_data.head()
country.head()
data = pd.merge(zomato_data, country, on='Country Code')
data.head()
Indian_rest = data[data.Country == 'India']
len(Indian_rest)

f, ax = plt.subplots(1,1, figsize = (15, 4))
ax = sns.countplot(Indian_rest[Indian_rest['Aggregate rating'] != 0]['Aggregate rating'])
total_cuisines = Indian_rest.Cuisines.value_counts()
cuisines = {}
cnt = 0
for i in total_cuisines.index:
    for j in i.split(', '):
        if j not in cuisines.keys():
            cuisines[j] = total_cuisines[cnt]
        else:
            cuisines[j] += total_cuisines[cnt]
    cnt += 1
sorted_cuisines = pd.Series(cuisines).sort_values(ascending=False)
sns.set(style="white")
f, g = plt.subplots(1,1, figsize = (15, 4))
g = sns.barplot(sorted_cuisines[:15].index, sorted_cuisines[:15].values, palette="inferno")
sns.set(style="white")
f, g = plt.subplots(1,1, figsize = (18, 4))
Indian_rest.Locality.value_counts()[:15].plot(kind='bar', rot=35)
print(Indian_rest['Has Online delivery'].value_counts())
sns.countplot(Indian_rest['Has Online delivery'])
Indian_rest.columns
Indian_rest['Rating color'].unique()
Indian_rest.head(5)
f, g = plt.subplots(1,1, figsize = (15, 4))
g = Indian_rest.agg('City').value_counts().sort_values()[:10].plot(kind='barh', rot=45)
f, g = plt.subplots(1,1, figsize = (10, 10))
g = sns.boxplot(x="Has Online delivery", y="Average Cost for two", data=Indian_rest)
f, g = plt.subplots(1,1, figsize = (10, 10))
g = sns.boxplot(x="Has Online delivery", y="Average Cost for two", data=Indian_rest)
g = sns.stripplot(x="Has Online delivery", y="Average Cost for two", jitter=True, data=Indian_rest, edgecolor="gray")
f, g = plt.subplots(1,1, figsize = (10, 5))
g = sns.violinplot(x="Has Online delivery", y="Average Cost for two", data=Indian_rest)
cnt_srs = Indian_rest.groupby('Rating color')['Rating color'].count()
# `cnt_srs
colors = ['Dark Green', 'Green', 'Orange', 'Red', 'White', 'Yellow']
total = cnt_srs.sum()

for i in colors:
    print('% of color {} is {}%'.format(i, round((cnt_srs[i]/total)* 100, 2)))