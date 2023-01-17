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
import matplotlib.pyplot as plt

import seaborn as sns
df_us = pd.read_csv(os.path.join(dirname, filename))
df_us
def basic_info(data):

    print(data.shape)

    print(data.size)

    print(data.info())

    cat, num = list(), list()

    for i in data.columns:

        if data[i].dtype == object:

            cat.append(i)

        else:

            num.append(i)

    print(cat, "\n")

    print(num)

    return cat, num
categorical, numerical = basic_info(df_us)
df_us2 = df_us.sort_values(by = ['id'])
df_us2
df_us2['date'] = pd.to_datetime(df_us2['date'])
plt.figure(figsize=(20,8))

sns.scatterplot(df_us2['id'][:250], df_us2['age'][:250], hue= df_us2['gender'])
df_us2.isnull().sum()
df_us2['manner_of_death'].value_counts()
labels = df_us2['manner_of_death'].value_counts().index.tolist()

sizes = df_us2['manner_of_death'].value_counts()

explot = (0, 0.5)

fig, ax = plt.subplots()

ax.pie(sizes, labels = labels, explode = explot, autopct = "%1.1f%%", shadow=True, startangle=90)

ax.axis('equal')

plt.show()
df_us2['armed'].value_counts()
plt.figure(figsize=(50,9))

sns.countplot(df_us2['armed'])

plt.xticks(rotation = -45)

plt.show()
plt.figure(figsize=(10,8))

plt.hist(df_us2['age'], edgecolor = "#7FFF00")

plt.show()
df_us2['gender'].value_counts()
labels = df_us2['gender'].value_counts().index.tolist()

sizes = df_us2['gender'].value_counts()

explot = (0, 0.5)

fig, ax = plt.subplots()

ax.pie(sizes, labels = labels, explode = explot, autopct = "%1.1f%%", shadow=True, startangle=90)

ax.axis('equal')

plt.show()
sns.countplot(df_us2['gender'], hue=df_us2['manner_of_death'])
df_us2['race'].value_counts()
fig, ax = plt.subplots(figsize=(20,10))

graph = sns.countplot(ax = ax, x = 'race', data = df_us2)

i=0

for p in graph.patches:

    print(p)

    height = p.get_height()

    graph.text(p.get_x()+p.get_width()/2., height + 0.1,

        df_us2['race'].value_counts()[i],ha="center")

    i += 1
fig, ax = plt.subplots(figsize = (10,8))

#sns.countplot(df_us2['race'], hue= df_us2['gender'])

graph = sns.countplot(df_us2['race'], hue = df_us2['gender'])

for p in graph.patches:

    for p in graph.patches:

        height = p.get_height()

        graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")
plt.figure(figsize=(10,8))

graph = sns.countplot(df_us2['race'], hue = df_us2['manner_of_death'])

for p in graph.patches:

    for p in graph.patches:

        height = p.get_height()

        graph.text(p.get_x()+p.get_width()/2., height + 0.3,height ,ha="center")
df_us2['state'].value_counts()
plt.figure(figsize=(40,9))

sns.countplot(df_us2['state'])

plt.xticks(rotation = -45)

plt.show()
df_us2['signs_of_mental_illness'].value_counts()
labels = df_us2['signs_of_mental_illness'].value_counts().index.tolist()

sizes = df_us2['signs_of_mental_illness'].value_counts()

explot = (0, 0.10)

fig, ax = plt.subplots()

ax.pie(sizes, labels = labels, explode = explot, autopct = "%1.1f%%", shadow=True, startangle=90)

ax.axis('equal')

plt.show()
plt.figure(figsize=(10,8))

graph = sns.countplot(df_us2['signs_of_mental_illness'], hue = df_us2['gender'])

for p in graph.patches:

    for p in graph.patches:

        height = p.get_height()

        graph.text(p.get_x()+p.get_width()/2., height + 0.3,height ,ha="center")
plt.figure(figsize=(10,7))

graph = sns.countplot(df_us2['signs_of_mental_illness'], hue = df_us2['race'])

for p in graph.patches:

    for p in graph.patches:

        height = p.get_height()

        graph.text(p.get_x()+p.get_width()/2., height + 0.3,height ,ha="center")
df_us2['threat_level'].value_counts()
fig, ax = plt.subplots(figsize=(10,6))

countplot = sns.countplot(ax = ax, x = 'threat_level', data = df_us2)

i = 0

for p in countplot.patches:

    height = p.get_height()

    countplot.text(p.get_x() + p.get_width()/2.0, height + 0.1, height, ha = 'center')

    i +=1
plt.figure(figsize=(10,8))

countplot = sns.countplot(df_us2['threat_level'], hue = df_us2['gender'])

i = 0

for p in countplot.patches:

    height = p.get_height()

    countplot.text(p.get_x() + p.get_width()/2.0, height + 0.1, height, ha = 'center')

    i +=1
df_us2['flee'].value_counts()
labels = df_us2['flee'].value_counts().index.tolist()

sizes = df_us2['flee'].value_counts()



fig, ax=plt.subplots()

patches, texts = ax.pie(sizes,shadow=True, startangle=90)



labels = ['{0} - {1:1.2f}'.format(i,j) for i, j in zip(labels,sizes)]

sort_legend = False



plt.legend(patches, labels, loc= 'best', bbox_to_anchor=(-0.1, 1.), fontsize = 10)

ax.axis('equal')

fig = plt.gcf()

plt.show()
plt.figure(figsize=(10,8))

countplot = sns.countplot(df_us2['flee'], hue = df_us2['gender'])

i = 0

for p in countplot.patches:

    height = p.get_height()

    countplot.text(p.get_x() + p.get_width()/2.0, height + 0.1, height, ha = 'center')

    i +=1
df_us2['body_camera'].value_counts()
labels = df_us2['body_camera'].value_counts().index.tolist()

series = df_us2['body_camera'].value_counts()



plt.pie(series, labels = labels, autopct = "%1.1f%%", shadow=True, startangle=90)

plt.axis('equal')

plt.show()
df_us2['arms_category'].value_counts()
plt.figure(figsize=(10,8))

countplot = sns.countplot(df_us2['arms_category'])

i = 0

for p in countplot.patches:

    height = p.get_height()

    countplot.text(p.get_x() + p.get_width()/2.0, height + 0.1, height, ha = 'center')

    i +=1

plt.xticks(rotation = -90)

plt.show()
plt.figure(figsize=(15, 9))

countplot = sns.countplot(df_us2['arms_category'], hue = df_us2['gender'])

i = 0

for p in countplot.patches:

    height = p.get_height()

    countplot.text(p.get_x() + p.get_width()/2.0, height + 0.1, height, ha = 'center')

    i +=1

plt.xticks(rotation = -90)

plt.legend(loc = 'upper right')

plt.show()
plt.figure(figsize=(30, 9))

countplot = sns.countplot(df_us2['arms_category'], hue = df_us2['race'])

i = 0

for p in countplot.patches:

    height = p.get_height()

    countplot.text(p.get_x() + p.get_width()/2.0, height + 0.1, height, ha = 'center')

    i +=1

plt.xticks(rotation = -90)

plt.legend(loc = 'upper right')

plt.show()