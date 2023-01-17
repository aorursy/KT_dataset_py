# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from pprint import pprint

import random



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/others/MBA_ADMISSIONS.csv')

df.head()
print("Columns names")

print('-'*75)

print(df.columns)

print()

print("Dataset Information")

print('-'*75)

print(df.info())

print()

print('-'*75)

pprint(df.describe())
categories = []

numerical = []



for cols in df.columns:

    if df[cols].dtype == 'object':

        categories.append(cols)

    else:

        numerical.append(cols)

print("categories columns : ", categories)

print()

print("numerical columns: ", numerical)
for items in categories:

    print(items, ":", df[items].unique())
f, ax = plt.subplots(4,2, figsize=(25, 25))



for items, subplot in zip(categories, ax.flatten()):

    sns.countplot(x=items, data=df, ax=subplot, palette = 'hsv').set_title('Distribution of {}'.format(items))

plt.show()

plt.tight_layout()

f, ax = plt.subplots(4,2, figsize=(25, 25))

x = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]



for items, subplot in zip(categories, ax.flatten()):

    labels = df[items].unique()

    size = df[items].value_counts()

    explode = x[0:len(labels)]

    subplot.pie(size, labels = labels,explode = explode,shadow = True, autopct = '%.2f%%')

    subplot.text(-0.4, 1.3, items, fontsize = 20)

    subplot.legend(loc=3)

plt.show()

plt.tight_layout()

plt.axis('off')
f, ax = plt.subplots(4,2, figsize=(25, 25))

color = ['g','b','r','y']

sns.set(style = 'whitegrid')

for items, subplot in zip(numerical, ax.flatten()):

    c = color[random.randint(0,3)]

    sns.distplot(df[items], norm_hist=False, kde=True, rug=False,color=c, hist_kws={"alpha": 1},ax=subplot).set_title('Distribution of {}'.format(items))

plt.show()

plt.tight_layout()

# More Bins for scrutiny



f, ax = plt.subplots(4,2, figsize=(25, 25))

color = ['g','b','r','y']

sns.set(style = 'whitegrid')

for items, subplot in zip(numerical, ax.flatten()):

    c = color[random.randint(0,3)]

    sns.distplot(df[items], norm_hist=False, kde=True, rug=False,color=c, hist_kws={"alpha": 1}, bins=100, ax=subplot).set_title('Distribution of {}'.format(items))

    plt.tight_layout()



plt.show()
f, ax = plt.subplots(4,2, figsize=(25, 25))

sns.set_style("white")



for items, subplot in zip(numerical, ax.flatten()):

    sns.scatterplot(data=df, y=items,x=df.index, hue="Gender", style="Gender", ax=subplot).set_title("Distribution of {}".format(items))

plt.tight_layout()

plt.show()
for i,items in enumerate(categories):

    sns.swarmplot(data=df, x=items, y="percentage_MBA", hue="Gender")

    plt.tight_layout()

    plt.show()
for i,items in enumerate(categories):

    hue = categories[0]

    if items != hue:

        sns.countplot(x=items, hue=hue, data=df).set_title("Distribution of {} vs {}".format(items, hue))

        plt.show()
for i,items in enumerate(categories):

    hue = categories[1]

    if items != hue:

        sns.countplot(x=items, hue=hue, data=df).set_title("Distribution of {} vs {}".format(items, hue))

        plt.show()
for i,items in enumerate(categories):

    hue = categories[2]

    if items != hue:

        sns.countplot(x=items, hue=hue, data=df).set_title("Distribution of {} vs {}".format(items, hue))

        plt.show()
for i,items in enumerate(categories):

    hue = categories[3]

    if items != hue:

        sns.countplot(x=items, hue=hue, data=df).set_title("Distribution of {} vs {}".format(items, hue))

        plt.show()
for i,items in enumerate(categories):

    hue = categories[4]

    if items != hue:

        sns.countplot(x=items, hue=hue, data=df).set_title("Distribution of {} vs {}".format(items, hue))

        plt.show()
for i,items in enumerate(categories):

    hue = categories[5]

    if items != hue:

        sns.countplot(x=items, hue=hue, data=df).set_title("Distribution of {} vs {}".format(items, hue))

        plt.show()
for i,items in enumerate(categories):

    hue = categories[6]

    if items != hue:

        sns.countplot(x=items, hue=hue, data=df).set_title("Distribution of {} vs {}".format(items, hue))

        plt.show()
f, ax = plt.subplots(4,2, figsize=(15, 15))

for var, subplot in zip(categories, ax.flatten()):

    sns.boxplot(x=var, y='Age_in_years', data=df, ax=subplot).set_title("Analysis of {} vs Age_in_years".format(var))

    plt.tight_layout()
f, ax = plt.subplots(4,2, figsize=(15, 15))

for var, subplot in zip(categories, ax.flatten()):

    sns.boxplot(x=var, y='percentage_MBA', data=df, ax=subplot).set_title("Analysis of {} vs percentage_MBA".format(var))

    plt.tight_layout()
df.head(5)
f, ax = plt.subplots(4,2, figsize=(25, 25))

sns.set_style("white")



for items, subplot in zip(numerical, ax.flatten()):

    sns.violinplot(df['Gender'], df[items], palette = 'hsv', ax=subplot).set_title('Gender vs {}'.format(items))

plt.tight_layout()

plt.show()
f, ax = plt.subplots(4,2, figsize=(25, 25))

sns.set_style("white")



for items, subplot in zip(numerical, ax.flatten()):

    sns.stripplot(df['Gender'], df[items], palette = 'hsv', ax=subplot).set_title('Gender vs {}'.format(items))

plt.tight_layout()

plt.show()
f, ax = plt.subplots(4,2, figsize=(20, 20))

sns.set_style("white")



for i, subplot in zip(sorted(df.Age_in_years.unique()), ax.flatten()):

        sns.countplot(x="STATE", hue="Gender", data=df[df.Age_in_years == i], palette='hsv', ax=subplot).set_title("Distribution of {} vs {} for age {}".format('STATE', 'Gender', i))

plt.tight_layout()

plt.show()
corrMatrix = df.corr()

sns.heatmap(corrMatrix, annot=True)

plt.show()