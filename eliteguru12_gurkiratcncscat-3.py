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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
df=pd.read_csv("/kaggle/input/okcupid-profiles/okcupid_profiles.csv")

df=df.iloc[:,0:21]

df.head
df.describe()
total_nrows = df.shape[0] 

total_ncols = df.shape[1]

print('Total rows: ', total_nrows, ' Total columns: ', total_ncols)

print(df.info())
plt.figure(figsize=(12, 5))

df.isna().sum().plot(kind="bar") 

plt.xticks(rotation=50)
df1 = df.dropna(subset=['diet', 'drugs','education','age','status','sex','orientation','body_type','ethnicity','job','pets'])
def rmissingvaluecol(df1,threshold):

    l = []

    l = list(df1.drop(df1.loc[:,list((100*(df1.isnull().sum()/len(df1.index))>=threshold))].columns, 1).columns.values)

    print("# Columns having more than %s percent missing values:"%threshold,(df1.shape[1] - len(l)))

    print("Columns:\n",list(set(list((df1.columns.values))) - set(l)))

    return l
l = rmissingvaluecol(df1,1)

df2 = df1[l]
df2.head()
print('body_type:') 

print(list(set(df1['body_type']))) 

print('diet:') 

print(list(set(df1['diet']))) 

print('Religion:') 

print(list(set(df1['religion'])))
print('body_type:') 

list(set(df1['body_type']))
def show_percetage(plot, feature): 

    total = len(feature)

    for p in ax.patches:

        percentage = '{:.1f}%'.format(100 * p.get_height()/total)

        x = p.get_x()+p.get_width()/2-0.1

        y = p.get_y()+p.get_height() 

        ax.annotate(percentage, (x, y), size=10)

import seaborn as sns

profile_age = df1[df1['age']<65]

ax = sns.violinplot(x='sex', y='age',data=profile_age,

palette="mako", split=True,

scale="count", inner="quartile")
plt.figure(figsize=(10, 5))

ax = sns.countplot(x='orientation', data=df1,

hue='sex',

palette='PRGn',

order=df1['orientation'].value_counts().iloc[:10].index)

total = float(len(df1))

show_percetage(ax, df1)

plt.show()
plt.figure(figsize=(12, 7))

ax = sns.countplot(x='body_type', data=df1,

hue='sex',

palette='rocket',

order=df1['body_type'].value_counts().iloc[:10].index)

ax.set_title("Body type by gender")

ax.set(xlabel='Count', ylabel='Body Type')

show_percetage(ax, df1)

plt.show()
plt.figure(figsize=(12, 7))

ax = sns.countplot(x='diet', data=df1,

hue='sex', palette='Paired',

order = df1['diet'].value_counts().iloc[:10].index)

plt.xticks(rotation = 30)

ax.set_title("Diet by gender")

show_percetage(ax, df1)
plt.figure(figsize=(12, 7))

sns.countplot(y='job', data=df1,

hue='sex', palette='Reds',

order = df1['job'].value_counts().iloc[:10].index)

plt.show()
plt.figure(figsize=(12, 7))

sns.countplot(y = 'pets', data=df1,

hue='sex', palette='PuRd',

order=df1['pets'].value_counts().iloc[:].index)

plt.show()