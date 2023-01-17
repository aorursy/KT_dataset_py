# Hello Viewer

# I'm a beginner in Data Science and looking forward to some valuable feedback.

# Please drop any suggestion or feedback you have in the comments.
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # data visualization



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/dogs-of-zurich/20170308hundehalter.csv')
df = df.rename(columns={'ALTER': 'Age','GESCHLECHT':'Gender',

                        'STADTKREIS': 'District','RASSE1': 'Primary Breed',

                        'RASSE2':'Secondary Breed','GEBURTSJAHR_HUND': 'Year of Birth',

                        'GESCHLECHT_HUND': 'Dog Gender','HUNDEFARBE':'Color',

                        'RASSENTYP':'Breed Type', 'HALTER_ID':'Holder_id'

                       })
df.dtypes
df.dropna(axis=1, how='all', inplace=True)

df.drop(columns=['Secondary Breed', 'RASSE1_MISCHLING'], inplace=True)

df = df[df.Age.notnull()]
df
label = ['11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90','91-100']

x = np.arange(len(label))

width = 0.35

# df.Age.unique()
men, women = [], []

for age in label:

    men.append(len(df[(df.Gender == 'm') & (df.Age == age)]))

    women.append(len(df[(df.Gender == 'w') & (df.Age == age)]))

    

print(men)

print(women)
fig,ax = plt.subplots(figsize=(12,7))

rec = ax.bar(x-width/2, men, width, label='Men')

rect = ax.bar(x+width/2, women, width, label='Women')

ax.set_xticks(x)

ax.set_xticklabels(label)

ax.set_ylabel('No. of people')

ax.set_title('Men vs Women Ownership by Age Group')

ax.legend()
df['Breed Type'].unique()
t1, t2, t3 = [], [], []

for age in label:

    t1.append(len(df[(df['Breed Type'] == 'K') & (df.Age == age)]))

    t2.append(len(df[(df['Breed Type'] == 'I') & (df.Age == age)]))

    t3.append(len(df[(df['Breed Type'] == 'II') & (df.Age == age)]))

print(t1)

print(t2)

print(t3)
fig1,ax1 = plt.subplots(figsize=(12,7))

rec = ax1.bar(x-width/2, t1, width, label='Type K')

rect = ax1.bar(x+width/2, t2, width, label='Type I')

rects = ax1.bar(x, t3, width, label='Type II')

ax1.set_xticks(x)

ax1.set_xticklabels(label)

ax1.set_ylabel('No. of Dogs')

ax1.set_title('Type of Breed Ownership by Age Group')

ax1.legend()
df.District.unique()
type_k = df[df['Breed Type'] == 'II']

type_k.Color.unique()
colors = ['schwarz', 'blue', 'braun', 'wildfarbig', 'weiss', 'orange']

y = np.arange(len(colors))

d1, d2, d3 = [], [], []

for c in colors:

    d1.append(len(df[(df['Breed Type'] == 'K') & (df.Color == c)]))

    d2.append(len(df[(df['Breed Type'] == 'I') & (df.Color == c)]))

    d3.append(len(df[(df['Breed Type'] == 'II') & (df.Color == c)]))

    

print(d1,d2,d3)

    
fig2, ax2 = plt.subplots(figsize=(12,7))

type_one = ax2.plot(y, d1, '-o', label='Type K')

type_two = ax2.plot(y, d2, '-o', label='Type I')

type_three = ax2.plot(y, d3, '-o', label='Type II')

ax2.set_xticks(y)

ax2.set_xticklabels(colors)

ax2.set_ylabel('No. of Dogs')

ax2.set_title('Selected Colors Distribution among Breed Type')

ax2.legend()