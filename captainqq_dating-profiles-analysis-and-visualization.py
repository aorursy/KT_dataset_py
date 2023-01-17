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
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
raw_data = pd.read_csv('../input/okcupid-profiles/okcupid_profiles.csv')
okcupid_profiles = raw_data.iloc[:,0:21] # Ignore all the "essay" text columns for now
# print(okcupid_profiles.shape)

total_nrows = okcupid_profiles.shape[0]

total_ncols = okcupid_profiles.shape[1]

print('Total rows: ', total_nrows, ' Total columns: ', total_ncols)

print(okcupid_profiles.info())

# print(okcupid_profiles.describe())
total_count = okcupid_profiles.isna().count()

count_na = okcupid_profiles.isna().sum().sort_values(ascending = False)

pct_na =(okcupid_profiles.isna().sum()/total_count).sort_values(ascending = False)

pd.options.display.float_format = '{:.1%}'.format

missing_data = pd.concat([count_na, pct_na], axis=1, keys=['Count', 'Percent'])

# Only show columns with more than 5% missing data

missing_data[missing_data['Percent']>=0.05].head(20)
plt.figure(figsize=(12, 5))

okcupid_profiles.isna().sum().plot(kind="bar")

plt.xticks(rotation=50)
# okcupid_profiles.groupby('sex').count()

print(okcupid_profiles['sex'].value_counts())

print(okcupid_profiles['ethnicity'].value_counts().sort_values(ascending=False)[:5])
# Check unique values

print('body_type:')

print(list(set(okcupid_profiles['body_type'])))

print('diet:')

print(list(set(okcupid_profiles['diet'])))

print('Religion:')

print(list(set(okcupid_profiles['religion'])))
# Check unique values

print('body_type:')

list(set(okcupid_profiles['body_type']))
# sns.countplot(y = 'age', data = my_data)

plt.figure(figsize=(15, 7))

sns.catplot(x='age', data = okcupid_profiles, kind='count', palette='pastel', hue='sex', 

            height=5, # make the plot 5 units high

            aspect=2)

plt.xticks(rotation = 25)

plt.title('Age distribution')

plt.show()
profile_age = okcupid_profiles[okcupid_profiles['age']<65]

ax = sns.violinplot(x='sex', y='age',data=profile_age,

                    palette="muted", split=True, 

                    scale="count", inner="quartile")
# Define a function to show pentages on catplot

def show_percetage(plot, feature):

    total = len(feature)

    for p in ax.patches:

        percentage = '{:.1f}%'.format(100 * p.get_height()/total)

        x = p.get_x()+p.get_width()/2-0.1

        y = p.get_y()+p.get_height()

        ax.annotate(percentage, (x, y), size=10)
plt.figure(figsize=(9, 5))

ax = sns.countplot(x='orientation', data=okcupid_profiles, 

                   hue='sex', 

                   palette='PRGn',

                   order=okcupid_profiles['orientation'].value_counts().iloc[:10].index) 

total = float(len(okcupid_profiles))

show_percetage(ax, okcupid_profiles)

plt.show()
plt.figure(figsize=(9, 5))

ax = sns.countplot(x='body_type', data=okcupid_profiles, 

                   hue='sex', 

                   palette='rocket',

                   order=okcupid_profiles['body_type'].value_counts().iloc[:10].index)

ax.set_title("Body type by gender")

ax.set(xlabel='Count', ylabel='Body Type') 

show_percetage(ax, okcupid_profiles)

plt.show()
plt.figure(figsize=(9, 5))

ax = sns.countplot(x='diet', data=okcupid_profiles, 

                   hue='sex', palette='Paired',

                   order = okcupid_profiles['diet'].value_counts().iloc[:10].index) 

plt.xticks(rotation = 30)

ax.set_title("Diet by gender")

show_percetage(ax, okcupid_profiles)
plt.figure(figsize=(9, 5))

sns.countplot(y='job', data=okcupid_profiles, 

              hue='sex', palette='Reds',

              order = okcupid_profiles['job'].value_counts().iloc[:10].index)

plt.show()
# Remove invalid data

plt.figure(figsize=(9, 5))

okcupid_profiles_income = okcupid_profiles[okcupid_profiles.income != -1]

sns.countplot(y='income', data=okcupid_profiles_income, hue='sex', palette='Set1')

sns.despine() # Removes the spines from the right and upper portion of the plot

plt.show()
plt.figure(figsize=(9, 5))

sns.countplot(y = 'pets', data=okcupid_profiles, 

              hue='sex', palette='PuRd',

              order=okcupid_profiles['pets'].value_counts().iloc[:].index)

plt.show()