# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import seaborn as sns

from collections import Counter

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/tv-shows-on-netflix-prime-video-hulu-and-disney/tv_shows.csv')
print(data['type'].value_counts().values[0])
# We can remove the type column

data = data.iloc[:,1:-1]
data['ID'] = data.index
data.head()
total_platforms = data[['Netflix', 'Hulu', 'Prime Video', 'Disney+']].apply(pd.Series.value_counts).reset_index()

total_platforms.columns = ['Present', 'Netflix', 'Hulu', 'Prime Video', 'Disney+']
total_platforms
labels = ['Netflix', 'Hulu', 'Prime Video', 'Disney+']
plt.figure(figsize=(12,8))

p = plt.pie(total_platforms[total_platforms.Present == 1].iloc[:,1:], labels=labels, explode = (0.1, 0.1, 0.1, 0.1), 

            autopct='%1.0f%%', labeldistance=1.1, shadow=True)
for i in labels:

    data['rating_'+i] = data['IMDb'] * data[i]
imdb_platforms = data.replace(0, np.nan)
imdb_platforms = imdb_platforms[['rating_Netflix', 'rating_Hulu', 'rating_Prime Video', 'rating_Disney+']].mean().reset_index()

imdb_platforms.columns = ['ratings', 'mean']

imdb_platforms.sort_values(by='mean', ascending=False)
plt.figure(figsize=(12,8))

ax = sns.barplot(x="ratings", y="mean", data=imdb_platforms)

l = ax.set(ylim=(6.8, 7.3))
# Less TV Shows on Disney+ and the lowest rates
fig, axes = plt.subplots(2, 2,figsize=(22,15))



ax1 = data[data.Netflix == 1].groupby('IMDb')['ID'].count().plot(xlim=(0, 10), ax=axes[0,0])

ax2 = data[data.Hulu == 1].groupby('IMDb')['ID'].count().plot(xlim=(0, 10),ax=axes[0,1])

ax3 = data[data['Prime Video'] == 1].groupby('IMDb')['ID'].count().plot(xlim=(0, 10),ax=axes[1,0])

ax4 = data[data['Disney+'] == 1].groupby('IMDb')['ID'].count().plot(xlim=(0, 10),ax=axes[1,1])



ax1.title.set_text(labels[0])

ax2.title.set_text(labels[1])

ax3.title.set_text(labels[2])

ax4.title.set_text(labels[3])



plt.show()
# Disney+ with a discontinuous curve
fig, axes = plt.subplots(2, 2,figsize=(22,15))



ax1 = data[data.Netflix == 1].groupby('Year')['ID'].count().plot(xlim=(data['Year'].min(), data['Year'].max()), ax=axes[0,0])

ax2 = data[data.Hulu == 1].groupby('Year')['ID'].count().plot(xlim=(data['Year'].min(), data['Year'].max()),ax=axes[0,1])

ax3 = data[data['Prime Video'] == 1].groupby('Year')['ID'].count().plot(xlim=(data['Year'].min(), data['Year'].max()),ax=axes[1,0])

ax4 = data[data['Disney+'] == 1].groupby('Year')['ID'].count().plot(xlim=(data['Year'].min(), data['Year'].max()),ax=axes[1,1])



ax1.title.set_text(labels[0])

ax2.title.set_text(labels[1])

ax3.title.set_text(labels[2])

ax4.title.set_text(labels[3])



plt.show()
# As expected, more tv shows from the 2000s
data['Age'].value_counts()
fig = plt.figure(figsize=(35,15))



age_netflix = data[data.Netflix == 1].groupby(['Age', 'Netflix']).count()['ID'].reset_index()[['Age', 'ID']]

age_hulu = data[data.Hulu == 1].groupby(['Age', 'Hulu']).count()['ID'].reset_index()[['Age', 'ID']]

age_prime = data[data['Prime Video'] == 1].groupby(['Age', 'Prime Video']).count()['ID'].reset_index()[['Age', 'ID']]

age_disney = data[data['Disney+'] == 1].groupby(['Age', 'Disney+']).count()['ID'].reset_index()[['Age', 'ID']]



ax = plt.subplot2grid((2,4),(0,0))

p = plt.pie(age_netflix['ID'], labels=age_netflix.Age, explode = (0.05, 0.05, 0.05, 0.05, 0.05), autopct='%1.0f%%', labeldistance=1.1, shadow=True)

plt.title('Netflix')

ax = plt.subplot2grid((2,4),(0,1))

p = plt.pie(age_hulu['ID'], labels=age_hulu.Age, explode = (0.05, 0.05, 0.05, 0.05), autopct='%1.0f%%', labeldistance=1.1, shadow=True)

plt.title('Hulu')

ax = plt.subplot2grid((2,4),(1,0))

p = plt.pie(age_prime['ID'], labels=age_prime.Age, explode = (0.05, 0.05, 0.05, 0.05, 0.05), autopct='%1.0f%%', labeldistance=1.1, shadow=True)

plt.title('Prime Video')

ax = plt.subplot2grid((2,4),(1,1))

p = plt.pie(age_disney['ID'], labels=age_disney.Age, explode = (0.05, 0.05, 0.05), autopct='%1.0f%%', labeldistance=1.1, shadow=True)

plt.title('Disney+')

plt.show()
# Disney has a lot more accessible contents (98% of all or 7+)

# The 3 others has ~60% of adult content (16+ and 18+)
plt.figure(figsize=(20,8))

g = sns.countplot(data.IMDb)

t = plt.xticks(rotation=90)
age = data.groupby('Age').count()['ID'].reset_index()

age.columns = ['Age', 'count']
age.sort_values(by='count', ascending=False)
plt.figure(figsize=(12,8))

p = plt.pie(age['count'], labels=age.Age, explode = (0.05, 0.05, 0.05, 0.05, 0.05), autopct='%1.0f%%', labeldistance=1.1, shadow=True)
# 16+ is the most frequent age used for TV Shows