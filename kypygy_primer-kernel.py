import numpy as np

import pandas as pd
df = pd.read_csv('/kaggle/input/spaceflightnow-news-articles/space_flight_news.csv')
df.head()
# We don't really need this column

df.drop(columns=['Unnamed: 0'], inplace=True)
# Percentage of article titles that are at least partially related to SpaceX

len(df[df['title'].str.contains('SpaceX')])/len(df)*100
# Let's get some time series data

from datetime import datetime

df['datetime'] = df.date.apply(lambda x: datetime.strptime(x, "%B %d, %Y"))

df['month'] = df.datetime.apply(lambda x: x.month)

df['year'] = df.datetime.apply(lambda x: x.year)

df['day'] = df.datetime.apply(lambda x: x.day)

df['day_of_week'] = df.datetime.apply(lambda x: x.weekday())
import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style='darkgrid')



# Let's plot some of that time series data

# Monthly Breakdown

plt.figure(figsize=(16,6))

ax = sns.countplot(data=df, x='month')

ax.set(xlabel='Month', ylabel='Number of Articles')

ax.set_title('Monthly Breakdown')
# Yearly Breakddown

plt.figure(figsize=(16,6))

ax = sns.countplot(data=df, x='year')

ax.set(xlabel='Year', ylabel='Number of Articles')

ax.set_title('Yearly Breakdown')
# Day of Week Breakdown (0 is Monday and 6 is Sunday)



plt.figure(figsize=(16,6))

ax = sns.countplot(data=df, x='day_of_week')

ax.set(xlabel='Day of Week', ylabel='Number of Articles')

ax.set_title('Day of Week Breakdown')