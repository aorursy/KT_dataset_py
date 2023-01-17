# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

% matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/911.csv')

df.head()
df.info()
df.describe()
print('Most occuring zip codes: \n')

print(df['zip'].value_counts().head())

print('\n\nMost occuring townships: \n')

print(df['twp'].value_counts().head())

print('\n\nAmount of unique titles:')

print(df['title'].nunique())
df['Reason'] = df['title'].apply(lambda x: x.split(':')[0])

print(df['Reason'].value_counts())



sns.countplot(x = 'Reason', data = df)

plt.show()
# From string to datetime object

df['timeStamp'] = pd.to_datetime(df['timeStamp'])



# Splitting into hour, month and day

df['Hour'] = df['timeStamp'].apply(lambda x: x.hour)

df['Month'] = df['timeStamp'].apply(lambda x: x.month)

df['Day of Week'] = df['timeStamp'].apply(lambda x: x.dayofweek)



# Day of the week to label

dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

df['Day of Week'] = df['Day of Week'].map(lambda x: dmap[x])



# Plots, making sure the legend is next to the plot and doesn't cover the bars

sns.countplot(x = 'Day of Week', data = df, hue = 'Reason').set(title='Reasons for calling 911 per day of the week')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()

sns.countplot(x = 'Month', data = df, hue = 'Reason').set(title='Reasons for calling 911 per month')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()
df['Date'] = df['timeStamp'].apply(lambda x: x.date())

df.groupby('Date').count()['twp'].plot().set(title='# of townships per date')

plt.tight_layout()
df[df['Reason'] == 'EMS'].groupby('Date').count()['twp'].plot(color='green')

plt.title('EMS')

plt.tight_layout()
df[df['Reason'] == 'Fire'].groupby('Date').count()['twp'].plot(color='red')

plt.title('Fire')

plt.tight_layout()

plt.show()
df[df['Reason'] == 'Traffic'].groupby('Date').count()['twp'].plot(color='blue')

plt.title('Traffic')

plt.tight_layout()

plt.show()
dayHour = df.groupby(by=['Day of Week', 'Hour']).count()['Reason'].unstack()

dayHour.head()



fig, ax = plt.subplots(figsize=(12,10))

sns.heatmap(dayHour).set(title='Heatmap of the # of calls per day of the week and per hour')

plt.show()
# Create day/month

dayMonth = df.groupby(by=['Day of Week', 'Month']).count()['Reason'].unstack()



# Get color palette as cmap

cmap = sns.diverging_palette(240, 10, n=9, as_cmap=True)



fig, ax = plt.subplots(figsize=(12,10))

sns.heatmap(dayMonth, cmap=cmap).set(title='Heatmap of the # of calls per day of the week and per month')

plt.show()