# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/OPD_Crimes.csv')

data.head()
import seaborn as sns

import matplotlib.pyplot as plt



data.isna().sum()
data.drop(['Case Date Time'], axis=1, inplace=True)

data.drop(['Location'], axis=1, inplace=True)

data.drop(['Orlando Main Street Program Area'], axis=1, inplace=True)

data.drop(['Orlando Commissioner Districts'], axis=1, inplace=True)

data.drop(['Orlando Neighborhoods'], axis=1, inplace=True)

data.drop(['Case Number'], axis=1, inplace=True)



data.head()



df = data.copy(deep=1)

df.head()
df.isna().sum()
df['Case Offense Location Type'].value_counts().plot(kind='bar')

plt.title('Location Type of Crime Committed')

plt.yscale('symlog')



fig = plt.gcf()

fig.set_size_inches(17, 7)
df['Case Offense Charge Type'].value_counts().plot(kind='bar')

plt.title('Charge')

plt.yscale('symlog')



fig = plt.gcf()

fig.set_size_inches(17, 7)
x = []

x.append(df['Case Offense Charge Type'].value_counts()[0])

x.append(df['Case Offense Charge Type'].value_counts()[1])

x

pielabels = 'COMMITTED','ATTEMPTED'
plt.pie(x, labels=pielabels, autopct='%1.2f%%')



plt.axis('equal')

plt.show()
c = []

df['Case Disposition'].value_counts()
c1 = df['Case Disposition'].value_counts()[0]

c2 = df['Case Disposition'].value_counts()[1]

c3 = df['Case Disposition'].value_counts()[2]

c4 = df['Case Disposition'].value_counts()[3]

explode = [0,0,0,0.5]

l = [c1,c2,c3,c4]

labels = 'Closed','Arrest','Inactive','Open'
plt.pie(l, labels=labels, autopct='%1.2f%%', explode=explode)



plt.axis('equal')

plt.show()
s = df['Case Offense Category'].value_counts()
s
s.plot(kind='barh')

plt.title('Case Offense Categories')



fig = plt.gcf()

fig.set_size_inches(10, 5)
df['Case Offense Type'].value_counts()
df['Case Offense Type'].value_counts().plot(kind='bar')

plt.title('Charge')

plt.yscale('symlog')



fig = plt.gcf()

fig.set_size_inches(17, 7)