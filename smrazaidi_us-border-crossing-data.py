# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd #for file handling purposes

import numpy as np #for simple linear algebraic expressions 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LinearRegression

import calendar
df = pd.read_csv('../input/border-crossing-entry-data/Border_Crossing_Entry_Data.csv')
df.head()
print(df.info())
df.describe()
print(df['Border'].unique())

print(df['Measure'].unique())
amount_people=df[df['Measure'].isin(['Personal Vehicle Passengers','Bus Passengers','Pedestrians','Train Passengers'])]

amount_vehicles=df[df['Measure'].isin(['Trucks','Rail Containers Full','Trains','Truck Containers Empty','Rail containers Empty','Personal Vehicles','Buses','Truck Containers Full' ])]



amount_people[['Border','Value']].groupby(['Border']).sum()
amount_vehicles[['Border','Value']].groupby(['Border']).sum()
sns.barplot(x=df['Border'],y=df['Value'],palette='deep')

plt.xticks(rotation=90)

plt.ylabel('Activity at Entry Ports')

plt.title('Border Crossing')

plt.show()
sns.barplot(x='Measure',y='Value',data=df)

plt.xticks(rotation=90)

plt.title('Types of vehicle and passengers vs count')

plt.show()
X = df[['Port Code']]

#fits column details

Y = df['Value']

lm = LinearRegression()

lm

lm.fit(X,Y)

#makes linear Regression Model

lm.score(X, Y)

df['Year']=pd.DatetimeIndex(df['Date']).year
sns.barplot(x=df['Year'],y=df['Value'])

plt.xticks(rotation=90)

plt.title('Traffic')

plt.ylabel('Number Of Vehicles')

plt.xlabel('Years')

plt.show()
sns.boxplot(x='Measure',y='Year',data=df)

plt.xticks(rotation=90)

plt.title('Types of vehicle and passengers per Year')

plt.show()