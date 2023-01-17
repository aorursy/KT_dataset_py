import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set_palette('Set3')



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

df = pd.read_csv("../input/international-astronaut-database/International Astronaut Database.csv")

df.head()
others=0

for i in df.Country.value_counts():

    if i<10:

        others+=i

others

freqs = df['Country'].value_counts()[df['Country'].value_counts(normalize=False)>=10]

freqs = freqs.append(pd.Series([others],index=['Others']))

# freqs.plot(kind='pie')



fig1, ax1 = plt.subplots()

ax1.pie(freqs, labels=freqs.index, autopct='%1.f%%', shadow=True)

ax1.axis('equal')

plt.show()
gender = df['Gender'].value_counts()

fig1, ax1 = plt.subplots()

ax1.pie(gender, labels=gender.index, autopct='%1.f%%', shadow=True)

ax1.axis('equal')

plt.show()
# Lets take the common ones: Soyuz, STS, Apollo and Gemini

sns.set_palette('Set2')

Apollo= 0

STS = 0

Gemini = 0

Soyuz = 0

for i,craft in enumerate(df['Flights']):

    Apollo += craft.count('Apollo')

    STS += craft.count('STS')

    Gemini += craft.count('Gemini')

    Soyuz += craft.count('Soyuz')

crafts = pd.Series([Apollo, STS, Gemini, Soyuz],index=['Apollo', 'STS', 'Gemini', 'Soyuz'])

sns.barplot(x=crafts.index, y=crafts.values)
# Or in piechart form:

plt.pie(crafts, labels=crafts.index, autopct='%1.f%%', shadow=True)
flights = df['Total Flights'].value_counts()

sns.barplot(x=flights.index, y=flights.values)
df['Flight time (Hours)'] = [int(time[:3])*24 +int(time[4:6]) + int(time[7:])/60 for time in df['Total Flight Time (ddd:hh:mm)']]

df.head()
# Histogram of total flight hours

plt.hist(df["Flight time (Hours)"],color='skyblue',ec='black')

plt.show()
# Clipping for 900 hours, to see the small graphs better

plt.hist(df["Flight time (Hours)"][df["Flight time (Hours)"]>900],color='skyblue',ec='black')

plt.show()
print(f"Lowest Flight time: {round(df.iloc[:,6].min(),2)} hours")

print(f"Mean Flight time: {round(df.iloc[:,6].mean(),2)} hours")

print(f"Highest Flight time: {round(df.iloc[:,6].max(),2)} hours")