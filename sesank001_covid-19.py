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
import matplotlib.pyplot as plt
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv', index_col='ObservationDate')
df.head()
df['Country/Region'].unique().tolist()
US = df[df['Country/Region'] == "US"]

ITALY  = df[df['Country/Region'] == "Italy"]

CHINA = df[df['Country/Region']=="Mainland China"]

print("{}, {}".format(len(CHINA.index.unique()), CHINA.shape[0]))
len(US.index.unique()) == US.shape[0]
print("ITALY: number unique values: {}, number of Columns {}".format(len(ITALY.index.unique()), ITALY.shape[0]))

print("US:    number unique values: {}, number of Columns {}".format(len(US.index.unique()), US.shape[0]))

print("CHINA: number unique values: {}, number of Columns {}".format(len(CHINA.index.unique()), CHINA.shape[0]))
def getUnique(frame):

    out = {}

    for date in frame.index.unique():

        out[date] = frame[frame.index == date].Deaths.sum()

    return pd.DataFrame.from_dict(out, orient='index', columns=["Deaths"])
US_deaths = getUnique(US)

China_deaths = getUnique(CHINA)

print("US unique vals == number of columns --> {}".format(len(US_deaths.index.unique()) == US_deaths.shape[0]))

print("CHINA unique vals == number of ciolumns --> {}".format(len(China_deaths.index.unique()) == China_deaths.shape[0]))
plt.style.use("Solarize_Light2")
plt.figure(figsize=(20, 10))

plt.xlabel("Date")

plt.ylabel("Number of Deaths")

plt.xticks(rotation=60)

plt.title("Number of Daily Recorded Deaths in the US vs Italy vs CHINA")

l1 = plt.scatter(US_deaths.index, US_deaths)

l2 = plt.scatter(ITALY.index, ITALY.Deaths)

l3 = plt.scatter(China_deaths.index, China_deaths)

plt.legend( (l1, l2, l3), ('US', 'ITALY', 'CHINA'), loc='upper left')

plt.show()
recovered = {}

for Country in df['Country/Region'].unique():

    recovered[Country] = df[df['Country/Region'] == Country].Recovered.sum()

rec = pd.DataFrame.from_dict(recovered, orient='index', columns=['Number of Recoveries'])

rec = rec.sort_values(by='Number of Recoveries', ascending=False).head(25)
Top_25 = df[df['Country/Region'].isin(rec.index)]

top_25_deaths = {}

for country in Top_25['Country/Region'].unique():

    top_25_deaths[country] =  Top_25[Top_25['Country/Region'] == country].Deaths.sum()

top_deaths = pd.DataFrame.from_dict(top_25_deaths, orient='index', columns=['Deaths'])
plt.figure(figsize=(20, 10))

plt.title("Top 25 countries with the most recoveries")

plt.xlabel("Country")

plt.ylabel("Number of Recoveries in Millions")

plt.xticks(rotation=90)

b1 = plt.bar(rec.index, rec['Number of Recoveries'], width=.6, align='center')

b2 = plt.bar(top_deaths.index, top_deaths.Deaths, width=.6, align='center')

plt.legend((b1, b2), ("Recovered", "Deaths"))

plt.show()