import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import datetime as dt

from textblob import TextBlob

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
data = pd.read_csv('../input/Donald-Tweets!.csv')
tweets = data['Tweet_Text']
polarity = []

for i in tweets:

    txt = TextBlob(i)

    polarity.append( (txt.sentiment.polarity)*10 )
columns = ['Tweet','Polarity', 'Date', 'Time']

data = pd.DataFrame(data, columns=columns)

data.head()
data['Tweet'] = tweets

data['Polarity'] = pd.DataFrame(polarity)
data_by_polarity = data.sort_values(by='Polarity',ascending=False)
dt = data_by_polarity['Polarity']

fig, ax = plt.subplots(figsize=(10,7))

ax.set_title("Frequency of tweet sentiment!")

ax.set_xlabel("Sentiment amount")

ax.set_ylabel("Amount of tweets")

mean = np.mean(dt)

ax.hist(dt)

fig.tight_layout()

plt.show()
data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
data
dates = data.groupby(data['Date'].map(lambda x: x.date))
#data['Date'] = pd.to_datetime(data['Date'], yearfirst=True)

#data_by_date = pd.DataFrame(data.groupby(['Date']).size().sort_values(ascending=True).rename('Tweets'))
data_by_date = data.copy()

data_by_date['Date'] = pd.to_datetime(data['Date'], yearfirst=True)

data_by_date['Date'] = data_by_date['Date'].dt.month

data_by_date = pd.DataFrame(data_by_date.groupby(['Date']).size().sort_values(ascending=True).rename('Tweets'))

data_by_date
fig, ax = plt.subplots(figsize=(10, 10))

shap = data_by_date

labels = data_by_date.index.values

ax.pie(shap, labels=labels, shadow=True)

plt.title('Tweets per month')

plt.show()