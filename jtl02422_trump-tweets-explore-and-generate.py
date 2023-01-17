import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sb

import numpy as np

#https://github.com/jsvine/markovify

import markovify

from wordcloud import WordCloud
df = pd.read_csv('../input/better-donald-trump-tweets/Donald-Tweets!.csv')
df.head()
df.dtypes
df.isnull().sum()
df.nunique()
sb.countplot(x='Type', data=df)
hours = []

for value in df['Time'].values:

    hours.append(int(value.split(':')[0]))

    

df['Hour'] = pd.DataFrame(hours)

plt.figure(figsize=(15,10))

plt.xticks(fontsize=16)

plt.yticks(fontsize=16)

plt.xlabel("Hour", fontsize=20)

plt.ylabel("Number of Tweets", fontsize=20)

plt.title("Trumps Most Common Tweeting Time")

sb.countplot(x='Hour', data=df, order=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])
plt.figure(figsize=(15,10))

plt.ylabel("Number of Tweets")

plt.title("Most Tweets on Day")

df['Date'].value_counts().nlargest(10).plot.bar()
text = df['Tweet_Text'].to_numpy()
for i in range(len(text)):

    text[i] = " ".join(filter(lambda x:x[0]!='@', text[i].split()))

    text[i] = " ".join(filter(lambda x:x[0]!='#', text[i].split()))

    text[i] = " ".join(filter(lambda x:x[0:4]!='http', text[i].split()))
wordcloud = WordCloud(width=900, height=900,colormap='inferno').generate(' '.join(text))

plt.figure(figsize=(15,15))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
#https://github.com/jsvine/markovify

model = markovify.Text(text)
for i in range(10):

    print(model.make_sentence())