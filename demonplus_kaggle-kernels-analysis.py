import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



from wordcloud import WordCloud, STOPWORDS
data = pd.read_csv("../input/voted-kaggle-kernels.csv")

print(data.head())



data.shape
data[['Views', 'Forks']] = data[['Views', 'Forks']].fillna(0)



data[['Tags']] = data[['Tags']].fillna("")



data[['Views', 'Forks']] = data[['Views', 'Forks']].astype('int')
plt.figure(figsize=(15,10))

plt.hist(data['Votes'], 100)

plt.xticks(range(0, 2500, 250))

plt.show()
data['Votes'].max()
data['Votes'].idxmax()
from collections import Counter

words = Counter()

data['Tags'].str.lower().str.split().apply(words.update)

print(words.most_common(10))



stopwords = set(STOPWORDS)



wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stopwords,

                          max_words=200,

                          max_font_size=40, 

                          random_state=42

                         ).generate(str(data['Tags']))



print(wordcloud)

fig = plt.figure(1)

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
plt.figure(figsize=(15,10))

plt.hist(data['Comments'], 100)

plt.xticks(range(0, 1000, 100))

plt.show()
data['Comments'].max()
data['Comments'].idxmax()
plt.scatter(data['Votes'], data['Comments'], color="blue", linewidth=0.15)

plt.xlabel("Votes")

plt.ylabel("Comments")

plt.show()
plt.figure(figsize=(15,10))

plt.hist(data['Views'], 100)

plt.show()
data['Views'].max()
data['Views'].idxmax()
plt.figure(figsize=(15,10))

plt.hist(data['Forks'], 100)

plt.show()
data['Forks'].max()
data['Forks'].idxmax()
data['Code Type'].value_counts().plot(kind='pie')

plt.show()
data['Language'].value_counts().plot(kind='pie')

plt.show()