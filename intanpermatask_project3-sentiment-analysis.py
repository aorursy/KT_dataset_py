import numpy as np

import pandas as pd

import re, string, json, nltk

from pandas.io.json import json_normalize



import matplotlib.pyplot as plt 

import seaborn as sns

%matplotlib inline



import warnings 

warnings.filterwarnings("ignore", category=DeprecationWarning)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
key_norm = pd.read_csv("/kaggle/input/new sentiment/Sentiment/key_norm.csv")

key_norm.head()
dictionary = pd.read_csv("/kaggle/input/new sentiment/Sentiment/dictionary.csv")

dictionary.head()
t = open("/kaggle/input/new sentiment/Sentiment/stopword_list_TALA.txt", "r")

print(t.readline())

print(t.readline())

print(t.readline())

print(t.readline())

t.close()
with open('/kaggle/input/new sentiment/Sentiment/data_latih.json') as f:

    data = json.load(f)

    print(data)

    
df = json_normalize(data)

df.head()
df = df[['isi', 'sentimen']]

df.head()
# Convert tweet to lowercase

df['clean'] = df['isi'].map(lambda isi: isi.lower())



# Remove URL in tweets

df['clean'] = df['clean'].map(lambda clean: re.sub(r"http\S+", "", clean))



# Remove username in a tweet

df['clean'] = df['clean'].map(lambda clean: re.sub('@[^\s]+','',clean))



# Remove punctuation, numbers and special character in a tweet

df['clean'] = df['clean'].str.replace("[^a-zA-Z#]", " ")



# Remove hashtags in a tweet

df['clean'] = df['clean'].map(lambda clean: re.sub(r'#([^\s]+)','',clean))



df.head()
# tokenize

from nltk.tokenize import word_tokenize

df['clean'] = df['clean'].apply(word_tokenize) 

df.head(10)
dicti = pd.Series(key_norm.hasil.values,index=key_norm.singkat).to_dict()



words = s.split()

rfrm=[dict[word] if word in dicti else word for word in words]

rfrm= " ".join(rfrm)

print(rfrm)
# Remove stopwords

stop = pd.read_csv("/kaggle/input/new sentiment/Sentiment/stopword_list_TALA.txt", sep="\n", names=['stopwords'])
stop = stop.values
df['clean'] = df['clean'].apply(lambda x: [item for item in x if item not in stop])

df.head(10)
all_words = ' '.join([text for text in df['clean'][0]])

from wordcloud import WordCloud
positive = df[df['sentimen'] == 'positif']

positive_words = ' '.join([text for text in df['clean'][0]])



wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(positive_words)

plt.figure(figsize=(10, 7))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis('off')

plt.show()
negative = df[df['sentimen'] == 'negatif']

negative_words = ' '.join([text for text in df['clean'][0]])



wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(negative_words)

plt.figure(figsize=(10, 7))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis('off')

plt.show()