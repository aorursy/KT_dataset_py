import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

import nltk

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from wordcloud import WordCloud
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/imdb-dataset-sentiment-analysis-in-csv-format/Train.csv')

df.head()
df.info()
df.label.unique()
df.label.value_counts()
df.label.value_counts().plot(kind='pie', figsize=(20,8))

plt.show()
def text_prepare(text):

    wordnet = WordNetLemmatizer()

    STOPWORDS = set(stopwords.words('english'))

    text = text.lower()

    tokens = nltk.word_tokenize(text)

    tokens = [i for i in tokens if len(i)>2]

    tokens = [i for i in tokens if i.isalpha()]

    tokens = [i for i in tokens if i not in STOPWORDS]

    tokens = [wordnet.lemmatize(i) for i in tokens]

    return tokens
df['text'] = df['text'].apply(lambda x: text_prepare(x))
## A dictionary to count the frequency of words

freq_count = {}
for line in df['text']:

    for word in line:

        if word not in freq_count:

            freq_count[word] = 1

        else:

            freq_count[word] += 1

freq_count_sorted = {k: v for k, v in sorted(freq_count.items(), key=lambda item: item[1], reverse=True)}
SET_LIMIT = 5000
word_index_map = {v:k for k,v in enumerate(list(freq_count_sorted.keys())[:SET_LIMIT])}
def text_vector(text, label):

    x = np.zeros(len(word_index_map)+1)

    for word in text:

        if word in word_index_map:

            index = word_index_map[word]

            x[index] += 1

        

    x = x/x.sum()

    x[-1] = label

    return x 
data = np.zeros((len(df), len(word_index_map)+1))
idx = 0

index = 0

for idx in range(len(df)):

    tokens = df.iloc[idx,0]

    label = df.iloc[idx,1]

    data[index,:] = text_vector(tokens, label)

    index += 1
X = data[:,:-1]

y = data[:,-1]
model = LogisticRegression()

model.fit(X,y)
model.score(X,y)
test_data = pd.read_csv('/kaggle/input/imdb-dataset-sentiment-analysis-in-csv-format/Test.csv')

test_data.head(5)
test_data['text'] = test_data['text'].apply(lambda x: text_prepare(x))
data2 = np.zeros((len(test_data), len(word_index_map)+1))
idx = 0

index = 0

for idx in range(len(test_data)):

    tokens = test_data.iloc[idx,0]

    label = test_data.iloc[idx,1]

    data2[index,:] = text_vector(tokens, label)

    index += 1
X_test = data2[:,:-1]

y_test = data2[:,-1]
model.predict(X_test)
model.score(X_test,y_test)
threshold = 0.8

positive_score = {}

for word,index in word_index_map.items():

    weight = model.coef_[0][index]

    if weight > threshold:

        positive_score[word] = weight
positive_score = {k: v for k, v in sorted(positive_score.items(), key=lambda item: item[1], reverse=True)}
wordcloud = WordCloud(width = 1000, height = 500).generate(" ".join(list(positive_score.keys())))

plt.figure(figsize = (20, 20), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show() 
threshold = 1

negative_score = {}

for word,index in word_index_map.items():

    weight = model.coef_[0][index]

    if weight < -threshold:

        negative_score[word] = weight
negative_score = {k: v for k, v in sorted(negative_score.items(), key=lambda item: item[1], reverse=False)}
wordcloud = WordCloud(width = 1000, height = 500).generate(" ".join(list(negative_score.keys())))

plt.figure(figsize = (20, 20), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0)  

plt.show() 