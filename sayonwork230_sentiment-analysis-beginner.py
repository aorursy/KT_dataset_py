import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import nltk

from nltk.stem.wordnet import WordNetLemmatizer

from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
os.chdir('/kaggle/input/imdb-dataset-sentiment-analysis-in-csv-format/')
data = pd.read_csv('Train.csv')

data.head()
data.info()
data['label'].value_counts()
data['label'].value_counts().plot(kind='pie', figsize=(20, 5))

plt.show()
wordnet = WordNetLemmatizer()
def text_tokens(s):

    stopword = set(stopwords.words('english'))

    s = s.lower()

    tokens = word_tokenize(s)

    tokens = [wordnet.lemmatize(word) for word in tokens]

    tokens = [token for token in tokens if token not in stopword]

    tokens = [token for token in tokens if token >= 'a' and token <= 'z']

    return tokens
data['text'] = data['text'].apply(lambda x: text_tokens(x))
word_index_map = {}

current_index = 0
for text in data['text']:

    for token in text:

        if token not in word_index_map:

            word_index_map[token] = current_index

            current_index += 1
print("Length :",len(word_index_map))
def token_vector(tokens, label):

    x = np.zeros(len(word_index_map) + 1)

    for t in tokens:

        if t in word_index_map:

            index = word_index_map[t]

            x[index] += 1

    x = x/x.sum()

    x[-1] = label

    return x
trainset = data.iloc[:5001,:]
N = len(trainset)

_data = np.zeros((N, len(word_index_map)+1))

i = 0
idx = 0

for idx in range(len(trainset)):

    tokens = trainset.iloc[idx,0]

    label = trainset.iloc[idx,1]

    xy = token_vector(tokens,label)

    _data[i,:] = xy

    i += 1
X = _data[:,:-1]

y = _data[:,-1]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, y_train)
model.score(X_test,y_test)
threshold = 0.5

for word, index in (word_index_map).items():

    weight = model.coef_[0][index]

    if weight < -threshold:

        print(word, weight)
threshold = 0.5

for word, index in (word_index_map).items():

    weight = model.coef_[0][index]

    if weight > threshold:

        print(word, weight)