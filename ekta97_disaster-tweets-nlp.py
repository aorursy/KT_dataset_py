import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import re

import string
baseLoc = '/kaggle/input/nlp-getting-started/'

train_data = pd.read_csv(baseLoc + 'train.csv')

train_data.shape
test_data = pd.read_csv(baseLoc + 'test.csv')

test_data.shape
train_data.head()
test_data.head()
train_data.isnull().sum()
test_data.isnull().sum()
########  4342 are non-disasterous and 3271 are disasterous

target_val_counts = train_data['target'].value_counts()
non_disasterous = train_data[train_data['target'] ==0]

non_disasterous.head()
disasterous = train_data[train_data['target'] ==1]

disasterous.head()
sns.barplot(y=target_val_counts,x = target_val_counts.index)
keyword_val_counts = train_data['keyword'].value_counts()

keyword_val_counts[:20]
figure = plt.figure(figsize=(7,6))

sns.barplot(y= keyword_val_counts.index[:15], x= keyword_val_counts[:15])

location_val_counts = train_data['location'].value_counts()

location_val_counts[:20]
# Viewing first 20 most common locations

figure = plt.figure(figsize=(7,6))

sns.barplot(y= location_val_counts.index[:20], x= location_val_counts[:20])
train_data['text'] = train_data['text'].apply(lambda x: x.lower())

test_data['text'] = test_data['text'].apply(lambda x: x.lower())
train_data.head()
# Removing punctuation, html tags, symbols, numbers, etc.

def remove_noise(text):

    # Dealing with Punctuation

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\n', '', text)

    text = re.sub('\w*\d\w*', '', text)

    return text
train_data['text'] = train_data['text'].apply(lambda x: remove_noise(x))

test_data['text'] = test_data['text'].apply(lambda x: remove_noise(x))

train_data.head()
import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize 



stop = stopwords.words('english')



#filtered_sentence = [w for w in word_tokenize(example_sent) if not w in stop_words] 

#remove stopwords

def remove_stopwords(text):

    text = [item for item in text.split() if item not in stop]

    return ' '.join(text)



train_data['up_text'] = train_data['text'].apply(remove_stopwords)

test_data['up_text'] = test_data['text'].apply(remove_stopwords)



train_data.head()
from nltk.stem.snowball import SnowballStemmer



stemmer = SnowballStemmer("english")



def stemming(text):

    text = [stemmer.stem(word) for word in text.split()]

    return ' '.join(text)



train_data['stem_text'] = train_data['up_text'].apply(stemming)

test_data['stem_text'] = test_data['up_text'].apply(stemming)

train_data.head()
from wordcloud import WordCloud

import matplotlib.pyplot as plt

fig, (ax1) = plt.subplots(1, figsize=[7, 7])

wordcloud = WordCloud( background_color='white',

                        width=600,

                        height=600).generate(" ".join(train_data['stem_text']))

ax1.imshow(wordcloud)

ax1.axis('off')

ax1.set_title('Frequent Words',fontsize=16);
from sklearn.feature_extraction.text import CountVectorizer



# Using CountVectorizer to change the teweets to vectors

count_vectorizer = CountVectorizer(analyzer='word', binary=True)

count_vectorizer.fit(train_data['stem_text'])



train_vectors = count_vectorizer.fit_transform(train_data['stem_text'])

test_vectors = count_vectorizer.transform(test_data['stem_text'])





# Printing first vector

print(train_vectors[0].todense())

y = train_data['target']

from sklearn.naive_bayes import MultinomialNB

from sklearn import model_selection



model = MultinomialNB(alpha=1)



# Using cross validation to print out our scores

scores = model_selection.cross_val_score(model, train_vectors, y, cv=3, scoring="f1")

scores

from sklearn.linear_model import LogisticRegression



model = LogisticRegression()

scores = model_selection.cross_val_score(model, train_vectors, y, cv=3, scoring="f1")

scores

from sklearn import svm



model = svm.SVC()

scores = model_selection.cross_val_score(model, train_vectors, y, cv=3, scoring="f1")

scores

model
model.fit(train_vectors, y)



sample_submission = pd.read_csv(baseLoc + "sample_submission.csv")

# Predicting model with the test data that was vectorized (test_vectors)

sample_submission['target'] = model.predict(test_vectors)

sample_submission.to_csv("submission3.csv", index=False)
sample_submission.head()