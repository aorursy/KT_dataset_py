# import some usefull libraries and etc

import pandas as pd

import numpy as np

from sklearn import metrics

import nltk

import re

import matplotlib.pyplot as plt

import seaborn as sns



# import data

target = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')['target']

train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')[['text']]

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')[['text']]

ssub = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
train.head(5)
train.info()
test.head(5)
test.info()
ssub.head(5)
combined_df = train.append(test, ignore_index=True)

combined_df.info()
def remove_pattern(input_txt, pattern):

  r = re.findall(pattern, input_txt)

  for i in r:

    input_txt = re.sub(i, '', input_txt)



  return input_txt
combined_df['tidy_text'] = np.vectorize(remove_pattern)(combined_df['text'], "@[\w]*")

combined_df.head(5)
combined_df['tidy_text'] = combined_df['tidy_text'].str.replace("[^a-zA-Z#]", " " )

combined_df.head(5)
combined_df['tidy_text'] = combined_df['tidy_text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

combined_df.head(5)
tokenized_text = combined_df['tidy_text'].apply(lambda x: x.split())

tokenized_text.head(2)
from nltk.stem.porter import *

ps = PorterStemmer()



tokenized_text = tokenized_text.apply(lambda x: [ps.stem(i) for i in x])

tokenized_text.head(2)
for i in range(len(tokenized_text)):

    tokenized_text[i] = ' '.join(tokenized_text[i])

combined_df['tidy_text'] = tokenized_text

combined_df['tidy_text'].head(2)
from wordcloud import WordCloud

all_words = ' '.join([text for text in combined_df['tidy_text']])

wc = WordCloud(random_state=78).generate(all_words)



plt.figure(figsize=(10, 7))

plt.imshow(wc, interpolation='bilinear')

plt.axis('off')

plt.show()
from wordcloud import WordCloud

temp_df = combined_df

temp_df['target'] = target

all_words = ' '.join([text for text in temp_df['tidy_text'][temp_df['target'] == 0]])

wc = WordCloud(random_state=78).generate(all_words)



plt.figure(figsize=(10, 7))

plt.imshow(wc, interpolation='bilinear')

plt.axis('off')

plt.show()
from wordcloud import WordCloud

temp_df = combined_df

temp_df['target'] = target

all_words = ' '.join([text for text in temp_df['tidy_text'][temp_df['target'] == 1]])

wc = WordCloud(random_state=78).generate(all_words)



plt.figure(figsize=(10, 7))

plt.imshow(wc, interpolation='bilinear')

plt.axis('off')

plt.show()
def hashtag_extract(data):

  hashtags = []

  for i in data:

    ht = re.findall(r"#(\w+)", i)

    hashtags.append(ht)



  return hashtags
HT_false = hashtag_extract(temp_df['tidy_text'][temp_df['target'] == 0])

HT_true = hashtag_extract(temp_df['tidy_text'][temp_df['target'] == 1])



HT_false = sum(HT_false, [])

HT_true = sum(HT_true, [])
frq = nltk.FreqDist(HT_false)

df_for_plot = pd.DataFrame({

    'Hashtag': list(frq.keys()),

    'Count': list(frq.values())

})



df_for_plot = df_for_plot.nlargest(columns='Count', n=10)

plt.figure(figsize=(18, 5))

sns.barplot(data=df_for_plot, x='Hashtag', y='Count')
frq = nltk.FreqDist(HT_true)

df_for_plot = pd.DataFrame({

    'Hashtag': list(frq.keys()),

    'Count': list(frq.values())

})



df_for_plot = df_for_plot.nlargest(columns='Count', n=10)

plt.figure(figsize=(18, 5))

sns.barplot(data=df_for_plot, x='Hashtag', y='Count')
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

bagged_data = vectorizer.fit_transform(combined_df['tidy_text'])
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

tfidf_data = vectorizer.fit_transform(combined_df['tidy_text'])
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score



train_df = bagged_data[:7613, :]

test_df = bagged_data[7613:, :]



xtrain_df, xvalid_df, ytrain, yvalid = train_test_split(train_df, target, random_state=94, test_size=0.2)



lreg = LogisticRegression()

lreg.fit(xtrain_df, ytrain)



prediction = lreg.predict_proba(xvalid_df)

prediction_int = prediction[:,1] >= 0.5

prediction_int = prediction_int.astype(np.int)

print('F1 score on validation dataset (bag of words):')

f1_score(yvalid, prediction_int)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score



train_df = tfidf_data[:7613, :]

test_df = tfidf_data[7613:, :]



xtrain_df, xvalid_df, ytrain, yvalid = train_test_split(train_df, target, random_state=94, test_size=0.2)



lreg = LogisticRegression()

lreg.fit(xtrain_df, ytrain)



prediction = lreg.predict_proba(xvalid_df)

prediction_int = prediction[:,1] >= 0.4

prediction_int = prediction_int.astype(np.int)

print('F1 score on validation dataset (tfidf):')

f1_score(yvalid, prediction_int)
test_df = tfidf_data[7613:, :]

prediction = lreg.predict_proba(test_df)

prediction_int = prediction[:,1] >= 0.4

prediction_int = prediction_int.astype(np.int)



ssub["target"] = prediction_int

ssub.to_csv("submission.csv",index=False)