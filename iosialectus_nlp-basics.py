# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import nltk # A lot of useful text stuff

import re # regular expressions

from gensim.models import Word2Vec, Doc2Vec

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import Perceptron

from sklearn.svm import SVC, LinearSVC



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

print(df_train.head())
df_train['keyword'].value_counts()
df_train['location'].value_counts()
df_train.describe()
percent_true = 0.42966
df_test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

print(df_test.head())
combined_text = ""

combined_true_text = ""

combined_false_text = ""

for index, row in df_train.iterrows():

    row_text = row['text'].lower() + ' '

    combined_text += row_text

    if row['target']:

        combined_true_text += row_text

    else:

        combined_false_text += row_text

combined_text_words = nltk.word_tokenize(combined_text)

true_text_words = nltk.word_tokenize(combined_true_text)

false_text_words = nltk.word_tokenize(combined_false_text)

print(combined_text_words[:50])

print(true_text_words[:50])

print(false_text_words[:50])
freq_dist_combined = nltk.FreqDist(combined_text_words)

freq_dist_true = nltk.FreqDist(true_text_words)

freq_dist_false = nltk.FreqDist(false_text_words)

print(freq_dist_combined.most_common(100))

print()

print(freq_dist_true.most_common(100))

print()

print(freq_dist_false.most_common(100))
freq_dist_combined['http']
# Use Bayes' theorem to find the probabillity of being about a true disaster given the tweet contains a given word.

proba_true_list = [(key, freq_dist_true[key]*percent_true*len(combined_text_words)/(len(true_text_words)*freq_dist_combined[key])) for key in freq_dist_true.keys() if len(key)>3 and freq_dist_combined[key]>=25]

proba_true_list = sorted(proba_true_list, key=lambda x: np.absolute(percent_true - x[1]), reverse=True)

print(proba_true_list)
important_word_list = [tpl[0] for tpl in proba_true_list if np.absolute(tpl[1]-percent_true)>0.1]

for w in important_word_list:

    print(w)
len(important_word_list)
sent_list = [nltk.word_tokenize(row['text'].lower()) for (item, row) in df_train.iterrows()] + [nltk.word_tokenize(row['text'].lower()) for (item, row) in df_test.iterrows()]
alt_corpus = nltk.corpus.brown.sents(categories=['news', 'editorial', 'reviews','religion','hobbies','lore','belles_lettres','government','learned'])
alt_corpus[0]
print(len(alt_corpus))
print(len(sent_list))
word_embedding = Word2Vec(sent_list, size=75, min_count=5, workers=20, window=5,iter=1,sg=1)
word_embedding.train(alt_corpus, total_examples=word_embedding.corpus_count, epochs=25)
word_embedding.train(sent_list, total_examples=word_embedding.corpus_count, epochs=100)
print(word_embedding.wv.most_similar('russia'))

print()

print(word_embedding.wv.most_similar('fire'))

print()

print(word_embedding.wv.most_similar('bags'))

print()

print(word_embedding.wv.most_similar('helicopter'))

print()

print(word_embedding.wv.most_similar('bomb'))

print()

print(word_embedding.wv.most_similar('iran'))

print()

print(word_embedding.wv.most_similar('earthquake'))

print()

print(word_embedding.wv.most_similar('typhoon'))

print()

print(word_embedding.wv.most_similar('sick'))

print()

print(word_embedding.wv.most_similar('mosque'))
important_word_list_2 = [important_word_list[0]]

for w in important_word_list[1:]:

    add_word = True

    for w2 in important_word_list_2:

        if np.absolute(word_embedding.wv.similarity(w,w2))>0.65:

            add_word = False

    if add_word:

        important_word_list_2.append(w)

        

for w in important_word_list_2:

    print(w)
print(len(important_word_list_2))
important_word_list = important_word_list_2
def max_similarity(wd, text):

    ret = 0

    words = nltk.word_tokenize(text.lower())

    for w in words:

        x = 0

        if w in word_embedding.wv.vocab:

            x = word_embedding.wv.similarity(w,wd)

        if x>ret:

            ret=x

    return ret


def contains_one(s,lst):

    if not isinstance(s,str):

        s = s.decode('utf-8')

    for w in lst:

        if re.search(re.escape(w),s,re.IGNORECASE):

            return 1

    return 0
# Check for each of the important words

for w in important_word_list:

    df_train[w] = df_train.apply(lambda x: max_similarity(w, x['text']), axis=1)

    df_test[w] = df_test.apply(lambda x: max_similarity(w, x['text']), axis=1)
df_targets = df_train['target']

df_prediction = pd.DataFrame()

df_prediction['id'] = df_test['id']
df_train = df_train[[w for w in important_word_list]]

print(df_train.head())
df_test = df_test[[w for w in important_word_list]]

print(df_test.head())
for c in df_train.columns:

    print(c)
model = SVC()
model.fit(df_train, df_targets)
predictions = model.predict(df_test)
df_prediction['target'] = predictions
df_prediction.to_csv("predictions.csv", index=False)