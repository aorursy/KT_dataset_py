# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import bz2



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_file = bz2.BZ2File('../input/train.ft.txt.bz2')

test_file = bz2.BZ2File('../input/test.ft.txt.bz2')
train_file_lines = train_file.readlines()

test_file_lines = test_file.readlines()
print(len(train_file_lines))

print(len(test_file_lines))
train_file_lines[0]
del train_file, test_file
train_file_lines = [x.decode('utf-8') for x in train_file_lines]

test_file_lines = [x.decode('utf-8') for x in test_file_lines]
train_file_lines[0]
train_labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in train_file_lines]

train_sentences = [x.split(' ', 1)[1][:-1].lower() for x in train_file_lines]
import re
for i in range(len(train_sentences)):

    train_sentences[i] = re.sub('\d','0',train_sentences[i])
test_labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in test_file_lines]

test_sentences = [x.split(' ', 1)[1][:-1].lower() for x in test_file_lines]
for i in range(len(test_sentences)):

    test_sentences[i] = re.sub('\d','0',test_sentences[i])
train = pd.DataFrame({'text':train_sentences})
test = pd.DataFrame({'text':test_sentences})
train['labels'] = train_labels
test['labels'] = test_labels
del train_file_lines

del test_file_lines
import gc

gc.collect()
train = train.sample(frac=1)

test = test.sample(frac=1)
def clean_text(texts):

    texts = texts.replace('\n', ' ')

    if 'www.' in texts or 'http:' in texts or 'https:' in texts or '.com' in texts:

        texts = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "website", texts)

    return texts
train['text'] = train['text'].apply(lambda x: clean_text(x))

test['text'] = test['text'].apply(lambda x: clean_text(x))
train['body_len'] = train['text'].apply(lambda x: len(x) - x.count(" "))

test['body_len'] = test['text'].apply(lambda x: len(x) - x.count(" "))
import string

def count_punct(text):

    count = sum([1 for char in text if char in string.punctuation])

    return round(count/(len(text) - text.count(" ")), 3)*100
train['punct%'] = train['text'].apply(lambda x: count_punct(x))

test['punct%'] = test['text'].apply(lambda x: count_punct(x))
train['capitals'] = train['text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
test['capitals'] = test['text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
train['num_unique_words'] = train['text'].apply(lambda comment: len(set(w for w in comment.split())))

test['num_unique_words'] = test['text'].apply(lambda comment: len(set(w for w in comment.split())))
train['words_vs_unique'] = train['num_unique_words'] / train['body_len']

test['words_vs_unique'] = test['num_unique_words'] / test['body_len']
train['num_exclamation_marks'] = train['text'].apply(lambda comment: comment.count('!'))

test['num_exclamation_marks'] = test['text'].apply(lambda comment: comment.count('!'))
train['num_question_marks'] = train['text'].apply(lambda comment: comment.count('?'))

test['num_question_marks'] = test['text'].apply(lambda comment: comment.count('?'))
import matplotlib.pyplot as plt

import seaborn as sns



from wordcloud import WordCloud, STOPWORDS



stopwords = set(STOPWORDS)



plt.rcParams['figure.figsize'] = (15, 15)

wordcloud = WordCloud(background_color = 'white', width = 1200,  height = 1200, max_words = 10000, random_state=42).generate(str(train['text']))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
count = 0

for num in train['body_len']:

    count=count+num
count
plt.rcParams['figure.figsize'] = (15, 15)

wordcloud = WordCloud(background_color = 'white', width = 1200,  height = 1200, max_words = 10000, random_state=42).generate(str(test['text']))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
from keras.preprocessing.text import Tokenizer
tokenizer_obj = Tokenizer()

tokenizer_obj.fit_on_texts(list(train['text']) +list(test['text']))
word_index = tokenizer_obj.word_index

print('Found %s unique tokens.' % len(word_index))
print(len(train_sentences))

print(len(test_sentences))
print(type(train_sentences))

print(type(test_sentences))
bins = np.linspace(0, 200, 40)

plt.hist(train[train['labels']==1]['body_len'], bins, alpha=0.5, normed=True, label='positive')

plt.hist(train[train['labels']==0]['body_len'], bins, alpha=0.5, normed=True, label='negative')

plt.legend(loc='upper left')

plt.show()
bins = np.linspace(0, 200, 40)

plt.hist(train[train['labels']==1]['punct%'], bins, alpha=0.5, normed=True, label='positive')

plt.hist(train[train['labels']==0]['punct%'], bins, alpha=0.5, normed=True, label='negative')

plt.legend(loc='upper left')

plt.show()
bins = np.linspace(0, 200, 40)

plt.hist(train[train['labels']==1]['num_unique_words'], bins, alpha=0.5, normed=True, label='positive')

plt.hist(train[train['labels']==0]['num_unique_words'], bins, alpha=0.5, normed=True, label='negative')

plt.legend(loc='upper left')

plt.show()
import string

import nltk

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

wn = WordNetLemmatizer()
def clean_text(lines, review_lines):

    for line in lines:

        tokens = word_tokenize(line)

        #convert to lower case

        tokens = [w.lower() for w in tokens]

        #remove punctuation from each word

        table = str.maketrans('', '', string.punctuation)

        stripped = [w.translate(table) for w in tokens]

        #remove remaining tokens that are not alphabetic

        words = [word for word in stripped if word.isalpha()]

        #filter out stop words

        stop_words = set(stopwords.words('english'))

        words = [w for w in words if not w in stop_words]

        words = [wn.lemmatize(w) for w in words]

        review_lines.append(words)

    return review_lines
train_text = train_sentences[:200000]
Train_lines=list()
Train_lines = clean_text(train_text, Train_lines)
import gensim

from gensim import corpora
dictionary = corpora.Dictionary(Train_lines)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in Train_lines]
Lda = gensim.models.ldamodel.LdaModel
ldamodel = Lda(doc_term_matrix, num_topics=2, id2word = dictionary, passes=20)
print(ldamodel.print_topics())
ldamodel = Lda(doc_term_matrix, num_topics=3, id2word = dictionary, passes=20)
print(ldamodel.print_topics())
documents= doc_term_matrix[0]
print(documents)
vector = ldamodel[documents]
vector
print(train_text[0])

print(Train_lines[0])
ldamodel = Lda(doc_term_matrix, num_topics=4, id2word = dictionary, passes=30)
print(ldamodel.print_topics())
noun_texts=list()
from textblob import TextBlob
def noun_text(lines, review_lines):

    for line in lines:

        k=" ".join(line)

        blob = TextBlob(k)

        words = blob.noun_phrases

        words = list(words)

        words = " ".join(words)

        tokens = word_tokenize(words)

        tokens = [w.lower() for w in tokens]

        review_lines.append(tokens)

    return review_lines
for line in Train_lines:

    print(line)

    break
for line in Train_lines:

    k = " ".join(line)

    print(k)

    break
noun_texts = noun_text(Train_lines, noun_texts)
dictionary = corpora.Dictionary(noun_texts)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in noun_texts]
ldamodel = Lda(doc_term_matrix, num_topics=2, id2word = dictionary, passes=20)
print(ldamodel.print_topics())
ldamodel = Lda(doc_term_matrix, num_topics=3, id2word = dictionary, passes=30)
print(ldamodel.print_topics())
documents= doc_term_matrix[0]
vector = ldamodel[documents]
print(vector)
print(noun_texts[0])

print(train_text[0])
noun_adj_text=list()
def noun_adjective_text(lines, review_lines):

    for line in lines:

        k=str(" ".join(line))

        blob = TextBlob(k)

        sentence = blob.sentences[0]

        tokens=list()

        for word, pos in sentence.tags:

            if pos=='NN' or pos=='JJ':

                tokens.append(word)

        review_lines.append(tokens)

    return review_lines
noun_adj_text = noun_adjective_text(Train_lines, noun_adj_text)
dictionary = corpora.Dictionary(noun_adj_text)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in noun_adj_text]
ldamodel = Lda(doc_term_matrix, num_topics=4, id2word = dictionary, passes=40)
print(ldamodel.print_topics())
ldamodel = Lda(doc_term_matrix, num_topics=4, id2word = dictionary, passes=50)
print(ldamodel.print_topics())
documents= doc_term_matrix[0]
vector = ldamodel[documents]
print(vector)
print(noun_adj_text[0])

print(train_text[0])