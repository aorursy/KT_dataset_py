import pandas as pd
sample_train = pd.read_csv("data/sample_train.csv")

sample_train.head(10)
sample_test = pd.read_csv("data/sample_test.csv")

sample_test.head(10)
sample_sub = pd.read_csv("data/sample_submission.csv")

sample_sub.head(10)
import re

import pandas as pd

import numpy as np

import nltk

from nltk.stem import WordNetLemmatizer 

import tensorflow as tf

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score, accuracy_score

from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook as tqdm

#from bert_embedding import BertEmbedding

from gensim.models import FastText

import os

import string

nltk.download('wordnet')
# Почистить предложение (убрать из него все, кроме букв и цифр) + перевести в нижний регистр + разделить его на токены

def cleaner(sentence):

    return list(filter(lambda x: len(x) > 0, re.split('\W+', sentence.lower())))



# Почистить предложение (убрать из него все, кроме букв и цифр)

def split_sentence(text):

    return ' '.join([w for w in filter(re.compile("[a-zA-Z]+").match, split_sentence(text))])



# Собрать случаи с неправильным отделением "конца" слова после апострофа

def concat_apostrophe(text):

    short = ['t', 'm', 're', 's', 've']

    prev_token = ''

    for token in range(len(text)):

        if text[token] in short:

            if text[token] == 't':

                text[token-1] = text[token-1][:-1]

                text[token] = 'not'

            elif text[token] == 'm':

                text[token] = 'am'

            elif text[token] == 're':

                text[token] = 'are'

            # Потенциально ошибемся в случаях, когда 's - Present Perfect, т.е. he's come, и в случаях принадлежности

            # (person's happiness), но таких случаев не должно быть много

            elif text[token] == 's':

                text[token] = 'is'

            else:

                text[token] = 'have'

    return ' '.join(text)



# Загружаем стоп слова для английского языка

stop_words = nltk.corpus.stopwords.words('english')



# Если внимательно исследовать данный список стоп-слов, то окажется, что в нем отсутствует слово 'us'. Добавим его.

# Также удалим из списка слово 'not' - оно может сыграть важную роль.

stop_words.append('us')

stop_words.remove('not')



# Разбить на токены + удалить стоп слова + удалить слова, состоящие из 1 символа

def del_stop_words(text, stop_words=stop_words):

    return list(filter(lambda word: (word not in stop_words) & (len(word) > 1), text.split(' ')))



# Импортируем эмбеддинг-модель

import io

def load_vectors(fname):

    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')

    n, d = map(int, fin.readline().split())

    data = {}

    for line in fin:

        tokens = line.rstrip().split(' ')

        data[tokens[0]] = map(float, tokens[1:])

    return data

model_emb = load_vectors('/Users/raila/.mxnet/models/wiki-news-300d-1M.vec')
model_fast = {k: list(v) for k, v in model_emb.items()}
del model_emb
len(model_fast['apple'])
train = pd.read_csv('data/train.csv')

train.head()
train.describe()
X, y = train.comment_text, [train[col].values for col in train.columns[-6:]]
X = X.apply(cleaner)

X.head()
X = X.apply(concat_apostrophe)

X.head()
X = X.apply(del_stop_words)

X.head()
lemmatizer = WordNetLemmatizer()

X = X.apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

X.head()
# Устанавливаем min_df, чтобы избежать всякие кривые слова и опечатки; smooth_idf=True, чтобы idf считалось как

# log((1+n)/(1+df(word))) + 1 (см. документацию sklearn 4.2.3.4.)

tf_idf_vectorizer = TfidfVectorizer(min_df=1, smooth_idf=True)

matrix = tf_idf_vectorizer.fit_transform(X.apply(lambda x: ' '.join(w for w in x)))
tf_idf_dictionary = dict(zip(tf_idf_vectorizer.get_feature_names(), tf_idf_vectorizer.idf_))

print(np.array(list(tf_idf_dictionary.keys()))[[np.argsort(list(tf_idf_dictionary.values()))[-10:]]])
# Если слово встречается реже 100 раз, значит, либо это опечатка, либо очень специфичный термин,

# используемый в очень маленьком количестве документов.

count_vectorizer = CountVectorizer(min_df=100)

X_vect = count_vectorizer.fit_transform(X.apply(lambda x: ' '.join(w for w in x))).toarray()

X_vect
X_emb_df = pd.DataFrame(columns=range(900))

for num in tqdm(range(150000)):

    comment = X[num]

    words_idf = [tf_idf_dictionary[word] for word in comment]

    important_words = np.array(comment)[np.argsort(words_idf)[:-10:-1]]

    embeddings = np.array([])

    non_found = 0

    initial = 3

    count = 3

    for im_word in important_words[:count]:

        if im_word in model_fast: # Если слово присутствует в словаре

            embeddings = np.append(embeddings, np.array(model_fast[im_word]))

        else:

            non_found += 1

    while non_found != 0 and len(important_words)//300>count:

        im_word = important_words[count]

        if im_word in model_fast: # Если слово присутствует в словаре

            embeddings = np.append(embeddings, np.array(model_fast[im_word]))

            non_found -= 1

        else:

            continue

        count += 1

    if len(embeddings) // 300 < initial:

        embeddings = np.append(embeddings, np.repeat(np.repeat(0, 300), initial-len(embeddings)//300))

    X_emb_df.loc[num] = np.ravel(embeddings)

    # Чтобы не забивать оперативную память, будем добавлять по 1000

    if (num+1) % 1000 == 0:

        X_emb_df.to_csv('DataFrame' + str(num-1000) + '-' + str(num) + '.csv')

        print('done')

        del X_emb_df

        X_emb_df = pd.DataFrame(columns=range(900))
# Объединяем все df 

X_emb_df = pd.DataFrame(columns=map(str, range(900)))

dfs = sorted(filter(lambda x: x.startswith('DataFrame'), os.listdir()), key=lambda x: int(x[x.rfind('-')+1:-4]))

for one_df in tqdm(dfs):

    X_emb_df = X_emb_df.append(pd.read_csv(one_df).iloc[:, 1:])

    print(X_emb_df.shape)
X_ready = np.hstack((X_vect, X_emb_df.values))
del X_vect, X_emb_df
X_train, X_val = X_ready[:130000, :], X_ready[130000:, :]

y0_train, y0_val = y[0][:130000], y[0][130000:]

lr = LogisticRegression(C=0.5)

lr.fit(X_train, y0_train)

print('Roc auc: {0:5f}'.format(roc_auc_score(y0_val, lr.predict_proba(X_val)[:, 1])))
lr0 = LogisticRegression(C=0.5)

lr0.fit(X_ready, y[0])



lr1 = LogisticRegression(C=0.6)

lr1.fit(X_ready, y[1])



lr2 = LogisticRegression(C=0.4)

lr2.fit(X_ready, y[2])



lr3 = LogisticRegression(C=0.5)

lr3.fit(X_ready, y[3])



lr4 = LogisticRegression(C=0.45)

lr4.fit(X_ready, y[4])



lr5 = LogisticRegression(C=0.55)

lr5.fit(X_ready, y[5])
test = pd.read_csv('data/test.csv')

test.head()
X = test.comment_text
X = X.apply(cleaner)

X = X.apply(concat_apostrophe)

X = X.apply(del_stop_words)

X = X.apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

X.head()
tf_idf_vectorizer = TfidfVectorizer(min_df=1, smooth_idf=True)

matrix = tf_idf_vectorizer.fit_transform(X.apply(lambda x: ' '.join(w for w in x)))

tf_idf_dictionary = dict(zip(tf_idf_vectorizer.get_feature_names(), tf_idf_vectorizer.idf_))

print(np.array(list(tf_idf_dictionary.keys()))[[np.argsort(list(tf_idf_dictionary.values()))[-10:]]])
X_vect = count_vectorizer.transform(X.apply(lambda x: ' '.join(w for w in x))).toarray()

X_vect
X_emb_df = pd.DataFrame(columns=range(900))

for num in tqdm(range(len(X))):

    comment = X[num]

    words_idf = [tf_idf_dictionary[word] for word in comment]

    important_words = np.array(comment)[np.argsort(words_idf)[:-10:-1]]

    embeddings = np.array([])

    non_found = 0

    initial = 3

    count = 3

    for im_word in important_words[:count]:

        if im_word in model_fast: # Если слово присутствует в словаре

            embeddings = np.append(embeddings, np.array(model_fast[im_word]))

        else:

            non_found += 1

    while non_found != 0 and len(important_words)//300>count:

        im_word = important_words[count]

        if im_word in model_fast: # Если слово присутствует в словаре

            embeddings = np.append(embeddings, np.array(model_fast[im_word]))

            non_found -= 1

        else:

            continue

        count += 1

    if len(embeddings) // 300 < initial:

        embeddings = np.append(embeddings, np.repeat(np.repeat(0, 300), initial-len(embeddings)//300))

    X_emb_df.loc[num] = np.ravel(embeddings)

    # Чтобы не забивать оперативную память, будем добавлять по 1000

    if (num+1) % 1000 == 0:

        X_emb_df.to_csv('TestDataFrame' + str(num-1000) + '-' + str(num) + '.csv')

        print('done')

        del X_emb_df

        X_emb_df = pd.DataFrame(columns=range(900))
X_emb_df = pd.DataFrame(columns=map(str, range(900)))

dfs = sorted(filter(lambda x: x.startswith('TestDataFrame'), os.listdir()), key=lambda x: int(x[x.rfind('-')+1:-4]))

for one_df in tqdm(dfs):

    X_emb_df = X_emb_df.append(pd.read_csv(one_df).iloc[:, 1:])

    print(X_emb_df.shape)
X_ready = np.hstack((X_vect, X_emb_df.values))

del X_vect, X_emb_df
pred0 = lr0.predict_proba(X_ready)[:, 1]



pred1 = lr1.predict_proba(X_ready)[:, 1]



pred2 = lr2.predict_proba(X_ready)[:, 1]



pred3 = lr3.predict_proba(X_ready)[:, 1]



pred4 = lr4.predict_proba(X_ready)[:, 1]



pred5 = lr5.predict_proba(X_ready)[:, 1]
predictions = pd.read_csv('data/sample_submission.csv')

predictions.head()
predictions['toxic'] = pred0

predictions['severe_toxic'] = pred1

predictions['obscene'] = pred2

predictions['threat'] = pred3

predictions['insult'] = pred4

predictions['identity_hate'] = pred5

predictions.head()
predictions.to_csv('predictions.csv', columns=predictions.columns, index=False)