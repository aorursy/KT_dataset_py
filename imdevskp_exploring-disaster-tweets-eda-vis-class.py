# number



#  characters

#  words

#  sentence

#  paragraphs



#  hastags

#  mentions

#  web address



#  numbers

#  punctuations

#  emoji



#  stopwords

#  unique words

#  full 140 characters used ?



#  upper case words

#  title case words

 

#  avg. length

 

# unigram

# bigram

# trigram



# replace %20 by ' '



# remove

#  url

#  emoticons

#  html tags

#  punctuations



# correct spelling

 

# wordcloud

# visualize embeddings



# count vectorizer

# tf-idf



# embedding

#  lstm 

#  bidirectional

# glove

# bert

# automl
# def clean_text(text):

#     '''Make text lowercase, remove text in square brackets,remove links,remove punctuation

#     and remove words containing numbers.'''

#     text = text.lower()

#     text = re.sub('\[.*?\]', '', text)

#     text = re.sub('https?://\S+|www\.\S+', '', text)

#     text = re.sub('<.*?>+', '', text)

#     text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

#     text = re.sub('\n', '', text)

#     text = re.sub('\w*\d\w*', '', text)

#     return text
# basic

import numpy as np 

import pandas as pd



# plotting

import matplotlib.pyplot as plt

import seaborn as sns

import plotly_express as px



# palette

tw_pal = ['#55ACEE', '#292F33', '#66757F', '#CCD6DD', '#E1E8ED']

sns.set_style("whitegrid")



# machine learning

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

from sklearn.linear_model import RidgeClassifier, LogisticRegression

from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix



# nlp

from nltk.corpus import stopwords

from wordcloud import WordCloud, STOPWORDS 



# deep learning

import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences
# load data

train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
# labeling targets based on target column

train['label'] = train['target'].apply(lambda x: 'Real Disaster' if x==1 else 'Not a Real Disaster')
# random rows

train.sample(5)
# shape of the dataset

# print(train.shape)

# print(test.shape)



# info

# train.info()



# describe 

# train.describe()



# missing values

# train.isna().sum()
# %20 looks like it stands for character 'space' 

for table in [train, test]:

    for col in table.select_dtypes('object').columns:

        table[col] = table[col].str.replace('%20', ' ')
# derived tables



disaster = train[train['target']==1]

not_a_disaster = train[train['target']==0]
# target label count



sns.countplot(x='label', data=train, palette=tw_pal)

plt.legend()

plt.show()
# location count



print('Disaster location \n'+'-'*30)

print(disaster['location'].value_counts()[:6])

print('\nNot a disaster location \n'+'-'*30)

print(not_a_disaster['location'].value_counts()[:6])
# keyword count



print('Disaster keywords \n'+'-'*30)

print(disaster['keyword'].value_counts()[:6])

print('\nNot a disaster keywords \n'+'-'*30)

print(not_a_disaster['keyword'].value_counts()[:6])
# train['target_mean'] = train.groupby('keyword')['target'].transform('mean')



# plt.figure(figsize=(6, 50))

# sns.countplot(y = train.sort_values(by='target_mean', ascending=False)['keyword'],

#               hue = train.sort_values(by='target_mean', ascending=False)['target'], 

#               palette = tw_pal)

# plt.show()
# tweet length



# fig,ax = plt.subplots(1, 2, figsize=(12, 4))



# ax[0].hist(train[train['target']==1]['text'].str.len(), color='#55ACEE', 

#            bins=15, range=(0, 160))

# ax[0].set_title('Disaster')



# ax[1].hist(train[train['target']==0]['text'].str.len(), color='#66757F', 

#            bins=15, range=(0, 160))

# ax[1].set_title('No Disaster')



# fig.suptitle('Characters in tweets')

# plt.show()
# tweet length



plt.figure(figsize=(12, 5))

plt.hist(train[train['target']==1]['text'].str.len(), 

         color='#55ACEE', alpha=0.7, bins=70, 

         range=(0, 160), label='Disaster')

plt.hist(train[train['target']==0]['text'].str.len(), 

         color='#66757F', alpha=0.7, bins=70, 

         range=(0, 160), label='Not Disaster')

plt.suptitle('Characters in tweets')

plt.legend()

plt.show()
# no. of characters, words, sentences, paragraphs



train['no_chars'] = train['text'].apply(len)

train['no_words'] = train['text'].str.split().apply(len)

train['no_sent'] = train['text'].str.split('.').apply(len)

train['no_para'] = train['text'].str.split('\n').apply(len)



test['no_chars'] = test['text'].apply(len)

test['no_words'] = test['text'].str.split().apply(len)

test['no_sent'] = test['text'].str.split('.').apply(len)

test['no_para'] = test['text'].str.split('\n').apply(len)



cols = ['no_chars', 'no_words', 'no_sent', 'no_para']

col_names = ['Mean no. of Characters', 

             'Mean no. of Words', 

             'Mean no. of Sentences', 

             'Mean no. of Paragraphs']



fig, ax = plt.subplots(1, 4, figsize=(24, 4))

for ind, val in enumerate(cols):

    sns.barplot(x='label', y=val, palette=tw_pal, data=train, ax=ax[ind])

    ax[ind].set_title(col_names[ind])
# Characters per word, Characters per sentences, Words per sentences



train['chars_per_word'] = train['no_chars']/train['no_words']

train['chars_per_sent'] = train['no_chars']/train['no_sent']

train['words_per_sent'] = train['no_words']/train['no_sent']



test['chars_per_word'] = test['no_chars']/test['no_words']

test['chars_per_sent'] = test['no_chars']/test['no_sent']

test['words_per_sent'] = test['no_words']/test['no_sent']



cols = ['chars_per_word', 'chars_per_sent', 'words_per_sent']

col_names = ['Characters per word', 

             'Characters per sentences', 

             'Words per sentences']



fig, ax = plt.subplots(1, 3, figsize=(14, 4))

for ind, val in enumerate(cols):

    sns.barplot(x='label', y=val, palette=tw_pal, data=train, ax=ax[ind])

    ax[ind].set_title(col_names[ind])
# No. of hashtags, mentions, web addresses



def hash_count(tweet):

    w = tweet.split()

    return len([word for word in w if word.startswith('#')])



def mention_count(tweet):

    w = tweet.split()

    return len([word for word in w if word.startswith('@')])



def web_add(tweet):

    w = tweet.split()

    return len([word for word in w if word.startswith('http')])



train['no_hashtags'] = train['text'].apply(hash_count)

train['no_mentions'] = train['text'].apply(mention_count)

train['no_web_add'] = train['text'].apply(web_add)



test['no_hashtags'] = test['text'].apply(hash_count)

test['no_mentions'] = test['text'].apply(mention_count)

test['no_web_add'] = test['text'].apply(web_add)



cols = ['no_hashtags', 'no_mentions', 'no_web_add']

col_names = ['No. of hashtags', 

             'No. of mentions', 

             'No. of web addresses']



fig, ax = plt.subplots(1, 3, figsize=(14, 4))

for ind, val in enumerate(cols):

    sns.barplot(x='label', y=val, palette=tw_pal, data=train, ax=ax[ind])

    ax[ind].set_title(col_names[ind])


# def numbers(tweet):

#     w = tweet.split()

#     return sum([word.isnumeric() for word in w])

# cols = ['no_chars', 'no_words', 'no_sent', 'no_para', 'avg_word_len', 

#         'no_hashtags', 'no_mentions', 'no_web_add', 'numbers']



# titles = ['Average number of characters', 'Average number of words', 

#           'Average number of sentences', 'Average number of paragraphs', 

#           'Average word length', 'No. of hastags', 'No. of mentions', 

#           'No. of web addresses', 'No. of numbers']



# for ind, val in enumerate(cols):

#     plt.figure(figsize=(6,4))

#     sns.barplot(x='target', y=val, palette=tw_pal, data=train)

#     plt.suptitle(titles[ind])

#     plt.show()
# for ind, val in enumerate(cols):

#     plt.figure(figsize=(12, 5))

#     mn, mx = min(train[val]), max(train[val])

#     plt.hist(train[train['target']==1][val], color='#55ACEE', 

#              alpha=0.7, label='Disaster', bins=10, range=[mn, mx])

#     plt.hist(train[train['target']==0][val], color='#66757F', 

#              alpha=0.7, label='Not Disaster', bins=10, range=[mn, mx])

#     plt.suptitle(titles[ind])

#     plt.legend()

#     plt.show()
# for ind, val in enumerate(cols):

#     plt.figure(figsize=(12, 5))

#     sns.kdeplot(train[train['target']==1][val], alpha=0.4, shade=True, color="#55ACEE")

#     sns.kdeplot(train[train['target']==0][val], alpha=0.4, shade=True, color='#292F33')

#     plt.suptitle(titles[ind])

#     plt.legend()

#     plt.show()
# sns.countplot(x='keyword', data=train, ax=ax[ind])

# plt.show()
# train['tweet_target'].value_counts()
# # cv = TfidfVectorizer(stop_words='english')

# cv = CountVectorizer(stop_words = 'english')

# cv.fit(train["text"].append(test['text']))

# train_vectors = cv.transform(train["text"])

# test_vectors = cv.transform(test["text"])
# count_vect_df = pd.DataFrame(train_vectors.todense(), columns=cv.get_feature_names())

# train = pd.concat([train, count_vect_df], axis=1)
# count_vect_df = pd.DataFrame(test_vectors.todense(), columns=cv.get_feature_names())

# test = pd.concat([test, count_vect_df], axis=1)
# train.head()
# train.drop(['id', 'keyword', 'location', 'text'], axis=1, inplace=True)

# test.drop(['id', 'keyword', 'location', 'text'], axis=1, inplace=True)
# X = train.drop(['tweet_target'], axis=1)

# # X = train[['no_chars', 'no_para', 'avg_word_len', 'no_hashtags', 'no_mentions', 'no_web_add']]

# y = train['tweet_target']
# scaler = MinMaxScaler()

# X = scaler.fit_transform(X)

# test = scaler.transform(test)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# print(X.shape)

# print(y.shape)
# clf = KNeighborsClassifier()

# clf.fit(X_train, y_train)



# y_pred = clf.predict(X_test)

# print(accuracy_score(y_test, y_pred))

# print(confusion_matrix(y_test, y_pred))

# print(cross_val_score(clf, X, y, cv=5).mean())
# clf = BernoulliNB()

# clf.fit(X_train, y_train)



# y_pred = clf.predict(X_test)

# print(accuracy_score(y_test, y_pred))

# print(confusion_matrix(y_test, y_pred))

# # print(cross_val_score(clf, X, y, cv=5).mean())
# clf = BernoulliNBnoulliNB()

# clf.fit(X_train, y_train)



# y_pred = clf.predict(X_test)

# print(accuracy_score(y_test, y_pred))

# print(confusion_matrix(y_test, y_pred))

# print(cross_val_score(clf, X, y, cv=5).mean())
# clf = LogisticRegression(solver='lbfgs')

# clf.fit(X_train, y_train)



# y_pred = clf.predict(X_test)

# print(accuracy_score(y_test, y_pred))

# print(confusion_matrix(y_test, y_pred))

# print(cross_val_score(clf, X, y, cv=10).mean())
# tfidf_model = TfidfVectorizer(stop_words='english')

# train_vectors = tfidf_model.fit_transform(train["text"])



# train_vectors = tfidf_model.fit_transform(train["text"])

# test_vectors = tfidf_model.transform(test["text"])



# clf = BernoulliNB()



# cross_val_score(clf, train_vectors, train["target"], cv=10, scoring="f1").mean()
# clf.fit(train.drop(['tweet_target'], axis=1), train["tweet_target"])

# sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

# sample_submission["target"] = clf.predict(test)

# sample_submission.head()

# sample_submission.to_csv("submission.csv", index=False)
# train.head()
# specific_wc = ['http', 'https']

# sw = list(set(stopwords.words('english')))

# sw = sw + specific_wc



# print(sw[:5])

# print(len(sw))
# # containers for features and labels



# sentences = []

# labels = []



# for ind, row in train.iterrows():

#     labels.append(row['tweet_target'])

#     sentence = row['text']

#     for word in sw: # removing stop words

#         token = " "+word+" "

#         sentence = sentence.replace(token, " ") # replacing stop words with space

#         sentence = sentence.replace(" ", " ")

#     sentences.append(sentence)
# # label encoding labels 



# enc = LabelEncoder()

# encoded_labels = enc.fit_transform(labels)



# print(enc.classes_)

# print(labels[:5])

# print(encoded_labels[:5])
# # word cloud on entire reviews

# wc = WordCloud(width = 600, height = 400, 

#                     background_color ='white', 

#                     stopwords = sw, 

#                     min_font_size = 10, colormap='Paired_r').generate(' '.join(sentences[:100]))

# plt.imshow(wc)
# # word cloud on positve reviews

# pos_rev = ' '.join(train[train['tweet_target']==0]['text'].to_list()[:10000])

# wc = WordCloud(width = 600, height = 400, 

#                     background_color ='white', 

#                     stopwords = sw, 

#                     min_font_size = 10, colormap='GnBu').generate(pos_rev)

# plt.imshow(wc)
# # word cloud on positve reviews

# pos_rev = ' '.join(train[train['tweet_target']==1]['text'].to_list()[:10000])

# wc = WordCloud(width = 600, height = 400, 

#                     background_color ='white', 

#                     stopwords = sw, 

#                     min_font_size = 10, colormap='RdGy').generate(pos_rev)

# plt.imshow(wc)
# # model parameters



# vocab_size = 1000

# embedding_dim = 16

# max_length = 120

# trunc_type='post'

# padding_type='post'

# oov_tok = "<OOV>"

# training_portion = .7

# train test split

# ---------------



# proportion of training dataset

train_size = int(len(sentences) * training_portion)



# training dataset

train_sentences = sentences[:train_size]

train_labels = encoded_labels[:train_size]



# validation dataset

validation_sentences = sentences[train_size:]

validation_labels = encoded_labels[train_size:]

# # tokenizing, sequencing, padding features



# tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)

# tokenizer.fit_on_texts(train_sentences)

# word_index = tokenizer.word_index



# train_sequences = tokenizer.texts_to_sequences(train_sentences)

# train_padded = pad_sequences(train_sequences, padding=padding_type, maxlen=max_length)



# validation_sequences = tokenizer.texts_to_sequences(validation_sentences)

# validation_padded = pad_sequences(validation_sequences, padding=padding_type, maxlen=max_length)
# print(train_padded.shape)

# print(validation_padded.shape)

# print(train_labels.shape)

# print(validation_labels.shape)
# # model initialization

# model = tf.keras.Sequential([

#     tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),

#     tf.keras.layers.GlobalAveragePooling1D(),

#     tf.keras.layers.Dense(24, activation='relu'),

#     tf.keras.layers.Dense(1, activation='sigmoid')

# ])



# # compile model

# model.compile(loss='binary_crossentropy',

#               optimizer='adam',

#               metrics=['accuracy'])



# # model summary

# model.summary()



# # train model

# num_epochs = 20

# history = model.fit(train_padded, train_labels, 

#                     epochs=num_epochs, verbose=1, 

#                     validation_data=(validation_padded, validation_labels))



# # loss and accuracy



# plt.figure(figsize=(10, 5))



# plt.subplot(1, 2, 1)

# plt.plot(history.history['accuracy'], label='Training Accuracy')

# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

# plt.xlabel("Epochs")

# plt.ylabel('Accuracy')

# plt.legend()



# plt.subplot(1, 2, 2)

# plt.plot(history.history['loss'], label='Training Loss')

# plt.plot(history.history['val_loss'], label='Validation Loss')

# plt.xlabel("Epochs")

# plt.ylabel('Loss')

# plt.legend()



# plt.show()
# # model initialization

# model = tf.keras.Sequential([

#     tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),

#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),

#     tf.keras.layers.Dense(24, activation='relu'),

#     tf.keras.layers.Dense(1, activation='sigmoid')

# ])



# # compile model

# model.compile(loss='binary_crossentropy',

#               optimizer='adam',

#               metrics=['accuracy'])



# # model summary

# model.summary()



# # model fit

# num_epochs = 20

# history = model.fit(train_padded, train_labels, 

#                     epochs=num_epochs, verbose=1, 

#                     validation_data=(validation_padded, validation_labels))



# # accuracy and loss



# plt.figure(figsize=(10, 5))



# plt.subplot(1, 2, 1)

# plt.plot(history.history['accuracy'], label='Training Accuracy')

# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

# plt.xlabel("Epochs")

# plt.ylabel('Accuracy')

# plt.legend()



# plt.subplot(1, 2, 2)

# plt.plot(history.history['loss'], label='Training Loss')

# plt.plot(history.history['val_loss'], label='Validation Loss')

# plt.xlabel("Epochs")

# plt.ylabel('Loss')

# plt.legend()



# plt.show()
# # model initialization

# model = tf.keras.Sequential([

#     tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),

#     tf.keras.layers.Conv1D(128, 5, activation='relu'),

#     tf.keras.layers.GlobalMaxPooling1D(),

#     tf.keras.layers.Dense(24, activation='relu'),

#     tf.keras.layers.Dense(1, activation='sigmoid')

# ])



# # compile model

# model.compile(loss='binary_crossentropy',

#               optimizer='adam',

#               metrics=['accuracy'])



# # model summary

# model.summary()



# # model fit

# num_epochs = 20

# history = model.fit(train_padded, train_labels, 

#                     epochs=num_epochs, verbose=1, 

#                     validation_data=(validation_padded, validation_labels))

# # accuracy and loss



# plt.figure(figsize=(10, 5))



# plt.subplot(1, 2, 1)

# plt.plot(history.history['accuracy'], label='Training Accuracy')

# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

# plt.xlabel("Epochs")

# plt.ylabel('Accuracy')

# plt.legend()



# plt.subplot(1, 2, 2)

# plt.plot(history.history['loss'], label='Training Loss')

# plt.plot(history.history['val_loss'], label='Validation Loss')

# plt.xlabel("Epochs")

# plt.ylabel('Loss')

# plt.legend()



# plt.show()