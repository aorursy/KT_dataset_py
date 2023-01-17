import random # Random generators

import numpy as np

import pandas as pd # Pandas dataframe

import matplotlib.pyplot as plt

import re # Text cleaning

import nltk # Text processing

nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem.snowball import SnowballStemmer

from bs4 import BeautifulSoup # Text cleaning

import tensorflow as tf # Tensorflow

from tensorflow.keras import preprocessing # Text preprocessing

from tensorflow.keras.preprocessing.text import Tokenizer # Text preprocessing

from tensorflow.keras.preprocessing.sequence import pad_sequences # Text preprocessing

from tensorflow.keras.models import Sequential # modeling neural networks

from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Embedding, SpatialDropout1D, LSTM

from tensorflow.keras.initializers import Constant

from tensorflow.keras import optimizers, metrics # Neural Network

from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score

random.seed(10) # Set seed for the random generators

print(f"Tensorflow version: {tf.__version__}")
df_train = pd.read_csv("../input/kuc-hackathon-winter-2018/drugsComTrain_raw.csv", parse_dates=['date'])

df_test = pd.read_csv("../input/kuc-hackathon-winter-2018/drugsComTest_raw.csv", parse_dates=['date'])

print(f"Train DataFrame Shape: {df_train.shape}") # We have 161297 rows and 7 columns for training

print(f"Test DataFrame Shape: {df_test.shape}")   # We have 53766 rows and 7 columns for testing
df_train.head()
df_all = pd.concat([df_train, df_test])
df_drugCondition = df_all.groupby(['condition'])['drugName'].nunique().sort_values(ascending=False)

df_drugCondition[0:20].plot(kind="bar", figsize=(15,7), fontsize=15, color="blue")

plt.xlabel("Conditions", fontsize=20)

plt.ylabel("Drags", fontsize=20)

plt.title("Top 20: The number of drugs per condition.", fontsize=20)
df_drugCondition = df_all.groupby(['condition'])['drugName'].nunique().sort_values(ascending=False)

df_drugCondition[df_drugCondition.shape[0] - 20:df_drugCondition.shape[0]].plot(kind="bar", figsize=(15,7), fontsize=15, color="blue")

plt.xlabel("Conditions", fontsize=20)

plt.ylabel("Drugs", fontsize=20)

plt.title("Bottom 20 : The number of drugs per condition.", fontsize=20)
df_all[df_all['condition'].astype(str).str.contains('</span>')].head()
print(f"Number of total rows contain the HTML <span> tag: {len(df_all[df_all['condition'].astype(str).str.contains('</span>')])} rows")
df_train['review'][2]
df_rating = df_all['rating'].value_counts().sort_values(ascending=False)

df_rating.plot(kind="bar", figsize=(15,7), fontsize=15, color="blue")

plt.xlabel("Number of Rates", fontsize=20)

plt.ylabel("Values", fontsize=20)

plt.title("Total Count of Rating Values between One to Ten", fontsize=20)
df_train.loc[(df_train['rating'] > 6), 'ReviewSentiment'] = 2

df_train.loc[((df_train['rating'] >= 5) & (df_train['rating'] <= 6)), 'ReviewSentiment'] = 1

df_train.loc[(df_train['rating'] < 5), 'ReviewSentiment'] = 0



df_test.loc[(df_test['rating'] > 6), 'ReviewSentiment'] = 2

df_test.loc[((df_test['rating'] >= 5) & (df_test['rating'] <= 6)), 'ReviewSentiment'] = 1

df_test.loc[(df_test['rating'] < 5), 'ReviewSentiment'] = 0



df_all.loc[(df_all['rating'] > 6), 'ReviewSentiment'] = 2

df_all.loc[((df_all['rating'] >= 5) & (df_all['rating'] <= 6)), 'ReviewSentiment'] = 1

df_all.loc[(df_all['rating'] < 5), 'ReviewSentiment'] = 0



print(df_train['ReviewSentiment'].value_counts())

print(df_test['ReviewSentiment'].value_counts())
df_all['ReviewSentiment'].value_counts().plot(kind="bar", figsize=(15,7), fontsize=15, color="blue")

plt.xlabel("Sentiment Value", fontsize=20)

plt.ylabel("Number of Review", fontsize=20)

plt.title("Total Number of Reviews for each Sentiment value", fontsize=20)
df_all[df_all.isna().any(axis=1)].head()
print(f"Total rows with missing value: {len(df_all[df_all.isna().any(axis=1)])} rows")
print(f"The percentage of missing value: {(1194 / df_all.shape[0]) * 100} %")
df_train = df_train.dropna(axis=0)

df_test = df_test.dropna(axis=0)
df_train = df_train[df_train['condition'].astype(str).str.contains('</span>') == False]

df_test = df_test[df_test['condition'].astype(str).str.contains('</span>') == False]
df_train_1 = df_train.groupby(['condition'])['drugName'].nunique().sort_values(ascending=False)

df_train_1_list = df_train_1[df_train_1 == 1].index.to_list()

df_train = df_train[~df_train['condition'].isin(df_train_1_list)]



df_test_1 = df_test.groupby(['condition'])['drugName'].nunique().sort_values(ascending=False)

df_test_1_list = df_test_1[df_test_1 == 1].index.to_list()

df_test = df_test[~df_test['condition'].isin(df_test_1_list)]
# removing some stopwords from the list of stopwords as they are important for drug recommendation



stops = set(stopwords.words('english'))

not_stop = ["aren't","couldn't","didn't","doesn't","don't","hadn't","hasn't","haven't","isn't","mightn't",

            "mustn't","needn't","no","nor","not","shan't","shouldn't","wasn't","weren't","wouldn't"]



for i in not_stop:

    stops.remove(i)
stemmer = SnowballStemmer('english')



def reviewWords(review):

    # 1. Delete HTML 

    reviewText = BeautifulSoup(review, 'html.parser').get_text()

    # 2. Make a space

    lettersOnly = re.sub('[^a-zA-Z]', ' ', reviewText)

    # 3. Lowercase only

    words = lettersOnly.lower().split()

    # 4. Remove stopwords 

    meaningfulWords = [w for w in words if not w in stops]

    # 5. Stemming

    stemmedWords = [stemmer.stem(w) for w in meaningfulWords]

    # 6.Join words

    return( ' '.join(stemmedWords))
%%time 

df_train['cleanedReview'] = df_train['review'].apply(reviewWords)

df_test['cleanedReview'] = df_test['review'].apply(reviewWords)
%%time

text_train = df_train['cleanedReview']

tokenizer = Tokenizer(num_words = 5000)

tokenizer.fit_on_texts(text_train)

sequences = tokenizer.texts_to_sequences(text_train)

word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=200)
from tensorflow.keras.utils import to_categorical



y_raw = df_train['ReviewSentiment']



y_labels = to_categorical(np.asarray(y_raw))

print('Shape of data tensor:', data.shape)

print('Shape of label tensor:', y_labels.shape)
# split the data into a training set and a validation set

indices = np.arange(data.shape[0])

np.random.shuffle(indices)

data = data[indices]

labels = y_labels[indices]

nb_validation_samples = int(0.25 * data.shape[0])



x_train = data[:-nb_validation_samples]

y_train = labels[:-nb_validation_samples]

x_validation = data[-nb_validation_samples:]

y_validation = labels[-nb_validation_samples:]
model = Sequential()

model.add(Embedding(5000, 128, input_length=200))

model.add(SpatialDropout1D(0.1))

model.add(LSTM(128, dropout=0.1, recurrent_dropout=0.1, return_sequences=True))

model.add(LSTM(128, dropout=0.1, recurrent_dropout=0.1))

model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
%%time

history = model.fit(x_train, y_train,

          batch_size=128,

          epochs=12,

          validation_data=(x_validation, y_validation),

          verbose=0,

          use_multiprocessing=True)
history.history.keys()
# Plot the accuracy and loss

acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']

e = np.arange(len(acc)) + 1



plt.plot(e, acc, label = 'train')

plt.plot(e, val_acc, label = 'validation')

plt.title('Training and validation accuracy')

plt.xlabel('Epoch')

plt.grid()

plt.legend()



plt.figure()



plt.plot(e, loss, label = 'train')

plt.plot(e, val_loss, label = 'validation')

plt.title('Training and validation loss')

plt.xlabel('Epoch')

plt.grid()

plt.legend()



plt.show()
# Find the predicted values for the validation set

y_pred = np.argmax(model.predict(x_validation), axis = 1)

y_true = np.argmax(y_validation, axis = 1)
y_pred
# Calculate the classification report

cr = classification_report(y_true, y_pred)

print(cr)
cm = confusion_matrix(y_true, y_pred).T

print(cm)
# Calculate the cohen's kappa, both with linear and quadratic weights

k = cohen_kappa_score(y_true, y_pred)

print(f"Cohen's kappa (linear)    = {k:.3f}")

k2 = cohen_kappa_score(y_true, y_pred, weights = 'quadratic')

print(f"Cohen's kappa (quadratic) = {k2:.3f}")
%%time

text_test = df_test['cleanedReview']

tokenizer = Tokenizer(num_words = 5000)

tokenizer.fit_on_texts(text_test)

sequences = tokenizer.texts_to_sequences(text_test)

word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=200)
y_raw = df_test['ReviewSentiment']



y_labels = to_categorical(np.asarray(y_raw))

print('Shape of data tensor:', data.shape)

print('Shape of label tensor:', y_labels.shape)
indices = np.arange(data.shape[0])

np.random.shuffle(indices)

data = data[indices]

labels = y_labels[indices]



x_test = data

y_test = labels
# Find the predicted values for the test set

y_pred = np.argmax(model.predict(x_test), axis = 1)

y_true = np.argmax(y_test, axis = 1)
y_pred
# Calculate the classification report

cr = classification_report(y_true, y_pred)

print(cr)
cm = confusion_matrix(y_true, y_pred).T

print(cm)
# Calculate the cohen's kappa, both with linear and quadratic weights

k = cohen_kappa_score(y_true, y_pred)

print(f"Cohen's kappa (linear)    = {k:.3f}")

k2 = cohen_kappa_score(y_true, y_pred, weights = 'quadratic')

print(f"Cohen's kappa (quadratic) = {k2:.3f}")
%%time

text_test = df_test['cleanedReview']

tokenizer = Tokenizer(num_words = 5000)

tokenizer.fit_on_texts(text_test)

sequences = tokenizer.texts_to_sequences(text_test)

word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=200)
y_raw = df_test['ReviewSentiment']



y_labels = to_categorical(np.asarray(y_raw))

print('Shape of data tensor:', data.shape)

print('Shape of label tensor:', y_labels.shape)
indices = np.arange(data.shape[0])

np.random.shuffle(indices)

data = data[indices]

labels = y_labels[indices]



x_test = data

y_test = labels
freshmodel = Sequential()

freshmodel.add(Embedding(5000, 128, input_length=200))

freshmodel.add(SpatialDropout1D(0.1))

freshmodel.add(LSTM(128, dropout=0.1, recurrent_dropout=0.1, return_sequences=True))

freshmodel.add(LSTM(128, dropout=0.1, recurrent_dropout=0.1))

freshmodel.add(Dense(3, activation='softmax'))

freshmodel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
%%time

freshhistory = freshmodel.fit(x_test, y_test,

          batch_size=128,

          epochs=6,

          verbose=0,

          use_multiprocessing=True)
freshhistory.history.keys()
# Plot the accuracy and loss

acc = freshhistory.history['accuracy']

loss = freshhistory.history['loss']

e = np.arange(len(acc)) + 1



plt.plot(e, acc, label = 'test')

plt.title('Test accuracy')

plt.xlabel('Epoch')

plt.grid()

plt.legend()



plt.figure()



plt.plot(e, loss, label = 'test')

plt.title('Test loss')

plt.xlabel('Epoch')

plt.grid()

plt.legend()



plt.show()
# Find the predicted values for the test set

y_pred = np.argmax(freshmodel.predict(x_test), axis = 1)

y_true = np.argmax(y_test, axis = 1)
y_pred
# Calculate the classification report

cr = classification_report(y_true, y_pred)

print(cr)
cm = confusion_matrix(y_true, y_pred).T

print(cm)
# Calculate the cohen's kappa, both with linear and quadratic weights

k = cohen_kappa_score(y_true, y_pred)

print(f"Cohen's kappa (linear)    = {k:.3f}")

k2 = cohen_kappa_score(y_true, y_pred, weights = 'quadratic')

print(f"Cohen's kappa (quadratic) = {k2:.3f}")