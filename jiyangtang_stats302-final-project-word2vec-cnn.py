import numpy as np

import pandas as pd

import os





data_true = pd.read_csv('../input/fake-and-real-news-dataset/True.csv')

data_false = pd.read_csv('../input/fake-and-real-news-dataset/Fake.csv')



data_true = data_true.drop_duplicates()

data_false = data_false.drop_duplicates()



data_true['real_fake'] = 1

data_false['real_fake'] = 0



data = pd.concat([data_true, data_false])

data['date'] = pd.to_datetime(data['date'], errors='coerce')

data['year'] = data['date'].dt.year

data['month'] = data['date'].dt.month

data['day'] = data['date'].dt.day

data.drop('date', axis=1, inplace=True)

data.dropna(inplace=True)



data.info()
from nltk import word_tokenize

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer



stops = set(stopwords.words('english'))



def preprocess(s):

    # remove stopwords

    tokens = word_tokenize(s)

    ret = [tokens[i] for i, w in enumerate(tokens) if w.lower() not in stops]



    # lemmatization

    lemm = WordNetLemmatizer()

    ret = [lemm.lemmatize(w) for w in ret]

    return ret





X = [preprocess(s) for s in data.text.values]

y = data.real_fake.values
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer

import gensim





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)



# train word2vec model

embedding_size = 100

w2v_model = gensim.models.Word2Vec(sentences=X_train, size=embedding_size, window=5, min_count=1)
from tensorflow.keras.preprocessing.sequence import pad_sequences





def texts_to_sequences(x, tokenizer, max_words=700):

    ret = tokenizer.texts_to_sequences(x)

    ret = pad_sequences(ret, maxlen=max_words)

    return np.asarray(ret)





# get weight matrix from word2vec model

tokenizer = Tokenizer(num_words=700)

tokenizer.fit_on_texts(X)



X_train_seq = texts_to_sequences(X_train, tokenizer)

X_test_seq = texts_to_sequences(X_test, tokenizer)

word_index = tokenizer.word_index



vocab_size = len(word_index) + 1

weight_matrix = np.zeros((vocab_size, embedding_size))

for word, i in word_index.items():

    weights = w2v_model.wv[word] if word in w2v_model.wv else np.zeros(embedding_size)

    weight_matrix[i] = weights
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras.layers import Conv1D, Dense, GlobalMaxPooling1D, Embedding

from tensorflow.keras.models import Sequential





model = Sequential()

# Non-trainable embeddidng layer using weight matrix calculated above

model.add(Embedding(vocab_size, output_dim=embedding_size, weights=[weight_matrix], trainable=False))

model.add(Conv1D(128, 5, activation='relu'))

model.add(GlobalMaxPooling1D())

model.add(Dense(128, activation='relu'))

model.add(Dense(1, activation='sigmoid'))



# callbacks

early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=5)

model_checkpoint = ModelCheckpoint(

    filepath='best.pth',

    save_weights_only=True,

    monitor='val_acc',

    mode='max',

    save_best_only=True

)



# train

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

print(model.summary())

model.fit(X_train_seq, y_train, validation_split=0.3, epochs=200, callbacks=[early_stopping, model_checkpoint])
from sklearn.metrics import confusion_matrix, classification_report





model.load_weights('best.pth')

y_pred = (model.predict(X_test_seq) >= 0.5).astype("int")

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred, digits=4))
import pandas as pd





testset = pd.read_csv('../input/source-based-news-classification/news_articles.csv')

testset = testset[testset.language == 'english']

testset.drop_duplicates(inplace=True, subset='text')

testset.dropna(inplace=True, subset=['text', 'label'])

# remove news whose text contains less than 5 characters

testset = testset[testset.text.str.len() > 5]

testset.info()
from sklearn.metrics import accuracy_score





X = testset.text.values

X = texts_to_sequences(X, tokenizer)

y = testset.label.map({'Real': 1, 'Fake': 0}).values



model.load_weights('best.pth')

y_pred = (model.predict(X_test_seq) >= 0.5).astype("int")

accuracy_score(y_test, y_pred)