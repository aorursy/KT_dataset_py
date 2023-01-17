import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import string
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, SpatialDropout1D
from sklearn.model_selection import train_test_split
df = pd.DataFrame(pd.read_csv('/kaggle/input/restaurant-reviews-in-dhaka-bangladesh/reviews.csv'))
pd.set_option('display.max_colwidth',150)
df.head()
df.isnull().sum()
df = df.drop('Review', axis = 1)
df = df.drop('Recommends', axis = 1)
def lower_case(txt):
    return txt.lower()

def remove_punctuation(txt):
    txt_clean = "".join([c for c in txt if c not in string.punctuation])
    return txt_clean

def remove_non_ascii_chars(txt):
    txt_fullclean = "".join(i for i in txt if ord(i)<128)
    return txt_fullclean
df['lower_case_review'] = df['Review Text'].apply(lambda x: lower_case(x))
df['punctuation_free_review'] = df['lower_case_review'].apply(lambda x: remove_punctuation(x))
df['clean_review'] = df['punctuation_free_review'].apply(lambda x: remove_non_ascii_chars(x))
df = df[['clean_review']]
s_i_a = SentimentIntensityAnalyzer()

print(df['clean_review'][50])
s_i_a.polarity_scores(df['clean_review'].iloc[50])
df['polarity_score'] = df['clean_review'].apply(lambda x: s_i_a.polarity_scores(x))
df['compound_score'] = df['polarity_score'].apply(lambda x: x['compound'])
df['sentiment_tag'] = df['compound_score'].apply(lambda x: 'positive' if x>0.20 else 'negative')
df['sentiment_tag'].value_counts()
sb.set(rc={'figure.figsize':(7.5,4.27)})

sb.countplot(x = 'sentiment_tag', data = df)
df['labels'] = df['sentiment_tag'].apply(lambda x: 0 if x == 'negative' else 1)
df = df.drop('polarity_score', axis = 1)
df = df.drop('compound_score', axis = 1)
df = df.drop('sentiment_tag', axis = 1)
(X, y) = (df['clean_review'].values, df['labels'].values)
print("Type of X: ", type(X), "\nType of y: ",type(y))
print("\nShape of X: ",X.shape, "\nShape of y: ", y.shape)
print("\nExample:", X[0], '=', y[0])
max_length = 100
embedding_dim = 50
trunc_type='post'
padding_type='post'
oov_token = "<OOV>"
tokenizer = Tokenizer(oov_token = oov_token)
tokenizer.fit_on_texts(X)

word_index = tokenizer.word_index
vocab_size = len(word_index)

X_seq = tokenizer.texts_to_sequences(X)
X_padded = pad_sequences(X_seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)

print('Vocab Size = ', vocab_size)
print(X_padded[0])
train_padded, test_padded, train_label, test_label = train_test_split(X_padded, y, test_size = 0.15, random_state = 10)
train_padded = np.array(train_padded)
test_padded = np.array(test_padded)
train_label = np.array(train_label)
test_label = np.array(test_label)

print(train_padded.shape, test_padded.shape, train_label.shape, test_label.shape)
embeddings_index = dict()
f = open('../input/glove6b50dtxt/glove.6B.50d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
embedding_matrix = np.zeros((vocab_size, 50))
for word, index in tokenizer.word_index.items():
    if index > vocab_size - 1:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length = max_length, weights = [embedding_matrix], trainable = False))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(128, return_sequences = True, input_shape = train_padded.shape))
model.add(LSTM(64, dropout = 0.2, recurrent_dropout = 0.2))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

print(model.summary())
num_epochs = 5

history = model.fit(train_padded, train_label, epochs=num_epochs, validation_data=(test_padded, test_label), verbose=2)
import matplotlib.pyplot as plt


acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc))


plt.plot(epochs, acc, 'r')
plt.plot(epochs, val_acc, 'b')
plt.title('Training and validation accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["Accuracy", "Validation Accuracy"])

plt.figure()


plt.plot(epochs, loss, 'r')
plt.plot(epochs, val_loss, 'b')
plt.title('Training and validation loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Loss", "Validation Loss"])

plt.figure()
