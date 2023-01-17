from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
dataset = pd.read_csv('train.csv')
dataset.head()
x_text = np.array(dataset.iloc[1:,1])
y = np.array(dataset.iloc[1:,2:], dtype='float32')
from nltk.corpus import stopwords
import string

def normalize(x_text):
    stop = stopwords.words('english')
    res = [' '.join([t for t in doc.split() if t not in string.punctuation and t not in  stop]) for doc in x_text]
    return res
def tokenize_data(x_text):
    tokenizer = Tokenizer(lower=True)
    tokenizer.fit_on_texts(x_text)
    x_tokenized_doc = []
    for doc in x_text:
        tk_doc = tokenizer.texts_to_sequences(doc)
        x_tk_optimized = [x_doc for x_doc in tk_doc if len(x_doc)>0]
        x_tokenized_doc.append(x_tk_optimized)
    return x_tokenized_doc, tokenizer
x_text = normalize(x_text)
x_tokenized, x_tk= tokenize_data(x_text)
index_to_words = {id: word for word, id in x_tk.word_index.items()}
max_vocab = len(set([word for word, id in x_tk.word_index.items()]))
max_len = max([len(item) for item in x_tokenized])
print(max_len)
print(max_vocab)
words_to_index = {word: id for word, id in x_tk.word_index.items()}
x_array = []
for x_doc in x_tokenized:
    x_list = [x_item[0] for x_item in x_doc]
    x_array.append(x_list)

x_all =pad_sequences(x_array, maxlen=2000, dtype='int32', padding='post', truncating='post', value=0.0)
print(x_all)
x = np.array(x_all)
print(x.shape)
print(y.shape[1])
learning_rate=0.001

model = Sequential()
model.add(Embedding(max_vocab, 100, input_length=2000))
model.add(Dense(64, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='tanh'))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(y.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate), metrics=['accuracy'])

model.summary()
num_epoch=50
checkpoint = ModelCheckpoint(filepath='best_model_1.hdf5', save_best_only=True)
hist = model.fit(x, y, batch_size=200, epochs=num_epoch, validation_split=0.2, callbacks=[checkpoint], shuffle=True, verbose=2)
train_loss = hist.history['loss']
val_loss   = hist.history['val_loss']
train_acc  = hist.history['acc']
val_acc    = hist.history['val_acc']
xc         = range(num_epoch)
plt.figure()
plt.plot(xc, train_loss, color='red')
plt.plot(xc, val_loss, color='green')
plt.show()
test_dataset = pd.read_csv('test.csv')
test_dataset.head()
x_text_test = np.array(test_dataset.iloc[1:,1])
x_text_test = normalize(x_text_test)
x_tokenized_doc_test = []
for doc in x_text_test:
    tk_doc = x_tk.texts_to_sequences(doc)
    x_tk_optimized = [x_doc for x_doc in tk_doc if len(x_doc)>0]
    x_tokenized_doc_test.append(x_tk_optimized)
x_tokenized_doc_test[:1]
x_array_test = []
for x_doc in x_tokenized_doc_test:
    x_list = [x_item[0] for x_item in x_doc]
    x_array_test.append(x_list)
x_array_test[0]
len(x_array_test)
model.load_weights('best_model_1.hdf5')
final_sub = []
for i in range(10):
    x_all_test =pad_sequences([x_array_test[i]], maxlen=2000, dtype='int32', padding='post', truncating='post', value=0.0)
    x_test = np.array(x_all_test)
    probab = model.predict_proba(x_test)
    print(test_dataset.iloc[i][0])
    print(test_dataset.iloc[i][1])
    print(probab)

