%matplotlib inline
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
# from tf.keras.models import Sequential  # This does not work!
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import gc
tf.__version__
tf.keras.__version__
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
test_data.isnull().any()
test_data.isnull().any()
test_data.head()
x_train_text = train_data.question_text
y_train = train_data.target
x_test_text = test_data.question_text
print("Train-set size: ", len(x_train_text))
print("Test-set size:  ", len(x_test_text))
data_text = list(train_data['question_text'].values) + list(test_data['question_text'].values)
x_train_text[1]
y_train[1]
num_words = 50000
tokenizer = Tokenizer(num_words=num_words)
%%time
tokenizer.fit_on_texts(data_text)
if num_words is None:
    num_words = len(tokenizer.word_index)
tokenizer.word_index
x_train_tokens = tokenizer.texts_to_sequences(x_train_text)
x_train_text[1]
np.array(x_train_tokens[1])
x_test_tokens = tokenizer.texts_to_sequences(x_test_text)
num_tokens = [len(tokens) for tokens in x_train_tokens + x_test_tokens]
num_tokens = np.array(num_tokens)
np.mean(num_tokens)
np.max(num_tokens)
np.min(num_tokens)
max_tokens = 100
pad = 'pre'
x_train_pad = pad_sequences(x_train_tokens, maxlen=max_tokens,
                            padding=pad, truncating=pad)
x_test_pad = pad_sequences(x_test_tokens, maxlen=max_tokens,
                           padding=pad, truncating=pad)
x_train_pad.shape
x_test_pad.shape
np.array(x_train_tokens[1])
x_train_pad[1]
idx = tokenizer.word_index
inverse_map = dict(zip(idx.values(), idx.keys()))
def tokens_to_string(tokens):
    # Map from tokens back to words.
    words = [inverse_map[token] for token in tokens if token != 0]
    
    # Concatenate all words.
    text = " ".join(words)

    return text
x_train_text[1]
tokens_to_string(x_train_tokens[1])
EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(num_words, len(word_index))
embedding_matrix_1 = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= num_words: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix_1[i] = embedding_vector

del embeddings_index; gc.collect() 
EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(num_words, len(word_index))
embedding_matrix_2 = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= num_words: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix_2[i] = embedding_vector
del embeddings_index; gc.collect()
EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(num_words, len(word_index))
embedding_matrix_3 = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= num_words: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix_3[i] = embedding_vector
        
del embeddings_index; gc.collect()  
embedding_matrix = np.concatenate((embedding_matrix_1, embedding_matrix_2, embedding_matrix_3), axis=1)  
del embedding_matrix_1, embedding_matrix_2, embedding_matrix_3
gc.collect()
np.shape(embedding_matrix)

model = Sequential()
embedding_size = 300
model.add(Embedding(num_words, embedding_size * 3, weights=[embedding_matrix], trainable=False))
model.add(GRU(units=16, return_sequences=True))
model.add(GRU(units=8, return_sequences=True))
model.add(GRU(units=4))
model.add(Dense(1, activation='sigmoid'))
optimizer = Adam(lr=1e-3)
model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
model.summary()
%%time
model.fit(x_train_pad, y_train,
          validation_split=0.05, epochs=2, batch_size=512)
out = model.predict(x_test_pad,batch_size=256)
out_df = pd.DataFrame({"qid":test_data["qid"].values})
out_pred = (out>0.35).astype(int)
out_df['prediction'] = out_pred
out_df.to_csv("submission.csv", index=False)