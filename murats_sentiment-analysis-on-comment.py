import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, GRU, Embedding, CuDNNGRU

from tensorflow.python.keras.optimizers import Adam

from tensorflow.python.keras.preprocessing.text import Tokenizer

from tensorflow.python.keras.preprocessing.sequence import pad_sequences
df = pd.read_csv("../input/hepsiburada.csv")

df.head()
X = df["Review"].values.tolist()

y = df["Rating"].values.tolist()
cutoff = int(len(df) *.8)

X_train, X_test = X[:cutoff], X[cutoff:]

y_train, y_test = y[:cutoff], y[cutoff:]

#cutoff our threshold value. we divide, as before cutoff and after cutoff.
X_train[800] # example of positive comment(turkish).
y_train[800] # positive number is 1, negative number is 0.
num_words = 10000

tokenizer = Tokenizer(num_words = num_words) #numwords default is "None"
tokenizer.fit_on_texts(X)

# X was comments, now we converting to token.
counter = 0

for word in tokenizer.word_index:

    counter = counter + 1

    print(counter, "-> ", word)

    if(counter == 20):

        break

# this code shows most using first 20 token word.
X_train_tokens = tokenizer.texts_to_sequences(X_train)

X_test_tokens = tokenizer.texts_to_sequences(X_test)
X_train[800]
np.array(X_train_tokens[800])
num_tokens = [len(tokens) for tokens in X_train_tokens + X_test_tokens]

num_tokens = np.array(num_tokens)

num_tokens
np.mean(num_tokens)
np.max(num_tokens)
np.argmax(num_tokens)
print("The longest comment: ", X_train[21941], "\n\n", "Token of the longest comment :", X_train_tokens[21941])
import math

max_tokens = np.mean(num_tokens) + 2*np.std(num_tokens)

max_tokens = math.ceil(max_tokens)

max_tokens
np.sum(num_tokens < max_tokens) / len(num_tokens)
X_train_pad = pad_sequences(X_train_tokens, maxlen = max_tokens)

X_test_pad = pad_sequences(X_test_tokens, maxlen = max_tokens)
print("X Train Shape : ", X_train_pad.shape, "\nX Test Shape : ", X_test_pad.shape)
np.array(X_train_tokens[800])
X_train_pad[800]
idx = tokenizer.word_index

inverse_map = dict(zip(idx.values(), idx.keys()))
def tokens_to_string(tokens):

    words = (inverse_map[token] for token in tokens if token != 0)

    text = " ".join(words)

    return text
X_train[800]
tokens_to_string(X_train_tokens[800])
model = Sequential()
embedding_size = 50 #vector size for each word.
model.add(Embedding(input_dim = num_words,

                   output_dim = embedding_size,

                   input_length = max_tokens,

                   name = "embedding_layer"))
model.add(CuDNNGRU(units = 16, return_sequences = True))

model.add(CuDNNGRU(units = 8, return_sequences = True))

model.add(CuDNNGRU(units = 4))

model.add(Dense(1, activation = "sigmoid"))
optimizer = Adam(lr = 1e-3)

model.compile(loss="binary_crossentropy",

             optimizer = optimizer,

             metrics = ["accuracy"])
model.summary()
model.fit(X_train_pad, y_train, epochs=5, batch_size = 256)
result = model.evaluate(X_test_pad, y_test)
result # loss value and hit rate
y_pred = model.predict(x = X_test_pad[0:1000])

y_pred = y_pred.T[0]
cls_pred = np.array([1.0 if p > 0.5 else 0.0 for p in y_pred])
cls_true = np.array(y_test[0:1000])
incorrect = np.where(cls_pred != cls_true)

incorrect = incorrect[0]

len(incorrect)
idx = incorrect[0]

idx
text = X_test[idx]

text
y_pred[idx]
text1 = "bu ??r??n ??ok iyi herkese tavsiye ederim"

text2 = "kargo ??ok h??zl?? ayn?? g??n elime ge??ti"

text3 = "tam bir fiyat performans ??r??n??"

text4 = "m??kemmel"

text5 = "k??t?? yorumlar g??z??m?? korkutmu??tu ancak hi??bir sorun ya??amad??m te??ekk??rler"

text6 = "bu ??r??n?? tasarlayan m??hendis izin vaktinde ??zene bezene yapm???? "

text7 = "b??y??k bir hayal k??r??kl?????? ya??ad??m bu ??r??n bu markaya yak????mam????"

text8 = "tasar??m?? harika ancak kargo ??ok ge?? geldi ve ??r??n a????lm????t?? tavsiye etmem"

text9 = "hi?? resimde g??sterildi??i gibi de??il"

text10 = "hi?? bu kadar k??t?? bir sat??c??ya denk gelmemi??tim ??r??n?? geri iade ediyorum"

text11 = "bekledi??im gibi ????kmad??"

text12 = "i??inden h??yar ????kt??, biliydim beyle elaca??n??"



texts = [text1, text2, text3, text4, text5, text6, text7, text8, text9, text10, text11, text12]
tokens = tokenizer.texts_to_sequences(texts)
tokens_pad = pad_sequences(tokens, maxlen=max_tokens)

tokens_pad.shape
z_pred = model.predict(tokens_pad)

pred = np.array(["a positive comment." if p > 0.5 else "a negative comment." for p in z_pred])

counter = 1

for i in pred:

    print(counter, ". comment is ",i)

    counter = counter +1