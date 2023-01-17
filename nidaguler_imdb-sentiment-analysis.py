import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats



from keras.datasets import imdb

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers.embeddings import Embedding

from keras.layers import SimpleRNN, Dense, Activation
(x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",

                                                      num_words=None,

                                                      skip_top=0,

                                                      maxlen=None,

                                                      seed=113,

                                                      start_char=1,

                                                      oov_char=2,

                                                      index_from=3)
print("Y Train Values:",np.unique(y_train))

print("Y Test Values:",np.unique(y_test))
unique, counts = np.unique(y_train, return_counts=True)

print("Y Train distrubution:",dict(zip(unique,counts)))
unique, counts = np.unique(y_test, return_counts=True)

print("Y Test distrubution:",dict(zip(unique,counts)))
plt.figure()

sns.countplot(y_train)

plt.xlabel("Classes")

plt.ylabel("Freq")

plt.title("y train")

plt.show()
plt.figure()

sns.countplot(y_test)

plt.xlabel("Classes")

plt.ylabel("Freq")

plt.title("y test")

plt.show()
d=x_train[0]

print(x_train[0])
print(len(d))
review_len_train=[]

review_len_test=[]

for i, ii in zip(x_train,x_test):

    review_len_train.append(len(i))

    review_len_test.append(len(ii))
sns.distplot(review_len_train, hist_kws={"alpha":0.3})

sns.distplot(review_len_test, hist_kws={"alpha":0.3})

plt.show()

print("Train mean:",np.mean(review_len_train))

print("Train median:",np.median(review_len_train))

print("Train mode:",stats.mode(review_len_train))
#number of words

word_index=imdb.get_word_index()

print(type(word_index))

print(len(word_index))
for keys, values in word_index.items():

    if values==1:

        print(keys)
for keys, values in word_index.items():

    if values==4:

        print(keys)
for keys, values in word_index.items():

    if values==123:

        print(keys)
def whatItSay(index=9):

    reverse_index = dict([(value,key) for(key,value) in word_index.items()])

    decode_review=" ".join([reverse_index.get(i-3, "!") for i in x_train[index]])

    print(decode_review)

    print(y_train[index])

    return decode_review

decoded_review = whatItSay()
#preprocess

num_words=15000

(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=num_words)

maxlen=130

x_train=pad_sequences(x_train,maxlen=maxlen)

x_test=pad_sequences(x_test,maxlen=maxlen)
print(x_train[5])
for i in x_train[0:10]:

    print(len(i))
decoded_review=whatItSay(5)
#rnn

rnn=Sequential()

rnn.add(Embedding(num_words, 32, input_length=len(x_train[0])))

rnn.add(SimpleRNN(16,input_shape = (num_words,maxlen), return_sequences=False,activation="relu"))

rnn.add(Dense(1))

rnn.add(Activation("sigmoid"))



print(rnn.summary())

rnn.compile(loss="binary_crossentropy", optimizer="rmsprop",metrics=["accuracy"])
history=rnn.fit(x_train,y_train, validation_data=(x_test,y_test),epochs=6, batch_size=128, verbose=1)

score=rnn.evaluate(x_test,y_test)

print("Accuracy: %",score[1]*100)
plt.figure()

plt.plot(history.history["accuracy"],label="train")

plt.plot(history.history["val_accuracy"], label="test")

plt.title("accuracy")

plt.ylabel("accuracy")

plt.xlabel("epochs")

plt.legend()

plt.show()
plt.figure()

plt.plot(history.history["loss"],label="train")

plt.plot(history.history["val_loss"], label="test")

plt.title("acc")

plt.ylabel("Acc")

plt.xlabel("epochs")

plt.legend()

plt.show()