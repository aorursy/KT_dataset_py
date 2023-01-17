import pandas as pd

import numpy as np



import nltk

from nltk.corpus import stopwords



import re



from sklearn.model_selection import train_test_split



import gensim



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



from keras.models import Sequential

from keras.layers import Activation, Dense, Dropout, LSTM, Embedding

from keras.utils import to_categorical
data = pd.read_csv("../input/train.tsv", sep="\t")
data.head(20)
data.shape
data["label"].value_counts()
# dropping id column



data = data.drop("id", axis=1)
stopwords = stopwords.words('english')
def tweet_cleaner(tweet):

    tweet = re.sub(r"@\w*", " ", str(tweet).lower()).strip() #removing username

    tweet = re.sub(r'https?://[A-Za-z0-9./]+', " ", str(tweet).lower()).strip() #removing links

    tweet = re.sub(r'[^a-zA-Z]', " ", str(tweet).lower()).strip() #removing sp_char

    tw = []

    

    for text in tweet.split():

        if text not in stopwords:

            tw.append(text)

    

    return " ".join(tw)
data.tweet = data.tweet.apply(lambda x: tweet_cleaner(x))
documents = [text.split() for text in data.tweet]
len(documents)
w2v_model = gensim.models.word2vec.Word2Vec(size = 256, window = 7, min_count = 5)
w2v_model.build_vocab(documents)
w2v_model.train(documents, total_examples=len(documents), epochs=32)
w2v_model.wv["books"]
tokenizer = Tokenizer()

tokenizer.fit_on_texts(data.tweet)
len(tokenizer.word_index)
x_train = pad_sequences(tokenizer.texts_to_sequences(data.tweet), maxlen=256, padding="post", truncating="post")
x_train
y_train = data.label



y_train_f = []

for x in y_train:

    if x == 1:

        y_train_f.append(1)

    elif x == 0:

        y_train_f.append(0)

    elif x == -1:

        y_train_f.append(2)

        

y_train_f = to_categorical(y_train_f)
y_train_f
embedding_matrix = np.zeros((14850,256))
for word, i in tokenizer.word_index.items():

    if word in w2v_model.wv:

        embedding_matrix[i] = w2v_model.wv[word]
embedding_layer = Embedding(14850, 256, weights=[embedding_matrix], input_length=256, trainable=False)
model = Sequential()

model.add(embedding_layer)

model.add(Dropout(0.25))

model.add(Dense(200, activation="relu"))

model.add(Dropout(0.25))

model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(50, activation="relu"))

model.add(Dense(3, activation="softmax"))
model.summary()
model.compile(loss='categorical_crossentropy',

              optimizer="adam",

              metrics=['accuracy'])
model.fit(x_train, y_train_f, batch_size=32, epochs=4, validation_split=0.1, verbose=1)
def sentiment(text):

    

    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=256)

    score = model.predict([x_test])[0]

    

    final = "Positive = %f ,Negative = %f, Neutral = %f" % (score[1], score[2], score[0])

    return print(final)
sentiment("I like reading books.")