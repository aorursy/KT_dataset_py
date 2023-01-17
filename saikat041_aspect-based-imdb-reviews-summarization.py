import re

import numpy as np

import pandas as pd 

from keras.models import Sequential

from keras.layers import Dense, Flatten, Embedding

from keras.layers import Bidirectional, GlobalMaxPool1D, LSTM, Dropout

from keras.layers import Conv1D, GlobalMaxPooling1D

from keras.layers import MaxPooling1D



from matplotlib import pyplot

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize, sent_tokenize

from nltk.stem import PorterStemmer



porter=PorterStemmer()

stop_words = stopwords.words('english')
def clean_text(text):

    text = re.sub(r'[^\w\s]','',text, re.UNICODE)

    text = text.lower()

    tokens = word_tokenize(text)

#     tokens = [word for word in tokens if word not in stop_words]

    tokens = [porter.stem(word) for word in tokens]

    tokens = " ".join(tokens)

    return tokens





def clean_corpus():



    train_pos = open('../input/imdb-dataset/train-pos.txt','r').read().split('\n')[0:12500]

    train_neg = open('../input/imdb-dataset/train-neg.txt','r').read().split('\n')[0:12500]

    test_pos = open('../input/imdb-dataset/test-pos.txt','r').read().split('\n')[0:12500]

    test_neg = open('../input/imdb-dataset/test-neg.txt','r').read().split('\n')[0:12500]

    

    train_pos = [clean_text(line) for line in train_pos]

    train_neg = [clean_text(line) for line in train_neg]

    test_pos = [clean_text(line) for line in test_pos]

    test_neg = [clean_text(line) for line in test_neg]

    

    return train_pos, train_neg, test_pos, test_neg



def prepare_data1(top_words):

    vocab = {}

    train_neg = open('../input/sentence-polarity-dataset/sent_neg.txt').read().split('\n')

    train_pos = open('../input/sentence-polarity-dataset/sent_pos.txt').read().split('\n')

    

    train_neg = [clean_text(line) for line in train_neg]

    train_pos = [clean_text(line) for line in train_pos]

    

    train = train_neg + train_pos

    

    for sent in train:

        for word in sent.split():

            if word in vocab:

                vocab[word]+=1

            else:

                vocab[word]=1

                    

    vocab_arr = sorted(vocab.items(), key=lambda x:x[1], reverse=True)

    vocab_arr = [x[0] for x in vocab_arr]

    top_vocab = {word:index for index,word in enumerate(vocab_arr[0:top_words])}  # index 0 reserved for padding and OOV words

    

    x_train = []

    y_train = []

    

    for index,sent in enumerate(train_neg):

        t = []

        for word in sent.split():

            if word in top_vocab:

                t.append(top_vocab[word])

            else:

                t.append(0)

                

        x_train.append(t)

        y_train.append(0)

        

    for index,sent in enumerate(train_pos):

        t = []

        for word in sent.split():

            if word in top_vocab:

                t.append(top_vocab[word])

            else:

                t.append(0)

                

        x_train.append(t)

        y_train.append(1)

        

        

    return top_vocab, x_train, y_train



def prepare_data2(top_words):                                                           # for deep learning

    vocab = {}

    

    train = train_pos + train_neg

    test = test_pos + test_neg

    

    for sent in train:

        for word in sent.split():

            if word in vocab:

                vocab[word]+=1

            else:

                vocab[word]=1

    vocab_arr = sorted(vocab.items(), key=lambda x:x[1], reverse=True)

    vocab_arr = [x[0] for x in vocab_arr]

    top_vocab = {word:index for index,word in enumerate(vocab_arr[0:top_words])}  # index 0 reserved for padding and OOV words

    

    x_train = []

    y_train = []

    

    x_test = []

    y_test = []

    

    for index,sent in enumerate(train):

        t = []

        for word in sent.split():

            if word in top_vocab:

                t.append(top_vocab[word])

            else:

                t.append(0)

                

        x_train.append(t)

        

        if index<12500:

            y_train.append(1)

        else:

            y_train.append(0)



    for index,sent in enumerate(test):

        t = []

        for word in sent.split():

            if word in top_vocab:

                t.append(top_vocab[word])

            else:

                t.append(0)

                

        x_test.append(t)

        

        if index<12500:

            y_test.append(1)

        else:

            y_test.append(0)

            

    return top_vocab, (x_train,y_train), (x_test, y_test)



def padSequences(arr, max_words):        

    ans = [x[0:max_words] if len(x)>=max_words else [x[i] if i<len(x) else 0 for i in range(max_words)] for x in arr]

    return ans



def shuffle(x, y):

    p = np.random.permutation(len(x))

    return x[p],y[p]



def give_phrases(sent):

    sent = sent.replace(',','.')

    sent = sent.replace('and','.')

    sent = sent.replace('but','.')

    

    return sent.split('.')



def check_sentences(model,word, movie_no, df):

    

    reviews = df.iloc[:,3].values

    movie_nos = df.iloc[:,1].values

    

    phrases = []

    

    for i in range(len(df)):

        if movie_nos[i]==movie_no:

            phrases+=give_phrases(reviews[i])

            

    pos_sen = []

    neg_sen = []



   

    for phrase in phrases:

        if word in phrase:

            sen = sentiment(model,phrase)

            if sen == 1:

                pos_sen.append(phrase)

            else:

                neg_sen.append(phrase)

              

    return pos_sen, neg_sen



def sentiment(model, sent):

    sent = clean_text(sent)

    

    vec = [top_vocab[w] if w in top_vocab else 0 for w in sent.split()]

    vec  = padSequences([vec], max_words)[0]

    

    pred = model.predict(np.array([vec]))[0][0]

    

    if pred>=0.5:

        return 1

    else:

        return 0
train_pos, train_neg, test_pos, test_neg = clean_corpus()
max_words = 500

vocab_size = 10000

embedding_dim = 32



top_vocab, (x_train, y_train), (x_test, y_test) = prepare_data2(vocab_size)

x_train = padSequences(x_train, max_words)

x_test = padSequences(x_test, max_words)

x_train,y_train,x_test,y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
print("Training CNN Model")                                                

model = Sequential()

model.add(Embedding(vocab_size, embedding_dim, input_length=max_words))



model.add(Conv1D(filters=32, kernel_size=8, padding='same', activation='relu')) 

model.add(MaxPooling1D(pool_size=2))



model.add(Flatten())                                                                                            #keras model for CNN

model.add(Dense(250, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=2, batch_size=128, verbose=1)

scores = model.evaluate(x_test, y_test, verbose=0)

print("Accuracy: %.2f%%" % (scores[1]*100))

cnnModel = model
# print("Training RNN Model")

# model = Sequential()

# model.add(Embedding(vocab_size, embedding_dim))

# model.add(Bidirectional(LSTM(32, return_sequences = True)))

# model.add(GlobalMaxPool1D())    

# model.add(Dense(32, activation="relu"))                                                                        #keras model RNN                          

# # model.add(Dropout(0.2)) 

# model.add(Dense(1, activation="sigmoid"))

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# history = model.fit(x_train, y_train,  validation_data=(x_test, y_test), epochs=5, batch_size=128, verbose=1)

# scores = model.evaluate(x_test, y_test, verbose=0)

# print("Accuracy: %.2f%%" % (scores[1]*100))

# rnnModel = model
df=pd.read_csv('../input/imdb-reviews-movie-wise/movies_250.csv',encoding="latin-1",header=None)
pos, neg = check_sentences(cnnModel,"acting", 4, df)

print('Positive : ', len(pos), 'Negative : ', len(neg))

print(pos,neg)