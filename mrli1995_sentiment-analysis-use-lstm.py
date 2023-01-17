import re

import numpy as np 

import pandas as pd 

import nltk



from sklearn.model_selection import train_test_split



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM, Dropout



from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau

data = pd.read_csv('../input/Sentiment.csv')

data = data[['text','sentiment']]

data.head()
data = data[data.sentiment != "Neutral"]

data.head()
data['text'] = data['text'].apply(lambda x: x.lower())

data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

data['text'] = data['text'].apply((lambda x: x.replace('rt',' ') ))

data.head()
print(data[ data['sentiment'] == 'Positive'].size)

print(data[ data['sentiment'] == 'Negative'].size)
Maxlen = 0

unique_words = set()

count = 0



for i in range(len(data['text'].values)):

    txt = data['text'].values[i]

    words = nltk.word_tokenize(txt)

    unique_words = unique_words.union(set(words))

    lenth = len(words)

    count += 1

    if lenth > Maxlen:

        Maxlen = lenth

print('all_sent:',count)

print('max length:',Maxlen)

print('total unique words:',len(unique_words))
max_fatures = 10000

Maxlen = 30



tokenizer = Tokenizer(num_words=max_fatures, split=' ')

tokenizer.fit_on_texts(data['text'].values)

X = tokenizer.texts_to_sequences(data['text'].values)

X = pad_sequences(X,maxlen = Maxlen)       #序列转化为经过填充以后的一个长度相同的新序列，用0填充。

X.shape
Y = np.zeros(len(data['sentiment'].values))



for i in range(len(data['sentiment'].values)):

    if data['sentiment'].values[i] == 'Positive':

        Y[i]=1
print(Y)

print(Y.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.25, random_state = 1)

print(X_train.shape,Y_train.shape)

print(X_test.shape,Y_test.shape)
embed_dim = 128

lstm_out = 64



model = Sequential()

model.add(Embedding(max_fatures, embed_dim,input_length = Maxlen))

model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))

model.add(Dropout(0.5))

model.add(Dense(1,activation='sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer='rmsprop',metrics = ['accuracy'])

model.summary()
# 当监测值不再改善时，该回调函数将中止训练

earlyStopping = EarlyStopping(monitor="val_loss", mode="min", patience=3,verbose=1)



# 在每个epoch后保存最优模型

modelCheckpoint = ModelCheckpoint('../my_model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')



# 当评价指标不再提升时，减少学习率

reduceLROnPlateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='min', min_delta=0.0001)



# callbacks是个列表形式

callbacks_list = [earlyStopping,modelCheckpoint,reduceLROnPlateau]
batch_size = 32

history = model.fit(X_train, Y_train, epochs = 20, batch_size=batch_size,validation_split=1/4,verbose = 1,callbacks=callbacks_list)
score,acc = model.evaluate(X_test, Y_test, verbose = 1, batch_size = batch_size)

print("loss: %.2f" % (score))

print("accuracy: %.2f" % (acc))
test1 = ['this is a great kernel']

print('the sentence is：',test1[0])



test1 = tokenizer.texts_to_sequences(test1)

test1 = pad_sequences(test1, maxlen=Maxlen, dtype='int32', value=0)



sentiment = model.predict(test1)[0][0]



if sentiment > 0.5:

    print("I am {:.2%} sure it's Positive".format(sentiment))

else:

    print("I am {:.2%}sure it's Negative".format(1- sentiment))
test2 = ['this is a bad kernel']

print('the sentence is：',test2[0])



test2 = tokenizer.texts_to_sequences(test2)

test2 = pad_sequences(test2, maxlen=Maxlen, dtype='int32', value=0)



sentiment = model.predict(test2)[0][0]



if sentiment > 0.5:

    print("I am {:.2%} sure it's Positive".format(sentiment))

else:

    print("I am {:.2%} sure it's Negative".format(1- sentiment))