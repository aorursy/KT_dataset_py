# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk

from nltk.corpus import stopwords

from keras.preprocessing.text import Tokenizer

from nltk.tokenize import TreebankWordTokenizer

from nltk.tokenize import MWETokenizer



%matplotlib inline

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

ts=pd.read_csv('../input/sms-spam-collection-dataset/spam.csv',encoding='latin-1')#문자열

del ts['Unnamed: 2']

del ts['Unnamed: 3']

del ts['Unnamed: 4']

ts['v1'] = ts['v1'].replace(['ham','spam'],[0,1])



x=ts['v2']

y=ts['v1']



ts2=pd.read_csv('../input/spam-filter/emails.csv')#1일 때 스팸

x2=ts2['text']

y2=ts2['spam']



ts3=pd.read_csv('../input/spam-or-not-spam-dataset/spam_or_not_spam.csv')#1일 때 스팸 , 메일 내용 중 비어있는 것이 있음



ts3 = ts3.dropna(axis=0) #Nan값 제거



x3=ts3['email']

y3=ts3['label']



ts4=pd.read_csv('../input/spam-mails-dataset/spam_ham_dataset.csv')#1일 때 스팸

x4=ts4['text']

y4=ts4['label_num']



train_x=pd.concat((x,x2,x3,x4),axis=0,ignore_index=True)

train_y=pd.concat((y,y2,y3,y4),axis=0,ignore_index=True)



#tok=Tokenizer()

##tok.fit_on_texts(train_x)

#print(tok.sequences_to_texts)

#sequences = tok.texts_to_sequences(train_x)







# Any results you write to the current directory are saved as output.
train_set=pd.concat((train_x,train_y),axis=1,ignore_index=True)

train_set=train_set.sample(frac=1)

train_x=train_set[0]

train_y=train_set[1]
from nltk.tokenize import sent_tokenize

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from nltk.tokenize import TreebankWordTokenizer

from nltk.tokenize import MWETokenizer

from collections import Counter

index=0

vocab=Counter()

train_x_tok={}

for text in train_x:

    tokenized_text = sent_tokenize(text)

    

    sentences = []

    stop_words = set(stopwords.words('english'))

    

    for i in tokenized_text:

        sentence = nltk.word_tokenize(i);

        #sentence = TreebankWordTokenizer().tokenize(i) # 단어 토큰화를 수행한다. -Treebank 사용

        result = []

    

        for word in sentence:

            word = word.lower() # 모든 단어를 소문자화하여 단어 개수를 줄인다

            if word not in stop_words: # 단어 토큰화된 결과에 대해서 불용어 제거

                if len(word) > 2: #길이가 2 이하인 경우에 대하여 추가로 단어 제거]

                    if len(word) < 20:

                        sentences.append(word) 

                        vocab[word] = vocab[word]+1 # 각 단어의 빈도를 Count한다

                    #word_count=word_count+1;

        #sentences.append(result)

    train_x_tok[index]=sentences

    index+=1

        

vocab_sorted = sorted(vocab.items(), key=lambda x:x[1], reverse=True)

    

word_to_index = {}

i = 0

for (word, frequency) in vocab_sorted:

        if frequency > 1: # 빈도수가 적은 단어는 제외

            i += 1

            word_to_index[word] = i

word_index=len(word_to_index)
print(train_x_tok[5])

train_x_idx=[]

for mail in train_x_tok:

    new_mail=[]

    for word in train_x_tok[mail]:

        if word in word_to_index.keys():

            new_mail.append(word_to_index[word])

    train_x_idx.append(new_mail)

    

print(train_x_idx[:5])
n_train = int(19470 * 0.9)

n_test = int(19470 - n_train)



#word_index = tok.word_index

#X_data=sequences

X_data=train_x_idx

print('단어 사전의 크기 :',word_index)

print('메일의 최대 길이 :',max(len(l) for l in X_data))

print('메일의 평균 길이 :',sum(map(len, X_data))/len(X_data))

plt.hist([len(s) for s in X_data], bins=50)

plt.xlabel('length of Data')

plt.ylabel('number of Data')

plt.show()
from keras.layers import SimpleRNN, Embedding, Dense, LSTM

from keras.models import Sequential

from keras.preprocessing.sequence import pad_sequences

vocab_size = word_index+1



max_len=7556



data = pad_sequences(X_data, maxlen=max_len)

print("data shape: ", data.shape)



x_test=data[n_train:]

y_test=train_y[n_train:]

x_train=data[:n_train]

y_train=train_y[:n_train]

print("x_train shape: ", x_train.shape)
model = Sequential()

model.add(Embedding(vocab_size, 32))

model.add(LSTM(32))

model.add(Dense(1,init='he_uniform', activation='sigmoid'))



model.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['acc'])

history = model.fit(x_train, y_train, epochs=4, batch_size=256, validation_split=0.1)
print("\n 테스트 정확도: %.4f" % (model.evaluate(x_test, y_test)[1]))
epochs = range(1, len(history.history['acc']) + 1)

plt.plot(epochs, history.history['loss'])

plt.plot(epochs, history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()