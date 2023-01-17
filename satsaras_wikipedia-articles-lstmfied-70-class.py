# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pickle



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import spacy

from gensim.parsing.preprocessing import remove_stopwords

from nltk.corpus  import stopwords

import re

from gensim.utils import lemmatize

import matplotlib.pyplot as plt

import keras

from keras.models import load_model

from keras.layers import Bidirectional

from keras.preprocessing.text import Tokenizer

from keras.preprocessing import text

from keras.preprocessing.sequence import pad_sequences

from keras.utils.np_utils import to_categorical
nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser",'ner'])
df_train=pd.read_csv('/kaggle/input/dbpedia-classes/DBPEDIA_train.csv',encoding='utf-8-sig')
df_val=pd.read_csv('/kaggle/input/dbpedia-classes/DBPEDIA_val.csv',encoding='utf-8-sig')
df_test=pd.read_csv('/kaggle/input/dbpedia-classes/DBPEDIA_test.csv',encoding='utf-8-sig')
import keras

from keras.preprocessing.text import Tokenizer

from keras.preprocessing import text

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Embedding, Flatten, Dense,LSTM,Dropout,Conv1D,MaxPooling1D
classes=dict(list(zip(set(df_train.l2.unique()),range(0,70))))
df_train['l2']=df_train['l2'].map(classes)

df_val['l2']=df_val['l2'].map(classes)

df_test['l2']=df_test['l2'].map(classes)

glove_dir="/kaggle/input/glove6b100dtxt"



embedding_index={}

f=open(os.path.join(glove_dir,'glove.6B.100d.txt'),encoding='utf8')

for line in f:

    values=line.split()

    word=values[0]

    coefs=np.asarray(values[1:],dtype='float32')

    embedding_index[word]=coefs

f.close()

print('Found %s word vectors ' % len(embedding_index))
stop = stopwords.words('english')
def cleaning(df):

    df.loc[:,'text']=pd.DataFrame(df.loc[:,'text'].str.lower())

    #df.loc[:,'text'] = [re.sub(r'\d+','', i) for i in df.loc[:,'text']]

    df.loc[:,'text'] = [re.sub(r'[^a-zA-Z]',' ', i) for i in df.loc[:,'text']]

    df.loc[:,'text'] = [re.sub(r"\b[a-zA-Z]\b", ' ', i) for i in df.loc[:,'text']]

    

    #df.loc[:,'text'] = [re.sub(r"[#|\.|_|\^|\$|\&|=|;|,|‐|-|–|(|)|//|\\+|\|*|\']+",'', i) for i in df.loc[:,'text']]

    df.loc[:,'text'] = [re.sub(' +',' ', i) for i in df.loc[:,'text']]

    return(df)

    
def lemmatization(df, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):

    texts_out = []

    for sent in df.loc[:,'text']:

        #print(len(sent))

        doc = nlp(sent)

        texts_out.append([token.lemma_ for token in doc])

    return(texts_out)
def textprocessing(df,Is_Test=1):

    df=cleaning(df)

    df['lemmatized_text_token']=lemmatization(df)

    df['lemmatized_text_token']=df['lemmatized_text_token'].apply(lambda x:[i for i in x if i not in (stop) ])

    #df_train_trunc=df_train[df_train['lemmatized_text_token'].apply(lambda c: len(c))>15]

    if Is_Test:

        with open('tokenizer.pickle', 'rb') as handle:

            tokenizer = pickle.load(handle)

    else:

        tokenizer=Tokenizer(oov_token='<unknown>')

        tokenizer.fit_on_texts(df['lemmatized_text_token'])

        with open('tokenizer.pickle', 'wb') as handle:

            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    word2id = tokenizer.word_index

    id2word = {v:k for k, v in word2id.items()}

    vocab_size = len(word2id) + 1 

    sequence=tokenizer.texts_to_sequences(df['lemmatized_text_token'])

    sequence=pad_sequences(sequence,maxlen=200)

    if Is_Test:

        return(sequence,word2id)

    else:

        return(sequence,vocab_size,word2id)
train_X,vocab_size,word2id=textprocessing(df_train,0)
val_X,word2id=textprocessing(df_val,1)
test_X,word2id=textprocessing(df_test,1)
maxlen=200

max_words=vocab_size



embedding_dim=100

embedding_matrix=np.zeros((max_words,embedding_dim))



for word, i in word2id.items():

    if i <max_words:

        embedding_vector=embedding_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[i]=embedding_vector
model_UniDirectional=Sequential()

model_UniDirectional.add(Embedding(max_words,embedding_dim,input_length=maxlen))

model_UniDirectional.add(Dropout(0.2))

model_UniDirectional.add(Conv1D(64, 5, activation='elu'))

model_UniDirectional.add(MaxPooling1D(pool_size=2))

model_UniDirectional.add(LSTM(196,return_sequences=True))

#model_UniDirectional.add(Dropout(0.2))

#model_UniDirectional.add(LSTM(132,return_sequences=True))

#model_UniDirectional.add(Dropout(0.2))

#model_UniDirectional.add(LSTM(64,return_sequences=True))

#model_UniDirectional.add(Dropout(0.2))

model_UniDirectional.add(LSTM(32))

model_UniDirectional.add(Dense(70,activation='softmax'))

model_UniDirectional.summary()
def model_training(model,train_X,train_Y,val_X,val_Y,model_name):

    model.layers[0].set_weights([embedding_matrix])

    model.layers[0].trainable=False

    model.compile(optimizer='adam',

             loss='categorical_crossentropy',

             metrics=['accuracy'])

    model.fit(train_X,train_Y,

                    batch_size=1024,

                    epochs=10,

                    validation_data=(val_X,val_Y)

                    )

    modelname_to_save=model_name+'.h5'

    model.save(modelname_to_save)

    return(model)
model_unidirectional=model_training(model_UniDirectional,train_X,to_categorical(df_train['l2']),val_X,to_categorical(df_val['l2']),'model_BiLSTM')
def plot_model(model):

    plt.plot(model.history.history['accuracy'])

    plt.plot(model.history.history['val_accuracy'])

    plt.title('model accuracy')

    plt.ylabel('accuracy')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.show()

    plt.plot(model.history.history['loss'])

    plt.plot(model.history.history['val_loss'])

    plt.title('model loss')

    plt.ylabel('loss')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.show()


plot_model(model_unidirectional)
m=model_unidirectional.evaluate(test_X,to_categorical(df_test['l2']))
print('Test Loss and Accuracy for unidirectional LSTM layer model is :',m)
model_BiDirectional=Sequential()

model_BiDirectional.add(Embedding(max_words,embedding_dim,input_length=maxlen))

model_BiDirectional.add(Dropout(0.2))

model_BiDirectional.add(Conv1D(64, 5, activation='elu'))

model_BiDirectional.add(MaxPooling1D(pool_size=2))

model_BiDirectional.add(Bidirectional(LSTM(196,return_sequences=True)))

model_BiDirectional.add(Bidirectional(LSTM(32)))

model_BiDirectional.add(Dense(70,activation='softmax'))

model_BiDirectional.summary()
model_BiDirectional_LSTM=model_training(model_BiDirectional,train_X,to_categorical(df_train['l2']),val_X,to_categorical(df_val['l2']),'model_BiLSTM')
plot_model(model_BiDirectional)
m1=model_BiDirectional.evaluate(test_X,to_categorical(df_test['l2']))
print('Test Loss and Accuracy for Bi-idirectional LSTM layer model is :',m1)