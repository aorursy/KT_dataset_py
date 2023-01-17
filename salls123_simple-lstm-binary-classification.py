from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.preprocessing.text import Tokenizer

from matplotlib import pyplot as plt

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Embedding,Dense,LSTM,Dropout,Flatten,BatchNormalization,Conv1D,GlobalMaxPooling1D,MaxPooling1D

from keras.optimizers import  SGD

import matplotlib.pyplot as plt

from keras.regularizers import l2

from keras.optimizers import Adam

from keras import regularizers

from keras.callbacks import EarlyStopping

from sklearn.preprocessing import OneHotEncoder

from keras.preprocessing import sequence

#from hyperas.distributions import uniform



from keras.utils.np_utils import to_categorical

from keras import regularizers

import pandas as pd

import string

import numpy as np

import matplotlib.pyplot as plt



import os
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



train= pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')[['text', 'target']]

test= pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')[['text','id']]



# train=pd.read_csv("train.csv")[['text', 'target']]

# test=pd.read_csv("test.csv")[['text','id']]
X=train["text"]

Y=train["target"].astype(int)
def clean_text(txt):

    txt = "".join(v for v in txt if v not in string.punctuation.lower())

    return txt

X=[clean_text(i) for i in X]
tokenizer = Tokenizer()

tokenizer.fit_on_texts(X)



def get_sequence_of_tokens(corpus):

    ## tokenization

    

    total_words = len(tokenizer.word_index) + 1

    

    ## convert data to sequence of tokens 

    input_sequences = []

    for line in corpus:

        token_list = tokenizer.texts_to_sequences([line])[0]

        input_sequences.append(token_list)

    return input_sequences, total_words



inp_sequences, total_words = get_sequence_of_tokens(X)
def get_sequence_of_tokens_pred(corpus):

    ## tokenization

    

    total_words = len(tokenizer.word_index) + 1

    

    ## convert data to sequence of tokens 

    #input_sequences = []

    token_list = tokenizer.texts_to_sequences([corpus])[0]

    return token_list, total_words
def generate_padded_sequences(input_sequences):

    max_sequence_len = max([len(x) for x in input_sequences])

    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='post'))

   

    predictors = input_sequences[:,:-1]

    return predictors, max_sequence_len



predictors, max_sequence_len = generate_padded_sequences(inp_sequences)
def generate_padded_sequences_pred(input_sequences,max_sequence_len):

    max_sequence_len = max([len(x) for x in input_sequences])

    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='post'))

   

    predictors = input_sequences[:,:-1]

    print(predictors)

    return predictors, max_sequence_len
x_train, x_test, y_train, y_test=train_test_split(predictors,Y, test_size=0.10, random_state=42)

y_train=to_categorical(y_train)

y_test=to_categorical(y_test)
def build_model():



  opt = Adam(lr=0.01)



  model = Sequential()

  model.add(Embedding(total_words, 16, input_length=x_train.shape[1], mask_zero=True))

  model.add(LSTM(12, dropout=0.7, recurrent_dropout=0.7))

  model.add(Dense(6, kernel_regularizer=regularizers.l1_l2(0.3)))

  model.add(Dropout(0.9))

  model.add(Dense(2, activation='softmax'))

  model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

  return model



model_lstm=build_model()
es = EarlyStopping(monitor='val_loss', mode='min',patience=5)

history_lstm = model_lstm.fit(x_train, y_train, epochs=120,batch_size=300,validation_data=(x_test,y_test),shuffle=False)
fig, axes = plt.subplots(1, 2, figsize = (16,6))

axes[0].plot(history_lstm.history['accuracy'])

axes[0].plot(history_lstm.history['val_accuracy'],'--')

axes[0].set_title('model accuracy')

axes[0].set_ylabel('accuracy')

axes[0].set_xlabel('epoch')

axes[0].legend(['train', 'val'], loc='lower right')



axes[1].plot(history_lstm.history['loss'])

axes[1].plot(history_lstm.history['val_loss'],"--")

axes[1].set_title('model loss')

axes[1].set_ylabel('accuracy')

axes[1].set_xlabel('epoch')

axes[1].legend(['val_loss', 'loss'], loc='lower right')
max_words = 100

max_len = 30

tok = Tokenizer(num_words=max_words)

tok.fit_on_texts(test["text"])

sequences = tok.texts_to_sequences(test["text"])

txts = sequence.pad_sequences(sequences, maxlen=max_len, padding='post')
preds = model_lstm.predict(txts)

len(sequences)
preds3=[ 1 if i<j else 0 for i,j in preds  ] 



sub = pd.DataFrame() 

sub["id"]=test["id"]

sub["target"]=preds3



print(sub)