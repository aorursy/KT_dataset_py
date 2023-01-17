#kaggle is not pre-installed with keras-bert

!pip install keras_bert 
import numpy # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras import layers,optimizers,models,callbacks

import re

import matplotlib.pyplot as plt

import codecs

from keras.preprocessing.sequence import pad_sequences

from keras_bert import load_trained_model_from_checkpoint, Tokenizer #this is bert for keras version

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

test = pd.read_csv("../input/nlp-getting-started/test.csv")

train = pd.read_csv("../input/nlp-getting-started/train.csv")

bert_pre = "../input/bert-pretraining/uncased_L-12_H-768_A-12/"



bert_config = bert_pre+'bert_config.json'

checkpoint_path = bert_pre+'bert_model.ckpt'

dict_path = bert_pre+'vocab.txt'
#step1-----preprocessing data



#load csv type data---text and label

train_data = train['text']

train_label = numpy.array(train['target'])

test_data = test['text']

test_id = test['id']

maxlen = 0





#remove unnecessary char and string

temp = []

for data in train_data:

    data = re.sub(r'http://t.co/.{10}',"",data).strip() #remove http link

    data = re.sub(r'<.*?>',"",data).strip() #remove html

    data = re.sub(r'[-+]?[.\d]*[\d]+[:,.\d]*',"",data).strip() #remove numbers

    data = re.sub(r'([!?.]){2,}',"",data).strip() #repalce duplicate punctuation

    data = re.sub(r'@\S+',"",data).strip().replace("#","") #remove @username and '#'

    temp.append(data)

train_data = temp



temp = []

for data in test_data:

    data = re.sub(r'http://t.co/.{10}',"",data).strip()

    data = re.sub(r'<.*?>',"",data).strip()

    data = re.sub(r'[-+]?[.\d]*[\d]+[:,.\d]*',"",data).strip()

    data = re.sub(r'([!?.]){2,}',"",data).strip()

    data = re.sub(r'@\S+',"",data).strip().replace("#","")

    temp.append(data)

test_data = temp
#build token dictionary

token_dict = {}

with codecs.open(dict_path, 'r', 'utf-8') as f:

    for line in f:

        token = line.strip()

        token_dict[token] = len(token_dict)
#encoding text,one word corresponds to one wordvector

#Use pad_sequences to supplement vectors（zero）

#'maxlen' is the maximum length of the texts

def encode_data(data,token_dict):

    global maxlen

    token = Tokenizer(token_dict)

    x1 = []

    x2 = []

    for seq in train_data:

        X1,X2 = token.encode(first=seq)

        if maxlen==0:

            maxlen = len(X1)

        elif maxlen<len(X1):

            maxlen=len(X1)

        x1.append(X1)

        x2.append(X2)

    print(maxlen)

    x1 = pad_sequences(x1, maxlen=maxlen, padding='post')

    x2 = pad_sequences(x2, maxlen=maxlen, padding='post')

    return x1,x2
#Use bert model to make predictions

#Save results for later use

x1,x2 = encode_data(train_data,token_dict)

bert_model = load_trained_model_from_checkpoint(bert_config,checkpoint_path,seq_len=None)

word_vec = bert_model.predict([x1,x2])

numpy.save('word_vec',word_vec)
#Here you can also try using your own design model

#but your model should not be too complicated

def build_model():

    model = models.Sequential()

    model.add(layers.Bidirectional(layers.LSTM(128)))

    model.add(layers.Dense(1,activation='sigmoid'))

    return model
#visualize results

def show_res(history):

    acc = history.history['acc']

    val_acc = history.history['val_acc']

    loss = history.history['loss']

    val_loss = history.history['val_loss']



    epochs = range(1,len(acc)+1)



    plt.plot(epochs,acc,'b',label='training acc')

    plt.plot(epochs, val_acc, 'r', label='val acc')

    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'b', label='training loss')

    plt.plot(epochs, val_loss, 'r', label='val loss')

    plt.legend()

    plt.show()
#start training!

word_vec = numpy.load('word_vec.npy')

model = build_model()

callback_list = [callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)] #callbacks is a dynamic monitoring in keras,if val_loss does not decrease for patience=n rounds, we multiply the learning rate by 0.2

model.compile(optimizer=optimizers.Adam(lr=1e-5),loss='binary_crossentropy',metrics=['acc'])

history = model.fit(word_vec,train_label,batch_size=16,epochs=20,validation_split=0.2,callbacks=callback_list)

model.save('tweets_binary.h5')

show_res(history)
#Use the trained model to predict the test_data

model = models.load_model('tweets_binary.h5')

x1,x2 = encode_data(test_data,token_dict)

bert_model = load_trained_model_from_checkpoint(bert_config,checkpoint_path,seq_len=None)

word_vec = bert_model.predict([x1,x2])

numpy.save('res_vec',word_vec)

res_vec = numpy.load('res_vec.npy')

res = model.predict(res_vec)

res = numpy.round(res)

numpy.save('res_vec',res)
#save classifition result

test_id = pd.DataFrame(test_id,dtype='int')

label = pd.DataFrame(res,dtype='int')

res_df = pd.concat([test_id,label],axis=1)

res_df.columns = ['id','target']

res_df.to_csv('sample_submission.csv',index=False)