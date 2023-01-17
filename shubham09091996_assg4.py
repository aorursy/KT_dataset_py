# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import bz2
import nltk
import chardet
# Any results you write to the current directory are saved as output.
trainfile=bz2.BZ2File("../input/train.ft.txt.bz2")
train_file_lines = trainfile.readlines()
train_file_lines[0]
train_file_lines = [x.decode('utf-8') for x in train_file_lines]
train_file_lines[0]
import re
train_labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in train_file_lines]
train_sentences = [x.split(' ', 1)[1][:-1].lower() for x in train_file_lines]

for i in range(len(train_sentences)):
    train_sentences[i] = re.sub('\d','0',train_sentences[i])
                                                           
for i in range(len(train_sentences)):
    if 'www.' in train_sentences[i] or 'http:' in train_sentences[i] or 'https:' in train_sentences[i] or '.com' in train_sentences[i]:
        train_sentences[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", train_sentences[i])
len(train_labels)
train_sentences[192]
import string
for i in range(len(train_sentences)):
    train_sentences[i] = train_sentences[i].translate(str.maketrans('','',string.punctuation))
train_sentences[192]
train = pd.DataFrame(data=list(zip(train_sentences, train_labels)), columns=['review_text', 'sentiment_class_label'])
train
train['word_count'] = [len(text.split()) for text in train.review_text]
train.head()
train = train[train.word_count < 20]
train.shape
train = train.drop(columns=['word_count'], axis=1)
train.head()
train.shape
train = train.set_index(np.arange(len(train)))
train.head()
mp={}
for i in train.review_text:
    for j in i.split():
        if j in mp:
            mp[j]+=1
        else:
            mp[j]=1
list_of_words=[]
for key, value in mp.items():
    if value>5:
        list_of_words.append(key)
mp
list_of_words
len(list_of_words)
data_info=[]
for i in train.iterrows():
    lis=[]
    text = i[1]['review_text'].split()
    for j in list_of_words:
        if j in text:
            lis.append(1)
        else:
            lis.append(0)
            
            
    data_info.append(lis)
data = pd.DataFrame(data_info, columns=list_of_words)
data.head()
data.shape
data['sentiment_class_label'] = train['sentiment_class_label']
data.head()
y = data['sentiment_class_label']
X = data.drop(['sentiment_class_label'], axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.1, random_state=10)
len(X_train), len(X_test)
X_train = X_train.set_index(np.arange(len(X_train)))
X_test = X_test.set_index(np.arange(len(X_test)))
X_train.head()
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import optimizers
from keras.utils import np_utils
import matplotlib.pyplot as plt

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
Y_train = np_utils.to_categorical(y_train, 2)
Y_test = np_utils.to_categorical(y_test, 2)
Y_train[:10]
model1 = Sequential()
model1.add(Dense(1024, activation='relu', input_dim=X_train.shape[1]))
model1.add(Dropout(0.5))
model1.add(Dense(256, activation='relu'))
model1.add(Dropout(0.5))
model1.add(Dense(32, activation='relu'))
model1.add(Dropout(0.5))
model1.add(Dense(2, activation='sigmoid'))
sgd = optimizers.SGD(lr=0.3, decay=5e-3, momentum=0.5, nesterov=True)
model1.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history1 = model1.fit(X_train, y_train, epochs=15, batch_size=64, verbose=1, validation_data=(X_test, y_test))
def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return 
    
    ## As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)
    
    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:    
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
plot_history(history1)
model2 = Sequential()
model2.add(Dense(1024, activation='relu', input_dim=X_train.shape[1]))
model2.add(Dropout(0.5))
model2.add(Dense(32, activation='relu'))
model2.add(Dropout(0.5))
model2.add(Dense(2, activation='sigmoid'))
sgd = optimizers.SGD(lr=0.3, decay=5e-3, momentum=0.5, nesterov=True)
model2.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history2 = model2.fit(X_train, y_train, epochs=15, batch_size=64, verbose=1, validation_data=(X_test, y_test))
plot_history(history2)
model3 = Sequential()
model3.add(Dense(256, activation='relu', input_dim=X_train.shape[1]))
model3.add(Dropout(0.5))
model3.add(Dense(2, activation='sigmoid'))
sgd = optimizers.SGD(lr=0.3, decay=5e-3, momentum=0.5, nesterov=True)
model3.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history3 = model3.fit(X_train, y_train, epochs=15, batch_size=64, verbose=1, validation_data=(X_test, y_test))
plot_history(history3)
model1.summary()
loss_curve1=history1.history['loss']
epoch_c=list(range(len(loss_curve1)))
loss_curve2=history2.history['loss']
#epoch_c=list(range(len(loss_curve)))
loss_curve3=history3.history['loss']
plt.xlabel('Epochs')
plt.ylabel('Loss value')
plt.plot(epoch_c,loss_curve1,label='3 hidden layer')
plt.plot(epoch_c,loss_curve2,label='2 hidden layer')
plt.plot(epoch_c,loss_curve3,label='1 hidden layer')
plt.legend()
plt.show()
loss_curve1=history1.history['acc']
epoch_c=list(range(len(loss_curve1)))
loss_curve2=history2.history['acc']
#epoch_c=list(range(len(loss_curve)))
loss_curve3=history3.history['acc']
plt.xlabel('Epochs')
plt.ylabel('Loss value')
plt.plot(epoch_c,loss_curve1,label='3 hidden layer')
plt.plot(epoch_c,loss_curve2,label='2 hidden layer')
plt.plot(epoch_c,loss_curve3,label='1 hidden layer')
plt.legend()
plt.show()
