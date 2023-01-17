# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





# Any results you write to the current directory are saved as output.



%matplotlib inline

import pandas as pd

import numpy as np



import os

from sklearn.metrics.classification import accuracy_score, log_loss

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM

from keras.utils.np_utils import to_categorical

from keras.callbacks import ModelCheckpoint

from keras.models import load_model

from keras.optimizers import Adam
data = pd.read_csv('../input/training_variants')

data.head()
# check nan value

data.isnull().sum()
# note the seprator in this file

data_text =pd.read_csv("../input/training_text",sep="\|\|",engine="python",names=["ID","TEXT"],skiprows=1)

data_text.head()

# check the null value in data_text

data_text.isnull().sum()
data_text.dropna(axis=0,how='any',inplace=True)
data_text.isnull().sum()
# preprocessing for data_text

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def nlp_preprocessing(total_text,index,column):

    if type(total_text) is not int:

        string = ""

        # replace every special char with space

        total_text = re.sub('[^a-zA-Z0-9\n]',' ',total_text)

        # replace multiple spaces with single space

        total_text = re.sub('\s+',' ',total_text)

        # converting all the chars into lower-case.

        total_text= total_text.lower()

        

        for word in total_text.split():

            # if the word is a not a stop word then retain that word from the data

            if not word in stop_words:

                string += word+' '

                

        data_text[column][index] = string

        

        

    
#text processing stage.

import time

import re

start_time = time.clock()

for index, row in data_text.iterrows():

    nlp_preprocessing(row['TEXT'], index, 'TEXT')

print('Time took for preprocessing the text :',time.clock() - start_time, "seconds")
#merging both gene_variations and text data based on ID

df_train = pd.merge(data,data_text,on= 'ID',how = 'left')

df_train.head()
df_train['full_text'] = df_train[['Gene','Variation']].apply(lambda x: ' '.join(x),axis=1)
df_train['TEXT'] = df_train['TEXT'].astype(str)
df_train['full_text'] = df_train[['full_text','TEXT']].apply(lambda x: ' '.join(x),axis=1)
df_train.head()
# Use the Keras tokenizer

num_words = 2000

tokenizer = Tokenizer(num_words=num_words)

tokenizer.fit_on_texts(df_train['full_text'].values)
# Pad the data 

X = tokenizer.texts_to_sequences(df_train['full_text'].values)

X = pad_sequences(X, maxlen=2000)
# Build out our simple LSTM

embed_dim = 128

lstm_out = 196



# Model saving callback

ckpt_callback = ModelCheckpoint('keras_model', 

                                 monitor='val_loss', 

                                 verbose=1, 

                                 save_best_only=True, 

                                 mode='auto')



model = Sequential()

model.add(Embedding(num_words, embed_dim, input_length = X.shape[1]))

model.add(LSTM(lstm_out, recurrent_dropout=0.2, dropout=0.2))

model.add(Dense(9,activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['categorical_crossentropy'])

print(model.summary())
from sklearn.model_selection import train_test_split

Y = pd.get_dummies(df_train['Class']).values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42, stratify=Y)

print(X_train.shape, Y_train.shape)

print(X_test.shape, Y_test.shape)
batch_size = 32

history = model.fit(X_train, Y_train, epochs=8, batch_size=batch_size, validation_split=0.2, 

                    callbacks=[ckpt_callback])
# Plot training & validation accuracy values

import matplotlib.pyplot as plt

%matplotlib inline

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'validation'], loc='best')

plt.show()
model = load_model('keras_model')
probas = model.predict(X_test)
pred_indices = np.argmax(probas, axis=1)

classes = np.array(range(1, 10))

preds = classes[pred_indices]

print('Log loss: {}'.format(log_loss(classes[np.argmax(Y_test, axis=1)], probas)))

print('Accuracy: {}'.format(accuracy_score(classes[np.argmax(Y_test, axis=1)], preds)))
# This function plots the confusion matrices given y_i, y_i_hat.

from sklearn.metrics import confusion_matrix

import seaborn as sns

def plot_confusion_matrix(test_y, predict_y):

    C = confusion_matrix(test_y, predict_y)

    # C = 9,9 matrix, each cell (i,j) represents number of points of class i are predicted class j

    

    A =(((C.T)/(C.sum(axis=1))).T)

    #divid each element of the confusion matrix with the sum of elements in that column

    

    # C = [[1, 2],

    #     [3, 4]]

    # C.T = [[1, 3],

    #        [2, 4]]

    # C.sum(axis = 1)  axis=0 corresonds to columns and axis=1 corresponds to rows in two diamensional array

    # C.sum(axix =1) = [[3, 7]]

    # ((C.T)/(C.sum(axis=1))) = [[1/3, 3/7]

    #                           [2/3, 4/7]]



    # ((C.T)/(C.sum(axis=1))).T = [[1/3, 2/3]

    #                           [3/7, 4/7]]

    # sum of row elements = 1

    

    B =(C/C.sum(axis=0))

    #divid each element of the confusion matrix with the sum of elements in that row

    # C = [[1, 2],

    #     [3, 4]]

    # C.sum(axis = 0)  axis=0 corresonds to columns and axis=1 corresponds to rows in two diamensional array

    # C.sum(axix =0) = [[4, 6]]

    # (C/C.sum(axis=0)) = [[1/4, 2/6],

    #                      [3/4, 4/6]] 

    

    labels = [1,2,3,4,5,6,7,8,9]

    # representing C in heatmap format

    print("-"*20, "Confusion matrix", "-"*20)

    plt.figure(figsize=(20,7))

    sns.heatmap(C, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)

    plt.xlabel('Predicted Class')

    plt.ylabel('Original Class')

    plt.show()



    # representing B in heatmap format FOR PRECISION

    print("-"*20, "Precision matrix (Columm Sum=1)", "-"*20)

    plt.figure(figsize=(20,7))

    sns.heatmap(B, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)

    plt.xlabel('Predicted Class')

    plt.ylabel('Original Class')

    plt.show()

    

    # representing A in heatmap format FOR RECALL

    print("-"*20, "Recall matrix (Row sum=1)", "-"*20)

    plt.figure(figsize=(20,7))

    sns.heatmap(A, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)

    plt.xlabel('Predicted Class')

    plt.ylabel('Original Class')

    plt.show()
plot_confusion_matrix(classes[np.argmax(Y_test, axis=1)], preds)