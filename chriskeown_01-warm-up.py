import os
import time
import numpy as np # linear algebra                                                                                                                                                                         
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)                                                                                                                                      
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers


import matplotlib.pylab as plt
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print("Train shape : ",train_df.shape)
print("Test shape : ",test_df.shape)
# Cross validation - create training and testing dataset
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=2018)
# Preprocess the data
## some config values                                                                                                                                                                                       
embed_size = 300 # how big is each word vector                                                                                                                                                              
max_features = 50000 # how many unique words to use (i.e num rows in embedding vector)                                                                                                                      
maxlen = 100 # max number of words in a question to use                                                                                                                                                     

## fill up the missing values                                                                                                                                                                               
train_X = train_df["question_text"].fillna("_na_").values
val_X = val_df["question_text"].fillna("_na_").values
test_X = test_df["question_text"].fillna("_na_").values

## Tokenize the sentences                                                                                                                                                                                   
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X))
train_X = tokenizer.texts_to_sequences(train_X)
val_X = tokenizer.texts_to_sequences(val_X)
test_X = tokenizer.texts_to_sequences(test_X)

## Pad the sentences                                                                                                                                                                                        
train_X = pad_sequences(train_X, maxlen=maxlen)
val_X = pad_sequences(val_X, maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen)

## Get the target values                                                                                                                                                                                    
train_y = train_df['target'].values
val_y = val_df['target'].values
inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size)(inp)
x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dense(16, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
model.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y))
pred_noemb_val_y = model.predict([val_X], batch_size=1024, verbose=1)

thresholds = np.arange(0.1, 0.501, 0.01)
f1s = np.zeros(thresholds.shape[0])

for ind, thresh in np.ndenumerate(thresholds):
    f1s[ind[0]] = metrics.f1_score(val_y, (pred_noemb_val_y > np.round(thresh, 2)).astype(int))

np.round(thresholds[np.argmax(f1s)], 2)
import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # print("Normalized confusion matrix")
    # else:
    # print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

opt_thresh = np.round(thresholds[np.argmax(f1s)], 2)
y_test = val_y
y_pred = (pred_noemb_val_y > opt_thresh).astype(int)

cnf_matrix = confusion_matrix(y_test, y_pred)

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Sincere','Insincere'],
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Sincere','Insincere'], normalize=True,
                      title='Normalized confusion matrix')

plt.show()

precision = cnf_matrix[1,1]/(cnf_matrix[1,1]+cnf_matrix[0,1])
recall = cnf_matrix[1,1]/(cnf_matrix[1,1]+cnf_matrix[1,0])
print("Precision: " + str(np.round(precision, 3)))
print("Recall: " + str(np.round(recall, 3)))
# Next step is to look at some that are correct and incorrect
pred_noemb_test_y = model.predict([test_X], batch_size=1024, verbose=1)
