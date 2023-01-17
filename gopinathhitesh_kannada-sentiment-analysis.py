import os 

import pandas as pd

import numpy as np

from nltk.tokenize import word_tokenize

from nltk import pos_tag

from sklearn.preprocessing import LabelEncoder

from collections import defaultdict

from nltk.corpus import wordnet as wn

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import model_selection, naive_bayes, svm

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix

path="C:\\Users\\hp\\Music\\project\\"

scrapdata = "../input/dataforkannadasentimentclassification//"

reviewdata = "../input/dataforkannadasentimentclassification//"

f1=pd.read_excel("..//input//dataforkannadasentimentclassification//collected_50k_new.xlsx")



print(len(f1))



f1.head()
f1.iloc[0]['kn_review']
f1.iloc[0]['En_review']
temp_f1 = f1.drop('En_review',axis=1)

temp_f1
#f2.to_excel(scrapdata+'reviewdata\\'+'ajaydata.xlsx')

f2 = pd.read_excel(scrapdata+'ajaydata.xlsx')
f2.drop('Unnamed: 0',axis=1,inplace=True)
f2.head()
f3 = pd.read_excel(reviewdata+"gadgetloka.xls")

len(f3)
f3.drop('Unnamed: 0',inplace=True,axis=1)
f3['label'].replace({'p':"ಪೋಸ್ಟ್",'N':'ನೆಗ್'},inplace=True)
f3.rename({'review':'kn_review'},axis = 1,inplace=True)

f3
f3.isnull().sum()
f4 = pd.read_excel(reviewdata+"vijaya_reviews.xls")

len(f4)
f4.drop('Unnamed: 0',inplace=True,axis=1)
f4['label'].replace('ಪೋಸ್','ಪೋಸ್ಟ್',inplace=True)
f4
f4.isnull().sum()
kannada_set = pd.concat([temp_f1,f2,f3,f4])

kannada_set.reset_index(inplace=True)
kannada_set.drop('index',inplace=True,axis=1)
kannada_set
#kannada_set.to_excel('complete_data.xlsx')
kannada_set['label'].value_counts()
kannada_set.isnull().sum()
f2=open("../input/kanstopwords//kannadastopwords(1).txt","r",encoding="utf-8")

kn_stopwords=f2.readlines()

for i in range(len(kn_stopwords)):

    kn_stopwords[i]=kn_stopwords[i].strip()

print(kn_stopwords)

f2.close()
kn_stopwords
kn_stopwords.append('.')
all_each = []

processed_review_list = []

for index,entry  in enumerate(kannada_set["kn_review"]):

    processed=[]

    processed_string_review = ""

    for word in word_tokenize(entry):

        if word not in kn_stopwords:

            processed.append(word)

            processed_string_review = processed_string_review + " " + word

    processed_review_list.append(processed_string_review)

    all_each.append(processed)
kannada_set["processed_review"] = all_each

print(kannada_set["processed_review"])
kannada_set['non_stop_review'] = processed_review_list
kannada_set.iloc[0][2]
print(len(kannada_set))

for i in range(len(kannada_set)):

    if len(kannada_set["processed_review"][i])<=2:

        kannada_set=kannada_set.drop(i)

print(len(kannada_set))
print(kannada_set["processed_review"].head())
kannada_set=kannada_set.reset_index()

kannada_set=kannada_set.drop(["index"],axis=1)

print(kannada_set)
kannada_set
#kannada_set.to_excel('complete_data.xlsx')

#kannada_set = pd.read_excel('../input/complete-data/complete_data.xls')



#kannada_set
kannada_set['kn_review'][0]
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(kannada_set['processed_review'],kannada_set['label'],test_size=0.2,random_state=3,stratify=kannada_set['label'])
val = [len(kannada_set['processed_review'][i]) for i in range(len(kannada_set['processed_review'])) ]
print('max para length: ',max(val))
Train_Y.value_counts()
Test_Y.value_counts()
import collections

from itertools import chain

Encoder = LabelEncoder()

Train_Y = Encoder.fit_transform(Train_Y)

Test_Y = Encoder.fit_transform(Test_Y)

print(Train_Y)
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm

from sklearn.feature_extraction.text import TfidfVectorizer

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense, Flatten,Concatenate, LSTM, Conv1D,Conv2D, MaxPooling1D,MaxPooling2D, Dropout, Activation,GRU

from keras.layers.embeddings import Embedding



from keras.preprocessing import text, sequence

from keras import layers, models, optimizers
from keras.layers.normalization import BatchNormalization

from keras.layers import TimeDistributed

from keras.layers import Bidirectional

from gensim.models import word2vec

import tensorflow as tf

from tensorflow.keras import layers , Input

from keras.callbacks import ModelCheckpoint
from keras.preprocessing import text, sequence

!pip install hickle 

import hickle as hkl
'''

# create a tokenizer 

token = text.Tokenizer()

token.fit_on_texts(kannada_set["non_stop_review"])

word_index = token.word_index

len(word_index)

'''





#word_index = hkl.load("../input/wordindex/wordindex")

token = hkl.load('../input/token-data/token')

#print(len(word_index))

len_word_index = 214976



train_seq_x = sequence.pad_sequences(token.texts_to_sequences(Train_X), maxlen=1529)

valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(Test_X), maxlen=1529)



train_seq_x.shape
valid_seq_x.shape
len(train_seq_x)
import pickle as pkl 
filehandler = open("../input/weightspickle/matrix.pkl", 'rb') 

embedding_matrix = pkl.load(filehandler)

#embedding_matrix = hkl.load('../input/weights/matrix_weights')
len(embedding_matrix)

np.random.seed(123)
model = Sequential()

model.add(Embedding(214976 + 1,  300, weights=[embedding_matrix], input_length=1529))

model.add(Dropout(0.2))

model.add(Conv1D(128, 5, activation='relu',padding='same'))

model.add(MaxPooling1D(pool_size=4))

model.add(Conv1D(64, 5, activation='relu',padding='same'))

model.add(MaxPooling1D(pool_size=2))

model.add(Dense(256))

model.add(Dropout(0.2))

model.add(LSTM(298, dropout=0.2, recurrent_dropout=0.2,return_sequences=True))

model.add(GRU(298, dropout=0.2,recurrent_dropout=0.2,return_sequences=False))

model.add(Dropout(0.2))

model.add(Dense(128))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
'''

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

mc = ModelCheckpoint('best_model_lstm2-{epoch:02d}-{val_accuracy:.2f}.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

history=model.fit(train_seq_x, Train_Y,validation_split = .2,batch_size =500, epochs=15,callbacks=[es,mc], verbose=1)

'''

model.summary()


model.load_weights("../input/weightslstm/best_model_lstm2-02-0.88.h5")

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
'''

self = history.history

N = np.arange(0, 3)

plt.figure()

plt.plot(N, self['loss'], label = "train_loss")

plt.plot(N, self['accuracy'], label = "train_acc")

plt.plot(N, self['val_loss'], label = "val_loss")

plt.plot(N, self['val_accuracy'], label = "val_acc")

plt.title("Training Loss and Accuracy [Epoch {}]".format(3))

plt.xlabel("Epoch #")

plt.ylabel("Loss/Accuracy")

plt.legend()

'''

print(model.evaluate(valid_seq_x,Test_Y))

print(model.metrics_names)
model2 = Sequential()

model2.add(Embedding(214976 + 1,  300, weights=[embedding_matrix], input_length=1529))

model2.add(Dropout(0.2))

model2.add(Conv1D(164, 5, activation='relu',padding='same'))

model2.add(MaxPooling1D(pool_size=4))

model2.add(LSTM(298, dropout=0.2, recurrent_dropout=0.2,return_sequences=True))

model2.add(GRU(298, dropout=0.2,recurrent_dropout=0.2,return_sequences=False))

model2.add(Dropout(0.2))

model2.add(Dense(128))

model2.add(Dropout(0.2))

model2.add(Dense(64))

model2.add(Dense(1, activation='sigmoid'))

model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
'''

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

mc = ModelCheckpoint('best_model_lstm3-{epoch:02d}-{val_accuracy:.2f}.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

history=model.fit(train_seq_x, Train_Y,validation_split = .2,batch_size =500, epochs=15,callbacks=[es,mc], verbose=1)

'''

model2.summary()
model2.load_weights("../input/weightslstm/best_model_lstm3-01-0.88.h5")

model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
'''

self = history.history

N = np.arange(0, 4)

plt.figure()

plt.plot(N, self['loss'], label = "train_loss")

plt.plot(N, self['accuracy'], label = "train_acc")

plt.plot(N, self['val_loss'], label = "val_loss")

plt.plot(N, self['val_accuracy'], label = "val_acc")

plt.title("Training Loss and Accuracy [Epoch {}]".format(3))

plt.xlabel("Epoch #")

plt.ylabel("Loss/Accuracy")

plt.legend()

'''

print(model2.evaluate(valid_seq_x,Test_Y))

print(model2.metrics_names)
def dummy_fun(doc):

    return doc



tfidf = TfidfVectorizer(

    analyzer='word',

    tokenizer=dummy_fun,

    preprocessor=dummy_fun,

    token_pattern=None)  



tfidf.fit(Train_X)

#print(tfidf.vocabulary_)



Train_X_Tfidf=tfidf.transform(Train_X)

Test_X_Tfidf = tfidf.transform(Test_X)

print(Train_X_Tfidf.shape)

print(Test_X_Tfidf.shape)

from sklearn.model_selection import cross_val_score
# fit the training dataset on the NB classifier

Naive = naive_bayes.MultinomialNB()

Naive.fit(Train_X_Tfidf,Train_Y)



m_score = cross_val_score(Naive, Train_X_Tfidf, Train_Y, cv=10).mean()
print('val score: ',m_score)
# predict the labels on validation dataset

predictions_NB = Naive.predict(Test_X_Tfidf)
from sklearn.metrics import roc_curve,roc_auc_score

y_pred_prob = Naive.predict_proba(Test_X_Tfidf)[:,1]

fpr,tpr,threshold = roc_curve(Test_Y,y_pred_prob)

score= roc_auc_score(Test_Y,y_pred_prob)

print(score)

plt.plot([0,1],[0,1],"k--")

plt.plot(fpr,tpr,label="Naive bayes")

plt.show()
print(accuracy_score(Test_Y, predictions_NB))

print(confusion_matrix(Test_Y,predictions_NB))

print(classification_report(Test_Y,predictions_NB))
from sklearn import model_selection, preprocessing, linear_model

solvers = ['newton-cg', 'lbfgs', 'liblinear']

penalty = ['l2']

c_values = [100, 10, 1.0, 0.1, 0.01]
lr = linear_model.LogisticRegression(penalty="l2")
lr.fit(Train_X_Tfidf,Train_Y)
# predict the labels on validation dataset

predictions_NB = lr.predict(Test_X_Tfidf)
from sklearn.metrics import roc_curve,roc_auc_score

y_pred_prob = lr.predict_proba(Test_X_Tfidf)[:,1]

fpr,tpr,threshold = roc_curve(Test_Y,y_pred_prob)

score= roc_auc_score(Test_Y,y_pred_prob)

print(score)

plt.plot([0,1],[0,1],"k--")

plt.plot(fpr,tpr,label="Naive bayes")

plt.show()
print(accuracy_score(Test_Y, predictions_NB))

print(confusion_matrix(Test_Y,predictions_NB))

print(classification_report(Test_Y,predictions_NB))
clfs = [model,model2]

pred = np.asarray([clf.predict(valid_seq_x) for clf in clfs])

print(pred.shape)
pred
avg_mo = np.average(pred, axis=0,weights = [2,7])
avg_mo
avg_mo.shape
maj2 = np.apply_along_axis(lambda x: x[0], axis=1, arr=avg_mo)

maj2
dl = []

for i in maj2 :

    if i>=0.55:

        dl.append(1)

    else:

        dl.append(0)

print(accuracy_score(Test_Y, dl))

print(confusion_matrix(Test_Y,dl))

print(classification_report(Test_Y,dl))
dl = np.asarray(dl)

dl
dl.shape
clfs = [Naive,lr]

pro = np.asarray([clf.predict(Test_X_Tfidf) for clf in clfs])

print(pro)

dl
result =np.asarray([(dl),pro[0],pro[1]])
result
result1 = np.asarray([np.argmax(np.bincount(result[:,c])) for c in range(result.shape[1])])

result1
result1.shape
print(accuracy_score(Test_Y, result1))

print(confusion_matrix(Test_Y,result1))

print(classification_report(Test_Y,result1))