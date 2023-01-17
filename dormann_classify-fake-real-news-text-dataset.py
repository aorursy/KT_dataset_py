import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.pipeline import Pipeline

from sklearn.svm import LinearSVC

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import BernoulliNB

from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder

from keras.models import Model

from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding

from keras.optimizers import RMSprop

from keras.preprocessing.text import Tokenizer

from keras.preprocessing import sequence

from keras.callbacks import EarlyStopping





import nltk

import nltk as nlp

import string

import re

real = pd.read_csv("../input/fake-and-real-news-dataset/True.csv")

fake = pd.read_csv("../input/fake-and-real-news-dataset/Fake.csv")

fake['target'] = 'fake'

real['target'] = 'real'

txt_news = pd.concat([fake, real]).reset_index(drop = True)

txt_news.head()

txt_news.isnull().all()
print('Number of examples : ' + str(txt_news.shape[0]))
target=txt_news.target

#txt_news=txt_news.drop(columns=['date', 'subject','target'])

txt_news['total']=txt_news['text']+' '+txt_news['subject']+txt_news['title']



target_binary=[]

for false_true in target :

    if false_true=="fake":

        target_binary.append(0)

    else :

        target_binary.append(1)

        
#tfidf

#transformer = TfidfTransformer(smooth_idf=False)

#count_vectorizer = CountVectorizer(ngram_range=(1, 2))

#counts = count_vectorizer.fit_transform(txt_news['total'].values)

#tfidf = transformer.fit_transform(counts)
x_train,x_test,y_train,y_test = train_test_split(txt_news['total'], target, test_size=0.25, random_state=42) #txt_news['text']
pipeline = Pipeline([('CountV', CountVectorizer()),

                 ('TfidfT', TfidfTransformer()),

                 ('clf', KNeighborsClassifier(n_neighbors = 30,weights = 'distance'))])#algorithm = 'brute'



model = pipeline.fit(x_train, y_train)

pred = model.predict(x_test)

print("accuracy KNeighborsClassifier: {}%".format(round(accuracy_score(y_test, pred)*100,3)))


pipeline = Pipeline([('CountV', CountVectorizer()),

                 ('TfidfT', TfidfTransformer()),

                 ('clf', LinearSVC(C=12, random_state=7))])#loss="hinge"



model = pipeline.fit(x_train, y_train)

pred = model.predict(x_test)

print("accuracy SVM: {}%".format(round(accuracy_score(y_test, pred)*100,3)))
from sklearn.ensemble import AdaBoostClassifier



pipeline = Pipeline([('CountV', CountVectorizer()),

                 ('TfidfT', TfidfTransformer()),

                 ('clf', AdaBoostClassifier())])



model = pipeline.fit(x_train, y_train)

pred = model.predict(x_test)

print("accuracy AdaBoostClassifier: {}%".format(round(accuracy_score(y_test, pred)*100,3)))
pipeline = Pipeline([('CountV', CountVectorizer()),

                 ('TfidfT', TfidfTransformer()),

                 ('clf', MultinomialNB(alpha=0.5))])



model = pipeline.fit(x_train, y_train)

pred = model.predict(x_test)

print("accuracy MultinomialNB: {}%".format(round(accuracy_score(y_test, pred)*100,3)))
pipeline = Pipeline([('CountV', CountVectorizer()),

                 ('TfidfT', TfidfTransformer()),

                 ('clf', XGBClassifier(

                                                   learning_rate = 0.015,

                                                   n_estimators = 18,

                                                   max_depth = 7,

                                                   random_state=42))])#loss = 'deviance',



model = pipeline.fit(x_train, y_train)

pred = model.predict(x_test)

print("accuracy XGBClassifier: {}%".format(round(accuracy_score(y_test, pred)*100,3)))
from lightgbm import LGBMClassifier



lgbm = LGBMClassifier(objective='multiclass', random_state=5)



pipe = Pipeline([('CountV', CountVectorizer()),

                 ('TfidfT', TfidfTransformer()),

                 ('clf', LGBMClassifier(

                                                   learning_rate = 0.01,

                                                   n_estimators = 18,

                                                   max_depth = 7,

                                                   random_state=42))])#loss = 'deviance',



model = pipe.fit(x_train, y_train)

pred = model.predict(x_test)

print("accuracy LGBMClassifier: {}%".format(round(accuracy_score(y_test, pred)*100,3)))





from sklearn.model_selection import GridSearchCV

from pprint import pprint

from time import time



pipeline = Pipeline([('CountV', CountVectorizer()),

                 ('TfidfT', TfidfTransformer()),

                 ('clf', XGBClassifier()),])



parameters = {}

parameters['clf__learning_rate'] = [0.01]

parameters['clf__n_estimators'] = [15,20]

parameters['clf__max_depth'] = [7,9]

parameters['clf__random_state'] = [42]







    

    

CV = GridSearchCV(pipe, parameters, n_jobs= 1) # scoring = 'mean_absolute_error'



print("Performing grid search...")

print("pipeline:", [name for name, _ in pipeline.steps])

print("parameters:")

pprint(parameters)

t0 = time()



model = CV.fit(x_train, y_train)

print("done in %0.3fs" % (time() - t0))

print()



print("Best score: %0.3f" % CV.best_score_)

print("Best parameters set:")

best_parameters = CV.best_estimator_.get_params()

for param_name in sorted(parameters.keys()):

    print("\t%s: %r" % (param_name, best_parameters[param_name]))





prediction = CV.predict(x_test)

print("accuracy LGBMClassifier with GridSearchCV: {}%".format(round(accuracy_score(y_test, prediction)*100,3)))
pipeline = Pipeline([('CountV', CountVectorizer()),

                 ('TfidfT', TfidfTransformer()),

                 ('clf', RandomForestClassifier(criterion= "entropy"))])



model = pipe.fit(x_train, y_train)

pred = model.predict(x_test)

print("accuracy RandomForestClassifier: {}%".format(round(accuracy_score(y_test, pred)*100,3)))
X = txt_news.total

Y = txt_news.target

le = LabelEncoder()

Y = le.fit_transform(Y)

Y = Y.reshape(-1,1)
len_size=0

for f in X:

    if len_size<len(f) :

        len_size=len(f)

        

print(str(len_size))
print(pd.Series({c: txt_news[c].map(lambda x: len(str(x))).max() for c in txt_news}).sort_values(ascending =False))

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15)

max_words = 500

max_len = 150

tok = Tokenizer(num_words=max_words)

tok.fit_on_texts(X_train)

sequences = tok.texts_to_sequences(X_train)

sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

def LSTM_MODEL():

    inputs = Input(name='inputs',shape=[max_len])

    layer = Embedding(max_words,50,input_length=max_len)(inputs)

    layer = LSTM(64)(layer)

    layer = Dense(256,name='FC1')(layer)

    layer = Activation('relu')(layer)

    layer = Dropout(0.5)(layer)

    layer = Dense(1,name='out_layer')(layer)

    layer = Activation('sigmoid')(layer)

    model = Model(inputs=inputs,outputs=layer)

    return model





model_lstm = LSTM_MODEL()

from tensorflow.keras.utils import plot_model 

plot_model(model_lstm, to_file='lstm_png1.png')

model_lstm.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
model_lstm.fit(sequences_matrix,Y_train,batch_size=256,epochs=10,validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
test_sequences = tok.texts_to_sequences(X_test)

test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)

accr = model_lstm.evaluate(test_sequences_matrix,Y_test)

print('Accuracy LSTM: {:0.2f}'.format(accr[1]))
from keras.layers.recurrent import LSTM, GRU



def GRU_model():

    inputs = Input(name='inputs',shape=[max_len])

    layer = Embedding(max_words,50,input_length=max_len)(inputs)

    layer = GRU(64)(layer)

    layer = Dense(256,name='FC1')(layer)

    layer = Activation('relu')(layer)

    layer = Dropout(0.5)(layer)

    layer = Dense(1,name='out_layer')(layer)

    layer = Activation('sigmoid')(layer)

    model = Model(inputs=inputs,outputs=layer)

    return model



model_gru = GRU_model()
plot_model(model_gru, to_file='gru_png1.png')

model_gru.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
model_gru.fit(sequences_matrix,Y_train,batch_size=256,epochs=10,validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
accr = model_gru.evaluate(test_sequences_matrix,Y_test)

print('Accuracy GRU: {:0.2f}'.format(accr[1]))