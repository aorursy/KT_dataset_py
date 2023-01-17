
import numpy as np # linear algebra
import pandas as pd # data processing

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
train_data= pd.read_csv('/kaggle/input/atis-airlinetravelinformationsystem/atis_intents_train.csv',
                       names= ["target", "text"])

test_data= pd.read_csv('/kaggle/input/atis-airlinetravelinformationsystem/atis_intents_test.csv',
                       names= ["target", "text"])

train_data
train_data.groupby("target").count()
train_data= train_data.append(train_data.loc[train_data.target.isin(["atis_flight_time", "atis_quantity"]), :])
from sklearn.preprocessing import OneHotEncoder as OHE

y_encoder= OHE().fit(np.array(train_data.target).reshape(-1,1))
ytr_encoded= y_encoder.transform(np.array(train_data.target).reshape(-1,1)).toarray()
yts_encoded= y_encoder.transform(np.array(test_data.target).reshape(-1,1)).toarray()
import nltk
train_data["lower_text"]= train_data.text.map(lambda x: x.lower())
test_data["lower_text"]= test_data.text.map(lambda x: x.lower())
from nltk import word_tokenize

train_data["tokenized"]= train_data.lower_text.map(word_tokenize)
test_data["tokenized"]= test_data.lower_text.map(word_tokenize)
from nltk.corpus import stopwords
from string import punctuation

def remove_stop(strings, stop_list):
    classed= [s for s in strings if s not in stop_list]
    return classed

stop= stopwords.words("english")
stop_punc= list(set(punctuation))+ stop

train_data["selected"]= train_data.tokenized.map(lambda df: remove_stop(df, stop_punc))
test_data["selected"]= test_data.tokenized.map(lambda df: remove_stop(df, stop_punc))
from nltk.stem import PorterStemmer

def normalize(text):
    return " ".join(text)

stemmer= PorterStemmer()

train_data["stemmed"]= train_data.selected.map(lambda xs: [stemmer.stem(x) for x in xs])
train_data["normalized"]= train_data.stemmed.apply(normalize)

test_data["stemmed"]= test_data.selected.map(lambda xs: [stemmer.stem(x) for x in xs])
test_data["normalized"]= test_data.stemmed.apply(normalize)
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer= Tokenizer(num_words= 10000)
tokenizer.fit_on_texts(train_data.normalized)

tokenized_train= tokenizer.texts_to_sequences(train_data.normalized)
tokenized_test= tokenizer.texts_to_sequences(test_data.normalized)
tokenizer.word_index.keys().__len__()
from tensorflow.keras.preprocessing.sequence import pad_sequences

train_padded= pad_sequences(tokenized_train, maxlen= 20, padding= "pre")
test_padded= pad_sequences(tokenized_test, maxlen= 20, padding= "pre")
train_padded.shape
#this function transform final processed text (columns padded) into 3D matrix (samples, steps, unique_words)
#matrix contents one hot encoded words. Encoding was done for each step and based on unique words

def transform_x(data, tokenizer):
    output_shape= [data.shape[0],
                  data.shape[1],
                  tokenizer.word_index.keys().__len__()]
    results= np.zeros(output_shape)
    
    for i in range(data.shape[0]):
        for ii in range(data.shape[1]):
            results[i, ii, data[i,ii]-1]= 1
    return results

xtr_transformed= transform_x(train_padded, tokenizer)
xts_transformed= transform_x(test_padded, tokenizer)
from tensorflow.keras.layers import Dense, LSTM, BatchNormalization, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy as CC
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.initializers import he_uniform, glorot_uniform
from tensorflow.keras.metrics import AUC
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2


class LSTMModel(object):
    
    def build_model(self, input_dim, output_shape, steps, dropout_rate, kernel_regularizer, bias_regularizer):
        input_layer= Input(shape= (steps, input_dim))
        
        #make lstm_layer
        lstm= LSTM(units= steps)(input_layer)
        dense_1= Dense(output_shape, kernel_initializer= he_uniform(),
                       bias_initializer= "zeros", 
                       kernel_regularizer= l2(l= kernel_regularizer),
                       bias_regularizer= l2(l= bias_regularizer))(lstm)
        x= BatchNormalization()(dense_1)
        x= relu(x)
        x= Dropout(rate= dropout_rate)(x)
        o= Dense(output_shape, kernel_initializer= glorot_uniform(),
                 bias_initializer= "zeros", 
                 kernel_regularizer= l2(l= kernel_regularizer), 
                 bias_regularizer= l2(l= bias_regularizer))(dense_1)
        o= BatchNormalization()(o)
        output= softmax(o, axis= 1)
        
        loss= CC()
        metrics= AUC()
        optimizer= Adam()
        self.model= Model(inputs= [input_layer], outputs= [output])
        self.model.compile(optimizer= optimizer, loss= loss, metrics= [metrics])
        
        
    def train(self, x, y, validation_split, epochs):
        self.model.fit(x, y, validation_split= validation_split, epochs= epochs)
        
    def predict(self, x):
        return self.model.predict(x)
steps= xtr_transformed.shape[1]
dim= xtr_transformed.shape[2]
output_shape= ytr_encoded.shape[1]

model= LSTMModel()
model.build_model(input_dim= dim,
                  output_shape= output_shape,
                  steps= steps, 
                  dropout_rate= 0.5, 
                  bias_regularizer= 0.3, 
                  kernel_regularizer= 0.3)
model.train(xtr_transformed, ytr_encoded,
           0.2, 0)
from sklearn.metrics import classification_report

prediction= y_encoder.inverse_transform(model.predict(xtr_transformed))
print(classification_report(train_data.target, prediction))
from sklearn.metrics import classification_report

prediction_test= y_encoder.inverse_transform(model.predict(xts_transformed))
print(classification_report(test_data.target, prediction_test))
model.train(xtr_transformed, ytr_encoded,
           0.2,1)
from sklearn.metrics import classification_report

prediction= y_encoder.inverse_transform(model.predict(xtr_transformed))
print(classification_report(train_data.target, prediction))
from sklearn.metrics import classification_report

prediction_test= y_encoder.inverse_transform(model.predict(xts_transformed))
print(classification_report(test_data.target, prediction_test))
model.train(xtr_transformed, ytr_encoded,
           0.1,0)
from sklearn.metrics import classification_report

prediction= y_encoder.inverse_transform(model.predict(xtr_transformed))
print(classification_report(train_data.target, prediction))
from sklearn.metrics import classification_report

prediction_test= y_encoder.inverse_transform(model.predict(xts_transformed))
print(classification_report(test_data.target, prediction_test))
model.train(xtr_transformed, ytr_encoded,
           0.5,0)
from sklearn.metrics import classification_report

prediction= y_encoder.inverse_transform(model.predict(xtr_transformed))
print(classification_report(train_data.target, prediction))
from sklearn.metrics import classification_report

prediction_test= y_encoder.inverse_transform(model.predict(xts_transformed))
print(classification_report(test_data.target, prediction_test))