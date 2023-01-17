# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder
embeddings_index = dict()

f = open('/kaggle/input/glove6b100dtxt/glove.6B.100d.txt',encoding="utf8")

for line in f:

    values = line.split()

    word = values[0]

    coefs = np.asarray(values[1:])

    embeddings_index[word] = coefs

f.close()

print('Loaded %s word vectors.' % len(embeddings_index))
data=pd.read_csv('/kaggle/input/donors-choose/preprocessed_data.csv')
y=data['project_is_approved']
data=data.drop(['project_is_approved'],axis=1)
preproc=[]

for row in data['project_grade_category']:

    row=row.replace('grades','')

    row=row.replace('_prek_2','prek2')

    row=row.replace('_3_5','3to5')

    row=row.replace('_6_8','6to8')

    row=row.replace('_9_12','9to12')

    preproc.append(row)

    

data['project_grade_category']=preproc
preproc=[]

for row in data['clean_categories']:

    row=row.replace(' ','')

    row=row.replace('_','')

    preproc.append(row)

    

data['clean_categories']=preproc
preproc=[]

for row in data['clean_subcategories']:

    row=row.replace(' ','')

    row=row.replace('_','')

    preproc.append(row)

    

data['clean_subcategories']=preproc
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.33, stratify=y)
X_train.columns
from keras.models import Sequential,Model

from keras.layers import Dense,BatchNormalization

from keras.preprocessing.text import Tokenizer

from keras.layers import LSTM,Input,Flatten

from keras.layers.embeddings import Embedding

from keras.preprocessing import sequence

from keras.optimizers import Adam

from keras.preprocessing.sequence import pad_sequences

from keras.layers import BatchNormalization

from keras.initializers import he_normal
#credit to https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/

token = Tokenizer()

token.fit_on_texts(X_train['essay'])

vocabulary_size = len(token.word_index)+1
embedding_matrix = np.zeros((vocabulary_size, 100))

for word, i in token.word_index.items():

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector
word_count = []



for e in data["essay"] :

    c = len(e.split())

    word_count.append(c)
max_word_count=max(word_count)
max_word_count
train_essay=token.texts_to_sequences(X_train['essay'])

test_essay=token.texts_to_sequences(X_test['essay'])
train_essay=pad_sequences(train_essay,maxlen=350)

test_essay=pad_sequences(test_essay,maxlen=350)
embedding_matrix.shape
max_length=350

input_essay = Input(shape=(max_length,))

embedding = Embedding(vocabulary_size, 100,weights=[embedding_matrix], input_length=max_length,trainable=False)(input_essay)

lstm=LSTM(100,return_sequences=True)(embedding)

flatten=Flatten()(lstm)
from scipy.sparse import hstack



train_price=X_train['price'].values.reshape(-1, 1)

train_number=X_train['teacher_number_of_previously_posted_projects'].values.reshape(-1, 1)



test_price=X_test['price'].values.reshape(-1, 1)

test_number=X_test['teacher_number_of_previously_posted_projects'].values.reshape(-1, 1)



rem_train=np.concatenate((train_price,train_number),axis=1)

rem_test=np.concatenate((test_price,test_number),axis=1)
from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

scaler.fit(rem_train)

rem_train_standard=scaler.transform(rem_train)

rem_test_standard=scaler.transform(rem_test)
rem_feat = Input(shape=(2,))

rem_feat_dense= Dense(128, activation='relu')(rem_feat)
# https://www.youtube.com/watch?v=2U6Jl7oqRkM

# https://www.tensorflow.org/tensorboard/r2/get_started



import tensorflow as tf

from keras.callbacks import TensorBoard,EarlyStopping

from time import time



earlystop_1 = EarlyStopping(monitor = 'val_loss', mode="max",min_delta = 0, patience = 4,verbose = 1,restore_best_weights = True)

tensorboard_1 = TensorBoard("logs1")
# https://github.com/pranaya-mathur/Donors-Choose-LSTM/blob/master/DonorsChoose_Model_1_13_Aug_19.ipynb

import tensorflow as tf

from sklearn.metrics import roc_auc_score



def auc_score(y_true, y_pred):

    if len(np.unique(y_true[:])) == 1:

        return 0.5

    else:

        return roc_auc_score(y_true, y_pred)



def auc_sc(y_true, y_pred):

    return tf.py_func(auc_score, (y_true, y_pred), tf.double)   
from keras.utils import np_utils

Y_train=np_utils.to_categorical(y_train)

Y_test=np_utils.to_categorical(y_test)
%load_ext tensorboard.notebook

%tensorboard --logdir logs1
model1.evaluate(test_1, Y_test,batch_size=128)
train_2 = [train_essay_tfidf,encoded_state_train,encoded_p_grade_train,encoded_cat_train,encoded_subcat_train,encoded_t_prefix_train,rem_train_standard]

test_2 = [test_essay_tfidf,encoded_state_test,encoded_p_grade_test,encoded_cat_test,encoded_subcat_test,encoded_t_prefix_test,rem_test_standard]
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer
my_counter=Counter()

for word in data['school_state'].values:

    my_counter.update(word.split())

vocab_dict=dict(my_counter)

sorted_vocab_dict=dict(sorted(vocab_dict.items(),key=lambda kv:kv[1]))
vectorizer=CountVectorizer(vocabulary=list(sorted_vocab_dict.keys()),lowercase=False,binary=True)

vectorizer.fit(X_train['school_state'].values)
train_state_ohe=vectorizer.transform(X_train['school_state'].values)

test_state_ohe=vectorizer.transform(X_test['school_state'].values)

cv_state_ohe=vectorizer.transform(X_cv['school_state'].values)
my_counter=Counter()

for word in data['project_grade_category'].values:

    my_counter.update(word.split())

vocab_dict=dict(my_counter)

sorted_vocab_dict=dict(sorted(vocab_dict.items(),key=lambda kv:kv[1]))
vectorizer=CountVectorizer(vocabulary=list(sorted_vocab_dict.keys()),lowercase=False,binary=True)

vectorizer.fit(X_train['project_grade_category'].values)

train_grade_ohe=vectorizer.transform(X_train['project_grade_category'].values)

test_grade_ohe=vectorizer.transform(X_test['project_grade_category'].values)

cv_grade_ohe=vectorizer.transform(X_cv['project_grade_category'].values)
my_counter=Counter()

for word in data['clean_categories'].values:

    my_counter.update(word.split())

vocab_dict=dict(my_counter)

sorted_vocab_dict=dict(sorted(vocab_dict.items(),key=lambda kv:kv[1]))
vectorizer=CountVectorizer(vocabulary=list(sorted_vocab_dict.keys()),lowercase=False,binary=True)

vectorizer.fit(X_train['clean_categories'].values)

train_cat_ohe=vectorizer.transform(X_train['clean_categories'].values)

test_cat_ohe=vectorizer.transform(X_test['clean_categories'].values)

cv_cat_ohe=vectorizer.transform(X_cv['clean_categories'].values)
my_counter=Counter()

for word in data['clean_subcategories'].values:

    my_counter.update(word.split())

vocab_dict=dict(my_counter)

sorted_vocab_dict=dict(sorted(vocab_dict.items(),key=lambda kv:kv[1]))
vectorizer=CountVectorizer(vocabulary=list(sorted_vocab_dict.keys()),lowercase=False,binary=True)

vectorizer.fit(X_train['clean_subcategories'].values)

train_sub_ohe=vectorizer.transform(X_train['clean_subcategories'].values)

test_sub_ohe=vectorizer.transform(X_test['clean_subcategories'].values)

cv_sub_ohe=vectorizer.transform(X_cv['clean_subcategories'].values)
my_counter=Counter()

for word in data['teacher_prefix'].values:

    my_counter.update(word.split())

vocab_dict=dict(my_counter)

sorted_vocab_dict=dict(sorted(vocab_dict.items(),key=lambda kv:kv[1]))
vectorizer=CountVectorizer(vocabulary=list(sorted_vocab_dict.keys()),lowercase=False,binary=True)

vectorizer.fit(X_train['teacher_prefix'].values)

train_prefix_ohe=vectorizer.transform(X_train['teacher_prefix'].values)

test_prefix_ohe=vectorizer.transform(X_test['teacher_prefix'].values)

cv_prefix_ohe=vectorizer.transform(X_cv['teacher_prefix'].values)
import scipy
train_data=scipy.sparse.hstack((train_state_ohe,train_grade_ohe,train_cat_ohe,train_sub_ohe,train_prefix_ohe,rem_train_standard)).tocsr().todense()

test_data=scipy.sparse.hstack(((test_state_ohe),(test_grade_ohe),(test_cat_ohe),(test_sub_ohe),(test_prefix_ohe),(rem_test_standard))).tocsr().todense()
train_data=np.expand_dims(train_data, axis=2)

test_data=np.expand_dims(test_data, axis=2)

cv_data=np.expand_dims(cv_data,axis=2)
train_data.shape
test_data.shape
cv_data.shape
from keras.layers import Conv1D

from keras.layers import MaxPooling1D



input_conv =  Input(shape=(cv_data.shape[1],1))

Layer1=Conv1D(filters=96,kernel_size=3,activation='relu')(input_conv)

Layer2=MaxPooling1D(3,padding='same')(Layer1)

Layer3=Dropout(0.5)(Layer2)



Layer4=Conv1D(filters=128,kernel_size=3,activation='relu',padding='same')(Layer3)

Layer5=MaxPooling1D(3,padding='same')(Layer4)

Layer6=Dropout(0.5)(Layer5)



Layer7=Flatten()(Layer6)
concat = concatenate([flatten,Layer7])



Layer8 = Dense(456, activation='relu',kernel_initializer=he_normal())(concat)

Layer9=Dropout(0.5)(Layer8)



Layer10= Dense(256, activation='relu',kernel_initializer=he_normal())(Layer9)

Layer11=Dropout(0.5)(Layer10)



Layer12 = Dense(128, activation='relu',kernel_initializer=he_normal())(Layer11)

Layer13=BatchNormalization()(Layer12)



main_output1 = Dense(1, activation='sigmoid')(Layer13)
model3 = Model(inputs=[input_essay,input_conv], outputs=[main_output1])
model3.summary()
model3.compile(optimizer='adam', loss='binary_crossentropy', metrics=[auc_sc])
train_data3=[train_essay,train_data]

test_data3=[test_essay,test_data]
earlystop_3 = EarlyStopping(monitor = 'val_loss', mode="min",min_delta = 0, patience = 5,verbose = 1,restore_best_weights = True)

tensorboard_3=TensorBoard("logs3")
model3.fit(train_data3,y_train,epochs=8, batch_size=64,verbose=1,validation_data=(cv_data3, y_cv),class_weight='balanced',callbacks=[tensorboard_3,earlystop_3])
%reload_ext tensorboard.notebook

%tensorboard --logdir logs3
model3.evaluate(test_data3, y_test,batch_size=128)
from prettytable import PrettyTable

x=PrettyTable()

x.field_names=["Model","Train AUC","CV AUC","Test AUC"]

x.add_row(["Model 1",0.71,0.73,0.69])

x.add_row(["Model 2",0.49,0.47,0.46])

x.add_row(["Model 3",0.53,0.58,0.59])
print(x)