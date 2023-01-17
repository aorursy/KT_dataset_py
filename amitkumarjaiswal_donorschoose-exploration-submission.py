import matplotlib.pyplot
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from time import time
from tqdm import tqdm_notebook
import re
import nltk
from nltk.stem import SnowballStemmer


import tensorflow as tf

from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Activation, Dropout, Flatten, Input
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.models import Model, Sequential 
from keras.layers.merge import concatenate
from keras.utils import plot_model

from keras import backend as K
train = pd.read_csv('../input/train.csv', low_memory=False, index_col='id')
test = pd.read_csv('../input/test.csv', low_memory=False, index_col='id')

res = pd.read_csv('../input/resources.csv', low_memory=False, index_col='id')
train['is_train'] = 1
test['is_train'] = 0
df = pd.concat([train, test], axis=0)
res_sum = res.groupby(res.index)['price','quantity'].sum()
res_mean = res.groupby(res.index)['price','quantity'].mean()
res_median = res.groupby(res.index)['price','quantity'].median()
df = pd.merge(df, res_sum,left_index = True, right_index=True)
df = pd.merge(df, res_mean,left_index=True, right_index=True, suffixes=('_sum', ''))
df = pd.merge(df, res_median,left_index=True, right_index=True, suffixes=('_mean', '_median'))
df.columns
target = 'project_is_approved'

cat_features  = ['school_state', 'teacher_prefix', 
                 'project_subject_categories', 
#                 'project_subject_subcategories', 
                 'project_grade_category']

text_features = ['project_title', 'project_resource_summary', 
                 'project_essay_1', 'project_essay_2', 
                 'project_essay_3', 'project_essay_4' ]

num_features  = ['teacher_number_of_previously_posted_projects', 
                 'price_sum', 'quantity_sum', 
                 'price_mean', 'quantity_mean',
                 'price_median', 'quantity_median']
test.columns
df.project_subject_categories.value_counts().head()
dummies = pd.get_dummies(df[cat_features])
dummies_list = pd.get_dummies(df[cat_features]).columns
df = df.merge(dummies,left_index=True,right_index=True)
train_cat = df[dummies_list].values[:182080]
test_cat = df[dummies_list].values[182080:]
df[num_features].isnull().sum()
prev_posts = df.groupby('id')['teacher_number_of_previously_posted_projects'].max()
prev_posts.isnull().sum()
df = df.drop('teacher_number_of_previously_posted_projects',axis=1)
df = df.join(prev_posts)
df[num_features].isnull().sum()
df[num_features] = df[num_features].fillna(0)
SS = StandardScaler()
df_scale = SS.fit_transform(df[num_features])

train_num = df_scale[:182080]
test_num = df_scale[182080:]
df[text_features].head(3)
df_text = df[text_features].fillna(' ')
df_text['full_text'] = ''
for f in text_features:
    df_text['full_text'] = df_text['full_text'] + df_text[f]
stemmer = SnowballStemmer('english',ignore_stopwords=True)

def clean(text):
    return re.sub('[!@#$:]', '', ' '.join(re.findall('\w{3,}', str(text).lower())))

def stem(text):
    return ' '.join([stemmer.stem(w) for w in text.split()])
df_text['full_text'] = df_text['full_text'].apply(lambda x: clean(x))
max_words = 500 #more words for more accuracy
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df_text['full_text'])

train_text = tokenizer.texts_to_matrix(df_text['full_text'][:182080], mode='binary')
test_text = tokenizer.texts_to_matrix(df_text['full_text'][182080:], mode='binary')
y = df[target].values[:182080]

len_cat = train_cat.shape[1]
len_num = train_num.shape[1]
len_text = train_text.shape[1]

size_embedding = 5000
# AUC for a binary classifier
def auc(y_true, y_pred):   
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)
# PFA, prob false alert for binary classifier
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)    
    return FP/N
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# P_TA prob true alerts for binary classifier
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)    
    return TP/P
# categorical NN 
inputs1 = Input(shape=(len_cat,))
dense_cat_1 = Dense(256, activation='relu')(inputs1)
dense_cat_2 = Dense(128, activation='relu')(dense_cat_1)
dense_cat_3 = Dense(64, activation='relu')(dense_cat_2)
dense_cat_4 = Dense(32, activation='relu')(dense_cat_3)
final_cat = Dense(32, activation='relu')(dense_cat_4)
# num channel
inputs2 = Input(shape=(len_num,))
dense_num_1 = Dense(256, activation='relu')(inputs2)
dense_num_2 = Dense(128, activation='relu')(dense_num_1)
dense_num_3 = Dense(64, activation='relu')(dense_num_2)
dense_num_4 = Dense(32, activation='relu')(dense_num_3)
final_num = Dense(32, activation='relu')(dense_num_4)
# text chanel
inputs3 = Input(shape=(len_text,))
embedding3 = Embedding(size_embedding, 36)(inputs3)
conv3 = Conv1D(filters=32, kernel_size=8, activation='relu')(embedding3)
drop3 = Dropout(0.1)(conv3)
pool3 = MaxPooling1D(pool_size=2)(drop3)
final_text = Flatten()(pool3)
#individual Model Tester
mod = Model(inputs1, Dense(1, activation='sigmoid')(final_cat))
mod.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy', auc])
batch_size = 100
mod.fit(train_cat, y, batch_size = batch_size,
        epochs=3, validation_split=0.2)
# merge
merged = concatenate([final_cat, final_num, final_text])

# Multi NN
dense1 = Dense(200, activation='relu')(merged)
dense2 = Dense(20, activation='relu')(dense1)
outputs = Dense(1, activation='sigmoid')(dense2)
model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
# Compile
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy', auc])
# Fitting
batch_size = 100
model.fit([train_cat, train_num, train_text], y, 
          batch_size=batch_size, epochs=2, validation_split=0.2)
#2.5. Submitting
submit = model.predict([test_cat, test_num, test_text], 
                       batch_size=batch_size,verbose=1)

submission = pd.read_csv('../input/sample_submission.csv')
submission['project_is_approved'] = submit
submission.to_csv('prediction.csv', index=False)
#Non-multichannel type of NN
train_all = np.hstack((train_cat, train_real, train_text))
train_all.shape
model2 = Sequential()
model2.add(Dense(256, input_shape=(train_all.shape[1],), activation='relu'))
model2.add(Dense(128, activation='relu'))
model2.add(Dense(1, activation='sigmoid'))
model2.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy', auc])
batch_size = 2000
model2.fit(train_all, y, 
           batch_size=batch_size, epochs=3, 
           validation_split=0.2)