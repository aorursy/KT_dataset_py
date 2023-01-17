import numpy as np

import pandas as pd

import matplotlib.pyplot as plt 

import seaborn as sns

import os

import re

import plotly.express as px

import plotly.figure_factory as ff

import plotly.graph_objects as go

from tqdm import tqdm



import tensorflow as tf

from tensorflow import keras

from keras import backend as k

from keras.utils import to_categorical

import transformers
DEVICE = 'TPU'
if DEVICE == 'TPU':

    print('Connecting to TPU...')

    try:

        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

        print('Running on TPU :',tpu.master())

    except ValueError:

        print('Could not connect to TPU')

        tpu = None

        

    if tpu:

        try:

            print('Initializing TPU...')

            tf.config.experimental_connect_to_cluster(tpu)

            tf.tpu.experimental.initialize_tpu_system(tpu)

            strategy = tf.distribute.experimental.TPUStrategy(tpu)

            print('TPU initialized!')

            

        except _:

            print('Failed to initialized TPU')

            

    else:

        DEVICE='GPU'



if DEVICE != 'TPU':

    print('Using default strategy for CPU and single GPU')

    strategy = tf.distribute.get_strategy()



if DEVICE == 'GPU':

    print('Num GPUs available : ',len(tf.config.experimental.list_physical_devices('GPU')))

    

AUTO = tf.data.experimental.AUTOTUNE

REPLICAS = strategy.num_replicas_in_sync

print('REPLICAS : ',REPLICAS)
Batch_size = 16 * strategy.num_replicas_in_sync

epochs = 13

AUTO = tf.data.experimental.AUTOTUNE



MODEL = 'jplu/tf-xlm-roberta-large'
train = pd.read_csv(r'../input/contradictory-my-dear-watson/train.csv')

test = pd.read_csv(r'../input/contradictory-my-dear-watson/test.csv')

submission = pd.read_csv(r'../input/contradictory-my-dear-watson/sample_submission.csv')
train.head()
test.head()
num_lang = train.groupby('language')['id'].count().sort_values(ascending=False).reset_index()

num_lang = pd.DataFrame(num_lang)

num_lang['count'] = num_lang['id']

num_lang = num_lang.drop('id',axis=1)

num_lang_data = num_lang.style.background_gradient(cmap='Greens')

num_lang_data
test_num_lang = test.groupby('language')['id'].count().sort_values(ascending=False).reset_index()

test_num_lang = pd.DataFrame(test_num_lang)

test_num_lang['count'] = test_num_lang['id']

test_num_lang = test_num_lang.drop('id',axis=1)

test_num_lang_data = test_num_lang.style.background_gradient(cmap='Oranges')

test_num_lang_data
fig = px.pie(num_lang,values='count',names='language',title='Language and their percentage in the train data :',color_discrete_sequence=px.colors.sequential.GnBu)

fig.update_traces(hoverinfo='label+percent', textfont_size=14,

                  marker=dict(line=dict(color='#000000', width=1.2)))

fig.show()



fig = px.pie(test_num_lang,values='count',names='language',title='Language and their percentage in the test data :',color_discrete_sequence=px.colors.sequential.RdBu)

fig.update_traces(hoverinfo='label+percent', textfont_size=14,

                  marker=dict(line=dict(color='#000000', width=1.2)))

fig.show()
fig = px.bar(num_lang,x='language',y='count')

fig.update_traces(marker_color='ivory', marker_line_color='black',

                  marker_line_width=1.3, opacity=0.5)

fig.update_layout(title_text='Languages and their count in the data')

fig.show()



fig = px.bar(test_num_lang,x='language',y='count')

fig.update_traces(marker_color='darkturquoise', marker_line_color='black',

                  marker_line_width=1.3, opacity=0.5)

fig.update_layout(title_text='Languages and their count in the data')

fig.show()
num_words_train_h = [None] * len(train)

for i in range(len(train)):

    num_words_train_h[i] = len(train['hypothesis'][i])

num_words_train_p = [None] * len(train)

for i in range(len(train)):

    num_words_train_p[i] = len(train['premise'][i])

    

num_words_test_h = [None] * len(test)

for i in range(len(test)):

    num_words_test_h[i] = len(test['hypothesis'][i])

num_words_test_p = [None] * len(test)

for i in range(len(test)):

    num_words_test_p[i] = len(test['premise'][i])
print('Maximum and minimum number of words in a single sentence in hypothesis in the train data :',(max(num_words_train_h),min(num_words_train_h)))

print('Maximum and minimum number of words in a single sentence in hypothesis in the test data :',(max(num_words_test_h),min(num_words_test_h)))

print('Maximum and minimum number of words in a single sentence in premise in the train data :',(max(num_words_train_p),min(num_words_train_p)))

print('Maximum and minimum number of words in a single sentence in premise in the test data :',(max(num_words_test_p),min(num_words_test_p)))
train['num_words_hypothesis'] = num_words_train_h

train['num_words_premise'] = num_words_train_p

test['num_words_hypothesis'] = num_words_test_h

test['num_words_premise'] = num_words_test_p
english_train = train[train['language']=='English']

english_test = test[test['language']=='English']





hist_data = [english_train['num_words_hypothesis'],english_train['num_words_premise']]

group_labels = ['hypothesis','premise']

fig = ff.create_distplot(hist_data,group_labels,colors=['ivory','teal'])

fig.update_layout(title_text='Number of words for English Language in train data')

fig.show()



hist_data = [english_test['num_words_hypothesis'],english_test['num_words_premise']]

group_labels = ['hypothesis','premise']

fig = ff.create_distplot(hist_data,group_labels,colors=['red','greenyellow'])

fig.update_layout(title_text='Number of words for English Language in test data')

fig.show()
labels = train['label'].sort_values().value_counts().reset_index()

labels = pd.DataFrame(labels)

labels.columns = ['label','count']

labels_ = labels.style.background_gradient(cmap='Blues')

labels_
target = train['label']

train = train.drop('label',axis=1)

train_text = [None] * len(train)

test_text = [None] * len(test)

for i in range(len(train)):

    train_text[i] = train['premise'][i] + ' ' + train['hypothesis'][i]

for i in range(len(test)):

    test_text[i] = test['premise'][i] + ' ' + test['hypothesis'][i]
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)
def roberta_encode(texts, maxlen=512):

    enc_di = tokenizer.batch_encode_plus(

        texts,  

        pad_to_max_length=True,

        max_length=maxlen

    )

    

    return np.array(enc_di['input_ids'])
train_input_ids = roberta_encode(train_text,maxlen=100)

test_input_ids = roberta_encode(test_text,maxlen=100)
from sklearn.model_selection import train_test_split

train_input_ids,validation_input_ids,train_labels,validation_labels = train_test_split(train_input_ids,target,test_size=0.2)
train_input_ids[7]
validation_input_ids[1]
test_input_ids[3]
validation_labels
train_labels
train_dataset = (

    tf.data.Dataset

    .from_tensor_slices((train_input_ids,train_labels))

    .repeat()

    .shuffle(2048)

    .batch(Batch_size)

    .prefetch(AUTO)

)



validation_dataset = (

    tf.data.Dataset

    .from_tensor_slices((validation_input_ids, validation_labels))

    .batch(Batch_size)

    .cache()

    .prefetch(AUTO)

)



test_dataset = (

    tf.data.Dataset

    .from_tensor_slices(test_input_ids)

    .batch(Batch_size)

)
def create_model(bert_model):

    input_ids = tf.keras.Input(shape=(100,),dtype='int32')

  

    output = bert_model(input_ids)[0]

    output = output[:,0,:]

    output = tf.keras.layers.Dense(3,activation='softmax')(output)

    model = tf.keras.models.Model(inputs = input_ids,outputs = output)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
with strategy.scope():

    bert_model = (

        

        transformers.TFAutoModel  

        .from_pretrained(MODEL)    

    )

    model = create_model(bert_model)   
model.summary()
history = model.fit(train_dataset,

                    validation_data = validation_dataset,

                    epochs = epochs,   

                    batch_size = Batch_size,

                    steps_per_epoch = len(train_input_ids)//Batch_size

                   )
plt.figure(figsize=(10,8))

plt.plot(history.history['accuracy'],color='orange')

plt.plot(history.history['val_accuracy'],color='green')

plt.legend(loc='best',shadow=True)

plt.grid()

plt.show()
plt.figure(figsize=(10,8))

plt.plot(history.history['loss'],color='orange')

plt.plot(history.history['val_loss'],color='green')

plt.legend(loc='best',shadow=True)

plt.grid()

plt.show()
pred = model.predict(test_dataset,verbose=1)

print(len(pred))

pred = pred.argmax(axis=1)

submission.prediction = pred      

submission.to_csv('submission.csv',index=False)    

submission.head()