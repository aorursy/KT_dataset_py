import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import tensorflow as tf

import tensorflow_hub as hub #to re use existing models of ML

import keras

import keras.backend as K

from keras.layers import *

from keras.callbacks import *

from keras.optimizers import *

from keras import Model



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import pickle    

import os
def save_obj(obj, name ):

    with open(name + '.pkl', 'wb') as f:

        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):

    with open(name + '.pkl', 'rb') as f:

        return pickle.load(f)

                

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# reading dataset

submission = pd.read_csv('/kaggle/input/google-quest-challenge/sample_submission.csv')

train = pd.read_csv("../input/google-quest-challenge/train.csv")

test = pd.read_csv("../input/google-quest-challenge/test.csv")

# concatenating both datasets

df = pd.concat([train, test])

df.head()
#loading module

module_url = "/kaggle/input/universalsentenceencoderlarge4/"

embed = hub.load(module_url)
# Shape of the Dataset

df.shape
# Checking Datatypes

df.dtypes
# Checking Missing Value

df.isnull().sum()
# Selecting Target Columns

targets = [

        'question_asker_intent_understanding',

        'question_body_critical',

        'question_conversational',

        'question_expect_short_answer',

        'question_fact_seeking',

        'question_has_commonly_accepted_answer',

        'question_interestingness_others',

        'question_interestingness_self',

        'question_multi_intent',

        'question_not_really_a_question',

        'question_opinion_seeking',

        'question_type_choice',

        'question_type_compare',

        'question_type_consequence',

        'question_type_definition',

        'question_type_entity',

        'question_type_instructions',

        'question_type_procedure',

        'question_type_reason_explanation',

        'question_type_spelling',

        'question_well_written',

        'answer_helpful',

        'answer_level_of_information',

        'answer_plausible',

        'answer_relevance',

        'answer_satisfaction',

        'answer_type_instructions',

        'answer_type_procedure',

        'answer_type_reason_explanation',

        'answer_well_written'    

    ]

#inputs 

input_columns = ['question_title','question_body','answer']
#question title column to list

X1 = train[input_columns[0]].values.tolist()

#question body column to list

X2 = train[input_columns[1]].values.tolist()

#answer column to list

X3 = train[input_columns[2]].values.tolist()

X1 = [x.replace('?','.').replace('!','.') for x in X1]

X2 = [x.replace('?','.').replace('!','.') for x in X2]

X3 = [x.replace('?','.').replace('!','.') for x in X3]



X = [X1,X2,X3]

y = train[targets].values.tolist()
# Plotting the channels where the Training queries comes from in the data

fig = plt.figure(figsize=(10,6))

ax = fig.add_subplot(111)

width = 0.4

df.host.value_counts().plot(kind='bar', color='blue', ax=ax, width=width, position=1)

ax.set_xlabel('Sites')

ax.set_ylabel('Question Counts')
# Plotting the category occurance of the queries

fig = plt.figure(figsize=(7,6))

ax = fig.add_subplot(111)

width = 0.2

df.category.value_counts().plot(kind='bar', color='green', ax=ax, width=width, position=1, legend=True)

ax.set_xlabel('Sites')

ax.set_ylabel('Question Counts')
def UniversalEmbedding(x):

    results = embed(tf.squeeze(tf.cast(x, tf.string)))["outputs"]

    print(results)

    return keras.backend.concatenate([results])
# build network

def swish(x):

    return K.sigmoid(x) * x



embed_size = 512 #must be 512 for univerasl embedding layer



input_text1 = Input(shape=(1,), dtype=tf.string)

embedding1 = Lambda(UniversalEmbedding, output_shape=(embed_size,))(input_text1)

input_text2 = Input(shape=(1,), dtype=tf.string)

embedding2 = Lambda(UniversalEmbedding, output_shape=(embed_size,))(input_text2)

input_text3 = Input(shape=(1,), dtype=tf.string)

embedding3 = Lambda(UniversalEmbedding, output_shape=(embed_size,))(input_text3)



x = Concatenate()([embedding1,embedding2,embedding3])

x = Dense(256, activation=swish)(x)

x = Dropout(0.4)(x)

x = BatchNormalization()(x)

x = Dense(64, activation=swish, kernel_regularizer=keras.regularizers.l2(0.001))(x)

x = Dropout(0.4)(x)

x = BatchNormalization()(x)

output = Dense(len(targets),activation='sigmoid',name='output')(x)
#model summary

model = Model(inputs=[input_text1,input_text2,input_text3], outputs=[output])

model.summary()
# Training, clean up and optimization

import gc

print(gc.collect())

# Train the network

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,

                              patience=2, min_lr=1e-7, verbose=1)

optimizer = Adadelta()



model.compile(optimizer=optimizer, loss='binary_crossentropy')

model.fit(X, [y], epochs=10, validation_split=.1,batch_size=32,callbacks=[reduce_lr])
# prep test data

X1 = test[input_columns[0]].values.tolist()

X2 = test[input_columns[1]].values.tolist()

X3 = test[input_columns[2]].values.tolist()

X1 = [x.replace('?','.').replace('!','.') for x in X1]

X2 = [x.replace('?','.').replace('!','.') for x in X2]

X3 = [x.replace('?','.').replace('!','.') for x in X3]



pred_X = [X1,X2,X3]

# Make a prediction

pred_y = model.predict(pred_X)

# Check the submission

submission = pd.read_csv('/kaggle/input/google-quest-challenge/sample_submission.csv')

submission[targets] = pred_y

submission.head()
# Save the result

submission.to_csv("submission.csv", index = False)