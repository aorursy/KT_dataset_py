# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.max_colwidth', -1)

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import plotly_express as px

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv("/kaggle/input/contradictory-my-dear-watson/train.csv")

test=pd.read_csv("/kaggle/input/contradictory-my-dear-watson/test.csv")
print("Shape of Training Data ",train.shape)

print("Shape of Testing Data ",test.shape)
train.head(10)
lang_count=train.groupby(['language']).size().reset_index().rename(columns={0:'count'})

print("Number of Languages in the Data ",train['language'].nunique())
fig = px.pie(lang_count, values='count', names='language', title='Distribution of Text Across Languages')

fig.show()
label_mapping={0:'entailment',1:'neutral',2:'contradiction'}
train['Relationship_Type']=train['label'].apply(lambda x:label_mapping[x])

train.head()
type_count=train.groupby(['Relationship_Type']).size().reset_index().rename(columns={0:'count'})

fig = px.pie(type_count, values='count', names='Relationship_Type', title='Distribution of Type of Relationship Between Premise and Hypothesis')

fig.show()
from transformers import BertTokenizer,TFBertModel

import tensorflow as tf
try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

    print("In TPU STRATERGY")

    print('Number of replicas:', strategy.num_replicas_in_sync)

except ValueError:

    strategy = tf.distribute.get_strategy() # for CPU and single GPU

    print('Number of replicas:', strategy.num_replicas_in_sync)
MODEL_NAME="bert-base-multilingual-cased"

MAX_LEN=64

BATCH_SIZE=32
tokeniser=BertTokenizer.from_pretrained(MODEL_NAME)

print("Vocab Size of the Bert Multilingual Tokeniser ",tokeniser.vocab_size)
from keras.preprocessing.sequence import pad_sequences

def convertToTokens(sentence,tokeniser):

    tokens=tokeniser.tokenize(sentence)

    tokens.append('[SEP]')

    return tokeniser.convert_tokens_to_ids(tokens)
def encode(hypothesis_list,premise_list,tokeniser,max_seq_len=128):

    num_examples=len(hypothesis_list)

    hypothesis=tf.ragged.constant([convertToTokens(s,tokeniser) for s in np.array(hypothesis_list)])

    premise=tf.ragged.constant([convertToTokens(s,tokeniser) for s in np.array(premise_list)])

    

    ### Add CLS Token to the beginning of the Hypothesis. 

    cls=[tokeniser.convert_tokens_to_ids(['[CLS]'])]*hypothesis.shape[0]

    

    

    input_ids=tf.concat([cls,hypothesis,premise],axis=-1) 

    print("Input IDS Type ",type(input_ids))

    

    ### mask should be zero for all 

    mask=tf.zeros_like(input_ids).to_tensor() ## Same shape with all elements filled with 0 as input_ids

    

    ### Token_Type_IDS => 0 for CLS and the Hypothesis and 1 for the premise

    cls_token_type=tf.zeros_like(cls)

    hypothesis_token_type=tf.zeros_like(hypothesis)

    premise_token_type=tf.ones_like(premise)

    

    input_type_ids=tf.concat([cls_token_type,hypothesis_token_type,premise_token_type],axis=-1).to_tensor()

    

    return {'input_word_ids': input_ids.to_tensor(),'input_mask': mask,'input_type_ids': input_type_ids}
X=train[['hypothesis','premise']]

y=train['label']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Shape of Train Data ",X_train.shape)

print("Shape of Test Data ",X_test.shape)

from transformers import TFAutoModel ## Will create Model Architecture, based on the path of the pretrained model
from tensorflow.keras import Input, Model, Sequential

from tensorflow.keras.layers import Dense, Dropout

from keras.optimizers import Adam
def createModel():

    with strategy.scope():

        model=TFAutoModel.from_pretrained(MODEL_NAME)

        input_ids = tf.keras.layers.Input(shape=(MAX_LEN,), name='input_token', dtype='int32')

        input_mask=tf.keras.layers.Input(shape=(MAX_LEN,), name='input_mask', dtype='int32')

        input_token_ids=tf.keras.layers.Input(shape=(MAX_LEN,), name='input_token_ids', dtype='int32')

        ### From the Model, we need to extract the Last Hidden Layer - this is the first element of the model output

        embedding=model([input_ids,input_mask,input_token_ids])[0]

        ### Extract the CLS Token from the Embedding Layer. CLS Token is aggregate of the entire sequence representation. It is the first token

        cls_token=embedding[:,0,:] ## embedding is of the size batch_size*MAX_LEN*768

    

        ### Add a Dense Layer, with three outputs 

        output_layer = Dense(3, activation='softmax')(cls_token)

    

        classification_model= Model(inputs=[input_ids, input_mask, input_token_ids], outputs = output_layer)

    

        classification_model.compile(Adam(lr=1e-5),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    

        #classification_model.summary()

    

    return classification_model

    

    

    

    
with strategy.scope():

    bert_model=createModel()

    bert_model.summary()
train_inputs=encode(X_train.hypothesis.values,X_train.premise.values,tokeniser)

val_inputs=encode(X_test.hypothesis.values,X_test.premise.values,tokeniser)

history = bert_model.fit([train_inputs['input_word_ids'],train_inputs['input_mask'],train_inputs['input_type_ids']],y_train,validation_data=([val_inputs['input_word_ids'],val_inputs['input_mask'],val_inputs['input_type_ids']],y_test),epochs = 15, batch_size = 16,shuffle = True)
test_inputs = encode(test.hypothesis.values, test.premise.values, tokeniser)
predictions = [np.argmax(i) for i in bert_model.predict([test_inputs['input_word_ids'],test_inputs['input_mask'],test_inputs['input_type_ids']])]

submission = pd.DataFrame()

submission['id']=test['id'].tolist()

submission['prediction'] = predictions
submission.head()
submission.to_csv("submission.csv", index = False)