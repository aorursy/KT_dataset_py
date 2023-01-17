import numpy as np

import pandas as pd

import os, time, random, warnings

from tqdm import tqdm



from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

import plotly.express as px

init_notebook_mode(connected=True)

import plotly.graph_objs as go



import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.layers import Dense,Input,Dropout

from tensorflow.keras.models import Model

import tensorflow_addons as tfa



from transformers import BertTokenizer, TFBertModel



try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

except ValueError:

    strategy = tf.distribute.get_strategy() # for CPU and single GPU

    print('Number of replicas:', strategy.num_replicas_in_sync)

    



CFG = {

'epochs' : 20,

'batch_size' : 32*strategy.num_replicas_in_sync,

'random_seed' : 42,

'max_len' : 100,

'validation_split':0.2

}







np.random.seed(CFG['random_seed'])

tf.random.set_seed(CFG['random_seed'])

os.environ["WANDB_API_KEY"] = "0" ## to silence warning

warnings.simplefilter("ignore")
train_df = pd.read_csv('../input/contradictory-my-dear-watson/train.csv')

train_df
iplot(px.pie(train_df,names='language',title='Languages counts'))
iplot(px.pie(train_df,names='label',title='Languages counts'))
plt.bar(['premise','hypothesis'],[np.mean([len(i) for i in train_df['premise'].apply(lambda x: x.split())]),np.mean([len(i) for i in train_df['hypothesis'].apply(lambda x: x.split())])])

plt.title('avg sentence len')

plt.show()
model_name = 'bert-base-multilingual-cased'

tokenizer = BertTokenizer.from_pretrained(model_name)
def encode_sentence(s):

    tokens = list(tokenizer.tokenize(s))

    tokens.append('[SEP]')

    return tokenizer.convert_tokens_to_ids(tokens)
def bert_encode(premises,hypothesis,tokenize):

    

    num_examples = len(hypothesis)

    

    sentence1 = tf.ragged.constant([

        encode_sentence(s) for s in np.array(premises)

    ])

    

    sentence2 = tf.ragged.constant([

        encode_sentence(s) for s in np.array(hypothesis)

    ])

    

    cls = [tokenizer.convert_tokens_to_ids(['[SEP]'])]*sentence1.shape[0]

    

    input_words_ids = tf.concat([cls,sentence1,sentence2],axis=-1)

    

    input_mask=tf.ones_like(input_words_ids)

    

    type_cls = tf.zeros_like(cls)

    type_sentence1 = tf.zeros_like(sentence1)

    type_sentence2 = tf.ones_like(sentence2)

    

    input_type_ids = tf.concat([type_cls,type_sentence1,type_sentence2],axis=-1)

    

    inputs={

        'input_words_ids' :input_words_ids.to_tensor(),

        'input_mask' : input_mask.to_tensor(),

        'input_type_ids' : input_type_ids.to_tensor()

    }

    

    return inputs
input_train = bert_encode(train_df.premise.values,train_df.hypothesis.values,tokenizer)



# input_val = dict()

# val_start = int(len(train_df.label.values)*CFG['validation_split'])



# for key in input_train.keys():

#     input_val[key] =input_train[key][val_start:]

#     input_train[key] = input_train[key][:val_start]
maxlen = CFG['max_len']



def build_model(l):

    bert_encoder = TFBertModel.from_pretrained(model_name)

    

    input_words_ids = Input(shape=(maxlen),dtype=tf.int32,name='input_words_ids')

    input_mask = Input(shape=(maxlen,),dtype=tf.int32,name='input_mask')

    input_type_ids = Input(shape=(maxlen,),dtype=tf.int32,name='input_type_ids')

    

    embedding = bert_encoder([input_words_ids,input_mask,input_type_ids])[0]

    

    fc1 = Dense(1024,activation='relu')(embedding[:,0,:])

    drop = Dropout(0.2)(fc1)

    

    output = Dense(3,activation='softmax')(drop)

    

    

    model = Model(inputs=[input_words_ids,input_mask,input_type_ids],outputs=output)

    



    model.compile(

        optimizer = keras.optimizers.Adam(lr=l),

        loss='sparse_categorical_crossentropy',

        metrics=['accuracy']

    )

    

    return model
# for e in range(3,4):

#     for l in [5e-5,3e-5,2e-5]:

        

#         with strategy.scope():

#             model = build_model(l)

        

#         print(f"----------------- epochs: {e} ,learning rate: {l} -----------------")

        

#         hist = model.fit(input_train,train_df.label.values,batch_size=CFG['batch_size'],epochs=e,verbose=1,validation_split=CFG['validation_split'])

        

        
with strategy.scope():

    model = build_model(l=1e-5)

    model.summary()

    

hist = model.fit(input_train,train_df.label.values,batch_size=CFG['batch_size'],epochs=CFG['epochs'],verbose=1)
test = pd.read_csv('../input/contradictory-my-dear-watson/test.csv')

input_test = bert_encode(test.premise.values,test.hypothesis.values,tokenizer)



preds = model.predict(input_test)
preds = [np.argmax(i) for i in preds]

preds
submission = pd.read_csv('../input/contradictory-my-dear-watson/sample_submission.csv')

submission
submission['prediction'] = preds

submission

submission.to_csv('submission.csv',index=False)