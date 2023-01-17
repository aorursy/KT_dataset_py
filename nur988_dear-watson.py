import numpy as np

import pandas as pd

import os

for dirname,_,filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname,filename))
os.environ['WANDB_API_KEY']="0"
from transformers import BertTokenizer,TFBertModel

import matplotlib.pyplot as plt

import tensorflow as tf
try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

except ValueError:

    strategy = tf.distribute.get_strategy() # for CPU and single GPU

    print('Number of replicas:', strategy.num_replicas_in_sync)
train=pd.read_csv('../input/contradictory-my-dear-watson/train.csv')

train.head()
train.premise.values[1]
train.hypothesis.values[1]
train.label.values[1]
labels,frequencies=np.unique(train.language.values,return_counts=True)

plt.figure(figsize=(10,10))

plt.pie(frequencies,labels=labels,autopct='%1.1f%%')

plt.show()
model_name='bert-base-multilingual-cased'

tokenizer=BertTokenizer.from_pretrained(model_name)
def encode_sentence(s):

    tokens=list(tokenizer.tokenize(s))

    tokens.append('[SEP]')

    return tokenizer.convert_tokens_to_ids(tokens)
encode_sentence("I Love Machine Learning")
def bert_encode(hypotheses,premises,tokenizer):

    num_examples=len(hypotheses)

    sentence1=tf.ragged.constant([encode_sentence(s) for s in np.array(hypotheses)])

    sentence2=tf.ragged.constant([encode_sentence(s) for s in np.array(premises)])

    cls=[tokenizer.convert_tokens_to_ids(['[CLS]'])]*sentence1.shape[0]

    input_word_ids = tf.concat([cls, sentence1, sentence2], axis=-1)

    input_mask=tf.ones_like(input_word_ids).to_tensor()

    type_cls=tf.zeros_like(cls)

    type_s1=tf.zeros_like(sentence1)

    type_s2=tf.ones_like(sentence2)

    input_type_ids=tf.concat(

    [type_cls,type_s1,type_s2],axis=-1).to_tensor()

    

    inputs={

        'input_word_ids':input_word_ids.to_tensor(),

        'input_mask':input_mask,

        'input_type_ids':input_type_ids

    }

    return inputs
train_input=bert_encode(train.premise.values,train.hypothesis.values,tokenizer)
train_input
max_len = 50



def build_model():

    bert_encoder = TFBertModel.from_pretrained(model_name)

    input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")

    input_type_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_type_ids")

    

    embedding = bert_encoder([input_word_ids, input_mask, input_type_ids])[0]

    output = tf.keras.layers.Dense(3, activation='softmax')(embedding[:,0,:])

    

    model = tf.keras.Model(inputs=[input_word_ids, input_mask, input_type_ids], outputs=output)

    model.compile(tf.keras.optimizers.Adam(lr=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    

    return model
with strategy.scope():

    model=build_model()

    model.summary()
model.fit(train_input,train.label.values,epochs=2,verbose=1,batch_size=64,validation_split=0.2)

test=pd.read_csv('../input/contradictory-my-dear-watson/test.csv')
test.head()
test_input=bert_encode(test.premise.values,test.hypothesis.values,tokenizer)
predictions=[np.argmax(i) for i in model.predict(test_input)]
submission=test.id.copy().to_frame()

submission['prediction']=predictions
submission.head()
submission.to_csv("submission.csv",index=False)