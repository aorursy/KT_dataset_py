import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import tensorflow.keras.layers as L

import tensorflow.keras.models as M

from keras.applications.resnet50 import ResNet50

import tensorflow as tf



from transformers import BertConfig,TFBertModel,BertModel

import matplotlib.pyplot as plt
AUTO = tf.data.experimental.AUTOTUNE

# Detect hardware, return appropriate distribution strategy

try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.



print("REPLICAS: ", strategy.num_replicas_in_sync)
train = pd.read_csv('/kaggle/input/conways-reverse-game-of-life-2020/train.csv')
y = train[[x for x in train.columns if 'start' in x]].replace(0,-1)

train = np.resize(train[[x for x in train.columns if 'stop' in x]].replace(0,-1).values,(50000,512))
train.shape
test= pd.read_csv('/kaggle/input/conways-reverse-game-of-life-2020/train.csv')

test = np.resize(test[[x for x in test.columns if 'stop' in x]].replace(0,-1).values,(50000,512))
train.shape,y.shape
def build_model(MAX_LEN = 512, NUM_BINS = len(np.unique(train))):

    ids = L.Input((MAX_LEN,), dtype=tf.int32)

    config = BertConfig() 

    config.vocab_size = NUM_BINS

    config.num_hidden_layers = 2

    bert_model = TFBertModel(config=config)



    x = bert_model(ids)[0]

    x = L.Flatten()(x)

    x = L.Dense(625,activation='tanh')(x)

    

    model = M.Model(inputs=ids, outputs=x)

#     optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    model.compile(loss=tf.keras.losses.Hinge(reduction="auto", name="hinge"), optimizer='adam',metrics = 'mae')



    return model
# def NN():

#     start = L.Input((625,))

#     embed = L.Embedding(2,25,input_length=625)

# #     image_tensor = L.Reshape((25,25,1))(start)

    

# #     conv = L.Conv2D(3,(3,3),padding='same')(image_tensor) 



# #     res = ResNet50(weights=None, include_top=False)

# #     res.layers.pop()

# #     for layer in res.layers:

# #         layer.trainable = True

# #     fe = M.Model(inputs=res.inputs, outputs=res.layers[-1].output)



# #     out = fe(conv)

# #     flat = L.Flatten()(embed)

# #     stop = L.Dense(625,activation='tanh')(flat)

# #     stop = L.Reshape((625,1))(stop)

# # # #     stop = L.Activation(activation='sigmoid')(stop)

#     stop = L.Flatten()(embed)

# #     stop = tf.math.round(stop)

#     stop = L.Dense(625,activation='relu')(stop)

#     model = M.Model(start,embed)

#     model.compile(loss='mae',optimizer='adam',metrics = 'mae')

#     return model
# with strategy.scope():

with tf.device('/gpu:0'):

    clf = build_model()

    clf.fit(train,y,epochs = 2)
clf.summary()
# with strategy.scope():

preds = clf.predict(test)
np.min(preds),np.max(preds)
np.unique(preds)
preds = np.where(preds>0,1,0).astype(int)
# preds = np.clip(np.round(preds),0,1).astype(int)
np.bincount(np.where(y.values.ravel()>0,1,0).astype(int))
26294051/4955949
np.bincount(preds.ravel().astype(int))
31200000/50000
pd.DataFrame(preds).describe()
sub = pd.read_csv('/kaggle/input/conways-reverse-game-of-life-2020/sample_submission.csv')
sub[[x for x in sub.columns.tolist() if 'start' in x]] = preds 
sub.to_csv("submission.csv", index=False)