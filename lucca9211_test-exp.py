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
# import library

import tensorflow as tf 



from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.utils import shuffle

import matplotlib.pyplot as plt
# # Detect hardware, return appropriate distribution strategy

# try:

#     # TPU detection. No parameters necessary if TPU_NAME environment variable is

#     # set: this is always the case on Kaggle.

#     tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

#     print('Running on TPU ', tpu.master())

# except ValueError:

#     tpu = None



# if tpu:

#     tf.config.experimental_connect_to_cluster(tpu)

#     tf.tpu.experimental.initialize_tpu_system(tpu)

#     strategy = tf.distribute.experimental.TPUStrategy(tpu)

# else:

#     # Default distribution strategy in Tensorflow. Works on CPU and single GPU.

#     strategy = tf.distribute.get_strategy()



# print("REPLICAS: ", strategy.num_replicas_in_sync)
#BATCH_SIZE = 18 * strategy.num_replicas_in_sync
df1 = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv", usecols=["comment_text", "toxic"])

df2 = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv",  usecols=["comment_text", "toxic"])

#df3 = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/validation.csv", usecols=["comment_text", "toxic"])

df3 = pd.read_csv("../input/jigsaw-multilingual-toxic-test-translated/jigsaw_miltilingual_valid_translated.csv", usecols=["translated", "toxic"])

df2.toxic = df2.toxic.round().astype(int)

df3.toxic = df3.toxic.round().astype(int)

df3.rename(columns={"translated":"comment_text"}, inplace = True)

df_train = pd.concat([

                      df1[["comment_text", "toxic"]],

                      df3[["comment_text", "toxic"]],

                      df2[["comment_text", "toxic"]].query('toxic==1'),

                      df2[["comment_text", "toxic"]].query('toxic==0').sample(n=200000, random_state=0)],

                      axis=0).reset_index(drop=True)

df_train = df_train.dropna()



df_train = shuffle(df_train, random_state=22)

df_train.head()
print("Training Shape:-", df_train.shape)
# Split our data into train and test sets

train_size = int(len(df_train) * .8)

print ("Train size: %d" % train_size)

print ("Test size: %d" % (len(df_train) - train_size))
# Split our labels into train and test sets

num_labels =1

train_toxic = df_train['toxic'].values[:train_size]

test_toxic = df_train['toxic'].values[train_size:]

%%writefile preprocess.py

# Pre-processing data: create our tokenizer class

from tensorflow.keras.preprocessing import text



class TextPreprocessor(object):

  def __init__(self, vocab_size):

    self._vocab_size = vocab_size

    self._tokenizer = None

  

  def create_tokenizer(self, text_list):

    tokenizer = text.Tokenizer(num_words=self._vocab_size)

    tokenizer.fit_on_texts(text_list)

    self._tokenizer = tokenizer



  def transform_text(self, text_list):

    text_matrix = self._tokenizer.texts_to_matrix(text_list)

    return text_matrix
# Create vocab from training corpus

from preprocess import TextPreprocessor



VOCAB_SIZE=400 # This is a hyperparameter, try out different values for your dataset



train_text = df_train['comment_text'].values[:train_size]

test_text = df_train['comment_text'].values[train_size:]



processor = TextPreprocessor(VOCAB_SIZE)

processor.create_tokenizer(train_text)



body_train = processor.transform_text(train_text) 

body_test = processor.transform_text(test_text)
# Save the processor state of the tokenizer

import pickle

with open('processor_state.pkl', 'wb') as f:

  pickle.dump(processor, f)
def create_model(vocab_size, num_labels):

  

  model = tf.keras.models.Sequential()

  model.add(tf.keras.layers.Dense(50, input_shape=(VOCAB_SIZE,), activation='relu'))

  model.add(tf.keras.layers.Dense(25, activation='relu'))

  model.add(tf.keras.layers.Dense(num_labels, activation='sigmoid'))



  #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

  model.compile(

  optimizer=tf.keras.optimizers.Adam(1e-5),

  loss=tf.keras.losses.BinaryCrossentropy(),#losses.SparseCategoricalCrossentropy(from_logits=True),

  metrics=[tf.keras.metrics.AUC()]#metrics.SparseCategoricalAccuracy(name="acc")]

  )

  return model

%%time

# with strategy.scope():

#     model = create_model(VOCAB_SIZE, num_labels)

    

# model.summary()



model = create_model(VOCAB_SIZE, num_labels)

    

#model.summary()

%%time

#TPU_WORKER = 'grpc://10.0.0.2:8470'

#tf.logging.set_verbosity(tf.logging.INFO)





mirrored_strategy = tf.distribute.MirroredStrategy()



with mirrored_strategy.scope():

    tpu_model = model



# tpu_model = tf.distribute.cluster_resolver.contrib.tpu.keras_to_tpu_model(

#     model,

#     strategy=tf.contrib.tpu.TPUDistributionStrategy(

#         tf.contrib.cluster_resolver.TPUClusterResolver(TPU_WORKER)))





tpu_model.summary()







# tf.config.experimental_connect_to_host('grpc://' + os.environ['COLAB_TPU_ADDR'])

# resolver = tf.distribute.cluster_resolver.TPUClusterResolver('grpc://' + os.environ['COLAB_TPU_ADDR'])

# tf.tpu.experimental.initialize_tpu_system(resolver)

# strategy = tf.distribute.experimental.TPUStrategy(resolver) 
%%time

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

#n_steps = train_text.shape[0] // BATCH_SIZE



earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')

mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')

reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, min_delta=1e-5, mode='min')





# # Train and evaluate the model

# history =model.fit(body_train, train_toxic, 

#           epochs=500,

#           batch_size=21751,

#           validation_split=0.1,

#           callbacks=[earlyStopping, mcp_save, reduce_lr_loss]

#          )

%%time

# Train and evaluate the model

history =tpu_model.fit(body_train, train_toxic, 

          epochs=1500,

          batch_size=21751,

          validation_split=0.1,

          callbacks=[earlyStopping, mcp_save, reduce_lr_loss]

         )

%%time

print('Eval loss/accuracy:{}'.format(

  model.evaluate(body_test, test_toxic, batch_size=128)))

plt.plot(history.history['loss'], label='train loss')

plt.plot(history.history['val_loss'], label='val loss')

plt.xlabel("epoch")

plt.ylabel("Cross-entropy loss")

plt.legend();
test_df=pd.read_csv("../input/jigsaw-multilingual-toxic-test-translated/jigsaw_miltilingual_test_translated.csv")





test_df.rename(columns={"translated":"comment_text"}, inplace = True)

test_df.head()
test_df_text=test_df['comment_text'].values

sample_submission = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv")

body_submit = processor.transform_text(test_df_text) 

pred = model.predict(body_submit)

sample_submission['toxic'] =pred 

sample_submission.to_csv("submission.csv", index=False)

sample_submission.head()