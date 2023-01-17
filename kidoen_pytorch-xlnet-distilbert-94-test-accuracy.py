# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import pandas as pd
from random import shuffle
import nltk
from nltk.corpus import stopwords
import string
import re
df = pd.read_csv("/kaggle/input/60k-stack-overflow-questions-with-quality-rate/data.csv")
df.head()
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df['label_enc'] = labelencoder.fit_transform(df['Y'])
df = df.rename(columns={"Y":"label"})
df = df[['Title','Body','label','label_enc']]

df.rename(columns={'label':'label_desc'},inplace=True)
df.rename(columns={'label_enc':'label'},inplace=True)
# df.rename(columns={"text":"sentence"},inplace=True)

stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^(a-zA-Z)]',' ', text) # (a-zA-Z)\s]
    return text

df['Title'] = df['Title'].apply(clean_text)
df['Body'] = df['Body'].apply(clean_text)
df.head()
# shuffling the dataframe
df = df.sample(frac = 1)
df.head()
# lets split 5000 for test data ummm !!!
train = df.iloc[:55000,:]
test = df.iloc[55000:,:]
traintext = train[['Title','Body']].values.tolist()
testtext = test[['Title','Body']].values.tolist()
import tensorflow as tf
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
    print('Running on TPU ', tpu.master())
except ValueError:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)
import os
import numpy as np
import pandas as pd
from kaggle_datasets import KaggleDatasets
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import transformers
from transformers import TFAutoModel, AutoTokenizer
from tqdm.notebook import tqdm
import plotly.express as px
model = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model)
max_len = 300
batch_size = 32 * strategy.num_replicas_in_sync

trainencoded = tokenizer.batch_encode_plus(traintext,pad_to_max_length=True,max_length=max_len)
testencoded = tokenizer.batch_encode_plus(testtext,pad_to_max_length=True,max_length=max_len)

from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val = train_test_split(trainencoded['input_ids'],train['label'].values,test_size=0.2,random_state=42)
X_test = testencoded['input_ids']
auto = tf.data.experimental.AUTOTUNE
traindataset = tf.data.Dataset.from_tensor_slices((X_train,y_train)).repeat().shuffle(2048).batch(batch_size).prefetch(auto)
validdataset = (tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).cache().prefetch(auto))
testdataset = (tf.data.Dataset.from_tensor_slices(X_test).batch(batch_size))
with strategy.scope():
    transformer_encoder = TFAutoModel.from_pretrained(model)
    inputids = Input(shape=(max_len,),dtype=tf.int32, name="inputids")
    seqout = transformer_encoder(inputids)[0]
    # Only extract the token used for classification, which is <s>
    cls_token = seqout[:, 0, :]
    out = Dense(3,activation='softmax')(cls_token)
    model = Model(inputs=inputids, outputs=out)
    model.compile(Adam(lr=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(
    monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto',
    baseline=None, restore_best_weights=True
)

callbacks = [es]

nsteps = len(X_train)//batch_size
n_epochs = 40

history = model.fit(traindataset,steps_per_epoch=nsteps,validation_data=validdataset,epochs=n_epochs,callbacks=callbacks)
import plotly.express as px

hist = history.history
px.line(
    hist, x=range(1, len(hist['loss'])+1), y=['accuracy', 'val_accuracy'], 
    title='Model Accuracy', labels={'x': 'Epoch', 'value': 'Accuracy'}
)
px.line(
    hist, x=range(1, len(hist['loss'])+1), y=['loss', 'val_loss'], 
    title='Model Loss', labels={'x': 'Epoch', 'value': 'Loss'}
)
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
# PREDICTIONS ON UNSEEN TEST DATA 

predictions = model.predict(testdataset)
preds = np.argmax(predictions,axis=1)

ytest = test['label']
ytest = np.array(ytest)

print(classification_report(ytest,preds))
print("\n")
print(confusion_matrix(ytest,preds))
print("\n")
print(accuracy_score(ytest,preds))



model = 'xlnet-base-cased'
tokenizer = AutoTokenizer.from_pretrained(model)

max_len = 300
batch_size = 32 * strategy.num_replicas_in_sync

trainencoded = tokenizer.batch_encode_plus(traintext,pad_to_max_length=True,max_length=max_len)
testencoded = tokenizer.batch_encode_plus(testtext,pad_to_max_length=True,max_length=max_len)

from sklearn.model_selection import train_test_split

X_train,X_val,y_train,y_val = train_test_split(trainencoded['input_ids'],train['label'].values,test_size=0.2,random_state=42)
X_test = testencoded['input_ids']


auto = tf.data.experimental.AUTOTUNE
traindataset = tf.data.Dataset.from_tensor_slices((X_train,y_train)).repeat().shuffle(2048).batch(batch_size).prefetch(auto)
validdataset = (tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).cache().prefetch(auto))
testdataset = (tf.data.Dataset.from_tensor_slices(X_test).batch(batch_size))

with strategy.scope():
    transformer_encoder = TFAutoModel.from_pretrained(model)
    inputids = Input(shape=(max_len,),dtype=tf.int32, name="inputids")
    seqout = transformer_encoder(inputids)[0]
    # Only extract the token used for classification, which is <s>
    cls_token = seqout[:, 0, :]
    out = Dense(3,activation='softmax')(cls_token)
    model = Model(inputs=inputids, outputs=out)
    model.compile(Adam(lr=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    
    
from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(
    monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto',
    baseline=None, restore_best_weights=True
)

callbacks = [es]

nsteps = len(X_train)//batch_size
n_epochs = 40

history = model.fit(traindataset,steps_per_epoch=nsteps,validation_data=validdataset,epochs=n_epochs,callbacks=callbacks)

import plotly.express as px

hist = history.history
px.line(
    hist, x=range(1, len(hist['loss'])+1), y=['accuracy', 'val_accuracy'], 
    title='Model Accuracy', labels={'x': 'Epoch', 'value': 'Accuracy'}
)

px.line(
    hist, x=range(1, len(hist['loss'])+1), y=['loss', 'val_loss'], 
    title='Model Loss', labels={'x': 'Epoch', 'value': 'Loss'}
)
predictions = model.predict(testdataset)
preds = np.argmax(predictions,axis=1)

ytest = test['label']
ytest = np.array(ytest)

print(classification_report(ytest,preds))
print("\n")
print(confusion_matrix(ytest,preds))
print("\n")
print(accuracy_score(ytest,preds))

model = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model)

max_len = 300
batch_size = 32 * strategy.num_replicas_in_sync

trainencoded = tokenizer.batch_encode_plus(traintext,pad_to_max_length=True,max_length=max_len)
testencoded = tokenizer.batch_encode_plus(testtext,pad_to_max_length=True,max_length=max_len)

from sklearn.model_selection import train_test_split

X_train,X_val,y_train,y_val = train_test_split(trainencoded['input_ids'],train['label'].values,test_size=0.2,random_state=42)
X_test = testencoded['input_ids']


auto = tf.data.experimental.AUTOTUNE
traindataset = tf.data.Dataset.from_tensor_slices((X_train,y_train)).repeat().shuffle(2048).batch(batch_size).prefetch(auto)
validdataset = (tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).cache().prefetch(auto))
testdataset = (tf.data.Dataset.from_tensor_slices(X_test).batch(batch_size))

with strategy.scope():
    transformer_encoder = TFAutoModel.from_pretrained(model)
    inputids = Input(shape=(max_len,),dtype=tf.int32, name="inputids")
    seqout = transformer_encoder(inputids)[0]
    # Only extract the token used for classification, which is <s>
    cls_token = seqout[:, 0, :]
    out = Dense(3,activation='softmax')(cls_token)
    model = Model(inputs=inputids, outputs=out)
    model.compile(Adam(lr=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    
    
from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(
    monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto',
    baseline=None, restore_best_weights=True
)

callbacks = [es]

nsteps = len(X_train)//batch_size
n_epochs = 40

history = model.fit(traindataset,steps_per_epoch=nsteps,validation_data=validdataset,epochs=n_epochs,callbacks=callbacks)

import plotly.express as px

hist = history.history
px.line(
    hist, x=range(1, len(hist['loss'])+1), y=['accuracy', 'val_accuracy'], 
    title='Model Accuracy', labels={'x': 'Epoch', 'value': 'Accuracy'}
)
px.line(
    hist, x=range(1, len(hist['loss'])+1), y=['loss', 'val_loss'], 
    title='Model Loss', labels={'x': 'Epoch', 'value': 'Loss'}
)
predictions = model.predict(testdataset)
preds = np.argmax(predictions,axis=1)

ytest = test['label']
ytest = np.array(ytest)

print(classification_report(ytest,preds))
print("\n")
print(confusion_matrix(ytest,preds))
print("\n")
print(accuracy_score(ytest,preds))

model = 'albert-base-v1'
tokenizer = AutoTokenizer.from_pretrained(model)

max_len = 300
batch_size = 32 * strategy.num_replicas_in_sync

trainencoded = tokenizer.batch_encode_plus(traintext,pad_to_max_length=True,max_length=max_len)
testencoded = tokenizer.batch_encode_plus(testtext,pad_to_max_length=True,max_length=max_len)

from sklearn.model_selection import train_test_split

X_train,X_val,y_train,y_val = train_test_split(trainencoded['input_ids'],train['label'].values,test_size=0.2,random_state=42)
X_test = testencoded['input_ids']


auto = tf.data.experimental.AUTOTUNE
traindataset = tf.data.Dataset.from_tensor_slices((X_train,y_train)).repeat().shuffle(2048).batch(batch_size).prefetch(auto)
validdataset = (tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).cache().prefetch(auto))
testdataset = (tf.data.Dataset.from_tensor_slices(X_test).batch(batch_size))

with strategy.scope():
    transformer_encoder = TFAutoModel.from_pretrained(model)
    inputids = Input(shape=(max_len,),dtype=tf.int32, name="inputids")
    seqout = transformer_encoder(inputids)[0]
    # Only extract the token used for classification, which is <s>
    cls_token = seqout[:, 0, :]
    out = Dense(3,activation='softmax')(cls_token)
    model = Model(inputs=inputids, outputs=out)
    model.compile(Adam(lr=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    
    
from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(
    monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto',
    baseline=None, restore_best_weights=True
)

callbacks = [es]

nsteps = len(X_train)//batch_size
n_epochs = 40

history = model.fit(traindataset,steps_per_epoch=nsteps,validation_data=validdataset,epochs=n_epochs,callbacks=callbacks)

import plotly.express as px

hist = history.history
px.line(
    hist, x=range(1, len(hist['loss'])+1), y=['accuracy', 'val_accuracy'], 
    title='Model Accuracy', labels={'x': 'Epoch', 'value': 'Accuracy'}
)
px.line(
    hist, x=range(1, len(hist['loss'])+1), y=['loss', 'val_loss'], 
    title='Model Loss', labels={'x': 'Epoch', 'value': 'Loss'}
)
predictions = model.predict(testdataset)
preds = np.argmax(predictions,axis=1)

ytest = test['label']
ytest = np.array(ytest)

print(classification_report(ytest,preds))
print("\n")
print(confusion_matrix(ytest,preds))
print("\n")
print(accuracy_score(ytest,preds))