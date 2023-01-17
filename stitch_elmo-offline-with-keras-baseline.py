# to be run the first time

#from google.colab import drive

#drive.mount('/content/drive')

#!pip install --upgrade tensorflow



#before restarting comment above

import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns



import os

#os.chdir('drive/My Drive/')

#os.chdir('kaggle/google_quest')



import tensorflow as tf

import tensorflow_hub as hub

import datetime

import keras.backend as K



#from tensorflow.keras.layers import Layer

#from keras.models import 

#from keras.layers import Dense, Embedding, LSTM, Bidirectional



import h5py



print("Version: ", tf.__version__)

print("Eager mode: ", tf.executing_eagerly())

print("Hub version: ", hub.__version__)

print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")
train = pd.read_csv("../input/google-quest-challenge/train.csv")

test = pd.read_csv("../input/google-quest-challenge/test.csv")

sub = pd.read_csv("../input/google-quest-challenge/sample_submission.csv")
_maxlen = 100



def gen_features(df,_maxlen):

    """

  dataset contains question_body & answer columns

  pads each of these columns to the maxlen provided

  concatenates qn and ans into one sectence. qn marked with "__qn__", answer marked with "__ans__"

  """

    qn = df['question_body'].tolist()

    qn = [' '.join(t.split()[0:np.int(_maxlen/2)-1]) for t in qn]

    #qn = tf.cast(qn,tf.string)#cast helps convert string into tensor string.

    

    ans = df['answer'].tolist()

    ans = [' '.join(t.split()[0:np.int(_maxlen/2)-1]) for t in ans]

    #qn = tf.cast(ans,tf.string)#cast helps convert string into tensor string.



  

  #[ " __qn__ "+ q + " __answer__ " + a for q,a in (zip(qn[:3],ans[:3]))] [" __qn__ After playing around with macro photography 

  #X_list = tf.cast(X_list_,tf.string) #cast helps convert string into tensor string.

    return [ " __qn__ "+ q + " __answer__ " + a for q,a in (zip(qn,ans))]



!ls ../input/elmo-v3-fromtfhub/
elmo_model = hub.load("../input/elmo-v3-fromtfhub")



#elmo_model(["the cat is on the mat", "dogs are in the fog"], signature= "default")

# the above will throw an auto-trackable error. this is problem due to loading tf1 hub in tf 2. 

# notes that help to solve the problem: https://www.tensorflow.org/hub/common_issues



elmo_model.signatures['default'](tf.cast(["the cat is on the mat", "dogs are in the fog"],tf.string))['elmo'].numpy().shape
def gen_elmo_embedding(x,store=True):

  # elmo is still available at tf 2.0, need to use hub.kerasLayer which works in tf 1.15

  # ref: https://colab.research.google.com/gist/gowthamkpr/f01a548c4faa4088e476c727f693091b/untitled235.ipynb

    elmo_model = hub.load("../input/elmo-v3-fromtfhub")

    #elmo_model = hub.KerasLayer("https://tfhub.dev/google/elmo/3", trainable=True, signature="default",output_key='elmo')

  #print(elmo_model(tf.cast(qn[:2],tf.string)))

    chunk_size=100

    dim = 1024 #elmo vector dimension

    

    elmo_embedding= np.zeros((1, _maxlen,dim))

    for i in np.arange(0,len(x),chunk_size):

          #if i < 3:

        e = elmo_model.signatures['default'](tf.cast(x[i:i+chunk_size],tf.string))['elmo'].numpy()

        elmo_embedding = np.vstack((elmo_embedding,e))



    if store:

        print("embed layer stored")

        with h5py.File('../input/elmo_100_maxlen.h5', 'w') as hf:

            hf.create_dataset("elmo_100_maxlen_npy",  data=elmo_embedding)

    return elmo_embedding[1:]

  
with h5py.File('../input/elmo_100_maxlen.h5', 'r') as hf:

    train_elmo = hf['elmo_100_maxlen_npy'][:]

y_list = np.asarray(train.iloc[:,11:]) #converting columns to numpy array
from tensorflow.keras.layers import Dense, Dropout,Embedding, LSTM, Bidirectional, Input, Dropout

from tensorflow.keras import Sequential

from tensorflow.keras.models import Model

from tensorflow.keras.preprocessing import sequence



def build_model():

    input_ = Input(shape=(_maxlen,1024),name = 'qn')

    X= Bidirectional(LSTM(50, activation='relu'), name='bLSTM')(input_) #Bilstm seems to increasing the loss exponentially. lstm requires 3 dim vectors and it converts into 2 dim vectors

    #X= LSTM(100, activation='relu',dropout=0.2, name='LSTM')(input_) #lstm requires 3 dim vectors and it converts into 2 dim vectors

    X= Dense(100, activation='relu')(X) # the above matrix with 300 units will be sparse. Convert to dense matrix of 100 hidden units

    output_= Dense(30, activation='sigmoid', name='output')(X)



    model = Model(input_,output_)

    model.summary()



    return model
model = build_model()

model.compile(optimizer = "adam",loss = "binary_crossentropy")

history = model.fit(train_elmo[1:],y_list,epochs=10,batch_size = 100,validation_split=0.2)
X_test = gen_features(df=test,_maxlen=_maxlen)

test_elmo =  gen_elmo_embedding(X_test,store=False)

test_preds = model.predict(test_elmo)
sub.iloc[:, 1:] = test_preds
sub.to_csv('submission.csv', index=False)