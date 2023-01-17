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
import numpy as np
import tensorflow as tf
import keras
import tqdm as tqdm
from tensorflow.keras.layers import Dense,concatenate,Activation,Dropout,Input,LSTM,Embedding,Flatten,Conv1D,BatchNormalization
from tensorflow.keras.models import Model
input_layer = Input(shape = (2,))
# here none in the output represent the sizes of batches and 2 means the input will be 2 dimensional vector
input_layer.shape
input_layer = Input(shape = (None,))
# here you are not defining the dimensions of input vector it can take any values
input_layer.shape
input_layer = Input(shape = (2,),batch_size = 50)
# now here you are fixing the size of batch and dimensions of input vector
input_layer.shape
output_dimensions = 10
vocab_size =200
input_layer = Input(shape=(2,))
embedding_layer = Embedding(vocab_size,output_dimensions)(input_layer)
# here we will get a 3d tensor where the 3rd dimension will be equal to output_dimensions in embedding layer
# vocab_size don't affect the output tensor shape
embedding_layer.shape
input_layer = Input(shape=(2,))
embedding_layer = Embedding(vocab_size,output_dimensions)(input_layer)
lstm_layer = LSTM(units=256)(embedding_layer)
lstm_layer.shape
input_layer = Input(shape=(2,))
embedding_layer = Embedding(vocab_size,output_dimensions)(input_layer)
lstm_layer = LSTM(units=256,return_sequences = True)(embedding_layer)
lstm_layer.shape
input_layer = Input(shape=(2,))
embedding_layer = Embedding(vocab_size,output_dimensions)(input_layer)
lstm_layer,state_h,state_c = LSTM(units=256,return_state = True)(embedding_layer)
lstm_layer.shape
state_h.shape
state_c.shape
input_layer = Input(shape=(2,))
embedding_layer = Embedding(vocab_size,output_dimensions)(input_layer)
lstm_layer = LSTM(units=256)(embedding_layer)
dense_layer = Dense(10,activation= 'softmax')(lstm_layer)
dense_layer.shape
input_layer = Input(shape=(2,))
embedding_layer = Embedding(vocab_size,output_dimensions)(input_layer)
lstm_layer = LSTM(units=256,return_sequences = True)(embedding_layer)
dense_layer = Dense(10,activation= 'softmax')(lstm_layer)
print(lstm_layer.shape)
print(dense_layer.shape)