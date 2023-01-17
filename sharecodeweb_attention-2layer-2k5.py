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
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (6,6)



from keras import backend as K

from keras.engine.topology import Layer

from keras import initializers, regularizers, constraints



from keras.layers import Dense, Input, LSTM, Bidirectional, Activation, Conv1D, GRU, TimeDistributed

from keras.layers import Dropout, Embedding, GlobalMaxPooling1D, MaxPooling1D, Add, Flatten, SpatialDropout1D

from keras.layers import GlobalAveragePooling1D, BatchNormalization, concatenate

from keras.layers import Reshape, merge, Concatenate, Lambda, Average

from keras.models import Sequential, Model, load_model

from keras.callbacks import ModelCheckpoint

from keras.initializers import Constant

from keras.layers.merge import add



from keras.preprocessing import sequence

from keras.preprocessing.text import Tokenizer, text_to_word_sequence

from keras.utils import np_utils



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from keras import initializers, regularizers

from keras import optimizers

from keras.engine.topology import Layer

from keras import constraints



############################################## 

"""

# ATTENTION LAYER

Cite these works 

1. Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]

"Hierarchical Attention Networks for Document Classification"

accepted in NAACL 2016

2. Winata, et al. https://arxiv.org/abs/1805.12307

"Attention-Based LSTM for Psychological Stress Detection from Spoken Language Using Distant Supervision." 

accepted in ICASSP 2018

Using a context vector to assist the attention

* How to use:

Put return_sequences=True on the top of an RNN Layer (GRU/LSTM/SimpleRNN).

The dimensions are inferred based on the output shape of the RNN.

Example:

	model.add(LSTM(64, return_sequences=True))

	model.add(AttentionWithContext())

	model.add(Addition())

	# next add a Dense layer (for classification/regression) or whatever...

"""

##############################################



def dot_product(x, kernel):

	"""

	Wrapper for dot product operation, in order to be compatible with both

	Theano and Tensorflow

	Args:

		x (): input

		kernel (): weights

	Returns:

	"""

	if K.backend() == 'tensorflow':

		return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)

	else:

		return K.dot(x, kernel)



class AttentionWithContext(Layer):

	"""

	Attention operation, with a context/query vector, for temporal data.

	Supports Masking.

	follows these equations:

	

	(1) u_t = tanh(W h_t + b)

	(2) \alpha_t = \frac{exp(u^T u)}{\sum_t(exp(u_t^T u))}, this is the attention weight

	(3) v_t = \alpha_t * h_t, v in time t

	# Input shape

		3D tensor with shape: `(samples, steps, features)`.

	# Output shape

		3D tensor with shape: `(samples, steps, features)`.

	"""



	def __init__(self,

				 W_regularizer=None, u_regularizer=None, b_regularizer=None,

				 W_constraint=None, u_constraint=None, b_constraint=None,

				 bias=True, **kwargs):



		self.supports_masking = True

		self.init = initializers.get('glorot_uniform')



		self.W_regularizer = regularizers.get(W_regularizer)

		self.u_regularizer = regularizers.get(u_regularizer)

		self.b_regularizer = regularizers.get(b_regularizer)



		self.W_constraint = constraints.get(W_constraint)

		self.u_constraint = constraints.get(u_constraint)

		self.b_constraint = constraints.get(b_constraint)



		self.bias = bias

		super(AttentionWithContext, self).__init__(**kwargs)



	def build(self, input_shape):

		assert len(input_shape) == 3



		self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],),

								 initializer=self.init,

								 name='{}_W'.format(self.name),

								 regularizer=self.W_regularizer,

								 constraint=self.W_constraint)

		if self.bias:

			self.b = self.add_weight(shape=(input_shape[-1],),

									 initializer='zero',

									 name='{}_b'.format(self.name),

									 regularizer=self.b_regularizer,

									 constraint=self.b_constraint)



		self.u = self.add_weight(shape=(input_shape[-1],),

								 initializer=self.init,

								 name='{}_u'.format(self.name),

								 regularizer=self.u_regularizer,

								 constraint=self.u_constraint)



		super(AttentionWithContext, self).build(input_shape)



	def compute_mask(self, input, input_mask=None):

		# do not pass the mask to the next layers

		return None



	def call(self, x, mask=None):

		uit = dot_product(x, self.W)



		if self.bias:

			uit += self.b



		uit = K.tanh(uit)

		ait = dot_product(uit, self.u)



		a = K.exp(ait)



		# apply mask after the exp. will be re-normalized next

		if mask is not None:

			# Cast the mask to floatX to avoid float64 upcasting in theano

			a *= K.cast(mask, K.floatx())



		# in some cases especially in the early stages of training the sum may be almost zero and this results in NaN's. 

		# Should add a small epsilon as the workaround

		# a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())

		a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())



		a = K.expand_dims(a)

		weighted_input = x * a

		

		return weighted_input



	def compute_output_shape(self, input_shape):

		return input_shape[0], input_shape[1], input_shape[2]

	

class Addition(Layer):

	"""

	This layer is supposed to add of all activation weight.

	We split this from AttentionWithContext to help us getting the activation weights

	follows this equation:

	(1) v = \sum_t(\alpha_t * h_t)

	

	# Input shape

		3D tensor with shape: `(samples, steps, features)`.

	# Output shape

		2D tensor with shape: `(samples, features)`.

	"""



	def __init__(self, **kwargs):

		super(Addition, self).__init__(**kwargs)



	def build(self, input_shape):

		self.output_dim = input_shape[-1]

		super(Addition, self).build(input_shape)



	def call(self, x):

		return K.sum(x, axis=1)



	def compute_output_shape(self, input_shape):

		return (input_shape[0], self.output_dim)
def recall_m(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (possible_positives + K.epsilon())

    return recall



def precision_m(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())

    return precision



def f1_m(y_true, y_pred):

    precision = precision_m(y_true, y_pred)

    recall = recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))
print('Loading data...')

#thay đổi dữ liệu đầu vào tùy thuộc vào tập dữ liệu cần chạy

path ='/kaggle/input/url-malicious-lstm/2k5.csv'

# load data

df =  pd.read_csv(path)



print(f'Data size: {df.shape}')



print(df['label'].value_counts())
#trộn dữ liệu để phục vụ tạo bộ dữ liệu train, test và validation

df = df.sample(frac=1).reset_index(drop=True)

df.head()
samples = df.url

labels = df.label

max_chars = 20000

maxlen = 128
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



tokenizer = Tokenizer(num_words=max_chars, char_level=True)

tokenizer.fit_on_texts(samples)

sequences = tokenizer.texts_to_sequences(samples)

word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=maxlen)

labels = [1 if i=='bad' else 0 for i in labels]

labels = np.asarray(labels)

print('Shape of data tensor:', data.shape)

print('Shape of label tensor:', labels.shape)
#phân chia tập dữ liệu thành train và validation

training_samples = int(len(samples) * 0.95)

validation_samples = int(len(labels) * 0.05)

print(training_samples, validation_samples)
indices = np.arange(data.shape[0])

np.random.shuffle(indices)

data = data[indices]

labels = labels[indices]
#tách bộ dữ liệu train thành train và test

x = data[:training_samples]

y = labels[:training_samples]

x_test = data[training_samples: training_samples + validation_samples]

y_test = labels[training_samples: training_samples + validation_samples]
num_chars = len(tokenizer.word_index)+1

embedding_vector_length = 128
import time

from tensorflow.keras.optimizers import Adam, SGD

def make_model(n_batch,num_chars, embedding_vector_length, maxlen):

    model = Sequential()

    model.add(Embedding(num_chars, embedding_vector_length, input_length=maxlen))

    model.add(SpatialDropout1D(0.2))

    model.add(Bidirectional(LSTM(64,  

                                 dropout=0.25, 

                                 recurrent_dropout=0.25,return_sequences=True)))

    model.add(BatchNormalization())

    model.add(Bidirectional(LSTM(64,  

                                 dropout=0.25, 

                                 recurrent_dropout=0.25, 

                                 return_sequences=True)))

    model.add(AttentionWithContext())

    model.add(Addition())

    model.add(Dense(1, activation='sigmoid'))



    model.summary()

    start = time.time()

    model.compile(optimizer='adam',

                loss='binary_crossentropy',

                metrics=['accuracy', f1_m, recall_m, precision_m])

    print("Compilation Time : ", time.time() - start)

    return model
def make_model_no_attention(n_batch,num_chars, embedding_vector_length, maxlen):

    model = Sequential()

    model.add(Embedding(num_chars, embedding_vector_length, input_length=maxlen))

    model.add(SpatialDropout1D(0.2))

    model.add(Bidirectional(LSTM(128, dropout=0.2,recurrent_dropout=0.2)))

    model.add(Dense(1, activation='sigmoid'))



    model.summary()

    start = time.time()

    model.compile(optimizer='adam',

                loss='binary_crossentropy',

                metrics=['accuracy', f1_m, recall_m, precision_m])

    print("Compilation Time : ", time.time() - start)

    return model
from keras.callbacks import EarlyStopping, ModelCheckpoint

def evaluate_model_with_n_layers(n_batch, x, y, x_test, y_test, num_chars, embedding_vector_length, maxlen,callbacks_list):

    model = make_model(n_batch,num_chars, embedding_vector_length, maxlen)

    hist=model.fit(x, y,

                epochs=100,

                batch_size=n_batch,

                callbacks=callbacks_list,

                validation_split=0.20,

                shuffle=True,

                verbose=1

                )

    test_acc = model.evaluate(x_test,y_test, verbose=0)

    return hist,test_acc





all_history = list ()

n_layers = 2

num_batch_size = [32]

for n_batch in num_batch_size:

    file_path = "BestModel_data_2k5_"+str(n_batch)+".hdf5"



    callbacks_list = [

        ModelCheckpoint(

            file_path,

            monitor='val_loss',

            verbose=1,

            save_best_only=True,

            mode='min'

        ),

        EarlyStopping(

            monitor='val_loss', 

            min_delta=0,

            patience=5, 

            verbose=1,

            mode='auto',

        )

    ]

    print('Starting test....')

    history, result = evaluate_model_with_n_layers(n_batch, x, y, x_test, y_test, num_chars, embedding_vector_length, maxlen,callbacks_list)

    print('Batch_Size =%d:' % n_batch)

    print(result)

    all_history.append(history)

    plt.plot(history.history['val_loss'], label='val_loss')

    plt.plot(history.history['val_accuracy'],label='val_accuracy')

plt.legend()

plt.show()