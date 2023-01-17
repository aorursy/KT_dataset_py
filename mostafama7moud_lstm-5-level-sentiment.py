from shutil import copyfile

# copy our file into the working directory (make sure it has .py suffix)

copyfile(src = "/kaggle/input/mylibs/datautils.py", dst = "../working/datautils.py")
from keras.preprocessing import sequence

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation

from keras.layers import Embedding

from keras.layers import LSTM

from keras.layers import Conv1D, MaxPooling1D

from keras.utils import to_categorical

from keras import optimizers

from sklearn.model_selection import train_test_split

import tensorflow as tf

import numpy as np

from sklearn.metrics import confusion_matrix

import datautils as du
X = np.load("../input/processed-data/processed-400k-x.npy", allow_pickle=True)

Y = np.load("../input/processed-data/processed-400k-y.npy")

#X_val = np.load("../input/valida/x-validation.npy")

#Y_val = np.load("../input/valida/y-validation.npy")



fname = '/kaggle/input/preprocess-dataset/LSTM-5-level-sentiment-200k-v2-weights.h5'



# Embedding

max_features = 116846

maxlen = 150

embedding_size = 128



# Convolution

kernel_size = 5

filters = 64

pool_size = 4



# LSTM

lstm_output_size = 128



# Training

batch_size = 17

epochs = 5



x = sequence.pad_sequences(X, maxlen=maxlen)

y = to_categorical(Y,num_classes=5)

print("Shape of x after sequence : ", x.shape)

print("Shape of y after to_categorical : ", y.shape)



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)



# detect and init the TPU

tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

tf.config.experimental_connect_to_cluster(tpu)

tf.tpu.experimental.initialize_tpu_system(tpu)



# instantiate a distribution strategy

tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

batch_size = batch_size * tpu_strategy.num_replicas_in_sync

optimizer = optimizers.Nadam(learning_rate=0.0029, beta_1=0.9, beta_2=0.999)

# instantiating the model in the strategy scope creates the model on the TPU

with tpu_strategy.scope():

    model = Sequential()

    model.add(Embedding(max_features, embedding_size, input_length=maxlen))

    model.add(LSTM(lstm_output_size))

    model.add(Dense(5))

    model.add(Activation('softmax'))



    model.compile(loss='categorical_crossentropy',

                  optimizer=optimizer,

                  metrics=['accuracy'])



# load weights from first model; will only affect the first layer, dense_1.

#model.load_weights(fname)

#print('Model loaded')

print('Training...')

history = model.fit(x_train, y_train,

          batch_size=batch_size,

          epochs=epochs,

          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)

print('Test score:', score)

print('Test accuracy:', acc)
model_name = "LSTM-5-level-sentiment-400k-v1"

print("Saving model as [%s] ....." % model_name)

model.save(model_name + '.h5')

model.save_weights(model_name + '-weights.h5')

string = model.to_json()

with open(model_name + '-architecture.json', 'w') as outfile:  

    outfile.write(string)

print("Model Saved.")
# Predict the values from the test dataset

Y_pred = model.predict(x_test)

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

Y_true = np.argmax(y_test,axis = 1) 

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

du.plot_confusion_matrix(confusion_mtx, classes = range(5)) 