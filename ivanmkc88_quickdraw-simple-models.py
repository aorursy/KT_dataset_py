# !pip install -U tensorflow==2.1.0rc0
# !pip uninstall -y keras
import tensorflow as tf

print(tf.__version__)
def get_Xy(in_path):
    X = HDF5Matrix(in_path, 'strokes')[:]
    y = to_categorical(word_encoder.transform(HDF5Matrix(in_path, 'word')[:]))
    return X, y
import h5py
import numpy as np
import os
from tensorflow.keras.utils import HDF5Matrix
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

base_dir = os.path.join('../input/quickdraw-overview')
train_path = os.path.join(base_dir, 'quickdraw_train.h5')
valid_path = os.path.join(base_dir, 'quickdraw_valid.h5')
# test_path = os.path.join(base_dir, 'quickdraw_test.h5')

words = HDF5Matrix(train_path, 'word')[:]
word_encoder = LabelEncoder()
word_encoder.fit(words)
X_train, y_train = get_Xy(train_path)
X_val, y_val = get_Xy(valid_path)
# X_test, y_test = get_Xy(test_path)
print(X_train.shape)
print(y_train.shape)
import matplotlib.pyplot as plt

fig, m_axs = plt.subplots(3,3, figsize = (16, 16))
rand_idxs = range(9)# np.random.choice(range(X_train.shape[0]), size = 9)

for c_id, c_ax in zip(rand_idxs, m_axs.flatten()):
    test_arr = X_train[c_id]
    test_arr = test_arr[test_arr[:,2]>0, :] # only keep valid points    
    lab_idx = np.cumsum(test_arr[:,2]-1)
                        
    for i in np.unique(lab_idx):
        c_ax.plot(test_arr[lab_idx==i,0], 
                test_arr[lab_idx==i,1], '.-')
    c_ax.axis('off')
    c_ax.set_title(word_encoder.classes_[np.argmax(y_train[c_id])].decode())

from tensorflow.keras.metrics import top_k_categorical_accuracy
def top_5_accuracy(x,y): return top_k_categorical_accuracy(x,y, 5)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv1D, Dense, Dropout, GlobalAveragePooling1D
mlp_stroke_model = Sequential()
mlp_stroke_model.add(BatchNormalization(input_shape = (None,)+X_train.shape[2:]))
# filter count and length are taken from the script https://github.com/tensorflow/models/blob/master/tutorials/rnn/quickdraw/train_model.py
mlp_stroke_model.add(Conv1D(48, (5,)))
mlp_stroke_model.add(Dropout(0.3))
mlp_stroke_model.add(Conv1D(64, (5,)))
mlp_stroke_model.add(Dropout(0.3))
mlp_stroke_model.add(Conv1D(128, (3,)))
mlp_stroke_model.add(Dropout(0.3))
mlp_stroke_model.add(GlobalAveragePooling1D())
mlp_stroke_model.add(Dropout(0.3))
mlp_stroke_model.add(Dense(256))
mlp_stroke_model.add(Dense(len(word_encoder.classes_), activation = 'softmax'))
mlp_stroke_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy', top_5_accuracy])
mlp_stroke_model.summary()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, Dense, Dropout, GlobalMaxPool1D, Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
import datetime
import os
from os import path

weight_path="{}_weights.best.hdf5".format('stroke_mlp_model')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)


reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)
early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=5) # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat]
mlp_stroke_model.fit(X_train, y_train,
                      validation_data = (X_val, y_val), 
                      batch_size = 1024,
                      epochs = 30,
                      shuffle = 'batch',
                      callbacks = callbacks_list)
mlp_stroke_model.load_weights(weight_path)
mlp_stroke_results = mlp_stroke_model.evaluate(test_X, test_y, batch_size = 2048)
print('Accuracy: %2.1f%%, Top 2 Accuracy %2.1f%%' % (100*mlp_stroke_results[1], 100*mlp_stroke_results[2]))
from keras.models import Sequential
from keras.layers import BatchNormalization, Conv1D, LSTM, Dense, Dropout
# if len(get_available_gpus())>0:
    # https://twitter.com/fchollet/status/918170264608817152?lang=en
#     from keras.layers import CuDNNLSTM as LSTM # this one is about 3x faster on GPU instances
stroke_read_model = Sequential()
stroke_read_model.add(BatchNormalization(input_shape = (None,)+train_X.shape[2:]))
# filter count and length are taken from the script https://github.com/tensorflow/models/blob/master/tutorials/rnn/quickdraw/train_model.py
stroke_read_model.add(Conv1D(48, (5,)))
stroke_read_model.add(Dropout(0.3))
stroke_read_model.add(Conv1D(64, (5,)))
stroke_read_model.add(Dropout(0.3))
stroke_read_model.add(Conv1D(96, (3,)))
stroke_read_model.add(Dropout(0.3))
stroke_read_model.add(LSTM(128, return_sequences = True))
stroke_read_model.add(Dropout(0.3))
stroke_read_model.add(LSTM(128, return_sequences = False))
stroke_read_model.add(Dropout(0.3))
stroke_read_model.add(Dense(256))
stroke_read_model.add(Dense(len(word_encoder.classes_), activation = 'softmax'))
stroke_read_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy', top_5_accuracy])
stroke_read_model.summary()

tf.keras.backend.set_learning_phase(1)
weight_path="{}_weights.best.hdf5".format('stroke_lstm_model')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)


reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)
early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=5) # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat]
stroke_read_model.fit(train_X, train_y,
                      validation_data = (valid_X, valid_y), 
                      batch_size = 1024,
                      epochs = 30,
                      shuffle = 'batch',
                      callbacks = callbacks_list)
stroke_read_model.load_weights(weight_path)
lstm_results = stroke_read_model.evaluate(test_X, test_y, batch_size = 2048)
print('Accuracy: %2.1f%%, Top 2 Accuracy %2.1f%%' % (100*lstm_results[1], 100*lstm_results[2]))