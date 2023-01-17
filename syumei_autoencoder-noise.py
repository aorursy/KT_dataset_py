#from google.colab import drive

#drive.mount('/content/gdrive')

import numpy as np

import pandas as pd

import random

from numba.decorators import jit



import matplotlib.pyplot as plt

import seaborn as sns

from pylab import rcParams



from keras.models import Model, load_model

from keras.layers import Input, Dense

from keras import regularizers

from sklearn.preprocessing import StandardScaler, StandardScaler

from keras.layers import Dense, BatchNormalization, Activation

import tensorflow as tf

from keras.models import Model, load_model

from keras.layers import Input, Dense, Layer, InputSpec

from keras.callbacks import ModelCheckpoint, TensorBoard

from keras import regularizers, activations, initializers, constraints, Sequential

from keras import backend as K

from keras.constraints import UnitNorm, Constraint

from keras.callbacks import ModelCheckpoint, EarlyStopping



from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,

                             roc_curve, recall_score, classification_report, f1_score,

                             precision_recall_fscore_support)

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.preprocessing import StandardScaler, MinMaxScaler



from sklearn.metrics import roc_auc_score

import lightgbm as lgb

 
#Autoencoder Optimization

class DenseTied(Layer):

    def __init__(self, units,

                 activation=None,

                 use_bias=True,

                 kernel_initializer='glorot_uniform',

                 bias_initializer='zeros',

                 kernel_regularizer=None,

                 bias_regularizer=None,

                 activity_regularizer=None,

                 kernel_constraint=None,

                 bias_constraint=None,

                 tied_to=None,

                 **kwargs):

        self.tied_to = tied_to

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:

            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super().__init__(**kwargs)

        self.units = units

        self.activation = activations.get(activation)

        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)

        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)

        self.bias_constraint = constraints.get(bias_constraint)

        self.input_spec = InputSpec(min_ndim=2)

        self.supports_masking = True

                

    def build(self, input_shape):

        assert len(input_shape) >= 2

        input_dim = input_shape[-1]



        if self.tied_to is not None:

            self.kernel = K.transpose(self.tied_to.kernel)

            self._non_trainable_weights.append(self.kernel)

        else:

            self.kernel = self.add_weight(shape=(input_dim, self.units),

                                          initializer=self.kernel_initializer,

                                          name='kernel',

                                          regularizer=self.kernel_regularizer,

                                          constraint=self.kernel_constraint)

        if self.use_bias:

            self.bias = self.add_weight(shape=(self.units,),

                                        initializer=self.bias_initializer,

                                        name='bias',

                                        regularizer=self.bias_regularizer,

                                        constraint=self.bias_constraint)

        else:

            self.bias = None

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})

        self.built = True



    def compute_output_shape(self, input_shape):

        assert input_shape and len(input_shape) >= 2

        output_shape = list(input_shape)

        output_shape[-1] = self.units

        return tuple(output_shape)



    def call(self, inputs):

        output = K.dot(inputs, self.kernel)

        if self.use_bias:

            output = K.bias_add(output, self.bias, data_format='channels_last')

        if self.activation is not None:

            output = self.activation(output)

        return output
class WeightsOrthogonalityConstraint (Constraint):

    def __init__(self, encoding_dim, weightage = 1.0, axis = 0):

        self.encoding_dim = encoding_dim

        self.weightage = weightage

        self.axis = axis

        

    def weights_orthogonality(self, w):

        if(self.axis==1):

            w = K.transpose(w)

        if(self.encoding_dim > 1):

            m = K.dot(K.transpose(w), w) - K.eye(self.encoding_dim)

            return self.weightage * K.sqrt(K.sum(K.square(m)))

        else:

            m = K.sum(w ** 2) - 1.

            return m



    def __call__(self, w):

        return self.weights_orthogonality(w)
class UncorrelatedFeaturesConstraint (Constraint):

    

    def __init__(self, encoding_dim, weightage = 1.0):

        self.encoding_dim = encoding_dim

        self.weightage = weightage

    

    def get_covariance(self, x):

        x_centered_list = []



        for i in range(self.encoding_dim):

            x_centered_list.append(x[:, i] - K.mean(x[:, i]))

        

        x_centered = tf.stack(x_centered_list)

        covariance = K.dot(x_centered, K.transpose(x_centered)) / tf.cast(x_centered.get_shape()[0], tf.float32)

        

        return covariance

            

    # Constraint penalty

    def uncorrelated_feature(self, x):

        if(self.encoding_dim <= 1):

            return 0.0

        else:

            output = K.sum(K.square(self.covariance - K.dot(self.covariance, K.eye(self.encoding_dim))))

            return output



    def __call__(self, x):

        self.covariance = self.get_covariance(x)

        return self.weightage * self.uncorrelated_feature(x)
#TensorFlowがGPUを認識しているか確認

from tensorflow.python.client import device_lib

device_lib.list_local_devices()
train = pd.read_csv("../input/eff09445/train_output_0.9445983706153531.csv")

test  = pd.read_csv("../input/eff09445/test_output_0.9445983706153531.csv")



useful = [t for t in train.columns.to_list() if ("count" not in t) and 

          ("w" not in t) and ("rows" not in t) 

          and ("umap_y" not in t)]

feature_names = list(set(useful) - set(['image_name', "patient_id", 'target']))

ycol = ["target"]

print(len(feature_names))



#mm = MinMaxScaler()

mm = StandardScaler()

mm.fit(train[feature_names])

train_mm = mm.transform(train[feature_names])

test_mm  = mm.transform(test[feature_names])



train[feature_names] = train_mm

test[feature_names] = test_mm
nb_epoch = 600

batch_size = 2048



input_dim = len(feature_names)

encoding_dim = 10

learning_rate = 1e-3
@jit

def noise(array):

  print('now noising') 

  height = len(array)

  width = len(array[0])

  print('start rand')  

  rands = np.random.uniform(0, 1, (height, width) )

  print('finish rand')  

  copy  = np.copy(array)

  for h in range(height):

    for w in range(width):

      if rands[h, w] <= 0.10:

        swap_target_h = random.randint(0,h)

        copy[h, w] = array[swap_target_h, w]

  print('finish noising') 

  return copy
def get_vanila_AE():



    # baseline_model

    encoder = Dense(encoding_dim, activation="relu", input_shape=(input_dim,), use_bias = True) 

    decoder = Dense(input_dim, activation="relu", use_bias = True)



    autoencoder_vanila = Sequential()

    autoencoder_vanila.add(encoder)

    autoencoder_vanila.add(decoder)

    

    autoencoder_vanila.compile(metrics=['mae'],

                    loss='mean_squared_error',

                    optimizer='adam')

    #autoencoder_vanila.summary()

    return autoencoder_vanila
def get_full_AE():

    encoder = Dense(encoding_dim, activation="relu", input_shape=(input_dim,), use_bias = True, kernel_constraint=UnitNorm(axis=0)) 

    decoder = Dense(input_dim, activation="relu", use_bias = True, kernel_constraint=UnitNorm(axis=1))

    autoencoder_full = Sequential()

    autoencoder_full.add(encoder)

    autoencoder_full.add(decoder)

    autoencoder_full.compile(metrics=['accuracy'],

                        loss='mean_squared_error',

                          optimizer='adam')

    return autoencoder_full
def train_AE(autoencoder, train_data, valid_data, 

             batch_size, epoch, callbacks, DO_TRAINING):

  if DO_TRANING:

     history = autoencoder.fit(train_data, train_data,

                               epochs=nb_epoch,

                               batch_size=batch_size,

                               shuffle=True,

                               validation_data=(valid_data, valid_data),

                               verbose=1,

                               callbacks=callbacks).history



     # Model loss

     plt.plot(history['loss'])

     plt.plot(history['val_loss'])

     plt.title('model loss')

     plt.ylabel('acc')

     plt.xlabel('epoch')

     plt.legend(['train', 'val'], loc='upper right');

     plt.show()

     

  return(autoencoder)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

oof_dfs = []

predicted_tests = []

for fold_id, (trn_idx, val_idx) in enumerate(skf.split(train, train[ycol])):

        # set callbacks

        DO_TRANING = True

        checkpointer = ModelCheckpoint(filepath="model_{}.h5".format(fold_id),

                                       verbose=0,

                                       save_best_only=True)

        earlystoping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')

        callbacks=[checkpointer, earlystoping]

        print("Fold ", fold_id)

        train_data = train.iloc[trn_idx]

        train_data = train_data[train_data["target"]==0.0][feature_names]

        train_data = pd.DataFrame(train_data.values, columns=feature_names)

        # add noise

        #train_data = pd.DataFrame(noise(train_data[feature_names].values), columns=feature_names)

        

        valid_data = train[feature_names].iloc[val_idx]



        autoencoder = get_vanila_AE()

        autoencoder_trained = train_AE(autoencoder, 

                                      train_data, 

                                      valid_data, 

                                      batch_size, 

                                      nb_epoch, 

                                      callbacks,

                                      DO_TRANING)

        

        #predicted_valid = autoencoder_trained.predict(valid_data)

        #predicted_test  = autoencoder_trained.predict(test[feature_names].values)

        

        #oof_dfs.append(pd.DataFrame(np.concatenate([predicted_valid, val_idx.reshape(len(val_idx), 1)], 1), 

        #                            columns=feature_names+["original_idx"]) 

        #                )

        #predicted_tests.append(predicted_test)
#oof_df = pd.concat(oof_dfs)

#oof_df["original_idx"] = oof_df["original_idx"].astype(int)

#oof_df = oof_df.sort_values("original_idx").reset_index()

#oof_df = pd.concat([train[["image_name", "patient_id", "target"]], oof_df[feature_names]], axis=1)

#oof_df.to_csv("oof_AE_with_noise.csv", index=False)
#sum(oof_df.groupby("target").mean().diff().iloc[1])
#test_output = pd.DataFrame(sum(predicted_tests) / len(predicted_tests), columns=feature_names)#

#test_output.to_csv("test_AE_with_noise.csv")