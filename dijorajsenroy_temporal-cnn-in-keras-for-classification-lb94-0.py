import tensorflow as tf
import numpy as np 
import pandas as pd
from sklearn import *
from sklearn.metrics import f1_score
from tensorflow.keras.models import Model
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import Input, Conv1D, Dense, Activation, Dropout,Multiply
from tensorflow.keras.layers import BatchNormalization,Bidirectional,GRU,Multiply, Add
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import datetime
import gc
GROUP_BATCH_SIZE = 4000
WINDOWS = [10, 50]


BASE_PATH = '/kaggle/input/liverpool-ion-switching'
DATA_PATH = '/kaggle/input/data-without-drift'
RFC_DATA_PATH = '/kaggle/input/ion-shifted-rfc-proba'
%%time

def create_rolling_features(df):
    for window in WINDOWS:
        df["rolling_mean_" + str(window)] = df['signal'].rolling(window=window).mean()
        df["rolling_std_" + str(window)] = df['signal'].rolling(window=window).std()
        df["rolling_var_" + str(window)] = df['signal'].rolling(window=window).var()
        df["rolling_min_" + str(window)] = df['signal'].rolling(window=window).min()
        df["rolling_max_" + str(window)] = df['signal'].rolling(window=window).max()
        df["rolling_min_max_ratio_" + str(window)] = df["rolling_min_" + str(window)] / df["rolling_max_" + str(window)]
        df["rolling_min_max_diff_" + str(window)] = df["rolling_max_" + str(window)] - df["rolling_min_" + str(window)]

    df = df.replace([np.inf, -np.inf], np.nan)    
    df.fillna(0, inplace=True)
    return df


def create_features(df, batch_size):
    
    df['group'] = df.groupby(df.index//batch_size, sort=False)['signal'].agg(['ngroup']).values
    df['group'] = df['group'].astype(np.uint16)
    for window in WINDOWS:    
        df['signal_shift_pos_' + str(window)] = df.groupby('group')['signal'].shift(window).fillna(0)
        df['signal_shift_neg_' + str(window)] = df.groupby('group')['signal'].shift(-1 * window).fillna(0)
        
    df['signal_2'] = df['signal'] ** 2
    return df   
## reading data
train = pd.read_csv(f'{DATA_PATH}/train_clean.csv', dtype={'time': np.float32, 'signal': np.float32, 'open_channels':np.int32})
test  = pd.read_csv(f'{DATA_PATH}/test_clean.csv', dtype={'time': np.float32, 'signal': np.float32})
sub  = pd.read_csv(f'{BASE_PATH}/sample_submission.csv', dtype={'time': np.float32})

# loading and adding shifted-rfc-proba features
y_train_proba = np.load(f"{RFC_DATA_PATH}/Y_train_proba.npy")
y_test_proba = np.load(f"{RFC_DATA_PATH}/Y_test_proba.npy")

for i in range(11):
    train[f"proba_{i}"] = y_train_proba[:, i]
    test[f"proba_{i}"] = y_test_proba[:, i]

    
train = create_rolling_features(train) ##trying to not use feature engg for TCN
test = create_rolling_features(test)   
    
## normalizing features
train_mean = train.signal.mean()
train_std = train.signal.std()
train['signal'] = (train.signal - train_mean) / train_std
test['signal'] = (test.signal - train_mean) / train_std


print('Shape of train is ',train.shape)
print('Shape of test is ',test.shape)
## create features

batch_size = GROUP_BATCH_SIZE

train = create_features(train, batch_size)
test = create_features(test, batch_size)

cols_to_remove = ['time','signal','open_channels','batch','batch_index','batch_slices','batch_slices2', 'group']
cols = [c for c in train.columns if c not in cols_to_remove]
X = train[cols]
y = train['open_channels']
X_pred = test[cols]
X_pred = X_pred.values

del train
del test
gc.collect()
import inspect
from typing import List
from tensorflow.keras import backend as K, Model, Input, optimizers
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation, SpatialDropout1D, Lambda, LSTM, Dropout
from tensorflow.keras.layers import Layer, Conv1D, Dense, BatchNormalization, LayerNormalization
def is_power_of_two(num):
    return num != 0 and ((num & (num - 1)) == 0)


def adjust_dilations(dilations):
    if all([is_power_of_two(i) for i in dilations]):
        return dilations
    else:
        new_dilations = [2 ** i for i in dilations]
        return new_dilations


class ResidualBlock(Layer):

    def __init__(self,
                 dilation_rate,
                 nb_filters,
                 kernel_size,
                 padding,
                 activation='relu',
                 dropout_rate=0,
                 kernel_initializer='he_normal',
                 use_batch_norm=False,
                 use_layer_norm=False,
                 **kwargs):

        # type: (int, int, int, str, str, float, str, bool, bool, bool, dict) -> None
        self.dilation_rate = dilation_rate
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.kernel_initializer = kernel_initializer
        self.layers = []
        self.layers_outputs = []
        self.shape_match_conv = None
        self.res_output_shape = None
        self.final_activation = None

        super(ResidualBlock, self).__init__(**kwargs)

    def _add_and_activate_layer(self, layer):
        """Helper function for building layer
        Args:
            layer: Appends layer to internal layer list and builds it based on the current output
                   shape of ResidualBlocK. Updates current output shape.
        """
        self.layers.append(layer)
        self.layers[-1].build(self.res_output_shape)
        self.res_output_shape = self.layers[-1].compute_output_shape(self.res_output_shape)

    def build(self, input_shape):

        with K.name_scope(self.name):  # name scope used to make sure weights get unique names
            self.layers = []
            self.res_output_shape = input_shape

            for k in range(2):
                name = 'conv1D_{}'.format(k)
                with K.name_scope(name):  # name scope used to make sure weights get unique names
                    self._add_and_activate_layer(Conv1D(filters=self.nb_filters,
                                                        kernel_size=self.kernel_size,
                                                        dilation_rate=self.dilation_rate,
                                                        padding=self.padding,
                                                        name=name,
                                                        kernel_initializer=self.kernel_initializer))

                with K.name_scope('norm_{}'.format(k)):
                    if self.use_batch_norm:
                        self._add_and_activate_layer(BatchNormalization())
                    elif self.use_layer_norm:
                        self._add_and_activate_layer(LayerNormalization())

                self._add_and_activate_layer(Activation('relu'))
                self._add_and_activate_layer(SpatialDropout1D(rate=self.dropout_rate))

            if self.nb_filters != input_shape[-1]:
                # 1x1 conv to match the shapes (channel dimension).
                name = 'matching_conv1D'
                with K.name_scope(name):
                    # make and build this layer separately because it directly uses input_shape
                    self.shape_match_conv = Conv1D(filters=self.nb_filters,
                                                   kernel_size=1,
                                                   padding='same',
                                                   name=name,
                                                   kernel_initializer=self.kernel_initializer)

            else:
                name = 'matching_identity'
                self.shape_match_conv = Lambda(lambda x: x, name=name)

            with K.name_scope(name):
                self.shape_match_conv.build(input_shape)
                self.res_output_shape = self.shape_match_conv.compute_output_shape(input_shape)

            self.final_activation = Activation(self.activation)
            self.final_activation.build(self.res_output_shape)  # probably isn't necessary

            # this is done to force Keras to add the layers in the list to self._layers
            for layer in self.layers:
                self.__setattr__(layer.name, layer)
            self.__setattr__(self.shape_match_conv.name, self.shape_match_conv)
            self.__setattr__(self.final_activation.name, self.final_activation)

            super(ResidualBlock, self).build(input_shape)  # done to make sure self.built is set True

    def call(self, inputs, training=None):
        """
        Returns: A tuple where the first element is the residual model tensor, and the second
                 is the skip connection tensor.
        """
        x = inputs
        self.layers_outputs = [x]
        for layer in self.layers:
            training_flag = 'training' in dict(inspect.signature(layer.call).parameters)
            x = layer(x, training=training) if training_flag else layer(x)
            self.layers_outputs.append(x)
        x2 = self.shape_match_conv(inputs)
        self.layers_outputs.append(x2)
        res_x = layers.add([x2, x])
        self.layers_outputs.append(res_x)

        res_act_x = self.final_activation(res_x)
        self.layers_outputs.append(res_act_x)
        return [res_act_x, x]

    def compute_output_shape(self, input_shape):
        return [self.res_output_shape, self.res_output_shape]
class TCN(Layer):
    def __init__(self,
                 nb_filters=32,
                 kernel_size=2,
                 nb_stacks=1,
                 dilations=(1, 2, 4, 8, 16),
                 padding='same',
                 use_skip_connections=False,
                 dropout_rate=0.0,
                 return_sequences=False,
                 activation='relu',
                 kernel_initializer='he_normal',
                 use_batch_norm=False,
                 use_layer_norm=False,
                 **kwargs):

        self.return_sequences = return_sequences
        self.dropout_rate = dropout_rate
        self.use_skip_connections = use_skip_connections
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.activation = activation
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.skip_connections = []
        self.residual_blocks = []
        self.layers_outputs = []
        self.build_output_shape = None
        self.lambda_layer = None
        self.lambda_ouput_shape = None

        if padding != 'causal' and padding != 'same':
            raise ValueError("Only 'causal' or 'same' padding are compatible for this layer.")

        if not isinstance(nb_filters, int):
            print('An interface change occurred after the version 2.1.2.')
            print('Before: tcn.TCN(x, return_sequences=False, ...)')
            print('Now should be: tcn.TCN(return_sequences=False, ...)(x)')
            print('The alternative is to downgrade to 2.1.2 (pip install keras-tcn==2.1.2).')
            raise Exception()

        # initialize parent class
        super(TCN, self).__init__(**kwargs)

    @property
    def receptive_field(self):
        assert_msg = 'The receptive field formula works only with power of two dilations.'
        assert all([is_power_of_two(i) for i in self.dilations]), assert_msg
        return self.kernel_size * self.nb_stacks * self.dilations[-1]

    def build(self, input_shape):

        # member to hold current output shape of the layer for building purposes
        self.build_output_shape = input_shape

        # list to hold all the member ResidualBlocks
        self.residual_blocks = []
        total_num_blocks = self.nb_stacks * len(self.dilations)
        if not self.use_skip_connections:
            total_num_blocks += 1  # cheap way to do a false case for below

        for s in range(self.nb_stacks):
            for d in self.dilations:
                self.residual_blocks.append(ResidualBlock(dilation_rate=d,
                                                          nb_filters=self.nb_filters,
                                                          kernel_size=self.kernel_size,
                                                          padding=self.padding,
                                                          activation=self.activation,
                                                          dropout_rate=self.dropout_rate,
                                                          use_batch_norm=self.use_batch_norm,
                                                          use_layer_norm=self.use_layer_norm,
                                                          kernel_initializer=self.kernel_initializer,
                                                          name='residual_block_{}'.format(len(self.residual_blocks))))
                # build newest residual block
                self.residual_blocks[-1].build(self.build_output_shape)
                self.build_output_shape = self.residual_blocks[-1].res_output_shape

        # this is done to force keras to add the layers in the list to self._layers
        for layer in self.residual_blocks:
            self.__setattr__(layer.name, layer)

        # Author: @karolbadowski.
        output_slice_index = int(self.build_output_shape.as_list()[1] / 2) if self.padding == 'same' else -1
        self.lambda_layer = Lambda(lambda tt: tt[:, output_slice_index, :])
        self.lambda_ouput_shape = self.lambda_layer.compute_output_shape(self.build_output_shape)

    def compute_output_shape(self, input_shape):
        """
        Overridden in case keras uses it somewhere... no idea. Just trying to avoid future errors.
        """
        if not self.built:
            self.build(input_shape)
        if not self.return_sequences:
            return self.lambda_ouput_shape
        else:
            return self.build_output_shape

    def call(self, inputs, training=None):
        x = inputs
        self.layers_outputs = [x]
        self.skip_connections = []
        for layer in self.residual_blocks:
            x, skip_out = layer(x, training=training)
            self.skip_connections.append(skip_out)
            self.layers_outputs.append(x)

        if self.use_skip_connections:
            x = layers.add(self.skip_connections)
            self.layers_outputs.append(x)

        if not self.return_sequences:
            x = self.lambda_layer(x)
            self.layers_outputs.append(x)
        return x
##running the model for each fold
skf = StratifiedKFold(n_splits=25, random_state=None, shuffle=True)
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
X_train = X_train.values
y_train = y_train.values
X_test = X_test.values
y_test = y_test.values

#One Hot Encoding Target Variables
onh = OneHotEncoder(sparse=False)
y_train = y_train.reshape(len(y_train),1)
y_train = onh.fit_transform(y_train)
y_test = y_test.reshape(len(y_test),1)
y_test = onh.fit_transform(y_test)
#Making 2D into 3D tensors
X_train= X_train.reshape((X_train.shape[0],X_train.shape[1],1))
X_test= X_test.reshape((X_test.shape[0],X_test.shape[1],1))
X_pred=X_pred.reshape((X_pred.shape[0],X_pred.shape[1],1))

print("X_train shape:",X_train.shape)
print("X_test shape:",X_test.shape)
print("y_train shape:",y_train.shape)
print("y_test shape:",y_test.shape)
print("X_pred shape:",X_pred.shape)
I = Input(shape=(X_train.shape[1],1))
model = TCN(nb_filters=32,kernel_size=6,dilations=[1, 2, 4, 8, 16, 32],return_sequences=True)(I)
model = GRU(256,return_sequences=True)(model)
model = Dropout(0.2)(model)
model = GRU(256)(model)
model = Dropout(0.2)(model)
#model = GRU(128)(model)
model = Dense(512,activation='relu',kernel_initializer='uniform')(model)
model = Dense(512,activation='relu',kernel_initializer='uniform')(model)
#model = Dense(512,activation='relu',kernel_initializer='uniform')(model)
model = Dense(11,activation='softmax')(model)

classifier = Model(inputs=[I], outputs=[model])
classifier.summary()
##loading weights from a pre-trained model
from tensorflow.keras.models import load_model
#classifier.save_weights('TCNweights.h5')
classifier.load_weights('/kaggle/input/weightstcn/TCNweights.h5')
##accuracy achieved was 96.80 on 16 epochs possibly not converged since val loss was decreasing slow--retrain##
##compiling model and fitting data
classifier.compile(optimizer = Adam(learning_rate=0.003, amsgrad=True),
                   loss = 'categorical_crossentropy',
                   metrics =['accuracy'])
classifier.fit(X_train,y_train,
          epochs=16,batch_size=4000,
          validation_data=[X_test,y_test])
## Prediction and reversing One Hot Encoding
y_pred=classifier.predict(X_pred)
y_pred =onh.inverse_transform(y_pred)
y_pred #Should be 10
# making submission
sub = pd.read_csv('../input/liverpool-ion-switching/sample_submission.csv')
sub.iloc[:,1] = y_pred[:,0]
sub.to_csv('submission.csv',index=False,float_format='%.4f')