# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from tqdm import tqdm

import seaborn as sns
x_train = pd.read_csv("../input/X_train.csv")

print(x_train.shape)

x_test = pd.read_csv("../input/X_test.csv")

print(x_test.shape)

y_train = pd.read_csv("../input/y_train.csv")

print(y_train.shape)



x_train.head()
train_seq = []

test_seq = []



for i in tqdm(sorted(x_train['series_id'].unique())) :

    train_seq.append(x_train[x_train['series_id']==i].drop(['row_id', 'series_id', 'measurement_number'], axis = 1).values)

    

for i in tqdm(sorted(x_test['series_id'].unique())) :

    test_seq.append(x_test[x_test['series_id']==i].drop(['row_id', 'series_id', 'measurement_number'], axis = 1).values)

    

train_seq = np.array(train_seq)

test_seq = np.array(test_seq)
from keras.layers import *

from keras.models import *

from keras.callbacks import *



from keras import backend as K

from keras.engine.topology import Layer

from keras import initializers, regularizers, constraints, optimizers, layers

from keras.callbacks import *



# https://www.kaggle.com/suicaokhoailang/lstm-attention-baseline-0-652-lb

class Attention(Layer):

    def __init__(self, step_dim,

                 W_regularizer=None, b_regularizer=None,

                 W_constraint=None, b_constraint=None,

                 bias=True, **kwargs):

        self.supports_masking = True

        self.init = initializers.get('glorot_uniform')



        self.W_regularizer = regularizers.get(W_regularizer)

        self.b_regularizer = regularizers.get(b_regularizer)



        self.W_constraint = constraints.get(W_constraint)

        self.b_constraint = constraints.get(b_constraint)



        self.bias = bias

        self.step_dim = step_dim

        self.features_dim = 0

        super(Attention, self).__init__(**kwargs)



    def build(self, input_shape):

        assert len(input_shape) == 3



        self.W = self.add_weight((input_shape[-1],),

                                 initializer=self.init,

                                 name='{}_W'.format(self.name),

                                 regularizer=self.W_regularizer,

                                 constraint=self.W_constraint)

        self.features_dim = input_shape[-1]



        if self.bias:

            self.b = self.add_weight((input_shape[1],),

                                     initializer='zero',

                                     name='{}_b'.format(self.name),

                                     regularizer=self.b_regularizer,

                                     constraint=self.b_constraint)

        else:

            self.b = None



        self.built = True



    def compute_mask(self, input, input_mask=None):

        return None



    def call(self, x, mask=None):

        features_dim = self.features_dim

        step_dim = self.step_dim



        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),

                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))



        if self.bias:

            eij += self.b



        eij = K.tanh(eij)



        a = K.exp(eij)



        if mask is not None:

            a *= K.cast(mask, K.floatx())



        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())



        a = K.expand_dims(a)

        weighted_input = x * a

        return K.sum(weighted_input, axis=1)



    def compute_output_shape(self, input_shape):

        return input_shape[0],  self.features_dim



def get_model(input_shape, class_num) :

    inp = Input((input_shape[1], input_shape[2]))

    x = Bidirectional(CuDNNGRU(128, return_sequences = True))(inp)

    x = Bidirectional(CuDNNGRU(64, return_sequences = True))(x)

    x = Attention(input_shape[1])(x)

    x = Dense(64, activation="relu")(x)

    x = Dense(32, activation="relu")(x)

    x = Dense(class_num, activation="softmax")(x)

    model = Model(inputs=inp, outputs=x)

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model
class_num = y_train['surface'].unique().shape[0]

y_group = y_train['group_id'].values

y_train = pd.get_dummies(y_train['surface'])

label_name = [col for col in y_train.columns]

y_train = y_train.values



from sklearn.model_selection import StratifiedKFold



random_seed = 2019

np.random.seed(random_seed)



fold = StratifiedKFold(5, shuffle = True, random_state = random_seed)
oof_train = np.zeros((train_seq.shape[0], class_num))

oof_test = np.zeros((test_seq.shape[0], class_num))



for i, (trn, val) in enumerate(list(fold.split(y_train, np.argmax(y_train, axis = 1)))) :

    model = get_model(train_seq.shape, class_num)

    chk = ModelCheckpoint("best_weight.wt", monitor='val_acc', mode = 'max', save_best_only = True, verbose = 1)

    model.fit(train_seq[trn], y_train[trn]

             , epochs = 100, batch_size = 32

             , validation_data = [train_seq[val], y_train[val]]

             , callbacks = [chk])

    model.load_weights("best_weight.wt")

    oof_train[val] = model.predict(train_seq[val])

    oof_test += model.predict(test_seq) / fold.n_splits
submission = pd.read_csv("../input/sample_submission.csv")

submission['surface'] = pd.DataFrame(np.argmax(oof_test, axis = 1))[0].apply(lambda x : label_name[x]).values.reshape(-1)

submission.to_csv("submission.csv", index = False)

submission.head()
sns.countplot(x="surface", data=pd.read_csv("../input/y_train.csv"))
sns.countplot(x="surface", data=submission)