import pandas as pd

import pyarrow.parquet as pq # Used to read the data

import os 

import numpy as np

from keras.layers import * # Keras is the most friendly Neural Network library, this Kernel use a lot of layers classes

from keras.models import Model

from tqdm import tqdm # Processing time measurement

from sklearn.model_selection import train_test_split 

from keras import backend as K # The backend give us access to tensorflow operations and allow us to create the Attention class

from keras import optimizers # Allow us to access the Adam class to modify some parameters

from sklearn.model_selection import GridSearchCV, StratifiedKFold # Used to use Kfold to train our model

from keras.callbacks import * # This object helps the model to train in a smarter way, avoiding overfitting
df_meta_train = pd.read_csv('../input/metadata_train.csv')

df_meta_test = pd.read_csv('../input/metadata_test.csv')
def obter_sinais_treino(num_amostra):

    signals = [str(i) for i in df_meta_train[df_meta_train['id_measurement'] == num_amostra]['signal_id']]

    return pd.read_parquet('../input/train.parquet', columns=signals)
def obter_sinais_teste(num_amostra):

    signals = [str(i) for i in df_meta_test[df_meta_test['id_measurement'] == num_amostra]['signal_id']]

    return pd.read_parquet('../input/test.parquet', columns=signals)
STEP = 160 #Constante para definir quantas amostras do sinal serão ignoradas



def preparar_dados_treino(num_amostra):

    X = np.array([])

    signals = obter_sinais_treino(num_amostra)

    for signal in signals:

        X = np.append(X, signals[signal].values[::STEP])

    y = np.asscalar(df_meta_train[(df_meta_train['id_measurement'] == num_amostra) & (df_meta_train['phase']==0)].head()['target'])

    return X, y



def preparar_dados_teste(num_amostra):

    X = np.array([])

    signals = obter_sinais_teste(num_amostra)

    for signal in signals:

        X = np.append(X, signals[signal].values[::STEP])

    return X
amostra_min = df_meta_train['id_measurement'].min()

amostra_max = df_meta_train['id_measurement'].max() + 1
X = []

y = []

for i in df_meta_train[df_meta_train['phase']==0]['id_measurement']:

    X_temp, y_temp = preparar_dados_treino(i)

    X.append(X_temp)

    y.append(y_temp)

    

print('100%')



X = np.asarray(X)

y = np.asarray(y)
sh1 = X.shape[0]

sh2 = int(X.shape[1] / 3)

sh3 = 3

X = X.reshape(sh1, sh2, sh3)

X.shape
import sys

print('Tamanho em memória: %d MB' % ((sys.getsizeof(X)/1024)/1024))
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix
# It is the official metric used in this competition

# below is the declaration of a function used inside the keras model, calculation with K (keras backend / thensorflow)

def matthews_correlation(y_true, y_pred):

    '''Calculates the Matthews correlation coefficient measure for quality

    of binary classification problems.

    '''

    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    y_pred_neg = 1 - y_pred_pos



    y_pos = K.round(K.clip(y_true, 0, 1))

    y_neg = 1 - y_pos



    tp = K.sum(y_pos * y_pred_pos)

    tn = K.sum(y_neg * y_pred_neg)



    fp = K.sum(y_neg * y_pred_pos)

    fn = K.sum(y_pos * y_pred_neg)



    numerator = (tp * tn - fp * fn)

    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))



    return numerator / (denominator + K.epsilon())
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
#clf = MLPClassifier(hidden_layer_sizes=(200, 100, 50), max_iter=50, alpha=20, solver='adam', verbose=True, random_state=42, tol=0.5)



# This is NN LSTM Model creation

def model_lstm(input_shape):

    # The shape was explained above, must have this order

    inp = Input(shape=(input_shape[1], input_shape[2]))

    # This is the LSTM layer

    # Bidirecional implies that the 160 chunks are calculated in both ways, 0 to 159 and 159 to zero

    # although it appear that just 0 to 159 way matter, I have tested with and without, and tha later worked best

    # 128 and 64 are the number of cells used, too many can overfit and too few can underfit

    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(inp)

    # The second LSTM can give more fire power to the model, but can overfit it too

    x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)

    # Attention is a new tecnology that can be applyed to a Recurrent NN to give more meanings to a signal found in the middle

    # of the data, it helps more in longs chains of data. A normal RNN give all the responsibility of detect the signal

    # to the last cell. Google RNN Attention for more information :)

    #x = Flatten()(x)

    x = Attention(input_shape[1])(x)

    x = Dense(64, activation="relu")(x)

    # A binnary classification as this must finish with shape (1,)

    x = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=x)

    # Pay attention in the addition of matthews_correlation metric in the compilation, it is a success factor key

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[matthews_correlation])

    



    return model
splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, y))



preds_val = []

y_val = []



for idx, (train_idx, val_idx) in enumerate(splits):

    train_X, train_y, val_X, val_y = X[train_idx], y[train_idx], X[val_idx], y[val_idx]

    

    model = model_lstm(train_X.shape)



    ckpt = ModelCheckpoint('weights_{}.h5'.format(idx), save_best_only=True, save_weights_only=True, verbose=1, monitor='val_matthews_correlation', mode='max')

    

    model.fit(train_X, train_y, batch_size=128, epochs=40, validation_data=[val_X, val_y], callbacks=[ckpt])

    model.load_weights('weights_{}.h5'.format(idx))

              

     # Add the predictions of the validation to the list preds_val

    preds_val.append(model.predict(val_X, batch_size=512))

    # and the val true y

    y_val.append(val_y)

              

preds_val = np.concatenate(preds_val)[...,0]

y_val = np.concatenate(y_val)

preds_val.shape, y_val.shape
import tensorflow as tf

def threshold_search(y_true, y_proba):

    best_threshold = 0

    best_score = 0

    for threshold in tqdm([i * 0.05 for i in range(20)]):

        score = K.eval(matthews_correlation(tf.convert_to_tensor(y_true, np.float64), tf.convert_to_tensor(y_proba > threshold,np.float64)))

        if score > best_score:

            best_threshold = threshold

            best_score = score

    search_result = {'threshold': best_threshold, 'matthews_correlation': best_score}

    return search_result
best_threshold = threshold_search(y_val, preds_val)['threshold']

best_threshold
amostra_min = df_meta_test['id_measurement'].min()

amostra_max = df_meta_test['id_measurement'].max() + 1

X = []

y = []

for i in range(amostra_min, amostra_max):

    X_temp = preparar_dados_teste(i)

    X.append(X_temp)

print('100%')

X = np.asarray(X)
sh1 = X.shape[0]

sh2 = int(X.shape[1] / 3)

sh3 = 3

X = X.reshape(sh1, sh2, sh3)

X.shape
preds_test = []

for i in range(5):

    model.load_weights('weights_{}.h5'.format(i))

    pred = model.predict(X, batch_size=300, verbose=1)

    preds_test.append(pred)



preds_test = (np.squeeze(np.mean(preds_test, axis=0)) > best_threshold).astype(np.int)
test_predicted = pd.DataFrame()

test_predicted['id_measurement'] = [i for i in range(amostra_min, amostra_max)]

test_predicted['target'] = preds_test
output = pd.merge(test_predicted, df_meta_test, on='id_measurement')

output.to_csv('submission-lstm-5fold.csv', index=False, columns=['signal_id', 'target'])