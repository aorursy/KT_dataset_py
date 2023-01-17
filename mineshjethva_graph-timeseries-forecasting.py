!pip install spektral
# !ls -lh 

TRAIN_FILE="/kaggle/input/demand-forecasting-kernels-only/train.csv"
""" spektral_utilities """

import numpy as np

from scipy import sparse as sp



import tensorflow as tf

from tensorflow.keras import backend as K

from tensorflow.python.ops.linalg.sparse import sparse as tfsp

from tensorflow.keras import backend as K



SINGLE  = 1   # Single         (rank(a)=2, rank(b)=2)

MIXED   = 2   # Mixed          (rank(a)=2, rank(b)=3)

iMIXED  = 3   # Inverted mixed (rank(a)=3, rank(b)=2)

BATCH   = 4   # Batch          (rank(a)=3, rank(b)=3)

UNKNOWN = -1  # Unknown





def transpose(a, perm=None, name=None):

    """

    Transposes a according to perm, dealing automatically with sparsity.

    :param a: Tensor or SparseTensor with rank k.

    :param perm: permutation indices of size k.

    :param name: name for the operation.

    :return: Tensor or SparseTensor with rank k.

    """

    if K.is_sparse(a):

        transpose_op = tf.sparse.transpose

    else:

        transpose_op = tf.transpose



    if perm is None:

        perm = (1, 0)  # Make explicit so that shape will always be preserved

    return transpose_op(a, perm=perm, name=name)





def reshape(a, shape=None, name=None):

    """

    Reshapes a according to shape, dealing automatically with sparsity.

    :param a: Tensor or SparseTensor.

    :param shape: new shape.

    :param name: name for the operation.

    :return: Tensor or SparseTensor.

    """

    if K.is_sparse(a):

        reshape_op = tf.sparse.reshape

    else:

        reshape_op = tf.reshape



    return reshape_op(a, shape=shape, name=name)





def autodetect_mode(a, b):

    """

    Return a code identifying the mode of operation (single, mixed, inverted mixed and

    batch), given a and b. See `ops.modes` for meaning of codes.

    :param a: Tensor or SparseTensor.

    :param b: Tensor or SparseTensor.

    :return: mode of operation as an integer code.

    """

    a_dim = K.ndim(a)

    b_dim = K.ndim(b)

    if b_dim == 2:

        if a_dim == 2:

            return SINGLE

        elif a_dim == 3:

            return iMIXED

    elif b_dim == 3:

        if a_dim == 2:

            return MIXED

        elif a_dim == 3:

            return BATCH

    return UNKNOWN





def filter_dot(fltr, features):

    """

    Wrapper for matmul_A_B, specifically used to compute the matrix multiplication

    between a graph filter and node features.

    :param fltr:

    :param features: the node features (N x F in single mode, batch x N x F in

    mixed and batch mode).

    :return: the filtered features.

    """

    mode = autodetect_mode(fltr, features)

    if mode == SINGLE or mode == BATCH:

        return dot(fltr, features)

    else:

        # Mixed mode

        return mixed_mode_dot(fltr, features)





def dot(a, b, transpose_a=False, transpose_b=False):

    """

    Dot product between a and b along innermost dimensions, for a and b with

    same rank. Supports both dense and sparse multiplication (including

    sparse-sparse).

    :param a: Tensor or SparseTensor with rank 2 or 3.

    :param b: Tensor or SparseTensor with same rank as a.

    :param transpose_a: bool, transpose innermost two dimensions of a.

    :param transpose_b: bool, transpose innermost two dimensions of b.

    :return: Tensor or SparseTensor with rank 2 or 3.

    """

    a_is_sparse_tensor = isinstance(a, tf.SparseTensor)

    b_is_sparse_tensor = isinstance(b, tf.SparseTensor)

    if a_is_sparse_tensor:

        a = tfsp.CSRSparseMatrix(a)

    if b_is_sparse_tensor:

        b = tfsp.CSRSparseMatrix(b)

    out = tfsp.matmul(a, b, transpose_a=transpose_a, transpose_b=transpose_b)

    if hasattr(out, 'to_sparse_tensor'):

        return out.to_sparse_tensor()



    return out





def mixed_mode_dot(a, b):

    """

    Computes the equivalent of `tf.einsum('ij,bjk->bik', a, b)`, but

    works for both dense and sparse input filters.

    :param a: rank 2 Tensor or SparseTensor.

    :param b: rank 3 Tensor or SparseTensor.

    :return: rank 3 Tensor or SparseTensor.

    """

    s_0_, s_1_, s_2_ = K.int_shape(b)

    B_T = transpose(b, (1, 2, 0))

    B_T = reshape(B_T, (s_1_, -1))

    output = dot(a, B_T)

    output = reshape(output, (s_1_, s_2_, -1))

    output = transpose(output, (2, 0, 1))



    return output





def degree_power(A, k):

    r"""

    Computes \(\D^{k}\) from the given adjacency matrix. Useful for computing

    normalised Laplacian.

    :param A: rank 2 array or sparse matrix.

    :param k: exponent to which elevate the degree matrix.

    :return: if A is a dense array, a dense array; if A is sparse, a sparse

    matrix in DIA format.

    """

    degrees = np.power(np.array(A.sum(1)), k).flatten()

    degrees[np.isinf(degrees)] = 0.

    if sp.issparse(A):

        D = sp.diags(degrees)

    else:

        D = np.diag(degrees)

    return D





def normalized_adjacency(A, symmetric=True):

    r"""

    Normalizes the given adjacency matrix using the degree matrix as either

    \(\D^{-1}\A\) or \(\D^{-1/2}\A\D^{-1/2}\) (symmetric normalization).

    :param A: rank 2 array or sparse matrix;

    :param symmetric: boolean, compute symmetric normalization;

    :return: the normalized adjacency matrix.

    """

    if symmetric:

        normalized_D = degree_power(A, -0.5)

        output = normalized_D.dot(A).dot(normalized_D)

    else:

        normalized_D = degree_power(A, -1.)

        output = normalized_D.dot(A)

    return output





def localpooling_filter(A, symmetric=True):

    r"""

    Computes the graph filter described in

    [Kipf & Welling (2017)](https://arxiv.org/abs/1609.02907).

    :param A: array or sparse matrix with rank 2 or 3;

    :param symmetric: boolean, whether to normalize the matrix as

    \(\D^{-\frac{1}{2}}\A\D^{-\frac{1}{2}}\) or as \(\D^{-1}\A\);

    :return: array or sparse matrix with rank 2 or 3, same as A;

    """

    fltr = A.copy()

    if sp.issparse(A):

        I = sp.eye(A.shape[-1], dtype=A.dtype)

    else:

        I = np.eye(A.shape[-1], dtype=A.dtype)

    if A.ndim == 3:

        for i in range(A.shape[0]):

            A_tilde = A[i] + I

            fltr[i] = normalized_adjacency(A_tilde, symmetric=symmetric)

    else:

        A_tilde = A + I

        fltr = normalized_adjacency(A_tilde, symmetric=symmetric)



    if sp.issparse(fltr):

        fltr.sort_indices()

    return fltr



""" spektral_gcn """

from tensorflow.keras import activations, initializers, regularizers, constraints

from tensorflow.keras import backend as K

from tensorflow.keras.layers import Layer



#from spektral_utilities import filter_dot, dot, localpooling_filter





class GraphConv(Layer):

    r"""

    A graph convolutional layer (GCN) as presented by

    [Kipf & Welling (2016)](https://arxiv.org/abs/1609.02907).

    **Mode**: single, mixed, batch.

    This layer computes:

    $$

        \Z = \hat \D^{-1/2} \hat \A \hat \D^{-1/2} \X \W + \b

    $$

    where \( \hat \A = \A + \I \) is the adjacency matrix with added self-loops

    and \(\hat\D\) is its degree matrix.

    **Input**

    - Node features of shape `([batch], N, F)`;

    - Modified Laplacian of shape `([batch], N, N)`; can be computed with

    `spektral.utils.convolution.localpooling_filter`.

    **Output**

    - Node features with the same shape as the input, but with the last

    dimension changed to `channels`.

    **Arguments**

    - `channels`: number of output channels;

    - `activation`: activation function to use;

    - `use_bias`: whether to add a bias to the linear transformation;

    - `kernel_initializer`: initializer for the kernel matrix;

    - `bias_initializer`: initializer for the bias vector;

    - `kernel_regularizer`: regularization applied to the kernel matrix;

    - `bias_regularizer`: regularization applied to the bias vector;

    - `activity_regularizer`: regularization applied to the output;

    - `kernel_constraint`: constraint applied to the kernel matrix;

    - `bias_constraint`: constraint applied to the bias vector.

    """



    def __init__(self,

                 channels,

                 activation=None,

                 use_bias=True,

                 kernel_initializer='glorot_uniform',

                 bias_initializer='zeros',

                 kernel_regularizer=None,

                 bias_regularizer=None,

                 activity_regularizer=None,

                 kernel_constraint=None,

                 bias_constraint=None,

                 **kwargs):



        super().__init__(activity_regularizer=activity_regularizer, **kwargs)

        self.channels = channels

        self.activation = activations.get(activation)

        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)

        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)

        self.bias_constraint = constraints.get(bias_constraint)

        self.supports_masking = False



    def build(self, input_shape):

        assert len(input_shape) >= 2

        input_dim = input_shape[0][-1]

        self.kernel = self.add_weight(shape=(input_dim, self.channels),

                                      initializer=self.kernel_initializer,

                                      name='kernel',

                                      regularizer=self.kernel_regularizer,

                                      constraint=self.kernel_constraint)

        if self.use_bias:

            self.bias = self.add_weight(shape=(self.channels,),

                                        initializer=self.bias_initializer,

                                        name='bias',

                                        regularizer=self.bias_regularizer,

                                        constraint=self.bias_constraint)

        else:

            self.bias = None

        self.built = True



    def call(self, inputs):

        features = inputs[0]

        fltr = inputs[1]



        # Convolution

        output = dot(features, self.kernel)

        output = filter_dot(fltr, output)



        if self.use_bias:

            output = K.bias_add(output, self.bias)

        if self.activation is not None:

            output = self.activation(output)

        return output



    def compute_output_shape(self, input_shape):

        features_shape = input_shape[0]

        output_shape = features_shape[:-1] + (self.channels,)

        return output_shape



    def get_config(self):

        config = {

            'channels': self.channels,

            'activation': activations.serialize(self.activation),

            'use_bias': self.use_bias,

            'kernel_initializer': initializers.serialize(self.kernel_initializer),

            'bias_initializer': initializers.serialize(self.bias_initializer),

            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),

            'bias_regularizer': regularizers.serialize(self.bias_regularizer),

            'kernel_constraint': constraints.serialize(self.kernel_constraint),

            'bias_constraint': constraints.serialize(self.bias_constraint)

        }

        base_config = super().get_config()

        return dict(list(base_config.items()) + list(config.items()))



    @staticmethod

    def preprocess(A):

        return localpooling_filter(A)



from spektral.layers import GraphConv
import numpy as np

import pandas as pd

from datetime import date, timedelta

import os

import random

import matplotlib.pyplot as plt

from tqdm import tqdm

from scipy.stats import skew, kurtosis



from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error



import tensorflow as tf

from tensorflow.keras.layers import *

from tensorflow.keras.models import *

from tensorflow.keras.optimizers import *

from tensorflow.keras.callbacks import *

from tensorflow.keras import backend as K
### IMPORT SPEKTRAL CLASSES ###



# from spektral_utilities import *

# from spektral_gcn import GraphConv
### READ DATA ###



df = pd.read_csv(TRAIN_FILE)

df['date'] = pd.to_datetime(df['date'])



print(df.shape)

df.head()
### SWITCH DATA FROM VERTICAL TO HORIZONTAL FORMAT ###



unstaked_df = df.copy()

unstaked_df['id'] = df['item'].astype(str)+'_'+df['store'].astype(str)

unstaked_df.set_index(['id','date'], inplace=True)

unstaked_df.drop(['store','item'], axis=1, inplace=True)

unstaked_df = unstaked_df.astype(float).unstack()

unstaked_df.columns = unstaked_df.columns.get_level_values(1)



print(unstaked_df.shape)

unstaked_df.head()
### UTILITY FUNCTIONS FOR FEATURE ENGINEERING ###



sequence_length = 14







def get_timespan(df, today, days):    

    df = df[pd.date_range(today - timedelta(days=days), 

            periods=days, freq='D')] # day - n_days <= dates < day    

    return df



def create_features(df, today, seq_len):

    

    all_sequence = get_timespan(df, today, seq_len).values

    

    group_store = all_sequence.reshape((-1, 10, seq_len))

    

    store_corr = np.stack([np.corrcoef(i) for i in group_store], axis=0)

    

    store_features = np.stack([

              group_store.mean(axis=2),

              group_store[:,:,int(sequence_length/2):].mean(axis=2),

              group_store.std(axis=2),

              group_store[:,:,int(sequence_length/2):].std(axis=2),

              skew(group_store, axis=2),

              kurtosis(group_store, axis=2),

              np.apply_along_axis(lambda x: np.polyfit(np.arange(0, sequence_length), x, 1)[0], 2, group_store)

            ], axis=1)

    

    group_store = np.transpose(group_store, (0,2,1))

    store_features = np.transpose(store_features, (0,2,1))

    

    return group_store, store_corr, store_features



def create_label(df, today):

    

    y = df[today].values

    

    return y.reshape((-1, 10))
### PLOT A SEQUENCE OF SALES FOR ITEM 10 IN ALL STORES ###



sequence = get_timespan(unstaked_df, date(2017,11,1), 30)

sequence.head(10).T.plot(figsize=(14,5))

plt.ylabel('sales')

plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
### DEFINE TRAIN, VALID, TEST DATES ###



train_date = date(2013, 1, 1)

valid_date = date(2015, 1, 1)

test_date = date(2016, 1, 1)
### CREATE TRAIN FEATURES ###



X_seq, X_cor, X_feat, y = [], [], [], []



for d in tqdm(pd.date_range(train_date+timedelta(days=sequence_length), valid_date)):

    seq_, corr_, feat_ = create_features(unstaked_df, d, sequence_length)

    y_ = create_label(unstaked_df, d)

    X_seq.append(seq_), X_cor.append(corr_), X_feat.append(feat_), y.append(y_)

    

X_train_seq = np.concatenate(X_seq, axis=0).astype('float16')

X_train_cor = np.concatenate(X_cor, axis=0).astype('float16')

X_train_feat = np.concatenate(X_feat, axis=0).astype('float16')

y_train = np.concatenate(y, axis=0).astype('float16')



print(X_train_seq.shape, X_train_cor.shape, X_train_feat.shape, y_train.shape)
### CREATE VALID FEATURES ###



X_seq, X_cor, X_feat, y = [], [], [], []



for d in tqdm(pd.date_range(valid_date+timedelta(days=sequence_length), test_date)):

    seq_, corr_, feat_ = create_features(unstaked_df, d, sequence_length)

    y_ = create_label(unstaked_df, d)

    X_seq.append(seq_), X_cor.append(corr_), X_feat.append(feat_), y.append(y_)

    

X_valid_seq = np.concatenate(X_seq, axis=0).astype('float16')

X_valid_cor = np.concatenate(X_cor, axis=0).astype('float16')

X_valid_feat = np.concatenate(X_feat, axis=0).astype('float16')

y_valid = np.concatenate(y, axis=0).astype('float16')



print(X_valid_seq.shape, X_valid_cor.shape, X_valid_feat.shape, y_valid.shape)
### CREATE TEST FEATURES ###



X_seq, X_cor, X_feat, y = [], [], [], []



for d in tqdm(pd.date_range(test_date+timedelta(days=sequence_length), date(2016,12,31))):

    seq_, corr_, feat_ = create_features(unstaked_df, d, sequence_length)

    y_ = create_label(unstaked_df, d)

    X_seq.append(seq_), X_cor.append(corr_), X_feat.append(feat_), y.append(y_)

    

X_test_seq = np.concatenate(X_seq, axis=0).astype('float16')

X_test_cor = np.concatenate(X_cor, axis=0).astype('float16')

X_test_feat = np.concatenate(X_feat, axis=0).astype('float16')

y_test = np.concatenate(y, axis=0).astype('float16')



print(X_test_seq.shape, X_test_cor.shape, X_test_feat.shape, y_test.shape)
### SCALE SEQUENCES ###



scaler_seq = StandardScaler()

scaler_feat = StandardScaler()



X_train_seq = scaler_seq.fit_transform(X_train_seq.reshape(-1,10)).reshape(X_train_seq.shape)

X_valid_seq = scaler_seq.transform(X_valid_seq.reshape(-1,10)).reshape(X_valid_seq.shape)

X_test_seq = scaler_seq.transform(X_test_seq.reshape(-1,10)).reshape(X_test_seq.shape)



y_train = scaler_seq.transform(y_train)

y_valid = scaler_seq.transform(y_valid)

y_test = scaler_seq.transform(y_test)



X_train_feat = scaler_feat.fit_transform(X_train_feat.reshape(-1,10)).reshape(X_train_feat.shape)

X_valid_feat = scaler_feat.transform(X_valid_feat.reshape(-1,10)).reshape(X_valid_feat.shape)

X_test_feat = scaler_feat.transform(X_test_feat.reshape(-1,10)).reshape(X_test_feat.shape)
### OBTAIN LAPLACIANS FROM CORRELATIONS ###



X_train_lap = localpooling_filter(1 - np.abs(X_train_cor))

X_valid_lap = localpooling_filter(1 - np.abs(X_valid_cor))

X_test_lap = localpooling_filter(1 - np.abs(X_test_cor))
def get_model():



    opt = Adam(lr=0.001)



    inp_seq = Input((sequence_length, 10))

    inp_lap = Input((10, 10))

    inp_feat = Input((10, X_train_feat.shape[-1]))



    x = GraphConv(32, activation='relu')([inp_feat, inp_lap])

    x = GraphConv(16, activation='relu')([x, inp_lap])

    x = Flatten()(x)



    xx = LSTM(128, activation='relu', return_sequences=True)(inp_seq)

    xx = LSTM(32, activation='relu')(xx)



    x = Concatenate()([x,xx])

    x = BatchNormalization()(x)

    x = Dropout(0.5)(x)

    x = Dense(128, activation='relu')(x)

    x = Dense(32, activation='relu')(x)

    x = Dropout(0.3)(x)

    out = Dense(1)(x)



    model = Model([inp_seq, inp_lap, inp_feat], out)

    model.compile(optimizer=opt, loss='mse', 

                  metrics=[tf.keras.metrics.RootMeanSquaredError()])



    return model
model = get_model()

model.summary() 
### TRAIN A MODEL FOR EACH STORES USING ALL THE DATA AVAILALBE FROM OTHER STORES ###





tf.random.set_seed(33)

os.environ['PYTHONHASHSEED'] = str(33)

np.random.seed(33)

random.seed(33)



session_conf = tf.compat.v1.ConfigProto(

    intra_op_parallelism_threads=1, 

    inter_op_parallelism_threads=1

)

sess = tf.compat.v1.Session(

    graph=tf.compat.v1.get_default_graph(), 

    config=session_conf

)

tf.compat.v1.keras.backend.set_session(sess)







pred_valid_all = np.zeros(y_valid.shape)

pred_test_all = np.zeros(y_test.shape)



for store in range(10):



    print('-------', 'store', store, '-------')

    

    es = EarlyStopping(patience=5, verbose=1, min_delta=0.001, monitor='val_loss', mode='auto', restore_best_weights=True)



    model = get_model()

    model.fit([X_train_seq, X_train_lap, X_train_feat], y_train[:,store], epochs=100, batch_size=256, 

              validation_data=([X_valid_seq, X_valid_lap, X_valid_feat], y_test[:,store]), callbacks=[es], verbose=2)



    pred_valid_all[:,store] = model.predict([X_valid_seq, X_valid_lap, X_valid_feat]).ravel()

    pred_test_all[:,store] = model.predict([X_test_seq, X_test_lap, X_test_feat]).ravel()





pred_valid_all = scaler_seq.inverse_transform(pred_valid_all)

reverse_valid = scaler_seq.inverse_transform(y_valid)

pred_test_all = scaler_seq.inverse_transform(pred_test_all)

reverse_test = scaler_seq.inverse_transform(y_test)
### RMSE ON TEST DATA ###



error = {}



for store in range(10):

    

    error[store] = np.sqrt(mean_squared_error(reverse_test[:,store], pred_test_all[:,store]))
### PLOT RMSE ###



plt.figure(figsize=(14,5))

plt.bar(range(10), error.values())

plt.xticks(range(10), ['store_'+str(s) for s in range(10)])

plt.ylabel('error')

np.set_printoptions(False)
### UTILITY FUNCTION TO PLOT PREDICTION ###



def plot_predictions(y_true, y_pred, store, item):

    

    y_true = y_true.reshape(50,-1,10)

    y_pred = y_pred.reshape(50,-1,10)

    

    plt.plot(y_true[item,:,store], label='true')

    plt.plot(y_pred[item,:,store], label='prediction')

    plt.title(f"store: {store} item: {item}"); plt.legend()

    plt.ylabel('sales'); plt.xlabel('date')
plt.figure(figsize=(11,5))

plot_predictions(reverse_test, pred_test_all, 7,0)