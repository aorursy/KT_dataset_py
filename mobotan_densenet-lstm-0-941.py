!pip list
!pip uninstall -y tensorflow                                                                                                 
!pip install tensorflow==2.2.0
import tensorflow as tf

print(tf.__version__)
!pip install tensorflow_addons

import tensorflow as tf

from tensorflow.keras.layers import *

import pandas as pd

import numpy as np

import random

from tensorflow.keras.callbacks import Callback, LearningRateScheduler

from tensorflow.keras.losses import categorical_crossentropy

from tensorflow.keras.optimizers import Adam

from tensorflow.keras import backend as K

from tensorflow.keras import losses, models, optimizers

import tensorflow_addons as tfa

import gc

from sklearn.model_selection import GroupKFold

from sklearn.metrics import f1_score

from keras.utils import plot_model

from keras.regularizers import l2

import warnings

from sklearn.decomposition import PCA

warnings.simplefilter('ignore')

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 1000)

pd.set_option('display.max_rows', 500)
# Any results you write to the current directory are saved as output.



# configurations and main hyperparammeters

EPOCHS = 300

NNBATCHSIZE = 16

GROUP_BATCH_SIZE = 500

SEED = 321

data_dim = 19

LR = 0.002

SPLITS = 6



def seed_everything(seed):

    random.seed(seed)

    np.random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    tf.random.set_seed(seed)
def Classifier_1(shape_): 



  def conv_block(input,growth_rate,dropout_rate = None,weight_decay = 1e-4):

    x = BatchNormalization(epsilon = 1.1e-5)(input)

    x = Activation('relu')(x)

    x = Conv1D(growth_rate,kernel_size = 3,dilation_rate = 1,kernel_initializer = 'he_normal', padding = 'same')(x)

    if(dropout_rate):

      x = Dropout(dropout_rate)(x)

    return x



  def dense_block(x,nb_layers,nb_filter,growth_rate,dropout_rate = 0.2,weight_decay = 1e-4):

    for i in range(nb_layers):

      cb = conv_block(x,growth_rate,dropout_rate,weight_decay)

      x = concatenate([x,cb],axis = -1)

      nb_filter += growth_rate

    return x ,nb_filter

  def transition_block(input,nb_filter,dropout_rate = None,pooltype = 1,weight_decay=1e-4):

    x = BatchNormalization(epsilon = 1.1e-5)(input)

    x = Activation('relu')(x)

    x = Conv1D(nb_filter,kernel_size = 1,kernel_initializer='he_normal',padding = 'same',use_bias = False,kernel_regularizer=l2(weight_decay))(x)



    if(dropout_rate):

          x = Dropout(dropout_rate)(x)

    if(pooltype==2):

          x = AveragePooling1D(1,strides=1)(x)

    elif(pooltype==1):

          x = ZeroPadding1D(padding=(0,1))(x)

          x = AveragePooling1D(2,strides = 2)(x)

    elif(pooltype==3):

        x = AveragePooling1D(2,strides=1)(x)

    return x,nb_filter

  

  def dense_cnn(input,nclass):

    _dropout_rate = 0.2 

    _weight_decay = 1e-4

    _nb_filter = 64

    x = Conv1D(_nb_filter,

          kernel_size = 5,

          strides = 1,

          kernel_initializer='he_normal',

          padding='same',

          use_bias=False, 

          kernel_regularizer=l2(_weight_decay))(input)

    x ,_nb_filter = dense_block(x,8,_nb_filter,8,None,_weight_decay)

    x_dense_1 = LSTM(128,return_sequences=True)(x)

    x ,_nb_filter = transition_block(x,128,_dropout_rate,2,_weight_decay)

    x ,_nb_filter = dense_block(x,8,_nb_filter,8,None,_weight_decay)

    x_dense_2 = LSTM(64,return_sequences=True)(x)

    x ,_nb_filter = transition_block(x,128,_dropout_rate,2,_weight_decay)

    x ,_nb_filter = dense_block(x,8,_nb_filter,8,None,_weight_decay)

    x_dense_3 = LSTM(32,return_sequences=True)(x)

    

    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)

    x = Activation('relu')(x) 

    x = Dense(nclass,activation='softmax')(x)

    

    x_lstms = concatenate([x_dense_1,x_dense_2,x_dense_3],axis = -1)

    x_lstms = BatchNormalization(axis=-1, epsilon=1.1e-5)(x_lstms)

    x_lstms = Conv1D(32,kernel_size = 5,strides = 1,kernel_initializer='he_normal', padding='same',use_bias=False, kernel_regularizer=l2(_weight_decay))(x_lstms)

    x_lstms =  Dense(nclass,activation='softmax')(x_lstms)

    x = Add()([x_lstms,x])

    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)

    x =  Dense(nclass,activation='softmax',name='DenseNet_out',)(x)

    

    return x

    

  inp = Input(shape = (shape_))

  

  out = dense_cnn(inp,11)

  

  model = models.Model(inputs = inp, outputs = out)



  opt = Adam(lr = LR)

  opt = tfa.optimizers.SWA(opt)

  model.compile(loss = losses.CategoricalCrossentropy(), optimizer = opt, metrics = ['accuracy'])

  return model
# read data

def read_data():

    train = pd.read_csv('/kaggle/input/competition-of-lwp/train_clean.csv', dtype={'time': np.float32, 'signal': np.float32, 'open_channels':np.int32})

    test  = pd.read_csv('./kaggle/input/competition-of-lwp/test_clean.csv', dtype={'time': np.float32, 'signal': np.float32})

    sub  = pd.read_csv('./kaggle/input/liverpool-ion-switching/sample_submission.csv', dtype={'time': np.float32})

    

    Y_train_proba = np.load("./kaggle/input/competition-of-lwp/Y_train_proba.npy")

    Y_test_proba = np.load("./kaggle/input/competition-of-lwp/Y_test_proba.npy")

    

    for i in range(11):

        train[f"proba_{i}"] = Y_train_proba[:, i]

        test[f"proba_{i}"] = Y_test_proba[:, i]



    return train, test, sub



def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2 # just added 

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2

    percent = 100 * (start_mem - end_mem) / start_mem

    print('Mem. usage decreased from {:5.2f} Mb to {:5.2f} Mb ({:.1f}% reduction)'.format(start_mem, end_mem, percent))

    return df



# create batches of 4000 observations

def batching(df, batch_size):

    df['group'] = df.groupby(df.index//batch_size, sort=False)['signal'].agg(['ngroup']).values

    df['group'] = df['group'].astype(np.uint16)

    return df



# normalize the data (standard scaler). We can also try other scalers for a better score!

def normalize(train, test):

    train_input_mean = train.signal.mean()

    train_input_sigma = train.signal.std()

    train['signal'] = (train.signal - train_input_mean) / train_input_sigma

    test['signal'] = (test.signal - train_input_mean) / train_input_sigma

    return train, test



# get lead and lags features

def lag_with_pct_change(df, windows):

    for window in windows:    

        df['signal_shift_pos_' + str(window)] = df.groupby('group')['signal'].shift(window).fillna(0)

        df['signal_shift_neg_' + str(window)] = df.groupby('group')['signal'].shift(-1 * window).fillna(0)

    return df

def add_window_feature(df):

    window_sizes = [10, 25, 50, 100, 500, 1000, 5000, 10000, 25000]

    for window in window_sizes:

        df["rolling_mean_" + str(window)] = df['signal'].rolling(window=window).mean()

        df["rolling_std_" + str(window)] = df['signal'].rolling(window=window).std()

        df["rolling_var_" + str(window)] = df['signal'].rolling(window=window).var()

        df["rolling_min_" + str(window)] = df['signal'].rolling(window=window).min()

        df["rolling_max_" + str(window)] = df['signal'].rolling(window=window).max()

        

    df = df.replace([np.inf, -np.inf], np.nan)

    df.fillna(0, inplace=True)

    

    return df

# main module to run feature engineering. Here you may want to try and add other features and check if your score imporves :).

def run_feat_engineering(df, batch_size):

    # create batches

    df = batching(df, batch_size = batch_size)

    # create leads and lags (1, 2, 3 making them 9 features)

    df = lag_with_pct_change(df, [1, 2, 3])

    #df = add_window_feature(df)

    # create signal ** 2 (this is the new feature)

    df['signal_2'] = df['signal'] ** 2

    

    return df



# fillna with the mean and select features for training

def feature_selection(train, test):

    features = [col for col in train.columns if col not in ['index', 'group', 'open_channels', 'time']]

    train = train.replace([np.inf, -np.inf], np.nan)

    test = test.replace([np.inf, -np.inf], np.nan)

    for feature in features:

        feature_mean = pd.concat([train[feature], test[feature]], axis = 0).mean()

        train[feature] = train[feature].fillna(feature_mean)

        test[feature] = test[feature].fillna(feature_mean)

    return train, test, features
# read data

def read_data():

    train = pd.read_csv('./train_clean.csv', dtype={'time': np.float32, 'signal': np.float32, 'open_channels':np.int32})

    test  = pd.read_csv('./test_clean.csv', dtype={'time': np.float32, 'signal': np.float32})

    sub  = pd.read_csv('./sample_submission.csv', dtype={'time': np.float32})

    

    Y_train_proba = np.load("./Y_train_proba.npy")

    Y_test_proba = np.load("./Y_test_proba.npy")

    

    for i in range(11):

        train[f"proba_{i}"] = Y_train_proba[:, i]

        test[f"proba_{i}"] = Y_test_proba[:, i]



    return train, test, sub



def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2 # just added 

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2

    percent = 100 * (start_mem - end_mem) / start_mem

    print('Mem. usage decreased from {:5.2f} Mb to {:5.2f} Mb ({:.1f}% reduction)'.format(start_mem, end_mem, percent))

    return df



# create batches of 4000 observations

def batching(df, batch_size):

    df['group'] = df.groupby(df.index//batch_size, sort=False)['signal'].agg(['ngroup']).values

    df['group'] = df['group'].astype(np.uint16)

    return df



# normalize the data (standard scaler). We can also try other scalers for a better score!

def normalize(train, test):

    train_input_mean = train.signal.mean()

    train_input_sigma = train.signal.std()

    train['signal'] = (train.signal - train_input_mean) / train_input_sigma

    test['signal'] = (test.signal - train_input_mean) / train_input_sigma

    return train, test



# get lead and lags features

def lag_with_pct_change(df, windows):

    for window in windows:    

        df['signal_shift_pos_' + str(window)] = df.groupby('group')['signal'].shift(window).fillna(0)

        df['signal_shift_neg_' + str(window)] = df.groupby('group')['signal'].shift(-1 * window).fillna(0)

    return df

def add_window_feature(df):

    window_sizes = [10, 25, 50, 100, 500, 1000, 5000, 10000, 25000]

    for window in window_sizes:

        df["rolling_mean_" + str(window)] = df['signal'].rolling(window=window).mean()

        df["rolling_std_" + str(window)] = df['signal'].rolling(window=window).std()

        df["rolling_var_" + str(window)] = df['signal'].rolling(window=window).var()

        df["rolling_min_" + str(window)] = df['signal'].rolling(window=window).min()

        df["rolling_max_" + str(window)] = df['signal'].rolling(window=window).max()

        

    df = df.replace([np.inf, -np.inf], np.nan)

    df.fillna(0, inplace=True)

    

    return df

# main module to run feature engineering. Here you may want to try and add other features and check if your score imporves :).

def run_feat_engineering(df, batch_size):

    # create batches

    df = batching(df, batch_size = batch_size)

    # create leads and lags (1, 2, 3 making them 9 features)

    df = lag_with_pct_change(df, [1, 2, 3])

    #df = add_window_feature(df)

    # create signal ** 2 (this is the new feature)

    df['signal_2'] = df['signal'] ** 2

    

    return df



# fillna with the mean and select features for training

def feature_selection(train, test):

    features = [col for col in train.columns if col not in ['index', 'group', 'open_channels', 'time']]

    train = train.replace([np.inf, -np.inf], np.nan)

    test = test.replace([np.inf, -np.inf], np.nan)

    for feature in features:

        feature_mean = pd.concat([train[feature], test[feature]], axis = 0).mean()

        train[feature] = train[feature].fillna(feature_mean)

        test[feature] = test[feature].fillna(feature_mean)

    return train, test, features
def Classifier(shape_):

    

    def cbr(x, out_layer, kernel, stride, dilation):

        x = Conv1D(out_layer, kernel_size=kernel, dilation_rate=dilation, strides=stride, padding="same")(x)

        x = BatchNormalization()(x)

        x = Activation("relu")(x)

        return x

    

    def wave_block(x, filters, kernel_size, n):

        dilation_rates = [2**i for i in range(n)]

        x = Conv1D(filters = filters,

                   kernel_size = 1,

                   padding = 'same')(x)

        res_x = x

        for dilation_rate in dilation_rates:

            tanh_out = Conv1D(filters = filters,

                              kernel_size = kernel_size,

                              padding = 'same', 

                              activation = 'tanh', 

                              dilation_rate = dilation_rate)(x)

            sigm_out = Conv1D(filters = filters,

                              kernel_size = kernel_size,

                              padding = 'same',

                              activation = 'sigmoid', 

                              dilation_rate = dilation_rate)(x)

            x = Multiply()([tanh_out, sigm_out])

            x = Conv1D(filters = filters,

                       kernel_size = 1,

                       padding = 'same')(x)

            res_x = Add()([res_x, x])

        return res_x

    

    inp = Input(shape = (shape_))

    x = cbr(inp, 64, 7, 1, 1)

    x = BatchNormalization()(x)

    x = wave_block(x, 16, 3, 12)

    x = BatchNormalization()(x)

    x = wave_block(x, 32, 3, 8)

    x = BatchNormalization()(x)

    x = wave_block(x, 64, 3, 4)

    x = BatchNormalization()(x)

    x = wave_block(x, 128, 3, 1)

    x = cbr(x, 32, 7, 1, 1)

    x = BatchNormalization()(x) 

    x = Dropout(0.2)(x)

    out = Dense(11, activation = 'softmax', name = 'out')(x)

    

    model = models.Model(inputs = inp, outputs = out)

    

    opt = Adam(lr = LR)

    opt = tfa.optimizers.SWA(opt)

    model.compile(loss = losses.CategoricalCrossentropy(), optimizer = opt, metrics = ['accuracy'])

    return model
# function that decrease the learning as epochs increase (i also change this part of the code)

def lr_schedule(epoch):

    if epoch < 10:

        lr = LR

    elif epoch < 15:

        lr = LR / 5

    elif epoch < 20:

        lr = LR / 7

    elif epoch < 25:

        lr = LR / 9

    elif epoch < 30:

        lr = LR / 11

    elif epoch < 35:

        lr = LR / 13

    elif epoch < 40:

        lr = LR / 15

    else:

        lr = LR / 100

    return lr



# class to get macro f1 score. This is not entirely necessary but it's fun to check f1 score of each epoch (be carefull, if you use this function early stopping callback will not work)

class MacroF1(Callback):

    def __init__(self, model, inputs, targets):

        self.model = model

        self.inputs = inputs

        self.targets = np.argmax(targets, axis = 2).reshape(-1)

        

    def on_epoch_end(self, epoch, logs):

        pred = np.argmax(self.model.predict(self.inputs), axis = 2).reshape(-1)

        score = f1_score(self.targets, pred, average = 'macro')

        print(f'F1 Macro Score: {score:.5f}')


# main function to perfrom groupkfold cross validation (we have 1000 vectores of 4000 rows and 8 features (columns)). Going to make 5 groups with this subgroups.

def run_cv_model_by_batch(train, test, splits, batch_col, feats, sample_submission, nn_epochs, nn_batch_size):

    seed_everything(SEED)

    K.clear_session()

    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)

    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=config)

    tf.compat.v1.keras.backend.set_session(sess)

    oof_ = np.zeros((len(train), 11)) # build out of folds matrix with 11 columns, they represent our target variables classes (from 0 to 10)

    preds_ = np.zeros((len(test), 11))

    target = ['open_channels']

    group = train['group']

    kf = GroupKFold(n_splits=5)

    splits = [x for x in kf.split(train, train[target], group)]

    

    new_splits = []

    for sp in splits:

        new_split = []

        new_split.append(np.unique(group[sp[0]]))

        new_split.append(np.unique(group[sp[1]]))

        new_split.append(sp[1])    

        new_splits.append(new_split)

    # pivot target columns to transform the net to a multiclass classification estructure (you can also leave it in 1 vector with sparsecategoricalcrossentropy loss function)

    tr = pd.concat([pd.get_dummies(train.open_channels), train[['group']]], axis=1)# [500 0000,12]

    tr.columns = ['target_'+str(i) for i in range(11)] + ['group']

    target_cols = ['target_'+str(i) for i in range(11)]

    train_tr = np.array(list(tr.groupby('group').apply(lambda x: x[target_cols].values))).astype(np.float32)

    train = np.array(list(train.groupby('group').apply(lambda x: x[feats].values)))

    test = np.array(list(test.groupby('group').apply(lambda x: x[feats].values)))



    for n_fold, (tr_idx, val_idx, val_orig_idx) in enumerate(new_splits[0:], start=0):

        train_x, train_y = train[tr_idx], train_tr[tr_idx]

        valid_x, valid_y = train[val_idx], train_tr[val_idx]

        print(f'Our training dataset shape is {train_x.shape}')

        print(f'Our validation dataset shape is {valid_x.shape}')



        gc.collect()

        shape_ = (None, train_x.shape[2]) # input is going to be the number of feature we are using (dimension 2 of 0, 1, 2)

        model = Classifier_1(shape_)

        # using our lr_schedule function

        cb_lr_schedule = LearningRateScheduler(lr_schedule)

        model.fit(train_x,train_y,

                  epochs = nn_epochs,

                  callbacks = [cb_lr_schedule, MacroF1(model, valid_x, valid_y)], # adding custom evaluation metric for each epoch

                  batch_size = nn_batch_size,verbose = 2,

                  validation_data = (valid_x,valid_y))

        preds_f = model.predict(valid_x)

        f1_score_ = f1_score(np.argmax(valid_y, axis=2).reshape(-1),  np.argmax(preds_f, axis=2).reshape(-1), average = 'macro') # need to get the class with the biggest probability

        print(f'Training fold {n_fold + 1} completed. macro f1 score : {f1_score_ :1.5f}')

        preds_f = preds_f.reshape(-1, preds_f.shape[-1])

        oof_[val_orig_idx,:] += preds_f

        te_preds = model.predict(test)

        te_preds = te_preds.reshape(-1, te_preds.shape[-1])           

        preds_ += te_preds / SPLITS

    # calculate the oof macro f1_score

    f1_score_ = f1_score(np.argmax(train_tr, axis = 2).reshape(-1),  np.argmax(oof_, axis = 1), average = 'macro') # axis 2 for the 3 Dimension array and axis 1 for the 2 Domension Array (extracting the best class)

    print(f'Training completed. oof macro f1 score : {f1_score_:1.5f}')

    sample_submission['open_channels'] = np.argmax(preds_, axis = 1).astype(int)

    sample_submission.to_csv('submission_wavenet_AddWindow_.csv', index=False, float_format='%.4f')
print('Reading Data Started...')

train, test, sample_submission = read_data()



x = [[(2000000,2500000),(4500000,5000000)]]

for j in range(len(x[0])): train.iloc[x[0][j][0]:x[0][j][1],1] = train.iloc[x[0][j][0]:x[0][j][1],1]+2.86



x = [[(500000,600000),(700000,800000)]]

for j in range(len(x[0])): test.iloc[x[0][j][0]:x[0][j][1],1] = test.iloc[x[0][j][0]:x[0][j][1],1]+2.86



train, test = normalize(train, test)

print('Reading and Normalizing Data Completed')

    

print('Creating Features')

print('Feature Engineering Started...')

train0 = run_feat_engineering(train, batch_size = GROUP_BATCH_SIZE)

test0 = run_feat_engineering(test, batch_size = GROUP_BATCH_SIZE)

train=reduce_mem_usage(train0)

test=reduce_mem_usage(test0)

del train0,test0

gc.collect()

train, test, features = feature_selection(train, test)

print('Feature Engineering Completed...')
print(f'Training Wavenet model with {SPLITS} folds of GroupKFold Started...')

run_cv_model_by_batch(train, test, SPLITS, 'group', features, sample_submission, EPOCHS, NNBATCHSIZE)

print('Training completed...')