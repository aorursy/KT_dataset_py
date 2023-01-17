from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization, Activation, concatenate

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder

from tensorflow.keras.utils import get_custom_objects

from tensorflow.keras.optimizers import Adam, Nadam

from scipy.stats import rankdata, spearmanr

from tensorflow.keras.models import Model

from sklearn.metrics import roc_auc_score

import tensorflow.keras.backend as K

from tqdm.notebook import tqdm

import random, os, gc, time

import tensorflow as tf

import tensorflow.keras

import numpy as np

import pandas as pd

import warnings



warnings.filterwarnings('ignore')

warnings.simplefilter('ignore')





SEED = 2020

batch_size_train = 2048

batch_size_pred  = 8000

verbose = 0

verboseB = False

low  = 10

high = 100
def set_seed(seed=SEED):

    random.seed(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed()
P = '../input/cat-in-the-dat-ii/'

train = pd.read_csv(P+'train.csv')

test = pd.read_csv(P+'test.csv')

sub = pd.read_csv(P+'sample_submission.csv')
useful_features = list(train.iloc[:, 1:24].columns)



y = train.sort_values('id')['target']

X = train.sort_values('id')[useful_features]

X_test = test[useful_features]

del train, test
categorical_features = [

'bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4', 'nom_0', 'nom_1', 'nom_2',

       'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9', 'day', 'month'

]



continuous_features = list(filter(lambda x: x not in categorical_features, X))
def OrdMapping(df, ordinal):

    ord_maps = {

        'ord_0': {val: i for i, val in enumerate([1, 2, 3])},

        'ord_1': {

            val: i

            for i, val in enumerate(

                ['Novice', 'Contributor', 'Expert', 'Master', 'Grandmaster']

            )

        },

        'ord_2': {

            val: i

            for i, val in enumerate(

                ['Freezing', 'Cold', 'Warm', 'Hot', 'Boiling Hot', 'Lava Hot']

            )

        },

        'ord_3': {val: i for i, val in enumerate(sorted(df['ord_3'].dropna().unique()))},

        'ord_4': {val: i for i, val in enumerate(sorted(df['ord_4'].dropna().unique()))},

        'ord_5': {val: i for i, val in enumerate(sorted(df['ord_5'].dropna().unique()))},

    }

    ord_cols = pd.concat([df[col].map(ord_map).fillna(max(ord_map.values())//2).astype('float32') for col, ord_map in ord_maps.items()], axis=1)

    ord_cols /= ord_cols.max()

    df[ordinal] = ord_cols

    return df
X['isna_sum'] = X.isna().sum(axis=1)

X_test['isna_sum'] = X_test.isna().sum(axis=1)



X['isna_sum'] = (X['isna_sum'] - X['isna_sum'].mean())/X['isna_sum'].std()

X_test['isna_sum'] = (X_test['isna_sum'] - X_test['isna_sum'].mean())/X_test['isna_sum'].std()
X = OrdMapping(X ,continuous_features)

X_test= OrdMapping(X_test, continuous_features)
class ContinuousFeatureConverter:

    def __init__(self, name, feature, log_transform):

        self.name = name

        self.skew = feature.skew()

        self.log_transform = log_transform

        

    def transform(self, feature):

        if self.skew > 1:

            feature = self.log_transform(feature)

        

        mean = feature.mean()

        std = feature.std()

        return (feature - mean)/(std + 1e-6)        
continuous_features +=['isna_sum']

feature_converters = {}

continuous_features_processed = []

continuous_features_processed_test = []



for f in tqdm(continuous_features):

    feature = X[f]

    feature_test = X_test[f]

    log = lambda x: np.log10(x + 1 - min(0, x.min()))

    converter = ContinuousFeatureConverter(f, feature, log)

    feature_converters[f] = converter

    continuous_features_processed.append(converter.transform(feature))

    continuous_features_processed_test.append(converter.transform(feature_test))

    

continuous_train = pd.DataFrame({s.name: s for s in continuous_features_processed}).astype(np.float32)

continuous_test = pd.DataFrame({s.name: s for s in continuous_features_processed_test}).astype(np.float32)
isna_columns = []

for column in tqdm(continuous_features):

    isna = continuous_train[column].isna()

    if isna.mean() > 0.:

        continuous_train[column + '_isna'] = isna.astype(int)

        continuous_test[column + '_isna'] = continuous_test[column].isna().astype(int)

        isna_columns.append(column)

        

continuous_train = continuous_train.fillna(-1.)

continuous_test = continuous_test.fillna(-1.)
def categorical_encode(df_train, df_test, categorical_features, n_values=140):

    df_train = df_train[categorical_features].astype(str)

    df_test = df_test[categorical_features].astype(str)

    

    categories = []

    for column in tqdm(categorical_features):

        categories.append(list(df_train[column].value_counts().iloc[: n_values - 1].index) + ['Other'])

        values2use = categories[-1]

        df_train[column] = df_train[column].apply(lambda x: x if x in values2use else 'Other')

        df_test[column] = df_test[column].apply(lambda x: x if x in values2use else 'Other')

        

    

    ohe = OneHotEncoder(categories=categories)

    ohe.fit(pd.concat([df_train, df_test]))

    df_train = pd.DataFrame(ohe.transform(df_train).toarray()).astype(np.float16)

    df_test = pd.DataFrame(ohe.transform(df_test).toarray()).astype(np.float16)

    return df_train, df_test
train_categorical, test_categorical = categorical_encode(X, X_test, categorical_features)
num_shape = continuous_train.shape[1]

cat_shape = train_categorical.shape[1]
X = pd.concat([continuous_train, train_categorical], axis=1)

del [[continuous_train, train_categorical]]

continuous_train = train_categorical = None

X_test = pd.concat([continuous_test, test_categorical], axis=1)

del [[continuous_test, test_categorical]]

continuous_test = test_categorical = None
test_rows = X_test.shape[0]

X = pd.concat([X, X_test], axis = 0)

del [[X_test]]

X_test = None

gc.collect()
class roc_callback(Callback):

    def __init__(self,training_data,validation_data):

        self.x = training_data[0]

        self.y = training_data[1]

        self.x_val = validation_data[0]

        self.y_val = validation_data[1]





    def on_train_begin(self, logs={}):

        return



    def on_train_end(self, logs={}):

        return



    def on_epoch_begin(self, epoch, logs={}):

        return



    def on_epoch_end(self, epoch, logs={}):

        y_pred_val = self.model.predict(self.x_val)

        roc_val = roc_auc_score(self.y_val, y_pred_val)

        print('\rroc-auc_val: %s' % (str(round(roc_val,4))),end=100*' '+'\n')

        return



    def on_batch_begin(self, batch, logs={}):

        return



    def on_batch_end(self, batch, logs={}):

        return

    

def focal_loss(gamma=2., alpha=.25):

    def focal_loss_fixed(y_true, y_pred):

        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))

        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(K.epsilon()+pt_1))-K.mean((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))

    return focal_loss_fixed



def custom_gelu(x):

    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))



get_custom_objects().update({'custom_gelu': Activation(custom_gelu)})

get_custom_objects().update({'focal_loss_fn': focal_loss()})
K.clear_session()

from tensorflow.keras.optimizers import Adam

def create_model():

    num_inp = Input(shape=(num_shape,))

    cat_inp = Input(shape=(cat_shape,))

    inps = concatenate([num_inp, cat_inp])

    x = Dense(512, activation=custom_gelu)(inps)

    x = Dense(256, activation=custom_gelu)(x)

    x = Dense(512, activation = custom_gelu)(x)

    x = Dropout(.2)(x)

    cat_out = Dense(cat_shape, activation = 'linear')(x)

    num_out = Dense(num_shape, activation = 'linear')(x)

    model = Model(inputs=[num_inp, cat_inp], outputs=[num_out, cat_out])

    model.compile(

        optimizer=Adam(.05, clipnorm = 1, clipvalue = 1),

        loss=['mse', 'mse']

    )

    return model
gc.collect()

model_mse = create_model()

model_mse.summary()
def inputSwapNoise(arr, p):

    n, m = arr.shape

    idx = range(n)

    swap_n = round(n*p)

    for i in range(m):

        col_vals = np.random.permutation(arr[:, i])

        swap_idx = np.random.choice(idx, size= swap_n)

        arr[swap_idx, i] = np.random.choice(col_vals, size = swap_n) 

    return arr
def auto_generator(X, swap_rate, batch_size):

    indexes = np.arange(X.shape[0])

    while True:

        np.random.shuffle(indexes)

        num_X = X[indexes[:batch_size], :num_shape] 

        num_y = inputSwapNoise(num_X, swap_rate)

        cat_X = X[indexes[:batch_size], num_shape:] 

        cat_y = inputSwapNoise(cat_X, swap_rate)

        yield [num_y, cat_y], [num_X, cat_X]
class WarmUpLearningRateScheduler(tf.keras.callbacks.Callback):

    """Warmup learning rate scheduler

    """



    def __init__(self, warmup_batches, init_lr, verbose=0):

        """Constructor for warmup learning rate scheduler



        Arguments:

            warmup_batches {int} -- Number of batch for warmup.

            init_lr {float} -- Learning rate after warmup.



        Keyword Arguments:

            verbose {int} -- 0: quiet, 1: update messages. (default: {0})

        """



        super(WarmUpLearningRateScheduler, self).__init__()

        self.warmup_batches = warmup_batches

        self.init_lr = init_lr

        self.verbose = verbose

        self.batch_count = 0

        self.learning_rates = []



    def on_batch_end(self, batch, logs=None):

        self.batch_count = self.batch_count + 1

        lr = K.get_value(self.model.optimizer.lr)

        self.learning_rates.append(lr)



    def on_batch_begin(self, batch, logs=None):

        if self.batch_count <= self.warmup_batches:

            lr = self.batch_count*self.init_lr/self.warmup_batches

            K.set_value(self.model.optimizer.lr, lr)

            if self.verbose > 0:

                print('\nBatch %05d: WarmUpLearningRateScheduler setting learning '

                      'rate to %s.' % (self.batch_count + 1, lr))

auto_ckpt = ModelCheckpoint('ae.model', monitor='loss', verbose=verbose, save_best_only=True, save_weights_only=True, mode='min', period=1)

warm_up_lr = WarmUpLearningRateScheduler(400, init_lr=0.001)

train_gen = auto_generator(X.values, .15, batch_size_train)



hist = model_mse.fit_generator(train_gen,

                               steps_per_epoch=len(X)//batch_size_train,

                               epochs=low,

                               verbose=verbose,

                               workers=-1, 

                               use_multiprocessing=True,

                               callbacks=[auto_ckpt, warm_up_lr])
del train_gen

gc.collect()

model_mse.load_weights('ae.model')
for layer in model_mse.layers:

    layer.trainable = False

model_mse.compile(

    optimizer='adam',

    loss=['mse', 'mse']

)



model_mse.summary()
def make_model(loss_fn):

    x1 = model_mse.layers[3].output

    x2 = model_mse.layers[4].output

    x3 = model_mse.layers[5].output

    x_conc = concatenate([x1,x2,x3])

    x = Dropout(.5)(x_conc)

    x = Dense(500, activation='relu')(x)

    x = Dropout(.5)(x)

    x = Dense(200, activation='relu')(x)

    x = Dropout(.5)(x)

    x = Dense(100, activation='relu')(x)

    x = Dropout(.5)(x)

    x = Dense(1, activation = 'sigmoid')(x)

    model = Model([model_mse.layers[0].input, model_mse.layers[1].input], x)

    model.compile(

        optimizer='adam',

        loss=[loss_fn]

    )

    return model
gc.collect()

bce_model = make_model('binary_crossentropy')

fcloss_model = make_model('focal_loss_fn')
X_test = X.iloc[-test_rows:, :]

X      = X.iloc[:-test_rows, :]

gc.collect()
split_ind = int(X.shape[0]*0.8)



X_tr  = X.iloc[:split_ind]

X_val = X.iloc[split_ind:]



y_tr  = y.iloc[:split_ind]

y_val = y.iloc[split_ind:]
ckpt = ModelCheckpoint('best_bce.model', monitor='val_loss', verbose=verbose, save_best_only=True, save_weights_only=True, mode='min', period=1)

bce_model.fit([X_tr.iloc[:, :num_shape], X_tr.iloc[:, num_shape:]], y_tr,

              epochs=high,

              batch_size=batch_size_train,

              validation_data = ([X_val.iloc[:, :num_shape], X_val.iloc[:, num_shape:]], y_val),

              verbose=verbose,

              workers=-1, 

              use_multiprocessing=True,

              callbacks=[ckpt])
valid_preds = bce_model.predict([X_val.iloc[:, :num_shape], X_val.iloc[:, num_shape:]], batch_size = batch_size_pred, verbose = verboseB)

print(f'ROC-AUC score {roc_auc_score(y_val, valid_preds)}')
bce_model.load_weights('best_bce.model')

valid_preds = bce_model.predict([X_val.iloc[:, :num_shape], X_val.iloc[:, num_shape:]], batch_size = batch_size_pred, verbose = verboseB)

print(f'ROC-AUC score {roc_auc_score(y_val, valid_preds)}')
ckpt2 = ModelCheckpoint('best_fcloss.model', monitor='val_loss', verbose=verbose, save_best_only=True, save_weights_only=True, mode='min', period=1)

fcloss_model.fit([X_tr.iloc[:, :num_shape], X_tr.iloc[:, num_shape:]], y_tr,

                 epochs=high,

                 batch_size=batch_size_train,

                 validation_data = ([X_val.iloc[:, :num_shape], X_val.iloc[:, num_shape:]], y_val),

                 verbose = verbose,

                 workers=-1, 

                 use_multiprocessing=True,

                 callbacks=[ckpt2])
gc.collect()

bce_model.load_weights('best_bce.model')

fcloss_model.load_weights('best_fcloss.model')
valid_preds = bce_model.predict([X_val.iloc[:, :num_shape], X_val.iloc[:, num_shape:]], batch_size = batch_size_pred, verbose = verboseB)

print(f'ROC-AUC score {roc_auc_score(y_val, valid_preds)}')
valid_preds  = bce_model.predict([X_val.iloc[:, :num_shape], X_val.iloc[:, num_shape:]], batch_size = batch_size_pred, verbose = verboseB)

valid_preds2 = fcloss_model.predict([X_val.iloc[:, :num_shape], X_val.iloc[:, num_shape:]], batch_size = batch_size_pred, verbose = verboseB)

gc.collect()

score  = roc_auc_score(y_val, valid_preds)

score2 = roc_auc_score(y_val, valid_preds2)

score_avg = roc_auc_score(y_val, (.5*valid_preds) + (.5*valid_preds2))

score_rank_avg = roc_auc_score(y_val, rankdata(valid_preds, method='dense') + rankdata(valid_preds2, method='dense'))

print(f'ROC-AUC score of BCE model {score}')

print(f'ROC-AUC score of Focal Loss model {score2}')

print(f'ROC-AUC score of Average of models {score_avg}')

print(f'ROC-AUC score of Rank Average of models {score_rank_avg}')
bce_model.fit([X.iloc[:, :num_shape], X.iloc[:, num_shape:]], y,

              epochs=low,

              batch_size=batch_size_train,

              verbose = verbose,

              workers=-1, 

              use_multiprocessing=True)



fcloss_model.fit([X.iloc[:, :num_shape], X.iloc[:, num_shape:]], y,

                 epochs=low,

                 batch_size=batch_size_train,

                 verbose = verbose,

                 workers=-1, 

                 use_multiprocessing=True)
test_preds  = bce_model.predict([X_test.iloc[:, :num_shape], X_test.iloc[:, num_shape:]], batch_size = batch_size_pred, verbose=verboseB)

test_preds2 = fcloss_model.predict([X_test.iloc[:, :num_shape], X_test.iloc[:, num_shape:]], batch_size = batch_size_pred, verbose=verboseB)
sub['target'] = rankdata(test_preds, method='dense') + rankdata(test_preds2, method='dense')

sub.target = sub.target/sub.target.max()

sub.to_csv('submission.csv', index=False)