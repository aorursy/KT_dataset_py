%matplotlib inline
import os
import gc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

DATA_DIR = '../input/avito-demand-prediction/'
EMB_PATH = '../input/fasttest-common-crawl-russian/cc.ru.300.vec'
#EMB_PATH = '../input/fasttext-russian-2m/wiki.ru.vec'
#EMB_PATH = '../input/russian-glove/multilingual_embeddings.ru'
target_col = 'deal_probability'
max_features = 50000
maxlen = 100
embed_size = 300
os.listdir(DATA_DIR)
usecols = ['region', 'city', 'parent_category_name', 'category_name', 'param_1',
           'param_2', 'param_3', 'title', 'description', 'price', 'activation_date',
           'item_seq_number', 'user_type', 'image_top_1']
train = pd.read_csv(DATA_DIR+'train.csv', usecols=usecols+[target_col])
test = pd.read_csv(DATA_DIR+'test.csv', usecols=usecols)
train = train[train['activation_date']<'2017-03-29']
test.loc[test['activation_date']>'2017-04-18', 'activation_date'] = '2017-04-18'
train['activation_date'].min(), train['activation_date'].max()
test['activation_date'].min(), test['activation_date'].max()
%%time
train['na_cnt'] = train.isnull().sum(axis=1)
train['na_price'] = train['price'].isnull().astype('int8')
test['na_cnt'] = test.isnull().sum(axis=1)
test['na_price'] = test['price'].isnull().astype('int8')
train['item_seq_number_log1p'] = train['item_seq_number'].apply(lambda x:np.log1p(x))
test['item_seq_number_log1p'] = test['item_seq_number'].apply(lambda x:np.log1p(x))
train['image_top_1'] = train['image_top_1'].fillna(train['image_top_1'].max()+1)
test['image_top_1'] = test['image_top_1'].fillna(test['image_top_1'].max()+1)

size_cols  = ['region', 'city', 'parent_category_name', 'category_name', 'image_top_1']
mean_cols = ['parent_category_name', 'category_name', 'image_top_1']

def add_group_size(df, by, y='price'):
    grp = df.groupby(by)[y].size().map(lambda x:np.log1p(x))
    grp = grp.rename('size_'+'_'.join(by)).reset_index()
    df = df.merge(grp, on=by, how='left')
    return df

def add_group_mean(df, by, y='price'):
    grp = df.groupby(by)[y].mean()
    grp = grp.rename('mean_price_'+'_'.join(by)).reset_index()
    df = df.merge(grp, on=by, how='left')
    return df

for c in size_cols:
    print('adding size by', [c, 'activation_date'])
    train = add_group_size(train, [c, 'activation_date'])
    test = add_group_size(test, [c, 'activation_date'])
    
for c in mean_cols:
    print('adding mean price by', (c,))
    train = add_group_mean(train, (c,))
    test = add_group_mean(test, (c,))
    
del train['activation_date'], test['activation_date']; gc.collect()
%%time
size_cols = [c for c in train.columns if 'size_' in c]
mean_cols = [c for c in train.columns if 'mean_' in c]

for c in ['price'] + mean_cols:
    train[c] = train[c].fillna(-1)#-1
    test[c] = test[c].fillna(-1)#-1
    train[c] = train[c].apply(lambda x:np.log1p(x))
    test[c] = test[c].apply(lambda x:np.log1p(x))

train = train.fillna('неизвестный')
test = test.fillna('неизвестный')
y = train[target_col].values
del train[target_col]; gc.collect()
train_num = len(train)

str_cols = [c for c in train.columns if c not in [
    'na_price', 'na_cnt', 'item_seq_number_log1p',
    'price', 'item_seq_number', 'image_top_1', 'user_type', target_col] + size_cols + mean_cols
           ]

cat_cols = ['region', 'city', 'parent_category_name', 'category_name', 'param_1', 'param_2', 'param_3', 
            'item_seq_number', 'user_type', 'image_top_1']

df = pd.concat([train, test], ignore_index=True)
del train, test; gc.collect()

df['text'] = ''
for i, c in enumerate(df.columns):
    if c in str_cols:
        df['text'] += ' ' + df[c].str.lower()
        if c not in cat_cols:
            del df[c]; gc.collect()
    if c in cat_cols:
        df[c] = pd.factorize(df[c])[0]
for c in cat_cols:
    print(c, df[c].min(), df[c].max())
print(df.info())
for c in cat_cols+['na_cnt']:
    if df[c].max()<2**7:
        df[c] = df[c].astype('int8')
    elif df[c].max()<2**15:
        df[c] = df[c].astype('int16')
    elif df[c].max()<2**31:
        df[c] = df[c].astype('int32')
    else:
        continue
print(df.info())
df['char_len'] = df['text'].apply(lambda x:np.log1p(len(x)))
df['char_len'] = df['char_len'].astype('float32')
df['word_len'] = df['text'].apply(lambda x:np.log1p(len(x.split(' '))))
df['word_len'] = df['word_len'].astype('float32')
df['char_len'].hist()
df['word_len'].hist()
df.head(3).T
df.describe().T
%%time
from keras.preprocessing import text, sequence
print('tokenizing...')
tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(df['text'].values.tolist())
%%time
print('getting embeddings')
def get_coefs(word, *arr, tokenizer=None):
    if tokenizer is None:
        return word, np.asarray(arr, dtype='float32')
    else:
        if word not in tokenizer.word_index:
            return None
        else:
            return word, np.asarray(arr, dtype='float32')
nb_words = min(max_features, len(tokenizer.word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for o in tqdm(open(EMB_PATH)):
    res = get_coefs(*o.rstrip().rsplit(' '), tokenizer=tokenizer)
    if res is not None:
        idx = tokenizer.word_index[res[0]]
        if idx < max_features:
            embedding_matrix[idx] = res[1]
gc.collect()
%%time
def fill_rand_norm(embedding_matrix):
    emb_zero_shape = embedding_matrix[embedding_matrix==0].shape
    emb_non_zero_mean = embedding_matrix[embedding_matrix!=0.].mean()
    emb_non_zero_std = embedding_matrix[embedding_matrix!=0.].std()
    embedding_matrix[embedding_matrix==0] = np.random.normal(emb_non_zero_mean, 
                                                             emb_non_zero_std, 
                                                             emb_zero_shape)
    return embedding_matrix
embedding_matrix = fill_rand_norm(embedding_matrix)
%%time
text = df['text'].values
del df['text']; gc.collect()
text = tokenizer.texts_to_sequences(text)
text = sequence.pad_sequences(text, maxlen=maxlen)
%%time
df_train = df[:train_num]
text_train = text[:train_num]
df_test = df[train_num:]
text_test = text[train_num:]
del text, df; gc.collect()
%%time
def get_keras_data(df, text):
    X = {}
    for c in df.columns:
        X[c] = df[c].values
    X['text'] = text
    return X
X_train = get_keras_data(df_train, text_train)
X_test = get_keras_data(df_test, text_test)
del df_train, text_train, df_test, text_test; gc.collect()
import keras.backend as K
from keras.layers import Input, Embedding, Dense, Flatten, Dropout, concatenate
from keras.layers import \
    BatchNormalization, SpatialDropout1D, GlobalAveragePooling1D, GRU, Bidirectional, GlobalMaxPooling1D, Conv1D
from keras.layers import CuDNNGRU
from keras.layers import add, dot
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from itertools import combinations

from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints

"""
from: https://www.kaggle.com/sermakarevich/hierarchical-attention-network
"""
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
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
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

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
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

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]
    
emb_cat_max = {}
for c in cat_cols:
    emb_cat_max[c] = max(X_train[c].max(), X_test[c].max())+1
params = {}
params['maxlen'] = maxlen
params['nb_words'] = nb_words
params['embedding_matrix'] = embedding_matrix
params['word_emb_size'] = embed_size
params['text_emb_dropout'] = 0.2
params['n_rnn'] = 64

params['emb_cat_max'] = emb_cat_max
params['emb_size'] = 32
params['n_output'] = 1
params['use_fm'] = True
params['use_deep'] = True
params['use_rnn'] = True
params['use_cnn'] = True
params['use_att'] = False
params['use_batch_norm'] = False #replace dropout """loss: inf ???"""
params['cnn_param'] = dict(filters=192, kernel_size=3)
params['deep_layers'] = [256, 256, 256]
params['drop_out'] = [0.5, 0.5, 0.5]
params['output_drop_out'] = 0.2
assert len(params['drop_out'])==len(params['deep_layers'])
params['cat_feats'] = cat_cols
params['num_feats'] = size_cols+mean_cols+['price', 'char_len', 'word_len', 'na_price', 'na_cnt', 'item_seq_number_log1p']
params['lr'] = 0.001
params['decay'] = 1e-6
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true))) 

"""
MIXED ARCH NN
"""
def get_model(params=params):
    cats = [Input(shape=[1], name=name) for name in params['cat_feats']]
    nums = [Input(shape=[1], name=name) for name in params['num_feats']]
    emb_fn = lambda name: Embedding(params['emb_cat_max'][name], params['emb_size'])
    embs = []
    for name, cat in zip(params['cat_feats'], cats):
        embs.append(emb_fn(name)(cat))
    
    texts = Input(shape=(params['maxlen'],), name='text')
    text_emb = Embedding(params['nb_words'], 
                         params['word_emb_size'], 
                         weights=[params['embedding_matrix']],
                         name='text_emb')(texts)
    
    outs = []
    if params['use_rnn']:
        x_text = SpatialDropout1D(params['text_emb_dropout'])(text_emb)
        x_rnn = Bidirectional(GRU(params['n_rnn'], return_sequences=True))(x_text)
        avg_pool_rnn = GlobalAveragePooling1D()(x_rnn)
        max_pool_rnn = GlobalMaxPooling1D()(x_rnn)
        outs += [avg_pool_rnn, max_pool_rnn]
    if params['use_cnn']:
        x_text = SpatialDropout1D(params['text_emb_dropout'])(text_emb)
        x_cnn = Conv1D(**params['cnn_param'])(x_text)
        avg_pool_cnn = GlobalAveragePooling1D()(x_cnn)
        max_pool_cnn = GlobalMaxPooling1D()(x_cnn)
        outs += [avg_pool_cnn, max_pool_cnn]
    if params['use_att']:
        x_text = SpatialDropout1D(params['text_emb_dropout'])(text_emb)
        x_att = AttentionWithContext()(x_text)
        outs += [x_att]
        if params['use_rnn']:
            x_rnn_att = AttentionWithContext()(x_rnn)
            outs += [x_rnn_att]
        if params['use_cnn']:
            x_cnn_att = AttentionWithContext()(x_cnn)
            outs += [x_cnn_att]
    if params['use_fm']:
        first_order = [Flatten()(emb0) for emb0 in embs]
        second_order = []
        for emb1, emb2 in combinations(embs, 2):
            dot_layer = dot([Flatten()(emb1), Flatten()(emb2)], axes=1)
            second_order.append(dot_layer)
        first_order = add(first_order)
        second_order = add(second_order)
        outs += [first_order, second_order]
    if params['use_deep']:
        all_in = [Flatten()(emb) for emb in embs] + nums + [Flatten()(text_emb)]
        x_in = concatenate(all_in)
        for idx, (drop_p, num_dense) in enumerate(zip(params['drop_out'], params['deep_layers'])):
            x_in = Dense(num_dense, activation='relu')(x_in)
            if params['use_batch_norm']:
                x_in = (BatchNormalization())(x_in)
            else:
                x_in = Dropout(drop_p)(x_in)
        deep = x_in
        outs += [deep]
        
    total_out = concatenate(outs) if len(outs)>1 else outs[0]
    
    if 0<params['output_drop_out']<1:
        total_out = Dropout(params['output_drop_out'])(total_out)
    #output = Dense(params['n_output'], activation='linear')(total_out)
    output = Dense(params['n_output'], activation='sigmoid')(total_out)
    model = Model(inputs=cats+nums+[texts], output=output)
    optimizer = Adam(lr=params['lr'], decay=params['decay'])
    model.compile(loss=root_mean_squared_error, #mean_squared_error, mean_absolute_error
                  optimizer=optimizer,
                  metrics=[root_mean_squared_error])
    return model

model = get_model()
#for k, v in params.items(): print(k, v)
file_path = "model.hdf5"
check_point = ModelCheckpoint(file_path, monitor='val_root_mean_squared_error', mode='min', 
                              save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_root_mean_squared_error', patience=2, mode='min')

batch_size = 2**10 * 2
epochs = 10

sample_weight = np.ones(y.shape)
sample_weight[y<1e-7] = 1 + len(y[y<1e-7])/len(y)
history = model.fit(X_train, y, sample_weight=sample_weight,
                    batch_size=batch_size, epochs=epochs, 
                    validation_split=0.05, verbose=1, 
                    callbacks=[check_point, early_stop])
model.load_weights(file_path)
pred = model.predict(X_test, batch_size=batch_size, verbose=1)
print('train', y.mean(), y.std())
print('pred', pred.mean(), pred.std())
plt.figure()
sns.distplot(y)
sns.distplot(pred)
plt.legend(['train', 'pred'])
plt.show()
sub = pd.read_csv(DATA_DIR+'sample_submission.csv')
sub[target_col] = pred
sub.head()
scr = min(history.history['val_root_mean_squared_error'])
print('save to '+f'nn_{scr}.csv')
sub.to_csv(f'nn_{scr}.csv', index=False)