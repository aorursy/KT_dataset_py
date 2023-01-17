import numpy as np
import pandas as pd
import gc
import os
print(os.listdir("../input"))
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

df_y_train = df_train['deal_probability']
df_x_train = df_train.drop(['deal_probability'], axis=1)
df_x_train.head(10)
df_x_train.info()
df_test.info()
# viewing # of unique value in each column 
for col in df_x_train.columns:
    print(col, len(df_x_train[col].unique()))
df_x_train['image_top_1'].fillna(value=3067, inplace=True)
df_test['image_top_1'].fillna(value=3067, inplace=True)
df_x_train['param_1'].fillna(value='_NA_', inplace=True)
df_test['param_1'].fillna(value='_NA_', inplace=True)

df_x_train['param_2'].fillna(value='_NA_', inplace=True)
df_test['param_2'].fillna(value='_NA_', inplace=True)

df_x_train['param_3'].fillna(value='_NA_', inplace=True)
df_test['param_3'].fillna(value='_NA_', inplace=True)
df_x_train['description'].fillna(value='_NA_', inplace=True)
df_test['description'].fillna(value='_NA_', inplace=True)
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import argparse

#create config init
config = argparse.Namespace()
def tknzr_fit(col, df_trn, df_test):
    tknzr = Tokenizer(filters='', lower=False, split='뷁', oov_token='oov' )
    tknzr.fit_on_texts(df_trn[col])
    return np.array(tknzr.texts_to_sequences(df_trn[col])), np.array(tknzr.texts_to_sequences(df_test[col])), tknzr
tr_reg, te_reg, tknzr_reg = tknzr_fit('region', df_x_train, df_test)
tr_pcn, te_pcn, tknzr_pcn = tknzr_fit('parent_category_name', df_x_train, df_test)
tr_cn, te_cn, tknzr_cn = tknzr_fit('category_name', df_x_train, df_test)
tr_ut, te_ut, tknzr_ut = tknzr_fit('user_type', df_x_train, df_test)
tr_city, te_city, tknzr_city = tknzr_fit('city', df_x_train, df_test)

tr_p1, te_p1, tknzr_p1 = tknzr_fit('param_1', df_x_train, df_test)
tr_p2, te_p2, tknzr_p2 = tknzr_fit('param_2', df_x_train, df_test)
tr_p3, te_p3, tknzr_p3 = tknzr_fit('param_3', df_x_train, df_test)
tr_week = pd.to_datetime(df_x_train['activation_date']).dt.weekday.astype(np.int32).values
te_week = pd.to_datetime(df_test['activation_date']).dt.weekday.astype(np.int32).values
tr_week = np.expand_dims(tr_week, axis=-1)
te_week = np.expand_dims(te_week, axis=-1)
tr_imgt1 = df_x_train['image_top_1'].astype(np.int32).values
te_imgt1 = df_test['image_top_1'].astype(np.int32).values
tr_imgt1 = np.expand_dims(tr_imgt1, axis=-1)
te_imgt1 = np.expand_dims(te_imgt1, axis=-1)
eps = 1e-10
tr_price = np.log(df_x_train['price']+eps)
te_price = np.log(df_test['price']+eps)
tr_price[tr_price.isna()] = -1.
te_price[te_price.isna()] = -1.

tr_price = np.expand_dims(tr_price, axis=-1)
te_price = np.expand_dims(te_price, axis=-1)
tr_itemseq = np.log(df_x_train['item_seq_number'])
te_itemseq = np.log(df_test['item_seq_number'])
# price_tr[price_tr.isna()] = -1.
# price_te[price_te.isna()] = -1.

tr_itemseq = np.expand_dims(tr_itemseq, axis=-1)
te_itemseq = np.expand_dims(te_itemseq, axis=-1)
config.len_desc = 100000
from keras.preprocessing.sequence import pad_sequences
tknzr_desc = Tokenizer(num_words=config.len_desc, lower='True')
tknzr_desc.fit_on_texts(df_x_train['description'].values)
tr_desc_seq = tknzr_desc.texts_to_sequences(df_x_train['description'].values)
te_desc_seq = tknzr_desc.texts_to_sequences(df_test['description'].values)
config.maxlen= 75
tr_desc_pad = pad_sequences(tr_desc_seq, maxlen=config.maxlen)
te_desc_pad = pad_sequences(te_desc_seq, maxlen=config.maxlen)
gc.collect()
## categorical
config.len_reg = len(tknzr_reg.word_index)
config.len_pcn = len(tknzr_pcn.word_index)
config.len_cn = len(tknzr_cn.word_index) 
config.len_ut = len(tknzr_ut.word_index)
config.len_city = len(tknzr_city.word_index) +1
config.len_week = 7
config.len_imgt1 = int(df_x_train['image_top_1'].max())+1
config.len_p1 = len(tknzr_p1.word_index)+1
config.len_p2 = len(tknzr_p2.word_index)+1
config.len_p3 = len(tknzr_p3.word_index)+1

## continuous
config.len_price = 1
config.len_itemseq = 1

#text
# config.len_desc = len(tknzr_desc.word_index)
## categorical
config.emb_reg = 8
config.emb_pcn = 4
config.emb_cn = 8
config.emb_ut = 2
config.emb_city = 16
config.emb_week = 4
config.emb_imgt1 = 16
config.emb_p1 = 8
config.emb_p2 = 16
config.emb_p3 = 16

#continuous
config.emb_price = 16
config.emb_itemseq = 16

#text
config.emb_desc = 100
valid_idx = df_y_train.sample(frac=0.2, random_state=1991).index
train_idx = df_y_train[np.invert(df_y_train.index.isin(valid_idx))].index
X = np.array([tr_reg, tr_pcn, tr_cn, tr_ut, tr_city, tr_week, tr_imgt1, tr_p1, tr_p2, tr_p3,
              tr_price, tr_itemseq])
X_test = np.array([te_reg, te_pcn, te_cn, te_ut, te_city, te_week, te_imgt1, te_p1, te_p2, te_p3,
                   te_price, te_itemseq])
Y = df_y_train
X_train = [x[train_idx] for x in X]
X_valid = [x[valid_idx] for x in X]
X_test = [x for x in X_test]

Y_train = Y[train_idx]
Y_valid = Y[valid_idx]
X_train.append(tr_desc_pad[train_idx])
X_valid.append(tr_desc_pad[valid_idx])
X_test.append(te_desc_pad)
gc.collect()
from keras.layers import Input, Embedding, Dense
from keras.layers import GlobalMaxPool1D, GlobalMaxPool2D
from keras.layers import concatenate
from keras.layers import LSTM, CuDNNGRU, CuDNNLSTM, GRU
from keras.models import Model

from keras import backend as K

from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

### rmse loss for keras
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true))) 
config.batch_size = 4096
def get_model():
    K.clear_session()
    inp_reg = Input(shape=(1, ), name='inp_region')
    emb_reg = Embedding(config.len_reg, config.emb_reg, name='emb_region')(inp_reg)
    
    inp_pcn = Input(shape=(1, ), name='inp_parent_category_name')
    emb_pcn = Embedding(config.len_pcn, config.emb_pcn, name='emb_parent_category_name')(inp_pcn)

    inp_cn = Input(shape=(1, ), name='inp_category_name')
    emb_cn = Embedding(config.len_cn, config.emb_cn, name="emb_category_name" )(inp_cn)
    
    inp_ut = Input(shape=(1, ), name='inp_user_type')
    emb_ut = Embedding(config.len_ut, config.emb_ut, name='emb_user_type' )(inp_ut)
    
    inp_city = Input(shape=(1, ), name='inp_city')
    emb_city = Embedding(config.len_city, config.emb_city, name='emb_city' )(inp_city)

    inp_week = Input(shape=(1, ), name='inp_week')
    emb_week = Embedding(config.len_week, config.emb_week, name='emb_week' )(inp_week)

    inp_imgt1 = Input(shape=(1, ), name='inp_imgt1')
    emb_imgt1 = Embedding(config.len_imgt1, config.emb_imgt1, name='emb_imgt1')(inp_imgt1)
    
    inp_p1 = Input(shape=(1, ), name='inp_p1')
    emb_p1 = Embedding(config.len_p1, config.emb_p1, name='emb_p1')(inp_p1)
    
    inp_p2 = Input(shape=(1, ), name='inp_p2')
    emb_p2 = Embedding(config.len_p2, config.emb_p2, name='emb_p2')(inp_p2)
    
    inp_p3 = Input(shape=(1, ), name='inp_p3')
    emb_p3 = Embedding(config.len_p3, config.emb_p3, name='emb_p3')(inp_p3)
    
    conc_cate = concatenate([emb_reg, emb_pcn,  emb_cn, emb_ut, emb_city, emb_week, emb_imgt1, emb_p1, emb_p2, emb_p3], axis=-1, name='concat_categorcal_vars')
    conc_cate = GlobalMaxPool1D()(conc_cate)
    
    inp_price = Input(shape=(1, ), name='inp_price')
    emb_price = Dense(config.emb_price, activation='tanh', name='emb_price')(inp_price)

    inp_itemseq = Input(shape=(1, ), name='inp_itemseq')
    emb_itemseq = Dense(config.emb_itemseq, activation='tanh', name='emb_itemseq')(inp_itemseq)
    
    conc_cont = concatenate([conc_cate, emb_price, emb_itemseq], axis=-1)
    x = Dense(200, activation='relu')(conc_cont)
    x = Dense(50, activation='relu')(x)

    ### text
    inp_desc = Input(shape=(config.maxlen, ), name='inp_desc')
    emb_desc = Embedding(config.len_desc, config.emb_desc, name='emb_desc')(inp_desc)
    
    desc_layer = GRU(40, return_sequences=False)(emb_desc)
    
    conc_desc = concatenate([x, desc_layer], axis=-1)
    ###

    outp = Dense(1, activation='sigmoid', name='output')(conc_desc)

    model = Model(inputs = [inp_reg, inp_pcn, inp_cn, inp_ut, inp_city, inp_week, inp_imgt1, inp_p1, inp_p2, inp_p3,
                            inp_price, inp_itemseq, inp_desc], outputs = outp)
    return model
model = get_model()
model.compile(optimizer='adam', loss = root_mean_squared_error, metrics=[root_mean_squared_error])
# model.compile(optimizer=RMSprop(lr=0.0005, decay=0.00001), loss = root_mean_squared_error, metrics=['mse', root_mean_squared_error])
model.summary()
### callbacks
checkpoint = ModelCheckpoint('best.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
early = EarlyStopping(patience=3, mode='min')
model.fit(x=X_train, y=np.array(Y_train), validation_data=(X_valid, Y_valid), batch_size=config.batch_size, epochs=6, callbacks=[checkpoint,early], verbose=1)
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))
model.load_weights('best.hdf5')
pred = model.predict(X_test)

subm = pd.read_csv("../input/sample_submission.csv")
subm['deal_probability'] = pred
subm.to_csv('submit_{}_{:.4f}.csv'.format('nn_p3', 0.226), index=False)
