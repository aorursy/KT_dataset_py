import warnings

warnings.simplefilter('ignore')



import numpy as np

import pandas as pd 

import os

from tqdm import tqdm

from sklearn.model_selection import train_test_split,KFold

from sklearn.utils import shuffle

import matplotlib.pyplot as plt

import seaborn as sns



import tensorflow as tf

import tensorflow_addons as tfa



np.random.seed(511)

tf.random.set_seed(511)
train = pd.read_json('../input/stanford-covid-vaccine/train.json',lines=True)

test = pd.read_json('../input/stanford-covid-vaccine/test.json',lines=True)

sample_sub = pd.read_csv('../input/stanford-covid-vaccine/sample_submission.csv')
sample_sub
train.head()
test.head()
def clear_data(df):

    df = df[df['signal_to_noise'] > 1]

    df = df[df['SN_filter'] == 1]

    

    return df
print(len(train))

train = clear_data(train)

print(len(train))
for col_name in ['reactivity_error','deg_error_Mg_pH10','deg_error_Mg_50C']:

    train[f'{col_name}_mean'] = train[col_name].apply(lambda x:np.mean(x))



train[['reactivity_error_mean','deg_error_Mg_pH10_mean','deg_error_Mg_50C_mean']].query('deg_error_Mg_50C_mean >1 or deg_error_Mg_pH10_mean>1 or deg_error_Mg_50C_mean>1')
tokenizer = {x:i for i,x in enumerate('GAUC.()ESHBIXM')}
def make_dataset(df,train=True):

    df['sequence'] = df['sequence'].apply(lambda x: [tokenizer[i] for i in x])

    df['structure'] = df['structure'].apply(lambda x: [tokenizer[i] for i in x])

    df['predicted_loop_type'] = df['predicted_loop_type'].apply(lambda x: [tokenizer[i] for i in x])

    

    X = np.array(df[['sequence','structure','predicted_loop_type']].values.tolist())

    X = np.transpose(X,(0,2,1))

    if not train:

        return X



    if train:

        y = np.array(df[['deg_Mg_pH10','deg_Mg_50C','reactivity']].values.tolist())

        

        

        return X,y

        

        

X,y= make_dataset(train.copy())

y = np.transpose(y,(0,2,1))
fig,ax = plt.subplots(1,3,figsize=(20,8))



sns.distplot(y[0],ax=ax[0])

ax[0].set_title('deg_Mg_pH10')

sns.distplot(y[1],ax=ax[1])

ax[1].set_title('deg_Mg_50C')

sns.distplot(y[2],ax=ax[2])

ax[2].set_title('reactivity')
def make_gru(num_hidden,rate=0.5):

    return tf.keras.layers.Bidirectional(

        tf.keras.layers.GRU(num_hidden,dropout=rate,return_sequences=True)

    )



def make_lstm(num_hidden,rate=0.5):

    return tf.keras.layers.Bidirectional(

        tf.keras.layers.LSTM(num_hidden,dropout=rate,return_sequences=True)

    )



EMB_DIM = 128



def make_model(lstm=True,num_hidden=256,seq_len=107,scored_len=68,spatial_dropout=True):

    inp = tf.keras.layers.Input(shape=(seq_len,3))

    

    embedding = tf.keras.layers.Embedding(len(tokenizer.keys()),EMB_DIM)(inp)

    

    reshaped = tf.keras.backend.reshape(embedding,(-1,seq_len,embedding.shape[2]*embedding.shape[3]))

    

    if spatial_dropout:        

        dropped = tf.keras.layers.SpatialDropout1D(0.2)(reshaped)

    else:

        dropped = tf.keras.layers.Dropout(0.2)(reshaped)

        

    if lstm:

        hs = make_lstm(num_hidden)(dropped)

        hs = make_lstm(num_hidden)(hs)

        hs = make_lstm(num_hidden)(hs)

    else:

        hs = make_gru(num_hidden)(dropped)

        hs = make_gru(num_hidden)(hs)

        hs = make_gru(num_hidden)(hs)

        

    hs = hs[:,:scored_len]

    

    hidden = tfa.layers.WeightNormalization(tf.keras.layers.Dense(512))(hs)

    hidden = tf.keras.layers.BatchNormalization()(hidden)

    hidden = tf.keras.layers.Activation('relu')(hidden)

    hidden = tf.keras.layers.Dropout(0.3)(hidden)

    

    output = tf.keras.layers.Dense(3)(hs)

    

    model = tf.keras.models.Model(inputs=inp,outputs=output)

    

    opt = tfa.optimizers.Lookahead(tf.keras.optimizers.Adam())

    

    model.compile(

        optimizer=opt,

        loss='mse',

        metrics=['mse']

    )

    

    return model

    

model  = make_model()

print(model.summary())
n_folds = 5



cv = KFold(n_folds)

lstm_hist = pd.DataFrame()

lstm_best_score = []

gru_hist = pd.DataFrame()

gru_best_score = []



for index,(train_ids,test_ids) in enumerate(cv.split(X)):

    X_train,X_test,y_train,y_test = X[train_ids],X[test_ids],y[train_ids],y[test_ids]

    

    # ----------------------------- LSTM --------------------------------

    

    tf.keras.backend.clear_session()



    lstm_model = make_model(lstm=True)

    

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau()

    mchpt = tf.keras.callbacks.ModelCheckpoint(

        filepath=f'lstm_{index}_fold.ckpt',

        save_best_only=False,

        save_weights_only=True

    )

    early_stopping = tf.keras.callbacks.EarlyStopping(patience=5)

    

    lstm_h = lstm_model.fit(X_train,y_train,batch_size=64,epochs=90,validation_data=(X_test,y_test),callbacks=[reduce_lr,mchpt,early_stopping])

    

    lstm_hist = lstm_hist.append(lstm_h.history,ignore_index=True)

    lstm_best_score.append(mchpt.best)

    

    # ----------------------------- GRU --------------------------------

    

    tf.keras.backend.clear_session()

    

    gru_model = make_model(lstm=False)

    

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau()

    mchpt = tf.keras.callbacks.ModelCheckpoint(

        filepath=f'gru_{index}_fold.ckpt',

        save_best_only=False,

        save_weights_only=True

    )

    early_stopping = tf.keras.callbacks.EarlyStopping(patience=5)

    

    gru_h = gru_model.fit(X_train,y_train,batch_size=64,epochs=90,validation_data=(X_test,y_test),callbacks=[reduce_lr,mchpt,early_stopping])

    

    gru_hist = gru_hist.append(gru_h.history,ignore_index=True)

    gru_best_score.append(mchpt.best)

    



    
fig,ax = plt.subplots(n_folds,figsize=(20,60))



for i in range(n_folds):

    sns.lineplot(np.arange(len(lstm_hist['val_loss'][i])),lstm_hist['val_loss'][i],ax=ax[i],markers=['o'])



    sns.lineplot(np.arange(len(gru_hist['val_loss'][i])),gru_hist['val_loss'][i],ax=ax[i],markers=['x'])

    

    ax[i].set_title(f' {i} fold val loss')

    ax[i].set_xlabel('epoch')



    ax[i].legend(['LSTM','GRU'])
fig,ax = plt.subplots(n_folds,figsize=(20,60))



for i in range(n_folds):

    sns.lineplot(np.arange(len(lstm_hist['val_mse'][i])),lstm_hist['val_mse'][i],ax=ax[i],markers=['o'])



    sns.lineplot(np.arange(len(gru_hist['val_mse'][i])),gru_hist['val_mse'][i],ax=ax[i],markers=['x'])

    

    ax[i].set_title(f' {i} fold val mse')

    ax[i].set_xlabel('epoch')



    ax[i].legend(['LSTM','GRU'])
fig,ax = plt.subplots(n_folds,figsize=(20,60))



for i in range(n_folds):

    sns.lineplot(np.arange(len(lstm_hist['lr'][i])),lstm_hist['lr'][i],ax=ax[i],markers=['o'])



    sns.lineplot(np.arange(len(gru_hist['lr'][i])),gru_hist['lr'][i],ax=ax[i],markers=['x'])

    

    ax[i].set_title(f' {i} fold lr')

    ax[i].set_xlabel('epoch')



    ax[i].legend(['LSTM','GRU'])
# Let's make submission only on GRU model
lstm_public = make_model(lstm=True,seq_len=107,scored_len = 107)

lstm_private = make_model(lstm=True,seq_len=130,scored_len = 130)



gru_public = make_model(lstm=False,seq_len=107,scored_len = 107)

gru_private = make_model(lstm=False,seq_len=130,scored_len = 130)
public_id,public_dataset =test[test['seq_length'] == 107]['id'], make_dataset(test[test['seq_length'] == 107],train=False)

private_id,private_dataset =test[test['seq_length'] == 130]['id'], make_dataset(test[test['seq_length'] == 130],train=False)
public_preds_lstm = np.zeros((len(public_dataset)*107,3))

private_preds_lstm = np.zeros((len(private_dataset)*130,3))

public_preds_gru =  np.zeros((len(public_dataset)*107,3))

private_preds_gru = np.zeros((len(private_dataset)*130,3))



for i in range(n_folds):

    

    lstm_public.load_weights(f'lstm_{i}_fold.ckpt')

    gru_public.load_weights(f'gru_{i}_fold.ckpt')

    

    public_preds_lstm +=lstm_public.predict(public_dataset).reshape(-1,3)

    public_preds_gru +=gru_public.predict(public_dataset).reshape(-1,3)

    

    lstm_private.load_weights(f'lstm_{i}_fold.ckpt')

    gru_private.load_weights(f'gru_{i}_fold.ckpt')

    

    private_preds_lstm +=lstm_private.predict(private_dataset).reshape(-1,3)

    private_preds_gru +=gru_private.predict(private_dataset).reshape(-1,3)

    

    

    
public_preds_lstm/=5.

private_preds_lstm/=5.

public_preds_gru/=5.

private_preds_gru/=5.



public_preds = .0*public_preds_lstm + 1.*public_preds_gru

private_preds = .0*private_preds_lstm + 1.*private_preds_gru
public_id_seqpos = []



for i in public_id:

    for j in range(107):

        public_id_seqpos.append(i+f"_{j}")

        

private_id_seqpos = []



for i in private_id:

    for j in range(130):

        private_id_seqpos.append(i+f"_{j}")
public_preds = pd.DataFrame({'id_seqpos': public_id_seqpos, 'deg_Mg_pH10' : public_preds[:,0],'deg_Mg_50C': public_preds[:,1],'reactivity': public_preds[:,2]})

private_preds = pd.DataFrame({'id_seqpos': private_id_seqpos, 'deg_Mg_pH10' : private_preds[:,0],'deg_Mg_50C': private_preds[:,1],'reactivity': private_preds[:,2]})
preds = pd.concat([public_preds,private_preds])
my_sub = sample_sub[['id_seqpos','deg_pH10','deg_50C']].merge(preds,on=['id_seqpos'])
my_sub.to_csv('submission.csv',index=False)