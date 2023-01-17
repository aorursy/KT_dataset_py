import numpy as np

import pandas as pd

import itertools

import gc



import tensorflow as tf
# detect and init the TPU

tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

tf.config.experimental_connect_to_cluster(tpu)

tf.tpu.experimental.initialize_tpu_system(tpu)



# instantiate a distribution strategy

tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
train = pd.read_json('/kaggle/input/stanford-covid-vaccine/train.json', lines=True)

test = pd.read_json('/kaggle/input/stanford-covid-vaccine/test.json', lines=True)

sample_sub = pd.read_csv('/kaggle/input/stanford-covid-vaccine/sample_submission.csv')
test.sample(3)
def get_encoder():

    seq = list('AGUC')

    stru = list('.()')

    loop = list('SEHIXMB')

    encoder = dict()

    for i, prod in enumerate(itertools.product(seq, stru, loop)):

        concat_prod = prod[0] + prod[1] + prod[2]

        encoder[concat_prod] = i

    return encoder    



def get_bpp_seqs(id_seqs):

    bpp = [np.load(f'../input/stanford-covid-vaccine/bpps/{seq}.npy') 

           for seq in id_seqs]

    # get quantiles

    bpp = [(1 - np.floor(bi.sum(axis=1)) * 100) for bi in bpp]

    return np.array(bpp)



def preprocess_train(df_train, cols=['sequence', 'structure', 'predicted_loop_type'],

                     targets=['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']):

    # transform into list

    vecs = np.array(df_train[cols].applymap(list).values.tolist()).transpose(0, 2, 1)

    # concatenate all 3 positions

    seqs = np.array([[vii[0] + vii[1] + vii[2] for vii in vi]

                     for vi in vecs])

    # encode

    assert df_train.seq_length.max() == df_train.seq_length.min()

    seq_len = df_train.seq_length.max()

    X = np.array([[encoder[sii] for sii in si] for si in seqs]).reshape(-1, seq_len, 1)

    

    # bpp

    Xbpp = get_bpp_seqs(df_train.id.values)

    

    # target

    y = np.array(df_train[targets].values.tolist()).transpose(0, 2, 1)

    

    return X, Xbpp, y, encoder



def preprocess_test(df_test, cols=['sequence', 'structure', 'predicted_loop_type']):

    # transform into list

    vecs = np.array(df_test[cols].applymap(list).values.tolist()).transpose(0, 2, 1)

    # concatenate all 3 positions

    seqs = np.array([[vii[0] + vii[1] + vii[2] for vii in vi]

                     for vi in vecs])

    # encode

    assert df_test.seq_length.max() == df_test.seq_length.min()

    seq_len = df_test.seq_length.max()

    X = np.array([[encoder[sii] for sii in si] for si in seqs]).reshape(-1, seq_len, 1)



    # bpp

    Xbpp = get_bpp_seqs(df_test.id.values)

    

    return X, Xbpp
# preprocess

encoder = get_encoder()

X_train, Xbpp_train, y_train, encoder = preprocess_train(train.loc[train.signal_to_noise >= 1])

X_test_pub, Xbpp_test_pub = preprocess_test(test.query("seq_length == 107"))

X_test_pvt, Xbpp_test_pvt = preprocess_test(test.query("seq_length == 130"))
# https://www.kaggle.com/c/stanford-covid-vaccine/discussion/183211

def mcrmse(y_true, y_pred):

    colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)

    return tf.reduce_mean(tf.sqrt(colwise_mse), axis=1)
# adapted from https://levelup.gitconnected.com/building-seq2seq-lstm-with-luong-attention-in-keras-for-time-series-forecasting-1ee00958decb

def get_model(seq_len=107, pred_len=68, loss='mse', opt='adam', emb_dim_seq=32, emb_dim_bpp=32):

    # input concat sequences

    input_seq = tf.keras.Input(shape=(seq_len, 1))

    

    # embedding

    x_seq = tf.keras.layers.Embedding(input_dim=len(encoder), output_dim=emb_dim_seq, input_length=seq_len)(input_seq)

    x_seq = tf.keras.layers.Reshape((x_seq.shape[1], x_seq.shape[2] * x_seq.shape[3]))(x_seq)

    x_seq = tf.keras.layers.SpatialDropout1D(0.2)(x_seq)

    # lstm

    x_seq = tf.keras.layers.Bidirectional(

                tf.keras.layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2, 

                                     return_sequences=True))(x_seq)

    x_seq = tf.keras.layers.Bidirectional(

                tf.keras.layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2, 

                                     return_sequences=True))(x_seq)

    x_seq = tf.keras.layers.Attention()([x_seq, x_seq])

    

    

    # bpp

    input_bpp = tf.keras.Input(shape=(seq_len, 1))

    

    # embedding

    x_bpp = tf.keras.layers.Embedding(input_dim=100, output_dim=emb_dim_bpp, input_length=seq_len)(input_bpp)

    x_bpp = tf.keras.layers.Reshape((x_bpp.shape[1], x_bpp.shape[2] * x_bpp.shape[3]))(x_bpp)

    x_bpp = tf.keras.layers.SpatialDropout1D(0.2)(x_bpp)

    

    # lstm

    x_bpp = tf.keras.layers.Bidirectional(

                tf.keras.layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2,

                                     return_sequences=True))(x_bpp)

    x_bpp = tf.keras.layers.Bidirectional(

                tf.keras.layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2, 

                                     return_sequences=True))(x_bpp)

    

    x_bpp = tf.keras.layers.Attention()([x_bpp, x_bpp])

    

    # concat

    x = tf.keras.layers.Concatenate(axis=-1)([x_seq, x_bpp])

    

    # truncate

    x = x[:, :pred_len, :]

    

    # dense

    x = tf.keras.layers.Dropout(0.3)(x)

    

    x = tf.keras.layers.Dense(32)(x)

    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Dropout(0.3)(x)

    

    x = tf.keras.layers.Dense(5, activation='linear')(x)

    

    model = tf.keras.Model(inputs=[input_seq, input_bpp], outputs=x)

    

    model.compile(optimizer=opt, loss=loss, metrics=[mcrmse])

    return model

    

    
get_model().summary()
from sklearn.model_selection import KFold
# params

n_folds = 5

seed = 12

epochs = 50

batch_size = 32

loss = mcrmse

emb_dim_seq = 64

emb_dim_bpp = 64



# placeholders

models = list()

val_metrics = list()

histories = list()

public_preds = np.zeros((test.query("seq_length == 107").shape[0], 107, 5))

private_preds = np.zeros((test.query("seq_length == 130").shape[0], 130, 5))





# folds

folds = KFold(n_folds, shuffle=True, random_state=seed).split(X_train, y_train)

for fold, (train_ix, test_ix) in enumerate(folds):

    print('-'*30, f'Fold {fold+1}/{n_folds}', '-'*30, '\n\n')

    tf.keras.backend.clear_session()

    

    # splits data

    xtrain_fold, xbpp_train_fold, y_train_fold = X_train[train_ix], Xbpp_train[train_ix], y_train[train_ix]

    xval_fold, xbpp_val_fold, y_val_fold = X_train[test_ix], Xbpp_train[test_ix], y_train[test_ix]

    # model

    opt = tf.keras.optimizers.Adam(learning_rate=0.01)

    with tpu_strategy.scope():

        m = get_model(emb_dim_seq=emb_dim_seq, emb_dim_bpp=emb_dim_bpp, opt=opt, loss=loss)

        # callback

        lr_callback = tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.2, verbose=1,

                                                           min_delta=0.007, monitor='val_loss')

        ckpt = tf.keras.callbacks.ModelCheckpoint(f'model-{fold}.h5')

    # fit

    gc.collect()

    h = m.fit((xtrain_fold, xbpp_train_fold), y_train_fold, 

              validation_data=((xval_fold, xbpp_val_fold), y_val_fold),

              epochs=epochs, batch_size=batch_size,

              callbacks=[lr_callback, ckpt])

    # save

    histories.append(h)

    

    # predict test

    model_short = get_model(seq_len=107, pred_len=107, emb_dim_seq=emb_dim_seq, 

                            emb_dim_bpp=emb_dim_bpp, opt=opt, loss=loss)

    model_short.load_weights(f'model-{fold}.h5')

    model_public_pred = model_short.predict((X_test_pub, Xbpp_test_pub)) / (n_folds)



    model_long = get_model(seq_len=130, pred_len=130, emb_dim_seq=emb_dim_seq, 

                           emb_dim_bpp=emb_dim_bpp, opt=opt, loss=loss)

    model_long.load_weights(f'model-{fold}.h5')

    model_private_pred = model_long.predict((X_test_pvt, Xbpp_test_pvt)) / (n_folds)



    public_preds += model_public_pred

    private_preds += model_private_pred

print(f" CV-mean fold loss: {np.mean([min(history.history['val_loss']) for history in histories])}")
# https://www.kaggle.com/tuckerarrants/openvaccine-gru-lstm



targets = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']

def format_predictions(public_preds, private_preds):

    preds = []

    

    for df, preds_ in [(test.query("seq_length == 107"), public_preds), 

                       (test.query("seq_length == 130"), private_preds)]:

        for i, uid in enumerate(df.id):

            single_pred = preds_[i]



            single_df = pd.DataFrame(single_pred, columns=targets)

            single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]



            preds.append(single_df)



    return pd.concat(preds)
preds = format_predictions(public_preds, private_preds)

submission = sample_sub[['id_seqpos']].merge(preds, on=['id_seqpos'])

submission.head()
submission.to_csv('submission.csv', index=False)