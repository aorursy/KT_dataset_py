import numpy as np

import pandas as pd



import tensorflow as tf

import tensorflow_addons as tfa



import gc



tfk = tf.keras

tfkl = tfk.layers

K = tfk.backend
train = pd.read_json('../input/stanford-covid-vaccine/train.json', lines=True)

train = train.loc[train.SN_filter >= 1]



test = pd.read_json('../input/stanford-covid-vaccine/test.json', lines=True)
train.head()
# 5 feature maps (1 seq paired, 1 seq unpaired, 1 loop paired, 1 loop unpaired, 1 structure)

# e.g. sequence paired features will have the corresponding code where there is a pair and '-' (fillna) where there is not.

fillna = '-'

encoder = {v: i for i, v in enumerate('AGUCSMIBHEX.(' + fillna)}

# we should treat ( and ) as the same

encoder[')'] = encoder['(']



def get_pair_unpair_mask(struct):

    pair_ixs = list()

    unpair_ixs = list()

    for ix, si in enumerate(struct):

        if si in ['(', ')']:

            pair_ixs.append(ix)

        else:

            unpair_ixs.append(ix)

    return (pair_ixs, unpair_ixs)



def str_replacer(string, newvalue, ixs):

    tmp = list(string)

    for ix in ixs:

        tmp[ix] = newvalue

    return ''.join(tmp)

    



def preprocess_inputs(df, encoder=encoder, fillna=fillna):

    # get pair mask

    pair_unpair_ixs = df.structure.apply(get_pair_unpair_mask).values

    

    seq = df.sequence.values.copy()

    # sequences

    seq_paired = seq.copy()

    seq_unpaired = seq.copy()

    for ix, (pair_ix, unpair_ix) in enumerate(pair_unpair_ixs):

        # paired

        seq_paired[ix] = str_replacer(seq_paired[ix], fillna, unpair_ix)

        seq_paired[ix] = [encoder[si] for si in seq_paired[ix]]

        # unpaired

        seq_unpaired[ix] = str_replacer(seq_unpaired[ix], fillna, pair_ix)

        seq_unpaired[ix] = [encoder[si] for si in seq_unpaired[ix]]

    

    loop = df.predicted_loop_type.values.copy()

    # sequences

    loop_paired = loop.copy()

    loop_unpaired = loop.copy()

    for ix, (pair_ix, unpair_ix) in enumerate(pair_unpair_ixs):

        # paired

        loop_paired[ix] = str_replacer(loop_paired[ix], fillna, unpair_ix)

        loop_paired[ix] = [encoder[si] for si in loop_paired[ix]]

        # unpaired

        loop_unpaired[ix] = str_replacer(loop_unpaired[ix], fillna, pair_ix)

        loop_unpaired[ix] = [encoder[si] for si in loop_unpaired[ix]]

        

    # structure

    structure = df.structure.apply(lambda s: [encoder[si] for si in s]).values

    

    # concat all

    X = np.vstack((seq_paired, seq_unpaired, loop_paired, loop_unpaired, structure))

    X = np.array(X.tolist()).transpose(1, 2, 0)

    return X





def get_bpp_mx(seq_ids):

    files_list = [f'../input/stanford-covid-vaccine/bpps/{seq_id}.npy' 

                  for seq_id in seq_ids]

    bpps = [np.load(f) for f in files_list]

    shape = bpps[0].shape

    return np.array(bpps).reshape(-1, shape[0], shape[1], 1)





target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']

def preprocess_labels(df, target_cols=target_cols):

    return np.array(df[target_cols].values.tolist()).transpose((0, 2, 1))





def preprocess(df, encoder=encoder, fillna=fillna, target_cols=target_cols):

    X = preprocess_inputs(df, encoder=encoder, fillna=fillna)

    Xbpp = get_bpp_mx(df.id.values)

    y = preprocess_labels(df, target_cols=target_cols)

    return (X, Xbpp, y)
def MCRMSE(y_true, y_pred):

    colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)

    return tf.reduce_mean(tf.sqrt(colwise_mse), axis=1)
# http://d2l.ai/chapter_convolutional-modern/resnet.html

class Residual1D(tfk.Model):  #@save

    def __init__(self, num_channels, use_1x1conv=False, strides=1):

        super().__init__()

        self.conv1 = tfkl.Conv1D(num_channels, padding='same',

                                 kernel_size=3, strides=strides)

        self.conv2 = tfkl.Conv1D(num_channels, padding='same',

                                 kernel_size=3)

        self.conv3 = None

        if use_1x1conv:

            self.conv3 = tfkl.Conv1D(num_channels, kernel_size=1,

                                     strides=strides)

        self.bn1 = tfkl.BatchNormalization()

        self.bn2 = tfkl.BatchNormalization()

        

    def call(self, X):

        y = tfk.activations.relu(self.bn1(self.conv1(X)))

        y = self.bn2(self.conv2(y))

        if self.conv3 is not None:

            X = self.conv3(X)

        y += X

        y = tfkl.LeakyReLU()(y)

        y = tfkl.Dropout(0.5)(y)

        return y

    

    def get_config(self):



        config = super().get_config().copy()

        config.update({

            'num_channels': self.num_channels,

            'use_1x1conv': self.use_1x1conv,

            'strides': self.strides,

        })

        return config

    

class ResidualBlock1D(tfkl.Layer):

    def __init__(self, num_channels, num_residuals, first_block=False, **kwargs):

        super(ResidualBlock1D, self).__init__(**kwargs)

        self.num_channels = num_channels

        self.num_residuals = num_residuals

        self.first_block = first_block

        

        self.residual_layers = list()

        for i in range(num_residuals):

            if i == 0 and not first_block:

                self.residual_layers.append(

                    Residual1D(num_channels, use_1x1conv=True, strides=1))

            else:

                self.residual_layers.append(Residual1D(num_channels))

                

    def call(self, X):

        for layer in self.residual_layers.layers:

            X = layer(X)

        return X

    

    def get_config(self):



        config = super().get_config().copy()

        config.update({

            'num_channels': self.num_channels,

            'num_residuals': self.num_residuals,

            'first_block': self.first_block,

        })

        return config

    

class Residual2D(tfk.Model):  #@save

    def __init__(self, num_channels, use_1x1conv=False, strides=1):

        super().__init__()

        self.conv1 = tfkl.Conv2D(num_channels, padding='same',

                                 kernel_size=3, strides=strides)

        self.conv2 = tfkl.Conv2D(num_channels, padding='same',

                                 kernel_size=3)

        self.conv3 = None

        if use_1x1conv:

            self.conv3 = tfkl.Conv2D(num_channels, kernel_size=1,

                                     strides=strides)

        self.bn1 = tfkl.BatchNormalization()

        self.bn2 = tfkl.BatchNormalization()

        

    def call(self, X):

        y = tfk.activations.relu(self.bn1(self.conv1(X)))

        y = self.bn2(self.conv2(y))

        if self.conv3 is not None:

            X = self.conv3(X)

        y += X

        y = tfkl.LeakyReLU()(y)

        y = tfkl.Dropout(0.5)(y)

        return y

    

    def get_config(self):



        config = super().get_config().copy()

        config.update({

            'num_channels': self.num_channels,

            'use_1x1conv': self.use_1x1conv,

            'strides': self.strides,

        })

        return config

    

class ResidualBlock2D(tfkl.Layer):

    def __init__(self, num_channels, num_residuals, first_block=False, **kwargs):

        super(ResidualBlock2D, self).__init__(**kwargs)

        self.num_channels = num_channels

        self.num_residuals = num_residuals

        self.first_block = first_block

        

        self.residual_layers = list()

        for i in range(num_residuals):

            if i == 0 and not first_block:

                self.residual_layers.append(

                    Residual2D(num_channels, use_1x1conv=True, strides=2))

            else:

                self.residual_layers.append(Residual2D(num_channels))

                

    def call(self, X):

        for layer in self.residual_layers.layers:

            X = layer(X)

        return X

    

    def get_config(self):



        config = super().get_config().copy()

        config.update({

            'num_channels': self.num_channels,

            'num_residuals': self.num_residuals,

            'first_block': self.first_block,

        })

        return config
def get_model(seq_len=107, pred_len=68, channels=5, embed_dim=32, loss=MCRMSE):

    # sequence model

    inputs = tfk.Input(shape=(seq_len, channels))

    # embedding

    x = tf.keras.layers.Embedding(input_dim=len(encoder), output_dim=embed_dim)(inputs)

    

    # reshape

    x = tf.reshape(

        x, shape=(-1, x.shape[1],  x.shape[2] * x.shape[3]))

    

    x = tfkl.SpatialDropout1D(.2)(x)

    

    # start

    x = tfkl.Conv1D(32, kernel_size=7, strides=1, padding='same')(inputs)

    x = tfkl.BatchNormalization()(x)

    x = tfkl.Activation('elu')(x)

    x = tfkl.MaxPool1D(pool_size=5, strides=1, padding='same')(x)

        

#     # residual blocks

    x = ResidualBlock1D(32, 2, first_block=True)(x)

    x = ResidualBlock1D(64, 2)(x)

    x = ResidualBlock1D(128, 2)(x)

        

    # sequence

    x = tfkl.Bidirectional(tfkl.GRU(128, dropout=0.1, return_sequences=True, 

                                     kernel_initializer='orthogonal'))(x)

    x = tfkl.Bidirectional(tfkl.GRU(128, dropout=0.1, return_sequences=True, 

                                     kernel_initializer='orthogonal'))(x)

    x = tfkl.Bidirectional(tfkl.GRU(128, dropout=0.1, return_sequences=True, 

                                     kernel_initializer='orthogonal'))(x)

    

    # bpp matrix model

    inputs_bpp = tfk.Input(shape=(seq_len, seq_len, 1))

    xbpp = tfkl.Conv2D(32, kernel_size=7, strides=2, padding='same')(inputs_bpp)

    xbpp = tfkl.BatchNormalization()(xbpp)

    xbpp = tfkl.Activation('elu')(xbpp)

    xbpp = tfkl.MaxPool2D(pool_size=3, strides=2, padding='same')(xbpp)

        

    # residual blocks

    xbpp = ResidualBlock2D(32, 2, first_block=True)(xbpp)

    xbpp = ResidualBlock2D(64, 2)(xbpp)

    xbpp = ResidualBlock2D(128, 2)(xbpp)

    xbpp = tfkl.GlobalAvgPool2D()(xbpp)

    xbpp = tfkl.RepeatVector(seq_len)(xbpp)

    

    

    # add

    x = tfkl.Concatenate(axis=-1)([x, xbpp])

    

    # truncate

    x = x[:, :pred_len]

    

    # dense

    x = tfkl.Dropout(0.2)(x)

    out = tfkl.Dense(5, activation='linear')(x)

    

    # model

    m = tfk.Model(inputs=[inputs, inputs_bpp], outputs=out)

    

    #some optimizers

    adam = tf.optimizers.Adam()

    radam = tfa.optimizers.RectifiedAdam()

    lookahead = tfa.optimizers.Lookahead(adam, sync_period=6)

    ranger = tfa.optimizers.Lookahead(radam, sync_period=6)

    

    # compile

    m.compile(optimizer=adam, loss=loss)

    return m
get_model().summary()
tfk.utils.plot_model(get_model())
from sklearn.model_selection import train_test_split



df_train, df_val = train_test_split(train, test_size=0.10, random_state=2020)



X_train, Xbpp_train, y_train = preprocess(df_train)

X_val, Xbpp_val, y_val = preprocess(df_val)
# fit

model = get_model(loss=MCRMSE)

epochs = 300

batch_size = 16



lr = tfk.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, min_delta=0.005,

                                     patience=10, min_lr=0.0000001, verbose=1)

ckpt = tfk.callbacks.ModelCheckpoint('model.h5')





gc.collect()

history = model.fit((X_train, Xbpp_train), y_train,

                    validation_data=((X_val, Xbpp_val), y_val),

                    epochs=epochs, batch_size=batch_size,

                    callbacks=[lr, ckpt])
public_df = test.query("seq_length == 107").copy()

private_df = test.query("seq_length == 130").copy()



public_inputs = preprocess_inputs(public_df)

bpp_public = get_bpp_mx(public_df.id.values)



private_inputs = preprocess_inputs(private_df)

bpp_private = get_bpp_mx(private_df.id.values)
private_preds = np.zeros((private_df.shape[0], 130, 5))

public_preds = np.zeros((public_df.shape[0], 107, 5))



#load best model and predict

model_short = get_model(seq_len=107, pred_len=107)

model_short.load_weights('model.h5')

public_preds = model_short.predict((public_inputs, bpp_public))



model_long = get_model(seq_len=130, pred_len=130)

model_long.load_weights('model.h5')

private_preds = model_long.predict((private_inputs, bpp_private))
preds_list = []



for df, preds in [(public_df, public_preds), (private_df, private_preds)]:

    for i, uid in enumerate(df.id):

        single_pred = preds[i]



        single_df = pd.DataFrame(single_pred, columns=target_cols)

        single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]



        preds_list.append(single_df)



preds_df = pd.concat(preds_list)

preds_df.head()
sample_sub = pd.read_csv('/kaggle/input/stanford-covid-vaccine/sample_submission.csv')



submission = sample_sub[['id_seqpos']].merge(preds_df, on=['id_seqpos'])

#sanity check

submission.head()
submission.to_csv('submission.csv', index=False)

print('Submission saved')