import sys

import os

# sys.path.insert(0,'../input/efficientnet/efficientnet-master/efficientnet-master/')

# from efficientnet import EfficientNetB5
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

import gc



import matplotlib.pyplot as plt



from joblib import Parallel, delayed



from sklearn.metrics import mean_absolute_error,mean_squared_error

from sklearn.preprocessing import quantile_transform,StandardScaler,MinMaxScaler



import tensorflow.keras.layers as L

import tensorflow as tf

from tensorflow.keras.applications.resnet50 import ResNet50

from tensorflow.keras.applications import EfficientNetB5
config = tf.compat.v1.ConfigProto()

config.gpu_options.allow_growth = True

session = tf.compat.v1.Session(config=config)
gc.collect()
npys_paths = '../input/stanford-covid-vaccine/bpps/'+pd.Series(os.listdir('../input/stanford-covid-vaccine/bpps'))

npys_ids = npys_paths.apply(lambda x : x.split('_')[1]).apply(lambda x : x.split('.')[0])

npys = pd.DataFrame([*zip(npys_paths,npys_ids)])
def load_npy(x):

    return np.resize(np.load(x),(130,130))
%%time

npys.iloc[:,0] = Parallel(n_jobs=4)(delayed(load_npy)(filename) for filename in npys.iloc[:,0].tolist())
npys.columns = ['genetic_probs','id_hash']
train = pd.read_json('../input/stanford-covid-vaccine/train.json',lines=True)

test = pd.read_json('../input/stanford-covid-vaccine/test.json', lines=True)

sub = pd.read_csv('../input/stanford-covid-vaccine/sample_submission.csv')

train['id_hash'] = train['id'].apply(lambda x : x.split('_')[1])

test['id_hash'] = test['id'].apply(lambda x : x.split('_')[1])
target_columns = ['reactivity', 'deg_Mg_pH10','deg_pH10', 'deg_Mg_50C', 'deg_50C']

y=np.array(train[target_columns].values.tolist()).transpose(0,2,1)
for df in [train,test]:

    df['Paired']=[sum([i=='(' or i==')' for i in j]) for j in df['structure']]

    df['Unpaired']=[sum([i=='.' for i in j]) for j in df['structure']]

    for col in ['E','S','H','I','G','A','U']:

        if col in ['E','S','H','I']:

            df[col]=[sum([i==col for i in j])/len(j) for j in df['predicted_loop_type']]

        else:

            df[col]=[sum([i==col for i in j])/len(j) for j in df['sequence']]

for a in [ 'G', 'A', 'C', 'U']:

    train[a+'_position']=[np.sum([i for i in range(len(j)) if j[i]==a])/len([i for i in range(len(j)) if j[i]==a]) for j in train['sequence']]

    test[a+'_position']=[np.sum([i for i in range(len(j)) if j[i]==a])/len([i for i in range(len(j)) if j[i]==a]) for j in test['sequence']]

for a in [ 'E', 'S', 'H',]:

    train[a+'_position']=[np.sum([i for i in range(len(j)) if j[i]==a])/len([i for i in range(len(j)) if j[i]==a]) for j in train['predicted_loop_type']]

    test[a+'_position']=[np.sum([i for i in range(len(j)) if j[i]==a])/len([i for i in range(len(j)) if j[i]==a]) for j in test['predicted_loop_type']]

for a in [ 'E', 'S', 'H',]:

    train[a+'']=[np.sum([i for i in range(len(j)) if j[i]==a])/len([i for i in range(len(j)) if j[i]==a]) for j in train['predicted_loop_type']]

    test[a+'_position']=[np.sum([i for i in range(len(j)) if j[i]==a])/len([i for i in range(len(j)) if j[i]==a]) for j in test['predicted_loop_type']]

target_columns = ['reactivity', 'deg_Mg_pH10','deg_pH10', 'deg_Mg_50C', 'deg_50C']

target_columns.extend(['SN_filter', 'signal_to_noise'])

target_columns.extend(['deg_error_pH10', 'deg_error_Mg_50C', 'deg_error_50C', 'reactivity_error', 'deg_error_Mg_pH10'] )

train.drop(target_columns,axis=1,inplace=True)

# SC = MinMaxScaler(feature_range=(-1, 1))

cols = pd.concat((train.select_dtypes('float64'),train.select_dtypes('int64')),axis=1).columns.tolist()

for col in ['seq_length','seq_scored','index']:

    cols.remove(col)

train_measurements = pd.DataFrame(quantile_transform(train[cols]),columns=cols)

public_measurements = pd.DataFrame(quantile_transform(test.query("seq_length == 107")[cols]),columns=cols)

private_measurements = pd.DataFrame(quantile_transform(test.query("seq_length == 130")[cols]),columns=cols)
train_im = np.array(train.merge(npys,on='id_hash')['genetic_probs'].values.tolist())

public_im =  np.array(test.merge(npys,on='id_hash').query("seq_length == 107")['genetic_probs'].values.tolist())

private_im = np.array(test.merge(npys,on='id_hash').query("seq_length == 130")['genetic_probs'].values.tolist())
public_im.shape,private_im.shape,public_measurements.shape,private_measurements.shape
train.shape,y.shape
def MCRMSE(y_true, y_pred):

    colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)

    return tf.reduce_mean(tf.sqrt(colwise_mse), axis=1)



def build_model(seq_len=107, pred_len=68, dropout=0.5, embed_dim=100, hidden_dim=128):

    image_tensor = L.Input(shape=(130,130), dtype=tf.float32)

    im = L.Reshape((130,130,1))(image_tensor)

#     conv = L.Conv2D(3,(3,3),padding='same')(im)

    efn = EfficientNetB5(input_shape=(130,130,1),weights=None,include_top=False)

#     res = ResNet50(weights='imagenet', include_top=False)

    classes = efn(im)

    classes = L.GlobalAveragePooling2D()(classes)

    classes = L.Dropout(dropout)(classes)

    csv_tensor = L.Input(shape=(16,))

    y = L.GaussianNoise(0.1)(csv_tensor)

    y = L.Concatenate()([classes, y]) 

    

    y = L.Dense(activation = 'linear',units=650)(y)

    y = L.Reshape((130,5))(y)

    y = y[:,:pred_len,:]

    model = tf.keras.Model(inputs=[image_tensor,csv_tensor], outputs=y)



    model.compile(tf.keras.optimizers.Adam(), loss=MCRMSE)

    

    return model
tf.config.optimizer.set_jit(True)

model = build_model()

model.summary()
with tf.device('/gpu'):

    model.fit([train_im,train_measurements],y,batch_size=64,

            epochs=100,

            validation_split=0.05,

             callbacks=[

            tf.keras.callbacks.ReduceLROnPlateau(),

            tf.keras.callbacks.ModelCheckpoint('model.h5')

        ])
mean_squared_error(model.predict([train_im,train_measurements]).reshape(2400,340),y.reshape(2400,340))
model_short = build_model(seq_len=107, pred_len=107)

model_long = build_model(seq_len=130, pred_len=130)



model_short.load_weights('model.h5')

model_long.load_weights('model.h5')



public_preds = model_short.predict([public_im,public_measurements])

private_preds = model_long.predict([private_im,private_measurements])
preds_ls = []

pred_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']

for df, preds, ids in [(public_im, public_preds,test.query("seq_length == 107")['id_hash']), (private_im, private_preds,test.query("seq_length == 130")['id_hash'])]:

    for i, uid in enumerate(ids):

        single_pred = preds[i]



        single_df = pd.DataFrame(single_pred, columns=pred_cols)

        single_df['id_seqpos'] = ['id_'+f'{uid}_{x}' for x in range(single_df.shape[0])]



        preds_ls.append(single_df)



preds_df = pd.concat(preds_ls)
preds_df
sample_df = pd.read_csv('/kaggle/input/stanford-covid-vaccine/sample_submission.csv')

submission = sample_df[['id_seqpos']].merge(preds_df, on=['id_seqpos'])

submission.to_csv('submission.csv', index=False)