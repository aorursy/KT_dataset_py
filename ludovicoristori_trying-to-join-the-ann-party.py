import pandas as pd

import numpy as np

import pyarrow.parquet as pq

np.random.seed(123456)
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

plt.style.use('ggplot')
from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split

from sklearn import metrics
import warnings

warnings.filterwarnings('ignore')

import os

print(os.listdir("../input"))
df_train = pd.read_csv('../input/vsb-data-prep-let-s-get-the-party-started/df_train.csv')

df_train.iloc[:,0:12].head()
df_test = pd.read_csv('../input/vsb-data-prep-let-s-get-the-party-started/df_test.csv')

df_test.iloc[:,0:12].head()
df_subm = pd.read_csv('../input/vsb-power-line-fault-detection/sample_submission.csv')

df_subm.head()
outname = 'target'

predictors = list(df_train.columns)

predictors.remove('signal_id')

predictors.remove('id_measurement')

predictors.remove(outname)
def remove_cols(df,col_to_delete):

    df_0=df[df['phase']==0]

    df_0.drop(col_to_delete,axis=1,inplace=True)

    df_1=df[df['phase']==1]

    df_1.drop(col_to_delete,axis=1,inplace=True)

    df_2=df[df['phase']==2]

    df_2.drop(col_to_delete,axis=1,inplace=True)

    df_merge=df_0.merge(df_1, on='id_measurement')

    df_merge=df_merge.merge(df_2, on='id_measurement')

    return(df_merge)
col_to_delete=['phase','signal_id','ErrFun','ErrGen','Amp0','Amp1','Pha0','Pha1','target']

df_train_r=remove_cols(df_train,col_to_delete)

df_test_r=remove_cols(df_test,col_to_delete)
X_df=df_train_r

XT_df=df_test_r

X_df.head()
y_df=df_train['target'].groupby(by=df_train['id_measurement']).first()
X_train_df, X_valid_df, Y_train_df, Y_valid_df = train_test_split(X_df, y_df, test_size=0.2, random_state=123)
absolute_max=max(max(df_train['max']),max(df_test['max']))

absolute_max
absolute_min=min(min(df_train['min']),min(df_test['min']))

absolute_min
absolute_std=np.mean(df_train['std'])

absolute_std
def damaged_ratio(Y, thr):

    dr = 100*sum(Y>=thr)/len(Y)

    return (dr)
def to_int_th(x,th,inverse):

    y = np.zeros(len(x))

    for i in range(0,len(x)):

        if (x[i]>=th) :

            y[i]=1

        else:

            y[i]=0

        if (inverse==1):

            y[i]=1-y[i]

    y = y.astype(int)

    return (y)
X_train_base=np.array(X_train_df.values, copy=True)

Y_train_base=np.array(Y_train_df, copy=True)

X_valid_base=np.array(X_valid_df.values, copy=True)

Y_valid_base=np.array(Y_valid_df, copy=True)

X_test_base=np.array(XT_df.values, copy=True)
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(X_train_base)

X_train = sc.transform(X_train_base) # label: Y_train_base (unscaled)

X_valid = sc.transform(X_valid_base) # label: Y_valid_base (unscaled)

X_test = sc.transform(X_test_base) # label: our goal ;-)
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers
model = keras.Sequential([

    layers.Dense(12, activation="relu"),

    layers.Dense(1, activation="relu")

])
NR_EPOCHS=5
model.compile(optimizer='adam', 

              loss='binary_crossentropy',

              metrics=['accuracy'])
history = model.fit(X_train, Y_train_base, 

                    validation_data=[X_valid, Y_valid_base], epochs=NR_EPOCHS)
Y_valid1 = model.predict(X_valid)

Y_valid1.shape
Y_valid1_df=pd.DataFrame(Y_valid1)

Y_valid1_df.describe()
sns.distplot(Y_valid1, color='blue',bins=10)
th1=(np.min(Y_valid1)+np.max(Y_valid1))/2 # damaged if > th1

Y_valid1_int=to_int_th(Y_valid1,th1,0)
metrics.confusion_matrix(Y_valid_base,Y_valid1_int)
def mmc(y_real_int, y_calc_int):

    cm = metrics.confusion_matrix(y_real_int,y_calc_int)

    tp = cm[0,0]

    tn = cm[1,1]

    fp = cm[0,1]

    fn = cm[1,0]

    num = tp*tn-fp*fn

    den = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))

    if den==0:

        mc=-1

    else:

        mc=num/den

    return np.float64(mc)
mmc(Y_valid_base,Y_valid1_int)
metrics.accuracy_score(Y_valid_base,Y_valid1_int)
Y_pred = model.predict(X_test)
Y_pred.shape
sns.distplot(Y_pred,bins=10)
Y_pred_int=to_int_th(Y_pred,th1,0)

np.unique(Y_pred_int,return_counts=True)
damaged_ratio(Y_train_base,0.5)
def find_thres(y_real, y_calc):

    thr_ndiv=100

    y_min=np.min(y_calc)

    y_max=np.max(y_calc)

    start_thres = (y_min+y_max)/2 # default, better than 0

    stop_thres = y_max

    opt_thres=start_thres

    opt_mmc = -1

    dthr=(stop_thres-start_thres)/thr_ndiv

    if (dthr==0):

        vec_thres = np.arange(start_thres, stop_thres+0.1,0.1)

    else:

        vec_thres = np.arange(start_thres,stop_thres, dthr)

    for thres in vec_thres:

        y_calc_int=to_int_th(y_calc,thres,0)

        m = mmc(y_real,y_calc_int)

        if (m > opt_mmc):

            opt_mmc = m

            opt_thres = thres

    print('opt. thres={t:.5f} mmc={m:.5f}'.format(t=opt_thres,m=opt_mmc))

    return opt_thres
import gc

import pyarrow.parquet as pq
metadata_train = pd.read_csv("../input/vsb-power-line-fault-detection/metadata_train.csv")

metadata_train.info()
row_nr=800000

row_nr
row_group_size=4000
time_sample_idx=np.arange(0,row_nr,row_group_size)

time_sample_idx[0:8]
metadata_test = pd.read_csv("../input/vsb-power-line-fault-detection/metadata_test.csv")

metadata_test.info()
sign_start=min(metadata_test['signal_id'])

sign_start
sign_stop=max(metadata_test['signal_id'])+1

sign_stop
sign_group_size=2000
def scale(val,orig_min,orig_max,des_min,des_max):

    X_std = (val - orig_min) / (orig_max - orig_min)

    X_scaled = X_std * (des_max - des_min) + des_min

    return(X_scaled)
def y_line(x,x1,y1,x2,y2):

    if (x1==x2):

        y=(y1+y2)/2

    else:

        m=(y1-y2)/(x1-x2)

        q=y1-m*x1

        y=m*x+q

    return (y)    
def extract_signal_features(signal_id,file_i,time_sample_idx,abs_max,abs_min):

    feat_nr=6

    signal_features=np.zeros((len(time_sample_idx),6))

    for j in range(0,len(time_sample_idx)-1):

        file_i_range_j = file_i.iloc[time_sample_idx[j]:time_sample_idx[j+1],signal_id]

        nr_ts=5

        x1=time_sample_idx[j]

        y1=np.mean(file_i.iloc[x1:x1+nr_ts,signal_id])

        x2=time_sample_idx[j+1]

        y2=np.mean(file_i.iloc[x2-nr_ts:x2,signal_id])

        x1=x1+nr_ts/2

        x2=x2-nr_ts/2

        range_mean = np.mean(file_i_range_j)

        x_min=file_i_range_j.idxmin()

        range_min = np.min(file_i_range_j)

        err_min =range_min-y_line(x_min,x1,y1,x2,y2)

        x_max=file_i_range_j.idxmax()

        range_max = np.max(file_i_range_j)

        err_max=range_max-y_line(x_max,x1,y1,x2,y2)

        range_std = np.std(file_i_range_j)

        if (range_std==0):

            err_rel_rng=0

            err_abs_rng=0

        else:

            err_rel_rng=(err_max-err_min)/range_std

            err_abs_rng=err_max-err_min

        prc_low=np.percentile(file_i_range_j,5)

        prc_high=np.percentile(file_i_range_j,95)

        sign_feat = np.array([range_mean,

                        range_std,

                        err_rel_rng,

                        err_abs_rng,

                        prc_low,

                        prc_high])

        signal_features[j]=sign_feat

    return signal_features    
def fill_ar_samples(filepath,sign_start,sign_stop,sign_group_size,row_nr,time_sample_idx,

                   abs_max,abs_min):

    time_samples_str=[str(idx) for idx in time_sample_idx]

    feat_nr=6

    samples_ar=np.zeros((sign_stop-sign_start,len(time_sample_idx),feat_nr))

    col_id_start=sign_start

    n_groups = int(np.round((sign_stop-sign_start)/sign_group_size))+1

    print('Steps = {}'.format(n_groups))

    for i in range(0,n_groups):

        col_id_stop = np.minimum(col_id_start+sign_group_size,sign_stop)

        col_numbers = np.arange(col_id_start,col_id_stop)

        print('Step {s} - cols = [{a},{b})'.format(s=i,a=col_id_start,b=col_id_stop))

        col_names = [str(c_num) for c_num in col_numbers]

        file_i = pq.read_pandas(filepath,columns=col_names).to_pandas()

        for c in col_names:

            if (int(c)%50==0):

                print('.',end='')

            col=int(c)-col_id_start

            feat = extract_signal_features(col,file_i,time_sample_idx,abs_max,abs_min)

            samples_ar[int(c)-sign_start] = feat

        col_id_start=col_id_stop

        print('')

    return (samples_ar)
%%time

train_samples=fill_ar_samples('../input/vsb-power-line-fault-detection/train.parquet',0,sign_start,sign_group_size,row_nr,time_sample_idx,absolute_max,absolute_min)

train_samples.tofile('train.npy')

train_samples.shape
def ar_compacted_phases(ar_samples,df_metadata,start_id_meas):

    nr_id_meas=int(ar_samples.shape[0]/3)

    nr_samples=ar_samples.shape[1]

    nr_feats_per_phase=ar_samples.shape[2]

    ar_measures=np.zeros((nr_id_meas,nr_samples,3*nr_feats_per_phase))

    for idx_signal in range(0,len(ar_samples)):

        idx_meas=df_metadata['id_measurement'].loc[idx_signal]-start_id_meas

        idx_sample=int(idx_signal%nr_samples)

        for phase in range(0,3):

            f_start=int(phase*nr_feats_per_phase)

            f_stop=int(f_start+nr_feats_per_phase)

            ar_measures[idx_meas,

                idx_sample,

                f_start:f_stop]=ar_samples[idx_signal,idx_sample,0:nr_feats_per_phase]

    return ar_measures
train_cf=ar_compacted_phases(train_samples,metadata_train,0)

train_cf.shape
y_cf=y_df.values

y_cf
%%time

test_samples=fill_ar_samples('../input/vsb-power-line-fault-detection/test.parquet',

                             sign_start,sign_stop,sign_group_size,row_nr,time_sample_idx,

                             absolute_max,absolute_min)

test_samples.tofile('test.npy')

test_samples.shape
id_meas_start=min(metadata_test['id_measurement'])

test_cf=ar_compacted_phases(test_samples,metadata_test,id_meas_start)

test_cf.shape
from keras.layers import *

from keras.models import Model

from keras.callbacks import ModelCheckpoint

from keras.callbacks import *
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
def matthews_correlation(y_true, y_pred):

    '''Calculates the Matthews correlation coefficient measure for quality

    of binary classification problems.

    '''

    y_pred = tf.convert_to_tensor(y_pred, np.float32)

    y_true = tf.convert_to_tensor(y_true, np.float32)

    

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
def model_lstm(input_shape):

    inp = Input(shape=(input_shape[1], input_shape[2]))

    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(inp)

    x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)

    x = Attention(input_shape[1])(x)

    x = Dense(64, activation="relu")(x)

    x = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=x)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[matthews_correlation])

    return model
CV_STEPS=8

NR_EPOCHS=40
ar_cv=np.arange(0,CV_STEPS)

ep_cv=np.arange(0,NR_EPOCHS)

mi=pd.MultiIndex.from_product([ar_cv,ep_cv], names=['cv','epoch'])

df=pd.DataFrame(index=mi,columns=['loss','val_loss','matthews_correlation','val_matthews_correlation'])
KF = KFold(n_splits=CV_STEPS, shuffle=True)
opt_thr=np.zeros(CV_STEPS)

for k in range(0,CV_STEPS):

    print('Step {}'.format(k))

    train_y=df_train[outname].values

    X_train, X_valid, y_train, y_valid = train_test_split(train_cf, y_cf, test_size=1/CV_STEPS)

    w_file_name='weights_best_{}.hdf5'.format(k)

    model = model_lstm(X_train.shape)

    ckpt = ModelCheckpoint(w_file_name, save_best_only=True, verbose=1,

                           save_weights_only=True, monitor='val_matthews_correlation', 

                           mode='max')

    history=model.fit(X_train, y_train, epochs=NR_EPOCHS, batch_size=128, shuffle=True,

          validation_data=[X_valid, y_valid],callbacks=[ckpt])

    if (os.path.exists(w_file_name)):

        print('weight file loaded...')

        model.load_weights(w_file_name)

    y_valid1=model.predict(X_valid)

    opt_thr[k]=find_thres(y_valid, y_valid1)

    df['loss'].loc[k].iloc[0:NR_EPOCHS]=history.history['loss']

    df['val_loss'].loc[k].iloc[0:NR_EPOCHS]=history.history['val_loss']

    df['matthews_correlation'].loc[k].iloc[0:NR_EPOCHS]=history.history['matthews_correlation']

    df['val_matthews_correlation'].loc[k].iloc[0:NR_EPOCHS]=history.history['val_matthews_correlation']
h=history.history

print(h.keys())
fig,ax = plt.subplots(1,2, figsize=(10,5))

for k in range(0,CV_STEPS):

    loss=df['loss'].loc[k]

    val_loss=df['val_loss'].loc[k]

    ax[0].plot(loss, color='red')

    ax[1].plot(val_loss, color='blue')
fig,ax = plt.subplots(1,2, figsize=(10,5))

mmc_train_last=np.zeros(CV_STEPS)

mmc_valid_last=np.zeros(CV_STEPS)

for k in range(0,CV_STEPS):

    mc=df['matthews_correlation'].loc[k]

    val_mc=df['val_matthews_correlation'].loc[k]

    ax[0].plot(mc, color='red')

    ax[1].plot(val_mc, color='blue')

    mmc_train_last[k]=df['matthews_correlation'].loc[k].loc[NR_EPOCHS-1]

    mmc_valid_last[k]=df['val_matthews_correlation'].loc[k].loc[NR_EPOCHS-1]

m=np.mean(mmc_train_last)

s=np.std(mmc_train_last) 

print('matthews_correlation mean={m} std={s}'.format(k=k,m=m,s=s))

m=np.mean(mmc_valid_last)

s=np.std(mmc_valid_last) 

print('val_matthews_correlation mean={m} std={s}'.format(k=k,m=m,s=s))
sns.distplot(y_valid1,color='green')
thr_avg=np.mean(opt_thr)

thr_std=np.std(opt_thr)

print('threshold={av} std={st}'.format(av=thr_avg,st=thr_std))

fin_thr=thr_avg
y_valid1_int=to_int_th(y_valid1,fin_thr,0)

metrics.confusion_matrix(y_valid,y_valid1_int)
%%time

ar_pred = np.zeros((len(test_cf),CV_STEPS))

for k in range(0,CV_STEPS):

    w_file_name='weights_best_{}.hdf5'.format(k)

    model1 = model_lstm(X_train.shape)

    model1.load_weights(w_file_name)

    y_pred1=model1.predict(test_cf)

    ar_pred[:,k]=np.squeeze(y_pred1)
y_pred=ar_pred.mean(axis=1)
sns.distplot(y_pred,color='green')
y_pred_int=to_int_th(y_pred,fin_thr,0)
np.unique(y_pred_int,return_counts=True)
XT_df['max']=y_pred_int

df_pred=XT_df[['id_measurement','max']]

df_pred.columns=['id_measurement','target']

df_pred.head()
df_subm=df_test[['signal_id','id_measurement']].merge(df_pred, on='id_measurement')

df_subm.drop('id_measurement',axis=1,inplace=True)

df_subm.head()
sum(df_subm['target'])
df_subm.to_csv('submission.csv', index=False)