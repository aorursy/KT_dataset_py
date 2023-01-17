RUN_FOLDS = 1

EPOCHS = 1

BATCHSIZE = 128

# please change to the following values offline

# RUN_FOLDS = 5

# EPOCHS = 50
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm, tqdm_notebook

import time

import glob

import os

print(os.listdir("../input"))

import gc





# Any results you write to the current directory are saved as output.

import tensorflow as tf

from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.preprocessing import StandardScaler

from keras import layers

from keras import backend as K

from keras import regularizers

from keras.constraints import max_norm

from keras.models import Sequential

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

from keras.models import load_model

from keras.models import Model

from keras.initializers import glorot_uniform

from keras.layers import Input,Dense,Activation,ZeroPadding2D,BatchNormalization,Flatten,Conv2D,AveragePooling2D,MaxPooling2D,Dropout,concatenate

from sklearn import preprocessing



import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve

#from sklearn.metrics import auc

from sklearn.metrics import roc_auc_score

from keras.utils import Sequence,to_categorical



GPU = 4

os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU)



import datetime

import os

import sys

import time



import random

import multiprocessing



import warnings

warnings.filterwarnings("ignore")
def reverse(tr,cols=None):

    reverse_list = [0,1,2,3,4,5,6,7,8,11,15,16,18,19,

                22,24,25,26,27,41,29,

                32,35,37,40,48,49,47,

                55,51,52,53,60,61,62,103,65,66,67,69,

                70,71,74,78,79,

                82,84,89,90,91,94,95,96,97,99,

                105,106,110,111,112,118,119,125,128,

                130,133,134,135,137,138,

                140,144,145,147,151,155,157,159,

                161,162,163,164,167,168,

                170,171,173,175,176,179,

                180,181,184,185,187,189,

                190,191,195,196,199]

    reverse_list = ['var_%d'%i for i in reverse_list]

    if cols is not None:

        for col in cols:

            colx = col.split('_')

            colx = '_'.join(colx[:2])

            if colx in reverse_list:

                print('reverse',col)

                tr[col] = tr[col]*(-1)

        return tr

    

    for col in reverse_list:

        tr[col] = tr[col]*(-1)

    return tr



def scale(tr,te=None):

    for col in tr.columns:

        if col.startswith('var_'):

            mean,std = tr[col].mean(),tr[col].std()

            tr[col] = (tr[col]-mean)/std

            if te is not None:

                te[col] = (te[col]-mean)/std

    if te is None:

        return tr

    return tr,te



def getp_vec_sum(x,x_sort,y,std,c=0.5):

    # x is sorted

    left = x - std/c

    right = x + std/c

    p_left = np.searchsorted(x_sort,left)

    p_right = np.searchsorted(x_sort,right)

    p_right[p_right>=y.shape[0]] = y.shape[0]-1

    p_left[p_left>=y.shape[0]] = y.shape[0]-1

    return (y[p_right]-y[p_left])



def get_pdf(tr,col,x_query=None,smooth=3):

    std = tr[col].std()

    tr = tr.dropna(subset=[col])

    df = tr.groupby(col).agg({'target':['sum','count']})

    cols = ['sum_y','count_y']

    df.columns = cols

    df = df.reset_index()

    df = df.sort_values(col)

    y,c = cols

    

    df[y] = df[y].cumsum()

    df[c] = df[c].cumsum()

    

    if x_query is None:

        rmin,rmax,res = -5.0, 5.0, 501

        x_query = np.linspace(rmin,rmax,res)

    

    dg = pd.DataFrame()

    tm = getp_vec_sum(x_query,df[col].values,df[y].values,std,c=smooth)

    cm = getp_vec_sum(x_query,df[col].values,df[c].values,std,c=smooth)+1

    dg['res'] = tm/cm

    dg.loc[cm<500,'res'] = 0.1

    return dg['res'].values



def get_pdfs(tr):

    y = []

    for i in range(200):

        name = 'var_%d'%i

        res = get_pdf(tr,name)

        y.append(res)

    return np.vstack(y)



def print_corr(corr_mat,col,bar=0.95):

    cols = corr_mat.loc[corr_mat[col]>bar,col].index.values

    return cols.tolist()



def get_group(df,cols,reverse=True,bar=0.9):

    if reverse:

        df = reverse(df,cols=cols)

    df = scale(df)

    pdfs = get_pdfs(df)

    df_pdf = pd.DataFrame(pdfs.T,columns=cols)

    corr_mat = df_pdf.corr(method='pearson')

    groups =[]

    skip_list = []

    for i in cols:

        if i not in skip_list:

            cols = print_corr(corr_mat,i,bar)

            if(len(cols)>1):

                groups.append(cols)

                for e,v in enumerate(cols):

                    skip_list.append(i)

    return groups
class DataGenerator(Sequence):

    'Generates data for Keras'

    def __init__(self, **kwargs):

        'Initialization'

        self.params = kwargs

        self.X = self.params['X']

        self.cols_info = self.params['cols_info']

        self.groups = self.params['groups']

        self.shuffle = self.params['shuffle']

        self.y = self.params['y']

        self.aug = self.params['aug']

        self.indexes = np.arange(self.y.shape[0])

        if self.shuffle:

            np.random.shuffle(self.indexes)



    def __len__(self):

        'Denotes the number of batches per epoch'

        batch_size = self.params['batch_size']

        return int(np.floor(self.indexes.shape[0] / batch_size))



    def __getitem__(self, index):

        'Generate one batch of data'

        # Generate indexes of the batch

        batch_size = self.params['batch_size']

        indexes = self.indexes[index*batch_size:(index+1)*batch_size]



        X, y = self.__data_generation(indexes)

        return X, y



    def on_epoch_end(self):

        'Updates indexes after each epoch'

        if self.shuffle == True:

            np.random.shuffle(self.indexes)



    def __data_generation(self, indexes):

        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        y = self.y[indexes]

        X = self.X[indexes]

        batch_size = self.params['batch_size']

        if self.aug:

            X,y = augment_fix_fast(X,y,groups=2,t1=2, t0=2)

        base_feats, noise_feats = self.cols_info

        allfeas = base_feats+noise_feats

        X = pd.DataFrame(X,columns=allfeas)

        X = get_keras_groups_data(X, self.cols_info,self.groups)

        return X, y

    

    def get_keras_data(self, dataset, cols_info):

        X = {}

        base_feats, noise_feats = cols_info

        X['base'] = np.reshape(dataset[:,:len(base_feats)], (-1, len(base_feats), 1))

        X['noise1'] = np.reshape(dataset[:,len(base_feats): len(base_feats) + len(noise_feats)], (-1, len(noise_feats), 1))

        

        return X

    

    def aug_(self,xb,xn1,y,t=2):

        xb_pos,xb_neg,xn1_pos,xn1_neg = [],[],[],[]

        for i in range(t):

            mask = y>0

            x1 = xb[mask].copy()

            x2 = xn1[mask].copy()

            ids = np.arange(x1.shape[0])

            for c in range(x1.shape[1]):

                np.random.shuffle(ids)

                x1[:,c] = x1[ids][:,c]

                x2[:,c] = x2[ids][:,c]

            xb_pos.append(x1)

            xn1_pos.append(x2)

        

        for i in range(t):

            mask = y==0

            x1 = xb[mask].copy()

            x2 = xn1[mask].copy()

            ids = np.arange(x1.shape[0])

            for c in range(x1.shape[1]):

                np.random.shuffle(ids)

                x1[:,c] = x1[ids][:,c]

                x2[:,c] = x2[ids][:,c]

            xb_neg.append(x1)

            xn1_neg.append(x2)

    



        xb_pos = np.vstack(xb_pos)

        xb_neg = np.vstack(xb_neg)

        xn1_pos = np.vstack(xn1_pos)

        xn1_neg = np.vstack(xn1_neg)



        ys = np.ones(xb_pos.shape[0])

        yn = np.zeros(xb_neg.shape[0])

        xb = np.vstack([xb,xb_pos,xb_neg])

        xn1 = np.vstack([xn1,xn1_pos,xn1_neg])

        y = np.concatenate([y,ys,yn])

        return xb,xn1,y
# define helper functions. auc, plot_history

def auc(y_true, y_pred):

    #auc = tf.metrics.auc(y_true, y_pred)[1]

    y_pred = y_pred.ravel()

    y_true = y_true.ravel()

    return roc_auc_score(y_true, y_pred)



def auc_2(y_true, y_pred):

    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)



def plot_history(histories, key='binary_crossentropy'):

    plt.figure(figsize=(16,10))

    #plt.plot([0, 1], [0, 1], 'k--')

    for name, history in histories:

        val = plt.plot(history.epoch, history.history['val_'+key], '--', label=name.title()+' Val')



    plt.plot(history.epoch, history.history[key], color=val[0].get_color(), label=name.title()+' Train')



    plt.xlabel('Epochs')

    plt.ylabel(key.replace('_',' ').title())

    plt.legend()



    plt.xlim([0,max(history.epoch)])

    plt.ylim([0, 0.4])

    plt.show()
def shuffle_col_vals_fix(x1, groups):

    group_size = x1.shape[1]//groups

    xs = [x1[:, i*group_size:(i+1)*group_size] for i in range(groups)]

    rand_x = np.array([np.random.choice(x1.shape[0], size=x1.shape[0], replace=False) for i in range(group_size)]).T

    grid = np.indices(xs[0].shape)

    rand_y = grid[1]

    res = [x[(rand_x, rand_y)] for x in xs]

    return np.hstack(res)



def augment_fix_fast(x,y,groups,t1=2, t0=2):

    # In order to make the sync version augment work, the df should be the form of:

    # var_1, var_2, var_3 | var_1_count, var_2_count, var_3_count | var_1_rolling, var_2_rolling, var_3_rolling

    # for the example above, 3 groups of feature, groups = 3

    xs,xn = [],[]

    for i in range(t1):

        mask = y>0

        x1 = x[mask].copy()

        x1 = shuffle_col_vals_fix(x1, groups)

        xs.append(x1)



    for i in range(t0):

        mask = (y==0)

        x1 = x[mask].copy()

        x1 = shuffle_col_vals_fix(x1, groups)

        xn.append(x1)



    xs = np.vstack(xs); xn = np.vstack(xn)

    ys = np.ones(xs.shape[0]);yn = np.zeros(xn.shape[0])

    x = np.vstack([x,xs,xn]); y = np.concatenate([y,ys,yn])

    return x,y
def build_magic_nn():

    train_df = pd.read_csv('../input/train.csv')

    test_df =  pd.read_csv("../input/test.csv")



        

    base_features = [x for x in train_df.columns.values.tolist() if x.startswith('var_')]

    train_df['real'] = 1



    for col in base_features:

        test_df[col] = test_df[col].map(test_df[col].value_counts())

    a = test_df[base_features].min(axis=1)



    test_df = pd.read_csv('../input/test.csv')

    test_df['real'] = (a == 1).astype('int')



    train = train_df.append(test_df).reset_index(drop=True)

    del test_df, train_df; gc.collect()

    for col in tqdm(base_features):

        train[col + '_size'] = train[col].map(train.loc[train.real==1, col].value_counts())

    cnt_features = [col + '_size' for col in base_features]



    for col in tqdm(base_features):

    #        train[col+'size'] = train.groupby(col)['target'].transform('size')

        train.loc[train[col+'_size']>1,col+'_no_noise'] = train.loc[train[col+'_size']>1,col]

    noise1_features = [col + '_no_noise' for col in base_features]



    train[noise1_features] = train[noise1_features].fillna(train[noise1_features].mean())



    train_df = train[train['target'].notnull()]

    test_df = train[train['target'].isnull()]

    all_features = base_features + noise1_features



    scaler = preprocessing.StandardScaler().fit(train_df[all_features].values)

    df_trn = pd.DataFrame(scaler.transform(train_df[all_features].values), columns=all_features)

    df_tst = pd.DataFrame(scaler.transform(test_df[all_features].values), columns=all_features)



    return df_trn,df_tst,train_df,test_df
df_trn,df_tst,train_df,test_df = build_magic_nn()
#%%time

y = train_df['target'].values

base_features = ['var_%d'%i for i in range(200)]

noise1_features = ['%s_no_noise'%i for i in base_features]

all_features = base_features + noise1_features

cols_info = [base_features, noise1_features]
df_trn = reverse(df_trn,cols=base_features + noise1_features)

df_tst = reverse(df_tst,cols=base_features + noise1_features)
def get_keras_data(dataset, cols_info):

    X = {}

    base_feats, noise_feats = cols_info

    X['base'] = np.reshape(np.array(dataset[base_feats].values), (-1, len(base_feats), 1))

    X['noise1'] = np.reshape(np.array(dataset[noise_feats].values), (-1, len(noise_feats), 1))

    return X



def get_keras_groups_data(dataset, cols_info, groups):

    X = {}

    base_feats, noise_feats = cols_info

    #X['base'] = np.reshape(np.array(dataset[base_feats].values), (-1, len(base_feats), 1))

    for c,g in enumerate(groups):

        X['group_%d'%c] = np.expand_dims(dataset[g].values,2)

    X['noise1'] = np.expand_dims(dataset[noise_feats].values,2)

    return X
groups = get_group(train_df[base_features+['target']].copy(),base_features,reverse=False,bar=0.9)
X_test = get_keras_groups_data(df_tst[all_features], cols_info, groups)
# define network structure -> 2D CNN

def Convnet(cols_info, groups, classes=1):

    base_feats, noise1_feats= cols_info

    

    xs = []

    ins = []

    for c,i in enumerate(groups):

        X = Input(shape=(len(i), 1), name='group_%d'%c)

        ins.append(X)

        X = Dense(1)(X)

        X = Activation('relu')(X)

        X = Flatten(name='group_%d_last'%c)(X)

        xs.append(X)

    

    for i,j in zip(cols_info,['noise1']):

        X = Input(shape=(len(i), 1), name=j)

        ins.append(X)

        X = Dense(16)(X)

        X = Activation('relu')(X)

        X = Flatten(name='%s_last'%j)(X)

        xs.append(X)

    

    X = concatenate(xs)

    X = Dense(classes, activation='sigmoid')(X)

    

    model = Model(inputs=ins,outputs=X)

    return model



model = Convnet(cols_info,groups)

model.summary()
try:

    del train, df_tst 

except:

    pass

gc.collect()
# parameters

SEED = 2019

n_folds = 5

debug_flag = True

skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
#%%time

#transformed_shape = tuple([-1] + list(shape))

#X_test = np.reshape(X_test, transformed_shape)



i = 0

result = pd.DataFrame({"ID_code": test_df.ID_code.values})

val_aucs = []

valid_X = train_df[['target']]

valid_X['predict'] = 0

for train_idx, val_idx in skf.split(df_trn, y):

    if i == RUN_FOLDS:

        break

    

    i += 1    

    X_train, y_train = df_trn.iloc[train_idx], y[train_idx]

    X_valid, y_valid = df_trn.iloc[val_idx], y[val_idx]

    

    #aug

    X_train, y_train = augment_fix_fast(X_train.values, y_train, groups=2, t1=2, t0=2)

    X_train = pd.DataFrame(X_train, columns=all_features)

    

    #X_train = get_keras_data(X_train, cols_info)

    #X_valid = get_keras_data(X_valid, cols_info)

    #X_train = np.reshape(X_train, transformed_shape)

    #X_valid = np.reshape(X_valid, transformed_shape)

    

    model_name = 'NN_fold{}_{}.h5'.format(str(i),GPU)

    

    model = Convnet(cols_info,groups)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'binary_crossentropy', auc_2])

    checkpoint = ModelCheckpoint(model_name, monitor='val_auc_2', verbose=1, 

                                 save_best_only=True, mode='max', save_weights_only = True)

    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=4, 

                                       verbose=1, mode='min', epsilon=0.0001)

    earlystop = EarlyStopping(monitor='val_auc_2', mode='max', patience=5, verbose=1)

    

    if 1:

        training_generator = DataGenerator(X=X_train.values,y=y_train,aug=1,groups=groups,

                                           batch_size=BATCHSIZE,shuffle=True,cols_info=cols_info)

        

        validation_generator = DataGenerator(X=X_valid.values,y=y_valid,aug=0,groups=groups,

                                             batch_size=BATCHSIZE,shuffle=False,cols_info=cols_info)

        

        history = model.fit_generator(generator=training_generator,

                        validation_data=validation_generator,

                        epochs=EPOCHS,  

                        callbacks=[checkpoint, reduceLROnPlat, earlystop])

    train_history = pd.DataFrame(history.history)

    train_history.to_csv('train_profile_fold{}_{}.csv'.format(str(i),GPU), index=None)

    

    # load and predict

    model.load_weights(model_name)

    

    #predict

    X_valid = get_keras_groups_data(X_valid, cols_info,groups)

    y_pred_keras = model.predict(X_valid).ravel()

    

    # AUC

    valid_X['predict'].iloc[val_idx] = y_pred_keras

    

    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_valid, y_pred_keras)

    auc_valid = roc_auc_score(y_valid, y_pred_keras)

    val_aucs.append(auc_valid)

    print('Fold %d auc %.4f'%(i,val_aucs[-1]))

    prediction = model.predict(X_test)

    result["fold{}".format(str(i))] = prediction
for i in range(len(val_aucs)):

    print('Fold_%d AUC: %.6f' % (i+1, val_aucs[i]))
val_aucs
# summary on results

auc_mean = np.mean(val_aucs)

auc_std = np.std(val_aucs)

auc_all = roc_auc_score(valid_X.target, valid_X.predict)

print('%d-fold auc mean: %.9f, std: %.9f. All auc: %6f.' % (n_folds, auc_mean, auc_std, auc_all))
y_all = result.values[:, 1:]

result['target'] = np.mean(y_all, axis = 1)

to_submit = result[['ID_code', 'target']]

to_submit.to_csv('NN_submission_{}.csv'.format(GPU), index=None)

result.to_csv('NN_all_prediction_{}.csv'.format(GPU), index=None)

valid_X['ID_code'] = train_df['ID_code']

valid_X = valid_X[['ID_code', 'target', 'predict']].to_csv('NN_oof_{}.csv'.format(GPU), index=None)