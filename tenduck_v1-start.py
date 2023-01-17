# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.model_selection import KFold

from scipy.stats import ttest_ind



import os 

import pandas as pds

import numpy as np



from matplotlib import pyplot as plt

import seaborn as sbn



import tensorflow as tf

import gc



import sys 



def printProgress (iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100): 

    formatStr = "{0:." + str(decimals) + "f}" 

    percent = formatStr.format(100 * (iteration / float(total))) 

    filledLength = int(round(barLength * iteration / float(total))) 

    bar = '#' * filledLength + '-' * (barLength - filledLength) 

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)), 

    if iteration == total: 

        sys.stdout.write('\n') 

    sys.stdout.flush() 

    

tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0],True)
train_set=pds.read_csv(os.path.join('/kaggle/input/lish-moa','train_features.csv'))

test_set=pds.read_csv(os.path.join('/kaggle/input/lish-moa','test_features.csv'))

target_nons=pds.read_csv(os.path.join('/kaggle/input/lish-moa','train_targets_nonscored.csv')).iloc[:,1:]

target_s=pds.read_csv(os.path.join('/kaggle/input/lish-moa','train_targets_scored.csv')).iloc[:,1:]

subs=pds.read_csv(os.path.join('/kaggle/input/lish-moa','sample_submission.csv'))
cp_time=train_set.cp_time

cp_dose=train_set.cp_dose
train_set_squares=(train_set.iloc[:,4:]**2).rename({i:i+'^2' for i in train_set.columns},axis=1)

test_set_squares=(test_set.iloc[:,4:]**2).rename({i:i+'^2' for i in test_set.columns},axis=1)



train_set=pds.concat([train_set,train_set_squares],1)

test_set=pds.concat([test_set,test_set_squares],1)



trs=train_set.iloc[:,4:]

tts=test_set.iloc[:,4:]
def gen_model(input_shape,output_shape):

    inputs=tf.keras.Input((input_shape))

    

    dense=tf.keras.layers.Dense(500)(inputs)

    batch=tf.keras.layers.BatchNormalization()(dense)

    drop=tf.keras.layers.Dropout(0.5)(batch)

    activ=tf.keras.activations.relu(drop,0.3)

    

    dense=tf.keras.layers.Dense(500)(activ)

    batch=tf.keras.layers.BatchNormalization()(dense)

    drop=tf.keras.layers.Dropout(0.5)(batch)

    activ=tf.keras.activations.relu(drop,0.3)

    

    dense=tf.keras.layers.Dense(500)(activ)

    batch=tf.keras.layers.BatchNormalization()(dense)

    drop=tf.keras.layers.Dropout(0.5)(batch)

    activ=tf.keras.activations.relu(drop,0.3)

    

    dense=tf.keras.layers.Dense(500)(activ)

    batch=tf.keras.layers.BatchNormalization()(dense)

    drop=tf.keras.layers.Dropout(0.5)(batch)

    activ=tf.keras.activations.relu(drop,0.3)

    

    dense=tf.keras.layers.Dense(500)(activ)

    batch=tf.keras.layers.BatchNormalization()(dense)

    drop=tf.keras.layers.Dropout(0.5)(batch)

    activ=tf.keras.activations.relu(drop,0.3)

    

    output=tf.keras.layers.Dense(output_shape,activation='sigmoid')(activ)

    

    model=tf.keras.models.Model(inputs,output)

    model.compile(loss=loss_fn,metrics='AUC')

    return model



def loss_fn(y_true,y_pred):

    return tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.transpose(y_true,[1,0]),

                                                              tf.transpose(y_pred,[1,0])),)
smooth_target_s=target_s.applymap(lambda x : 0.05 if x == 0 else 0.95)
kf=KFold(5,shuffle=True)



if not os.path.exists('model'):

    os.mkdir('model')



class_dict=dict()

    

for cpt,cpd in [[x,y] for x in cp_time.unique() for y in cp_dose.unique()]:

    

    

    idx=(cpt==cp_time) & (cpd==cp_dose)

    for n,(tr_idx,vl_idx) in enumerate(kf.split(train_set.loc[idx],smooth_target_s.loc[idx])):

        print(f'Start time : {cpt} / dose : {cpd} / fold : {n}')

        x_train,x_val=trs.loc[idx].values[tr_idx],trs.loc[idx].values[vl_idx]

        y_train,y_val=smooth_target_s.loc[idx].values[tr_idx],smooth_target_s.loc[idx].values[vl_idx]

        

        ys=y_train.sum(0)

        ys[ys==0]=y_train.shape[0]

        class_weights={n:i for n,i in enumerate((y_train.shape[0]-ys)/ys)}

        class_dict[f'{cpt}-{cpd}-{n}']=class_weights

        

        mod=gen_model(trs.shape[1],smooth_target_s.shape[1])

        hitory=mod.fit(x_train,y_train,

                batch_size=100,epochs=10000,verbose=0,

                class_weight=class_weights,

                callbacks=[tf.keras.callbacks.EarlyStopping(patience=50,

                                                            restore_best_weights=True),

                           tf.keras.callbacks.ModelCheckpoint(os.path.join('model',f'{cpt}-{cpd}-{n}.h5')),

                           tf.keras.callbacks.ReduceLROnPlateau()

                          ],

                validation_data=(x_val,y_val))

        print(f'End time : {cpt} / dose : {cpd} / fold : {n}')
for cpt,cpd in [[x,y] for x in cp_time.unique() for y in cp_dose.unique()]:

    for n in range(5):

        name=f'{cpt}-{cpd}-{n}'

        

        if n == 0 :

            cw = np.expand_dims(np.array([i for i in class_dict[name].values()]),-1)

        else:

            cw = np.concatenate([cw,

                                 np.expand_dims(np.array([i for i in class_dict[name].values()]),-1)

                                ],1)

        cw[cw.sum(1)!=0]=(cw[cw.sum(1)!=0]/cw.sum(1,keepdims=True)[cw.sum(1)!=0])

    

    for n in range(5):

        mod=gen_model(tts.shape[1],target_s.shape[1])

        mod.load_weights(os.path.join('model',name+'.h5'))

        pred=np.expand_dims(

            mod.predict(tts.loc[(cpt==test_set.cp_time) & (cpd==test_set.cp_dose)]),-1)

        if n == 0:

            preds=pred

        else:

            preds=np.concatenate([preds,pred],-1)

            

    subs.loc[(cpt==test_set.cp_time) & (cpd==test_set.cp_dose),

             subs.columns[1:]]=(preds*cw).sum(-1)

    

    del mod

    gc.collect()

    tf.keras.backend.clear_session()



subs.loc[test_set.cp_type=='ctl_vehicle',subs.columns[1:]]=0

subs.to_csv('submission.csv',index=False)