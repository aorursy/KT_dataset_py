import os

import cv2



import pydicom

import pandas as pd

import numpy as np 

import tensorflow as tf 

import matplotlib.pyplot as plt 



from tqdm.notebook import tqdm 
train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv') 
train.head()
train.SmokingStatus.unique()
def get_tab(df):

    vector = [(df.Age.values[0] - 30) / 30] 

    

    if df.Sex.values[0] == 'Male':

       vector.append(0)

    else:

       vector.append(1)

    

    if df.SmokingStatus.values[0] == 'Never smoked':

        vector.extend([0,0])

    elif df.SmokingStatus.values[0] == 'Ex-smoker':

        vector.extend([1,1])

    elif df.SmokingStatus.values[0] == 'Currently smokes':

        vector.extend([0,1])

    else:

        vector.extend([1,0])

    return np.array(vector) 
A = {} 

TAB = {} 

P = [] 

for i, p in tqdm(enumerate(train.Patient.unique())):

    sub = train.loc[train.Patient == p, :] 

    fvc = sub.FVC.values

    weeks = sub.Weeks.values

    if len(weeks) < 7:

        BAD_ID.append(p)

        continue

    c = np.vstack([weeks, np.ones(len(weeks))]).T

    a, b = np.linalg.lstsq(c, fvc)[0]

    

    A[p] = a

    TAB[p] = get_tab(sub)

    P.append(p)

len(P)
BAD_ID = ['ID00011637202177653955184', 'ID00052637202186188008618']
def get_img(path):

    d = pydicom.dcmread(path)

    return cv2.resize((d.pixel_array - d.RescaleIntercept) / (d.RescaleSlope * 1024), (512, 512))
from tensorflow.keras.utils import Sequence



class IGenerator(Sequence):

    BAD_ID = BAD_ID

    def __init__(self, keys, batch_size=32):

        self.keys = [k for k in keys if k not in self.BAD_ID]

        self.batch_size = batch_size

        

        self.train_data = {}

        for p in train.Patient.unique():

            ldir = os.listdir(f'../input/osic-pulmonary-fibrosis-progression/train/{p}/')

            numb = [float(i[:-4]) for i in ldir]

            self.train_data[p] = [i for i in os.listdir(f'../input/osic-pulmonary-fibrosis-progression/train/{p}/') 

                                  if int(i[:-4]) / len(ldir) < 0.8 and int(i[:-4]) / len(ldir) > 0.15]

    

    def __len__(self):

        return 1000

    

    def __getitem__(self, idx):

        x = []

        a_vector, tab_vector = [], [] 

        keys = np.random.choice(self.keys, size = self.batch_size)

        for k in keys:

            sub = train.loc[train.Patient == p, :]

            idx = np.random.choice(sub.index, size=int(len(sub) * 0.8))

            fvc = sub.loc[idx, 'FVC'].values

            weeks = sub.loc[idx, 'Weeks'].values



            c = np.vstack([weeks, np.ones(len(weeks))]).T

            a, b = np.linalg.lstsq(c, fvc)[0]

                

            i = np.random.choice(self.train_data[k], size=1)[0]

            img = get_img(f'../input/osic-pulmonary-fibrosis-progression/train/{k}/{i}')

            mask = cv2.resize(cv2.imread(f'../input/osic-pulmonary-fibrosis-progression-lungs-mask/mask_clear/mask_clear/{k}/{i[:-4]}.jpg', 0), (512, 512))> 0



            x.append(np.dstack([img, mask]))

            a_vector.append(a)

            tab_vector.append(get_tab(sub))

       

        x, a_vector, tab_vector = np.array(x), np.array(a_vector), np.array(tab_vector)

        return [x, tab_vector] , a_vector
from tensorflow.keras.layers import (

    Dense, Dropout, Activation, Flatten, Input, BatchNormalization, GlobalAveragePooling2D, Add, Conv2D, AveragePooling2D, 

    LeakyReLU, Concatenate 

)



from tensorflow.keras import Model

from tensorflow.keras.optimizers import Nadam



def get_model(shape=(512, 512, 2)):

    def res_block(x, n_features):

        _x = x

        x = BatchNormalization()(x)

        x = LeakyReLU(0.)(x)

    

        x = Conv2D(n_features, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)

        x = Add()([_x, x])

        return x

    

    inp = Input(shape=shape)

    

    # 512

    x = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same')(inp)

    x = BatchNormalization()(x)

    x = LeakyReLU(0.)(x)

    

    x = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)

    x = BatchNormalization()(x)

    x = LeakyReLU(0.)(x)

    

    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    

    # 256

    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)

    for _ in range(2):

        x = res_block(x, 32)

    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    

    # 128

    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)

    for _ in range(2):

        x = res_block(x, 64)

    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    

    # 64

    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)

    for _ in range(3):

        x = res_block(x, 128)

    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    

    # 32

    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)

    for _ in range(3):

        x = res_block(x, 128)

    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)    

    

    # 16

    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)

    for _ in range(3):

        x = res_block(x, 64)

    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    

    # 8

    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)

    for _ in range(3):

        x = res_block(x, 32)

        

    # 16

    x = GlobalAveragePooling2D()(x)

    

    inp2 = Input(shape=(4,))

    x2 = tf.keras.layers.GaussianNoise(0.2)(inp2)

    x = Dropout(0.5)(x)

    x = Concatenate()([x, x2])  

    x = Dense(1)(x)

    #x2 = Dense(1)(x)

    return Model([inp, inp2] , x)
model = get_model() 

model.summary() 
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.003), loss='mae') 
from sklearn.model_selection import train_test_split 



tr_p, vl_p = train_test_split(P, 

                              shuffle=True, 

                              train_size= 0.8) 
import seaborn as sns



sns.distplot(list(A.values()));
er = tf.keras.callbacks.EarlyStopping(

    monitor="val_loss",

    min_delta=1e-3,

    patience=10,

    verbose=0,

    mode="auto",

    baseline=None,

    restore_best_weights=True,

)
model.fit_generator(IGenerator(keys=tr_p), 

                    steps_per_epoch = 100,

                    validation_data=IGenerator(keys=vl_p),

                    validation_steps = 20, 

                    callbacks = [er], 

                    epochs=30)
def score(fvc_true, fvc_pred, sigma):

    sigma_clip = np.maximum(sigma, 70)

    delta = np.abs(fvc_true - fvc_pred)

    delta = np.minimum(delta, 1000)

    sq2 = np.sqrt(2)

    metric = (delta / sigma_clip)*sq2 + np.log(sigma_clip* sq2)

    return np.mean(metric)
from tqdm.notebook import tqdm



metric = []

for q in tqdm(range(1, 10)):

    m = []

    for p in vl_p:

        x = [] 

        tab = [] 

        

        if p in ['ID00011637202177653955184', 'ID00052637202186188008618']:

            continue

            

        ldir = os.listdir(f'../input/osic-pulmonary-fibrosis-progression/train/{p}/')

        for i in ldir:

            if int(i[:-4]) / len(ldir) < 0.8 and int(i[:-4]) / len(ldir) > 0.15:

                img = get_img(f'../input/osic-pulmonary-fibrosis-progression/train/{p}/{i}')

                mask = cv2.resize(cv2.imread(f'../input/osic-pulmonary-fibrosis-progression-lungs-mask/mask_clear/mask_clear/{p}/{i[:-4]}.jpg', 0), (512, 512))> 0



                x.append(np.dstack([img, mask]))

                tab.append(get_tab(train.loc[train.Patient == p, :])) 

        if len(x) < 1:

            continue

        tab = np.array(tab) 

    

        x = np.expand_dims(x, axis=-1) 

        _a = model.predict([x, tab]) 

        a = np.quantile(_a, q / 10)

        

        percent_true = train.Percent.values[train.Patient == p]

        fvc_true = train.FVC.values[train.Patient == p]

        weeks_true = train.Weeks.values[train.Patient == p]

        

        fvc = a * (weeks_true - weeks_true[0]) + fvc_true[0]

        percent = percent_true[0] - a * abs(weeks_true - weeks_true[0])

        m.append(score(fvc_true, fvc, percent))

    print(np.mean(m))

    metric.append(np.mean(m))
q = (np.argmin(metric) + 1)/ 10

q
sub = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv') 

sub.head() 
test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv') 

test.head()
A_test, B_test, P_test,W, FVC= {}, {}, {},{},{} 

STD, WEEK = {}, {} 

for p in test.Patient.unique():

    x = [] 

    tab = [] 

    ldir = os.listdir(f'../input/osic-pulmonary-fibrosis-progression/test/{p}/')

    for i in ldir:

        if int(i[:-4]) / len(ldir) < 0.8 and int(i[:-4]) / len(ldir) > 0.15:

            x.append(get_img(f'../input/osic-pulmonary-fibrosis-progression/test/{p}/{i}')) 

            tab.append(get_tab(test.loc[test.Patient == p, :])) 

    if len(x) <= 1:

        continue

    tab = np.array(tab) 

            

    x = np.expand_dims(x, axis=-1) 

    _a = model.predict([x, tab]) 

    a = np.quantile(_a, q)

    A_test[p] = a

    B_test[p] = test.FVC.values[test.Patient == p] - a*test.Weeks.values[test.Patient == p]

    P_test[p] = test.Percent.values[test.Patient == p] 

    WEEK[p] = test.Weeks.values[test.Patient == p]
for k in sub.Patient_Week.values:

    p, w = k.split('_')

    w = int(w) 

    

    fvc = A_test[p] * w + B_test[p]

    sub.loc[sub.Patient_Week == k, 'FVC'] = fvc

    sub.loc[sub.Patient_Week == k, 'Confidence'] = (

        P_test[p] + A_test[p] * (w - WEEK[p]) 

) 

    
sub.head()
sub[["Patient_Week","FVC","Confidence"]].to_csv("submission.csv", index=False)