Dropout_model = 0.385

FVC_weight = 0.2

Confidence_weight = 0.2

!pip install ../input/kerasapplications/keras-team-keras-applications-3b180cb -f ./ --no-index

!pip install ../input/efficientnet/efficientnet-1.1.0/ -f ./ --no-index

import os

import cv2

import pydicom

import pandas as pd

import numpy as np 

import tensorflow as tf 

import matplotlib.pyplot as plt 

import random

from tqdm.notebook import tqdm 

from sklearn.model_selection import train_test_split, KFold

from sklearn.metrics import mean_absolute_error

from tensorflow_addons.optimizers import RectifiedAdam

from tensorflow.keras import Model

import tensorflow.keras.backend as K

import tensorflow.keras.layers as L

import tensorflow.keras.models as M

from tensorflow.keras.optimizers import Nadam

import seaborn as sns

from PIL import Image
try: # detect TPUs

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect() # TPU detection

    strategy = tf.distribute.TPUStrategy(tpu)

except ValueError: # detect GPUs

    strategy = tf.distribute.MirroredStrategy() # for GPU or multi-GPU machines

    #strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU

    #strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy() # for clusters of multi-GPU machines



print("Number of accelerators: ", strategy.num_replicas_in_sync)
def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)

    

seed_everything(1)

config = tf.compat.v1.ConfigProto()

config.gpu_options.allow_growth = True

session = tf.compat.v1.Session(config=config)

train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv') 
tr_age_max = train.Age.values.max()

tr_age_min = train.Age.values.min()

tr_age_mean = train.Age.values.mean()

def expandFVC_Percent(DS):

    DS['FVC_Percent'] = DS['FVC'] / DS['Percent']

    DS['FVC_Percent'] = (DS['FVC_Percent'] - DS['FVC_Percent'].min()) / (DS['FVC_Percent'].max() - DS['FVC_Percent'].min())

    return DS
def expandMinFVC(data_set):

    data_set['MinWeeks'] = data_set.groupby('Patient')['Weeks'].transform('min')

    sub_tr = data_set.loc[data_set['Weeks']==data_set['MinWeeks']]



    sub_tr['MinFVC'] = sub_tr['FVC']



    sub_tr = sub_tr[['Patient', 'MinFVC']]



    merged = data_set.merge(sub_tr, on='Patient', how='outer')





    data_set=merged

    

    data_set['MinFVC'] = (data_set['MinFVC'] - data_set['MinFVC'].min())/(data_set['MinFVC'].max()-data_set['MinFVC'].min())

    print(data_set.sample(5))

    return data_set

train = expandMinFVC(train)

# train = expandFVC_Percent(train)



print(train.head())

print(train.sample(5))


def get_tab(df):

    # df: data frame

    # [age, male/female[0/1], SmokingStatus[00/11/01], BaseFvc] (5,)

    vector = [(df.Age.values[0] - tr_age_min) / (tr_age_max-tr_age_min)] 

    

    if df.Sex.values[0] == 'male':

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

    vector.extend([df['MinFVC'].values[0]])

#     vector.extend([df['FVC_Percent'].values[0]])

    return np.array(vector) 
# s = f'../input/osic-pulmonary-fibrosis-progression/train/{k}/{i}'

print(train.columns)

k =  train.Patient[102]

print(k)

i = 10

path = f'../input/osic-pulmonary-fibrosis-progression/train/{k}/{i}.dcm'

d = pydicom.dcmread(path)

print(d.pixel_array/(2**11))

# print(dir(d))

print(d)


A = {} 

TAB = {} 

P = [] 

for i, p in tqdm(enumerate(train.Patient.unique())):

    sub = train.loc[train.Patient == p, :] 

    fvc = sub.FVC.values

    weeks = sub.Weeks.values

    c = np.vstack([weeks, np.ones(len(weeks))]).T

    a, b = np.linalg.lstsq(c, fvc)[0]

    

    A[p] = a

    TAB[p] = get_tab(sub)

    P.append(p)



def get_img(path):

    d = pydicom.dcmread(path)

    return cv2.resize(d.pixel_array / 2**11, (512, 512))
from tensorflow.keras.utils import Sequence

import math



class IGenerator(Sequence):

    BAD_ID = ['ID00011637202177653955184', 'ID00052637202186188008618']

    def __init__(self, keys, a, tab, batch_size=4):

        self.keys = [k for k in keys if k not in self.BAD_ID]

        self.a = a

        self.tab = tab

        self.batch_size = batch_size

        

        self.train_data = {}  # imgs for each person; self.trian_data[pi] = [img1, img2, ..., imgn]

        for p in train.Patient.values:

            # filenames in "~/train/personid/"

            self.train_data[p] = os.listdir(f'../input/osic-pulmonary-fibrosis-progression/train/{p}/')

    

    def __len__(self):

        return 100

#         return math.ceil(len(self.keys) / float(self.batch_size)) 

    

    def __getitem__(self, idx):

        x = []

        a, tab = [], [] 

        keys = np.random.choice(self.keys, size = self.batch_size)  # persons in batch size

        for k in keys:

            #  chose a {[img, tab], a}for each person (only 1 img)

            try:

                self.train_data[k].sort()

                mx_len = len(self.train_data[k])

                i = np.random.choice(self.train_data[k][mx_len//3:mx_len//3*2], size=1)[0]

#                 i = self.train_data[k][max_len//2]

                img = get_img(f'../input/osic-pulmonary-fibrosis-progression/train/{k}/{i}')

                x.append(img)

                a.append(self.a[k])

                tab.append(self.tab[k])

            except:

                print(k, i)

       

        x, a, tab = np.array(x), np.array(a), np.array(tab)

        x = np.expand_dims(x, axis=-1)

        return [x, tab] , a
from tensorflow.keras.layers import (

    Dense, Dropout, Activation, Flatten, Input, BatchNormalization, GlobalAveragePooling2D, Add, Conv2D, AveragePooling2D, 

    LeakyReLU, Concatenate 

)

import efficientnet.tfkeras as efn



def get_efficientnet(model, shape):

    models_dict = {

        'b0': efn.EfficientNetB0(input_shape=shape,weights=None,include_top=False),

        'b1': efn.EfficientNetB1(input_shape=shape,weights=None,include_top=False),

        'b2': efn.EfficientNetB2(input_shape=shape,weights=None,include_top=False),

        'b3': efn.EfficientNetB3(input_shape=shape,weights=None,include_top=False),

        'b4': efn.EfficientNetB4(input_shape=shape,weights=None,include_top=False),

        'b5': efn.EfficientNetB5(input_shape=shape,weights=None,include_top=False),

        'b6': efn.EfficientNetB6(input_shape=shape,weights=None,include_top=False),

        'b7': efn.EfficientNetB7(input_shape=shape,weights=None,include_top=False)

    }

    return models_dict[model]



def build_model(shape=(512, 512, 1), model_class=None):

        inp = Input(shape=shape)

        base = get_efficientnet(model_class, shape)

        x = base(inp)

        x = GlobalAveragePooling2D()(x)

    #     inp2 = Input(shape=(4,))  # original is 4

        inp2 = Input(shape=(5,))  # add the feature of MinFVC

    #     inp2 = Input(shape=(6,))  # add the feature of MinFVC + FVC/Percent

        x2 = tf.keras.layers.GaussianNoise(0.2)(inp2)

        x = Concatenate()([x, x2]) 

        x = Dropout(Dropout_model)(x)

        x = Dense(1)(x)

        model = Model([inp, inp2] , x)



    #     weights = [w for w in os.listdir('../input/osic-model-weights') if model_class in w][0]

    #     model.load_weights('../input/osic-model-weights/' + weights)

        return model



def get_model(shape=(512, 512, 1)):

        def res_block(x, n_features):

            _x = x

            x = BatchNormalization()(x)

            x = LeakyReLU(0.05)(x)



            x = Conv2D(n_features, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)

            x = Add()([_x, x])

            return x



        inp = Input(shape=shape)



        # 512

        x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same')(inp)

        x = BatchNormalization()(x)

        x = LeakyReLU(0.05)(x)



        x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)

        x = BatchNormalization()(x)

        x = LeakyReLU(0.05)(x)



        x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)



        # 256

        x = Conv2D(8, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)

        for _ in range(2):

            x = res_block(x, 8)

        x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)



        # 128

        x = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)

        for _ in range(2):

            x = res_block(x, 16)

        x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)



        # 64

        x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)

        for _ in range(3):

            x = res_block(x, 32)

        x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)



        # 32

        x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)

        for _ in range(3):

            x = res_block(x, 64)

        x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)    



        # 16

        x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)

        for _ in range(3):

            x = res_block(x, 128)



        # 16

        x = GlobalAveragePooling2D()(x)



        inp2 = Input(shape=(5,))

        x2 = tf.keras.layers.GaussianNoise(0.2)(inp2)

        x = Concatenate()([x, x2]) 

        x = Dropout(0.6)(x) 

        x = Dense(1)(x)

        #x2 = Dense(1)(x)

        return Model([inp, inp2] , x)

    




model_classes = ['b3'] #['b0','b1','b2','b3',b4','b5','b6','b7']

models = [build_model(shape=(512, 512, 1), model_class=m) for m in model_classes]

print('Number of models: ' + str(len(models)))



model = models[0]



# model = get_model()

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mae')  # original is lr=0.001



from sklearn.model_selection import train_test_split 

tr_p, vl_p = train_test_split(P, shuffle=True, train_size = 0.8) 



er = tf.keras.callbacks.EarlyStopping(

    monitor="val_loss",

    min_delta=1e-5,

    patience=5,

    verbose=0,

    mode="auto",

    baseline=None,

    restore_best_weights=True,

)





model.fit_generator(IGenerator(keys=tr_p, 

                                   a = A, 

                                   tab = TAB), 

    #                     steps_per_epoch = 100,

                        validation_data=IGenerator(keys=vl_p, 

                                   a = A, 

                                   tab = TAB),

    #                     validation_steps = 20, 

    #                     callbacks = [er], 

                        epochs=300)



    # model.fit(Xtrain, Ytrain, batch_size = 32, epochs = 100)
weights_path = "./efn.ckpt"

if not os.path.exists(weights_path):

    os.mkdir(weights_path)

    

model.save_weights(weights_path)







def score(fvc_true, fvc_pred, sigma):

    sigma_clip = np.maximum(sigma, 70) # changed from 70, trie 66.7 too

    delta = np.abs(fvc_true - fvc_pred)

    delta = np.minimum(delta, 1000)

    sq2 = np.sqrt(2)

    metric = (delta / sigma_clip)*sq2 + np.log(sigma_clip* sq2)

    return np.mean(metric)

subs = []

from sklearn.model_selection import train_test_split 

tr_p, vl_p = train_test_split(P, shuffle=True, train_size = 0.8) 





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

                x.append(get_img(f'../input/osic-pulmonary-fibrosis-progression/train/{p}/{i}')) 

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

#         m.append(score(fvc_true, fvc, percent))

        m.append(score(fvc_true, fvc, np.sqrt(2) * (np.abs(fvc_true - fvc))))

    print(np.mean(m))

    metric.append(np.mean(m))

q = (np.argmin(metric) + 1)/ 10

print(q)

with open('q.txt', 'w') as f:

    f.write(str(q)+'\n')

# !ls

# !rm -r checkpoint
# model = build_model(shape=(512, 512, 1), model_class='b0')
# mode.load_weights(weigths_path)