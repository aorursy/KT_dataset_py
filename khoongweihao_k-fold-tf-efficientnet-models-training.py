!pip install ../input/kerasapplications/keras-team-keras-applications-3b180cb -f ./ --no-index

!pip install ../input/efficientnet/efficientnet-1.1.0/ -f ./ --no-index
import os

import cv2

import pydicom

import pandas as pd

import numpy as np 

import tensorflow as tf 

import matplotlib.pyplot as plt 

from tqdm.notebook import tqdm 

from tensorflow.keras.layers import (

    Dense, Dropout, Activation, Flatten, Input, BatchNormalization, GlobalAveragePooling2D, Add, Conv2D, AveragePooling2D, 

    LeakyReLU, Concatenate 

)

from tensorflow.keras import Model

from tensorflow.keras.utils import Sequence

import tensorflow.keras.backend as K

import tensorflow.keras.applications as tfa

import efficientnet.tfkeras as efn

from sklearn.model_selection import train_test_split, KFold

import seaborn as sns
config = tf.compat.v1.ConfigProto()

config.gpu_options.allow_growth = True

session = tf.compat.v1.Session(config=config)
EPOCHS = 2

BATCH_SIZE = 8

NFOLD = 5

LR = 0.003

SAVE_BEST = True

MODEL_CLASS = 'b1'
train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv') 
train.head()
train.SmokingStatus.unique()
def get_tab(df):

    vector = [(df.Age.values[0] - 30) / 30] 

    

    if df.Sex.values[0].lower() == 'male':

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

    c = np.vstack([weeks, np.ones(len(weeks))]).T

    a, b = np.linalg.lstsq(c, fvc)[0]

    

    A[p] = a

    TAB[p] = get_tab(sub)

    P.append(p)
def get_img(path):

    d = pydicom.dcmread(path)

    return cv2.resize((d.pixel_array - d.RescaleIntercept) / (d.RescaleSlope * 1000), (512, 512))
x, y = [], []

for p in tqdm(train.Patient.unique()):

    try:

        ldir = os.listdir(f'../input/osic-pulmonary-fibrosis-progression-lungs-mask/mask_noise/mask_noise/{p}/')

        numb = [float(i[:-4]) for i in ldir]

        for i in ldir:

            x.append(cv2.imread(f'../input/osic-pulmonary-fibrosis-progression-lungs-mask/mask_noise/mask_noise/{p}/{i}', 0).mean())

            y.append(float(i[:-4]) / max(numb))

    except:

        pass
class IGenerator(Sequence):

    BAD_ID = ['ID00011637202177653955184', 'ID00052637202186188008618']

    def __init__(self, keys, a, tab, batch_size=BATCH_SIZE):

        self.keys = [k for k in keys if k not in self.BAD_ID]

        self.a = a

        self.tab = tab

        self.batch_size = batch_size

        

        self.train_data = {}

        for p in train.Patient.values:

            self.train_data[p] = os.listdir(f'../input/osic-pulmonary-fibrosis-progression/train/{p}/')

    

    def __len__(self):

        return 1000

    

    def __getitem__(self, idx):

        x = []

        a, tab = [], [] 

        keys = np.random.choice(self.keys, size = self.batch_size)

        for k in keys:

            try:

                i = np.random.choice(self.train_data[k], size=1)[0]

                img = get_img(f'../input/osic-pulmonary-fibrosis-progression/train/{k}/{i}')

                x.append(img)

                a.append(self.a[k])

                tab.append(self.tab[k])

            except:

                print(k, i)

       

        x,a,tab = np.array(x), np.array(a), np.array(tab)

        x = np.expand_dims(x, axis=-1)

        return [x, tab] , a
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

    inp2 = Input(shape=(4,))

    x2 = tf.keras.layers.GaussianNoise(0.2)(inp2)

    x = Concatenate()([x, x2]) 

    x = Dropout(0.5)(x) 

    x = Dense(1)(x)

    model = Model([inp, inp2] , x)

    return model
kf = KFold(n_splits=NFOLD, random_state=42,shuffle=False)

P = np.array(P)

subs = []

folds_history = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(P)):

    print('#####################')

    print('####### Fold %i ######'%fold)

    print('#####################')

    print('Training...')

    

    er = tf.keras.callbacks.EarlyStopping(

        monitor="val_loss",

        min_delta=1e-3,

        patience=10,

        verbose=1,

        mode="auto",

        baseline=None,

        restore_best_weights=True,

    )



    cpt = tf.keras.callbacks.ModelCheckpoint(

        filepath='fold-%i.h5'%fold,

        monitor='val_loss', 

        verbose=1, 

        save_best_only=SAVE_BEST,

        mode='auto'

    )



    rlp = tf.keras.callbacks.ReduceLROnPlateau(

        monitor='val_loss', 

        factor=0.5,

        patience=5, 

        verbose=1, 

        min_lr=1e-8

    )

    model = build_model(model_class=MODEL_CLASS)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR), loss="mae") 

    history = model.fit_generator(IGenerator(keys=P[tr_idx], 

                                   a = A, 

                                   tab = TAB), 

                        steps_per_epoch = 32,

                        validation_data=IGenerator(keys=P[val_idx], 

                                   a = A, 

                                   tab = TAB),

                        validation_steps = 16, 

                        callbacks = [cpt, rlp], 

                        epochs=EPOCHS)

    folds_history.append(history.history)

    print('Training done!')
if SAVE_BEST:

    mean_val_loss = np.mean([np.min(h['val_loss']) for h in folds_history])

else:

    mean_val_loss = np.mean([h['val_loss'][-1] for h in folds_history])

print('Our mean CV MAE is: ' + str(mean_val_loss))