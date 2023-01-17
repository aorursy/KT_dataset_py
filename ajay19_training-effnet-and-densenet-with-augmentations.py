!pip install ../input/kerasapplications/keras-team-keras-applications-3b180cb -f ./ --no-index

!pip install ../input/efficientnet/efficientnet-1.1.0/ -f ./ --no-index
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import tensorflow as tf 

import matplotlib.pyplot as plt

import keras

import pydicom

import tqdm

import cv2

from tqdm.notebook import tqdm

from tensorflow.keras import Model

import tensorflow.keras.backend as K

import tensorflow.keras.layers as L

import tensorflow.keras.models as M
from sklearn.model_selection import train_test_split, KFold

from sklearn.metrics import mean_absolute_error

from tensorflow_addons.optimizers import RectifiedAdam

from tensorflow.keras.layers import (

    Dense, Dropout, Activation, Flatten, Input, BatchNormalization, GlobalAveragePooling2D, Add, Conv2D, AveragePooling2D, 

    LeakyReLU, Concatenate 

)

from tensorflow.keras.models import Model

import efficientnet.tfkeras as efn
import random

def seed_everything(seed=2020):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)

    

seed_everything(42)
config = tf.compat.v1.ConfigProto()

config.gpu_options.allow_growth = True

session = tf.compat.v1.Session(config=config)
train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv') 

print(train.shape)

train.head()
def get_tab(df):

    vector = [(df.Age.values[0] - 30) / 30] 

    

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
import albumentations as Alb



augs = {'Original': None,

             'Blur': Alb.Blur(p=1.0),

             #'MedianBlur': A.MedianBlur(blur_limit=5, p=1.0),

             'GaussianBlur': Alb.GaussianBlur(p=1.0),

             'MotionBlur': Alb.MotionBlur(p=1.0),

        'GridDropout': Alb.GridDropout(p=1.0),

        #'CenterCrop': A.CenterCrop(height=256, width=256, p=1.0),

        #'RandomRotate90': A.RandomRotate90(p=1.0),

        # 'ShiftScaleRotate': A.ShiftScaleRotate(p=1.0),

        #'Rotate': A.Rotate()

       }


image = get_img(f'../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/9.dcm')

print("Real SHape = ",image.shape)

for ite,(key, aug) in enumerate(augs.items()):

    if aug is not None:

        image = aug(image=image)['image']

        print("New Shape = ",image.shape)

        plt.imshow(image)
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
from tensorflow.keras.utils import Sequence



class IGenerator(Sequence):

    BAD_ID = ['ID00011637202177653955184', 'ID00052637202186188008618']

    def __init__(self, keys, a, tab, batch_size=32):

        self.keys = [k for k in keys if k not in self.BAD_ID]

        self.a = a

        self.tab = tab

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

        a, tab = [], [] 

        keys = np.random.choice(self.keys, size = self.batch_size)

        for k in keys:

            try:

                i = np.random.choice(self.train_data[k], size=1)[0]

                image = get_img(f'../input/osic-pulmonary-fibrosis-progression/train/{k}/{i}')

                for ite,(key, aug) in enumerate(augs.items()):

                    if aug is not None:

                        image = aug(image=image)['image']

                        x.append(image)

                        a.append(self.a[k])

                        tab.append(self.tab[k])

            except:

                print(k, i)

       

        x,a,tab = np.array(x), np.array(a), np.array(tab)

        #print(len(x),len(a),len(tab))

        x = np.expand_dims(x, axis=-1)

        return [x, tab] , a
def build_model(shape=(512,512,1), model_class=None):

    inp = Input(shape=shape)

    base = efn.EfficientNetB0(input_shape=shape,weights=None,include_top=False)

    base.trainable = False

    x = base(inp)

    x = GlobalAveragePooling2D()(x)

    inp2 = Input(shape=(4,))

    x2 = tf.keras.layers.GaussianNoise(0.2)(inp2)

    x = Concatenate()([x, x2]) 

    x = Dropout(0.5)(x) 

    x = Dense(1)(x)

    model = Model([inp, inp2] , x)

    return model



model = build_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mae')
from sklearn.model_selection import train_test_split 

tr_p, vl_p = train_test_split(P,shuffle=True,train_size= 0.8)
er = tf.keras.callbacks.EarlyStopping(

    monitor="val_loss",

    min_delta=1e-3,

    patience=10,

    verbose=0,

    mode="auto",

    baseline=None,

    restore_best_weights=True,

)



checkpoint_path = "../input/output/training_1/weights{epoch:08d}.h5"

checkpoint_dir = os.path.dirname(checkpoint_path)



# Create a callback that saves the model's weights

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,

                                                 save_weights_only=True,

                                                 verbose=1)
model.fit_generator(IGenerator(keys=tr_p, 

                               a = A, 

                               tab = TAB), 

                    steps_per_epoch = 500,

                    validation_data=IGenerator(keys=vl_p, 

                               a = A, 

                               tab = TAB),

                    validation_steps = 40, 

                    callbacks = [er,cp_callback], 

                    epochs=1)
from keras.applications import DenseNet121

densenet = DenseNet121(

    weights= None,

    include_top=False,

    input_shape=(512,512,1)

)
def build_densenet_model(densenet,shape=(512,512,1)):

    inp = Input(shape=shape)

    #base = efn.EfficientNetB0(input_shape=shape,weights=None,include_top=False)

    densenet.trainable = False

    x = densenet(inp)

    x = GlobalAveragePooling2D()(x)

    inp2 = Input(shape=(4,))

    x2 = tf.keras.layers.GaussianNoise(0.2)(inp2)

    x = Concatenate()([x, x2]) 

    x = Dropout(0.5)(x) 

    x = Dense(1)(x)

    model = Model([inp, inp2] , x)

    return model
model = build_densenet_model(densenet)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='mae')
densenet_path = "../input/output/densenet/weights{epoch:08d}.h5"

new_cp = tf.keras.callbacks.ModelCheckpoint(filepath=densenet_path,

                                                 save_weights_only=True,

                                                 verbose=1)
model.fit_generator(IGenerator(keys=tr_p, 

                               a = A, 

                               tab = TAB), 

                    steps_per_epoch = 500,

                    validation_data=IGenerator(keys=vl_p, 

                               a = A, 

                               tab = TAB),

                    validation_steps = 40, 

                    callbacks = [new_cp], 

                    epochs=1)