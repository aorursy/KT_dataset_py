import os

import json

import math



import cv2

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from skimage.transform import resize

from tqdm import tqdm_notebook as tqdm

import keras

from keras import layers

from keras.applications import DenseNet121

from keras.callbacks import Callback, ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Dense, Dropout, Activation, Flatten, Input, BatchNormalization, UpSampling2D, Add

from keras.layers import Conv2D, MaxPooling2D, LeakyReLU

from keras import Model

from keras.optimizers import Adam, Nadam

from sklearn.decomposition import PCA, KernelPCA

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, classification_report
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
def crop_image1(img,tol=5):

    # img is image data

    # tol  is tolerance

        

    mask = img>tol

    return img[np.ix_(mask.any(1),mask.any(0))]
train_resized_imgs = []



for image_id in tqdm(train_df['id_code']):

    path=f"../input/train_images/{image_id}.png"

    img = cv2.imread(path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



    img = crop_image1(img)

    img = cv2.resize(img, (224, 224))

    img=cv2.addWeighted (img,4, cv2.GaussianBlur(img , (0,0) , 224/10) ,-4 ,128)

    train_resized_imgs.append(img)
test_resized_imgs = []



for image_id in tqdm(test_df['id_code']):

    path=f"../input/test_images/{image_id}.png"

    img = cv2.imread(path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



    img = crop_image1(img)

    img = cv2.resize(img, (224, 224))

    img=cv2.addWeighted (img,4, cv2.GaussianBlur(img , (0,0) , 224/10) ,-4 ,128)

    test_resized_imgs.append(img)
y = train_df['diagnosis'].values
train_idx = len(train_resized_imgs)

train_resized_imgs.extend(test_resized_imgs)



train_resized_imgs = np.expand_dims(train_resized_imgs, axis=-1)
dataGen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        zca_epsilon=1e-06,  # epsilon for ZCA whitening

        rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        shear_range=0.,  # set range for random shear

        zoom_range=[0.75, 1.25],  # set range for random zoom

        channel_shift_range=0.05,  # set range for random channel shifts

        # set mode for filling points outside the input boundaries

        fill_mode='constant',

        cval=0.,  # value used for fill_mode = "constant"

        horizontal_flip=True,  # randomly flip images

        vertical_flip=True,  # randomly flip images

        rescale=1/255.,

        # set function that will be applied on each input

        preprocessing_function=None

    ).flow(np.array(train_resized_imgs), np.array(train_resized_imgs), batch_size=64)



def generator():

    for x, _ in dataGen:

        yield x, x
def get_encoder(shape=(224, 224, 2)):

    def res_block(x, n_features):

        _x = x

        x = BatchNormalization()(x)

        x = LeakyReLU()(x)

    

        x = Conv2D(n_features, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)

        x = Add()([_x, x])

        return x

    

    inp = Input(shape=shape)

    

    # 224

    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same')(inp)

    x = BatchNormalization()(x)

    x = LeakyReLU()(x)

    

    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)

    x = BatchNormalization()(x)

    x = LeakyReLU()(x)

    

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    

    # 112

    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)

    for _ in range(2):

        x = res_block(x, 32)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    

    # 56

    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)

    for _ in range(2):

        x = res_block(x, 32)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    

    # 28

    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)

    for _ in range(3):

        x = res_block(x, 64)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    

    # 14

    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)

    for _ in range(3):

        x = res_block(x, 64)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)    

    

    # 7

    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)

    for _ in range(3):

        x = res_block(x, 64)

    

    x = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)

    return Model(inp, x)
def get_decoder(shape=(7, 7, 128)):

    inp = Input(shape=shape)



    x = UpSampling2D((2, 2))(inp)

    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)

    x = BatchNormalization()(x)

    x = LeakyReLU()(x)

    

    x = UpSampling2D((2, 2))(x)

    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)

    x = BatchNormalization()(x)

    x = LeakyReLU()(x)

    

    x = UpSampling2D((2, 2))(x)

    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)

    x = BatchNormalization()(x)

    x = LeakyReLU()(x)

    

    x = UpSampling2D((2, 2))(x)

    x = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)

    x = BatchNormalization()(x)

    x = LeakyReLU()(x)

    

    x = UpSampling2D((2, 2))(x)

    x = Conv2D(9, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)

    x = BatchNormalization()(x)

    x = LeakyReLU()(x)

    

    x = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)

    return Model(inp, x)
from keras.callbacks import *



class CyclicLR(Callback):

    """This callback implements a cyclical learning rate policy (CLR).

    The method cycles the learning rate between two boundaries with

    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).

    The amplitude of the cycle can be scaled on a per-iteration or 

    per-cycle basis.

    This class has three built-in policies, as put forth in the paper.

    "triangular":

        A basic triangular cycle w/ no amplitude scaling.

    "triangular2":

        A basic triangular cycle that scales initial amplitude by half each cycle.

    "exp_range":

        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 

        cycle iteration.

    For more detail, please see paper.

    

    # Example

        ```python

            clr = CyclicLR(base_lr=0.001, max_lr=0.006,

                                step_size=2000., mode='triangular')

            model.fit(X_train, Y_train, callbacks=[clr])

        ```

    

    Class also supports custom scaling functions:

        ```python

            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))

            clr = CyclicLR(base_lr=0.001, max_lr=0.006,

                                step_size=2000., scale_fn=clr_fn,

                                scale_mode='cycle')

            model.fit(X_train, Y_train, callbacks=[clr])

        ```    

    # Arguments

        base_lr: initial learning rate which is the

            lower boundary in the cycle.

        max_lr: upper boundary in the cycle. Functionally,

            it defines the cycle amplitude (max_lr - base_lr).

            The lr at any cycle is the sum of base_lr

            and some scaling of the amplitude; therefore 

            max_lr may not actually be reached depending on

            scaling function.

        step_size: number of training iterations per

            half cycle. Authors suggest setting step_size

            2-8 x training iterations in epoch.

        mode: one of {triangular, triangular2, exp_range}.

            Default 'triangular'.

            Values correspond to policies detailed above.

            If scale_fn is not None, this argument is ignored.

        gamma: constant in 'exp_range' scaling function:

            gamma**(cycle iterations)

        scale_fn: Custom scaling policy defined by a single

            argument lambda function, where 

            0 <= scale_fn(x) <= 1 for all x >= 0.

            mode paramater is ignored 

        scale_mode: {'cycle', 'iterations'}.

            Defines whether scale_fn is evaluated on 

            cycle number or cycle iterations (training

            iterations since start of cycle). Default is 'cycle'.

    """



    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',

                 gamma=1., scale_fn=None, scale_mode='cycle'):

        super(CyclicLR, self).__init__()



        self.base_lr = base_lr

        self.max_lr = max_lr

        self.step_size = step_size

        self.mode = mode

        self.gamma = gamma

        if scale_fn == None:

            if self.mode == 'triangular':

                self.scale_fn = lambda x: 1.

                self.scale_mode = 'cycle'

            elif self.mode == 'triangular2':

                self.scale_fn = lambda x: 1/(2.**(x-1))

                self.scale_mode = 'cycle'

            elif self.mode == 'exp_range':

                self.scale_fn = lambda x: gamma**(x)

                self.scale_mode = 'iterations'

        else:

            self.scale_fn = scale_fn

            self.scale_mode = scale_mode

        self.clr_iterations = 0.

        self.trn_iterations = 0.

        self.history = {}



        self._reset()



    def _reset(self, new_base_lr=None, new_max_lr=None,

               new_step_size=None):

        """Resets cycle iterations.

        Optional boundary/step size adjustment.

        """

        if new_base_lr != None:

            self.base_lr = new_base_lr

        if new_max_lr != None:

            self.max_lr = new_max_lr

        if new_step_size != None:

            self.step_size = new_step_size

        self.clr_iterations = 0.

        

    def clr(self):

        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))

        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)

        if self.scale_mode == 'cycle':

            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)

        else:

            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)

        

    def on_train_begin(self, logs={}):

        logs = logs or {}



        if self.clr_iterations == 0:

            K.set_value(self.model.optimizer.lr, self.base_lr)

        else:

            K.set_value(self.model.optimizer.lr, self.clr())        

            

    def on_batch_end(self, epoch, logs=None):

        

        logs = logs or {}

        self.trn_iterations += 1

        self.clr_iterations += 1



        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))

        self.history.setdefault('iterations', []).append(self.trn_iterations)



        for k, v in logs.items():

            self.history.setdefault(k, []).append(v)

        

        K.set_value(self.model.optimizer.lr, self.clr())
encoder = get_encoder((224, 224, 1))

decoder = get_decoder((7, 7, 1))
inp = Input((224, 224, 1))

e = encoder(inp)

d = decoder(e)

model = Model(inp, d)
from keras import backend as K



def mask_mse(y_true, y_pred):

    mask = 1. - 1. / ( 1. + K.exp(-y_true**2))

    return K.abs(y_true*mask - y_pred*mask)
model.compile(optimizer=Nadam(lr=2*1e-3, schedule_decay=1e-5), loss='mse')
model.summary()
model.fit_generator(generator(), steps_per_epoch=500, epochs=5, callbacks=[

    CyclicLR(base_lr=8*1e-4, max_lr=6*1e-3, step_size=250, gamma=0.9)

])
vec = encoder.predict(np.array(train_resized_imgs)/255.)
avec = np.array([v.flatten()

                 for v in vec])
sc = MinMaxScaler()

avec = sc.fit_transform(avec)
train_idx = 3662
pca = PCA(n_components=3)

emb = pca.fit_transform(avec)

te_emb = emb[train_idx:]

emb = emb[:train_idx]
pca.explained_variance_ratio_[:5]
labelMap = {

    0:'No DR',

    1:'Mild',

    2:'Moderate',

    3:'Severe',

    4:'Proliferative DR'

}
plt.figure(figsize=(10, 10))



for t in list(set(y)):

    plt.plot(emb[np.array(y) == t, 0], emb[np.array(y) == t, 1], '.', label=labelMap[t], alpha=0.75)

plt.plot(te_emb[:, 0], te_emb[:, 1], '.', label='test data', color='gray', alpha=0.45)



plt.xlabel('component 0')

plt.ylabel('component 1')

plt.legend();
import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
data = [go.Scatter3d(

    x=emb[np.array(y) == t, 0],

    y=emb[np.array(y) == t, 1],

    z=emb[np.array(y) == t, 2],

    mode='markers',

    marker=dict(

        size=3,

        opacity=0.75

    ),

    name=labelMap[t]

)

    for t in list(set(y))

]



data.append(go.Scatter3d(

    x=te_emb[:, 0],

    y=te_emb[:, 1],

    z=te_emb[:, 2],

    mode='markers',

    marker=dict(

        color='#c0c0c0',

        size=2,

        opacity=0.75

    ),

    name='test data'

))



layout = go.Layout(

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0

    ),

)

fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='simple-3d-scatter')
len(emb), len(te_emb), len(train_resized_imgs), len(train_df), len(test_df)
dist_map = {}

count = 0

for i, tr in enumerate(emb):

    dist = np.mean(np.abs(te_emb - tr), axis=-1)

    dist_map[i] = np.where(dist < 1e-8)[0]

    if len(dist_map[i]) > 0:

        for j in dist_map[i]:

            if count < 10 :

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

            

                ax1.imshow(train_resized_imgs[i, ..., 0])

                ax1.set_title(f'Train img {i}: {labelMap[y[i]]}')

            

                ax2.imshow(train_resized_imgs[3662 + j, ..., 0])

                ax2.set_title(f'Test img {j}')

                plt.show()

            count += 1
count
count / len(train_df), count / len(test_df)