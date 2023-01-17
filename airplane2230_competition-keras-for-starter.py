import gc

import os

import warnings

import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



import math

from keras.callbacks import Callback

from keras import backend

from keras.models import Sequential, Model

from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, GlobalAveragePooling2D, BatchNormalization, Input

from keras.optimizers import Adam, SGD, Nadam

from keras.metrics import categorical_accuracy

from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, TensorBoard



import cv2

import PIL

from PIL import ImageOps, ImageFilter, ImageDraw



from keras import backend as K

warnings.filterwarnings(action='ignore')



# Efficient Net은 성능도 좋고, 가벼운 비교적 최신 모델입니다.

!pip install git+https://github.com/qubvel/efficientnet

from efficientnet import EfficientNetB3



K.image_data_format()
model_path = './model/'



if(not os.path.exists(model_path)):

    os.mkdir(model_path)

    

img_path = os.listdir('../input')

print(img_path)



CROP_PATH = '../input/' + img_path[0]

DATA_PATH = '../input/' + img_path[1]



# image folder path

TRAIN_IMG_PATH = CROP_PATH +'/train_crop/'

TEST_IMG_PATH = CROP_PATH +'/test_crop/'



# read csv

df_train = pd.read_csv(DATA_PATH + '/train.csv')

df_test = pd.read_csv(DATA_PATH + '/test.csv')

df_class = pd.read_csv(DATA_PATH + '/class.csv')



from sklearn.model_selection import train_test_split



df_train["class"] = df_train["class"].astype('str')



df_train = df_train[['img_file', 'class']]

df_test = df_test[['img_file']]



from keras.preprocessing.image import ImageDataGenerator



# Parameter

nb_test_samples = len(df_test)



# batch size는 제가 사용하기론 32가 최대였습니다. 양새원님 커널을 보면 128로 되어있는데,

# 어떻게 큰 배치를 사용하셧는지 신기할 따름입니다..

batch_size = 32



# Define Generator config

# ImageGenerator에는 여러가지 param이 존재합니다. 공식 API문서를 참고하시길 추천드립니다.

train_datagen = ImageDataGenerator(

    rotation_range = 60,

#     shear_range = 0.25,

    width_shift_range=0.30,

    height_shift_range=0.30,

    horizontal_flip = True, 

    vertical_flip = False,

    zoom_range=0.25,

    fill_mode = 'nearest',

    rescale = 1./255)



val_datagen = ImageDataGenerator(rescale = 1./255)

test_datagen = ImageDataGenerator(rescale = 1./255)



def get_steps(num_samples, batch_size):

    if (num_samples % batch_size) > 0 :

        return (num_samples // batch_size) + 1

    else :

        return num_samples // batch_size

    

base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))



# 데이터셋이 편향되어있다는 분석글을 본 후에 몇 가지 찾아보았습니다.

# sklearn을 활용하여 class weight를 적용해주는 방식입니다.

# 사실 있고 없고, 성능 향상은 가져다주지 않았습니다.

from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight('balanced',

                                                 np.unique(df_train['class']),

                                                 df_train['class'])
# init params

lr = 2e-4



epochs = 100

T_max = 10 # Cosine Annealing을 위한 param, 일부러 지우지 않았습니다.

n_cycles = epochs / 100



def get_callback(model_path):

    callback_list = [

              ModelCheckpoint(filepath=model_path, monitor='val_loss',

                      verbose=1, save_best_only=True),

              ReduceLROnPlateau(monitor='val_loss',

                        factor=0.2,

                        patience=3,

                        min_lr=1e-6,

                        cooldown=1,

                        verbose=1),

      EarlyStopping(monitor = 'val_f1_m', patience = 5)

              ]

    return callback_list



def get_model(input_size):

    inputs = Input(shape = (input_size, input_size, 3), name = 'input_1')

    x = base_model(inputs)

    x = GlobalAveragePooling2D()(x)

    x = Dense(2048, kernel_initializer='he_normal')(x)

    x = Dropout(0.3)(x)

    x = Activation('relu')(x)

    x = Dense(196, activation = 'softmax')(x)



    model = Model(inputs = inputs, outputs = x)

#     sgd = SGD(lr=lr, decay=1e-8, momentum=0.9, nesterov=True)

    nadam = Nadam(lr = lr)

    model.compile(optimizer= nadam, loss='categorical_crossentropy', metrics=[categorical_accuracy,f1_m, precision_m, recall_m])

    return model



from sklearn.model_selection import StratifiedKFold



k_folds = 5

img_size = (299, 299)

skf = StratifiedKFold(k_folds, random_state = 2019)



test_generator = test_datagen.flow_from_dataframe(

    dataframe=df_test,

    directory='../input/' + img_path[0] + '/test_crop',

    x_col='img_file',

    y_col=None,

    target_size= img_size,

    color_mode='rgb',

    class_mode=None,

    batch_size=batch_size,

    shuffle=False

)
j = 1



img_size = (299, 299)

model_names = []

for (train_index, valid_index) in skf.split(

    df_train['img_file'], 

    df_train['class']):

    

    traindf = df_train.iloc[train_index, :].reset_index()

    validdf = df_train.iloc[valid_index, :].reset_index()

    nb_train_samples = len(traindf)

    nb_validation_samples = len(validdf)



    print("=========================================")

    print("====== K Fold Validation step => %d/%d =======" % (j,k_folds))

    print("=========================================")



    # Make Generator

    train_generator_299 = train_datagen.flow_from_dataframe(

        dataframe=traindf, 

        directory=TRAIN_IMG_PATH,

        x_col = 'img_file',

        y_col = 'class',

        target_size = img_size,

        color_mode='rgb',

        class_mode='categorical',

        batch_size=batch_size,

        seed=42

    )



    validation_generator_299 = val_datagen.flow_from_dataframe(

        dataframe=validdf, 

        directory=TRAIN_IMG_PATH,

        x_col = 'img_file',

        y_col = 'class',

        target_size = img_size,

        color_mode='rgb',

        class_mode='categorical',

        batch_size=batch_size,

        shuffle=True

    )



    model_name = model_path + str(j) + '_EFFnet_f1.hdf5'

    model_names.append(model_name)

    model_EFF = get_model(img_size[0])

    

    try:

        model_EFF.load_weights(model_name)

    except:

        pass



    history = model_xception.fit_generator(

    train_generator_299,

    steps_per_epoch = get_steps(nb_train_samples, 32),

    epochs=epochs,

    validation_data = validation_generator_299,

    validation_steps = get_steps(nb_validation_samples, 32),

    callbacks =  get_callback(model_name),

    class_weight = class_weights

      )

        

    j+=1

    print(gc.collect())
# 생성자의 n_cycles를 주기로 작동하게 됩니다.

# 이 코드의 단점은 min_lr 파라미터가 없는점.

class CosineAnnealingLearningRateSchedule(Callback):

    # constructor

    def __init__(self, n_epochs, n_cycles, lrate_max, verbose = 0):

        self.epochs = n_epochs

        self.cycles=  n_cycles

        self.lr_max = lrate_max

        self.lrates = list()

    

    # caculate learning rate for an epoch

    def cosine_annealing(self, epoch, n_epochs, n_cycles, lrate_max):

        epochs_per_cycle = math.floor(n_epochs/n_cycles)

        cos_inner = (math.pi * (epoch % epochs_per_cycle)) / (epochs_per_cycle)

        return lrate_max/2 * (math.cos(cos_inner) + 1)

  

    # calculate and set learning rate at the start of the epoch

    def on_epoch_begin(self, epoch, logs = None):

        if(epoch < 101):

            # calculate learning rate

            lr = self.cosine_annealing(epoch, self.epochs, self.cycles, self.lr_max)

            print('\nEpoch %05d: CosineAnnealingScheduler setting learng rate to %s.' % (epoch + 1, lr))

        

        # 이 밑의 코드는 학습이 너무 느리고, cycle을 돌기떄문에 일정 epoch이상은 고정된 lr을 사용하려고 한 흔적입니다.

        #     elif((epoch >= 65) and (epoch < 75)):

        #       lr = 1e-5

        #       print('\n No CosineAnnealingScheduler set lr 1e-5')

        #     elif((epoch >= 75) and (epoch < 85)):

        #       lr = 1e-6

        #       print('\n No CosineAnnealingScheduler set lr 1e-6')

        #     elif((epoch >= 85)):

        #       lr = 1e-7

        #       print('\n No CosineAnnealingScheduler set lr 1e-7')



        # set learning rate

        backend.set_value(self.model.optimizer.lr, lr)

        # log value

        self.lrates.append(lr)
# T_max를 주기로 돌게 됩니다.

# 이때, T_mult는 제가 알기론 rewarmstarting을 위한 파라미터로서,

# 검색해보면 어느 블로그에서 이 parameter의 변화에 따른 성능변화를 정리한 글이 있습니다.



# 예를 들어서 T_max가 10이고, T_mult = 2이면,

# 첫 주기 10, 다음 주기 20, 그 다음주기 40

# 이런식으로 lr의 변화폭이 점점 줄어들게 됩니다.

# 위 코드보다 편리한점은 더 깔끔하고, lr_min(eta_min)을 적용할 수 있다는 점



class CosineAnnealingLearningRateSchedule(Callback):

    def __init__(self, n_epochs, init_lr, T_mult = 1, eta_min = 0,restart_decay = 0, verbose = 0):

        self.T_max = n_epochs

        self.T_mult = T_mult

        self.cycle_cnt = 0

        self.restart_decay = restart_decay

        self.init_lr = init_lr

        self.eta_min = eta_min

        self.lrates = list()

  # caculate learning rate for an epoch



    def cosine_annealing(self, epoch):

        lr = self.eta_min + (self.init_lr - self.eta_min) * (1 + math.cos(math.pi * (epoch / self.T_max))) / 2

        if(epoch == self.T_max):

            self.cycle_cnt += 1

            self.T_max = self.T_mult * self.T_max



        if(self.restart_decay >0):

            self.init_lr *= self.restart_decay

            print('change init learning rate {}'.format(self.init_lr))



    return lr

  # calculate and set learning rate at the start of the epoch



    def on_epoch_begin(self, epoch, logs = None):

        lr = self.cosine_annealing(epoch)

        print('\nEpoch %05d: CosineAnnealingScheduler setting learng rate to %s.' % (epoch + 1, lr))

        # set learning rate

        backend.set_value(self.model.optimizer.lr, lr)

        # log value

        self.lrates.append(lr)
from keras.models import load_model



incepres_model = ['_incepres.hdf5', InceptionResNetV2]

xception_model = ['_Xception.hdf5', Xception]

eff_model = ['_EFFnet.hdf5', EfficientNetB3]



model_list = [xception_model, incepres_model, eff_model]

# total predictions list

preds_list = []



lr = 1e-4

TTA_STEPS = 10



for model_name, base_model in model_list:

    print(model_name)

    

    # prediction each fold

    predictions = []

    if(model_name == '_EFFnet.hdf5'):

        model_load_dir = MODEL_EFF_PATH

    else:

        model_load_dir = MODEL_PATH

    for i in range(1, 6):

        model = get_model(base_model, 299, False)

        model.load_weights(os.path.join(model_load_dir, str(i)) + model_name)

        # tta prediction list

        tta_preds = []

        for _ in range(TTA_STEPS):

            test_generator_299.reset()

            pred = model.predict_generator(

            generator = test_generator_299, 

            steps = get_steps(nb_test_samples, batch_size),

            verbose = 1

            )

            tta_preds.append(pred) # (5, 6150, 196)

        tta_preds = np.mean(tta_preds, axis = 0) # (6150, 196)

        

        # for memory leaky

        del model # 별 효과 없음

        for _ in range(10): # 별 효과 없음

            gc.collect()

        K.clear_session() # 이 친구가 대장

        predictions.append(tta_preds) # (5, 6150, 196)

    preds_list.append(np.mean(predictions, axis = 0))