# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

print(os.listdir("../input/mytrain2/pretrain2"))



# Any results you write to the current directory are saved as output.
import warnings

import seaborn as sns

import matplotlib.pylab as plt

import PIL

from sklearn.model_selection import StratifiedKFold

import gc





from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator



from keras.applications import Xception



from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from keras import layers, models, optimizers



from keras.models import Sequential, Model

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, GlobalAveragePooling2D

from keras import layers

warnings.filterwarnings(action='ignore')

warnings.simplefilter(action='ignore', category=FutureWarning)
!pip install git+https://github.com/qubvel/efficientnet

import efficientnet.keras as efn
DATA_PATH = '../input/kakr3rdcropped'

print(os.listdir(DATA_PATH))

DATA_PATH2 = '../input/2019-3rd-ml-month-with-kakr'

os.listdir(DATA_PATH2)
# 이미지 폴더 경로

TRAIN_IMG_PATH = os.path.join(DATA_PATH, 'train')

TEST_IMG_PATH = os.path.join(DATA_PATH, 'test')



# CSV 파일 경로

df_train = pd.read_csv(os.path.join(DATA_PATH2, 'train.csv'))

df_test = pd.read_csv(os.path.join(DATA_PATH2, 'test.csv'))

df_class = pd.read_csv(os.path.join(DATA_PATH2, 'class.csv'))
img_size = (299, 299)

IMAGE_SIZE = 299

epochs = 25

BATCH_SIZE = 32

k_folds=5

seed = 9



train_datagen = ImageDataGenerator(

    rescale=1./255,

    #featurewise_center= True,  # set input mean to 0 over the dataset

    #samplewise_center=True,  # set each sample mean to 0

    #featurewise_std_normalization= True,  # divide inputs by std of the dataset

    #samplewise_std_normalization=True,  # divide each input by its std

    rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2,

    horizontal_flip=True,

    vertical_flip=False,

    zoom_range=0.2,

    shear_range=0.2,

    #brightness_range=(1, 1.2),

    fill_mode='nearest'

    )

val_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)



#아래 세줄 실행 안하면 오류발생

df_train['class'] = df_train['class'].astype('str')

df_train = df_train[['img_file', 'class']]

df_test = df_test[['img_file']]



#모델 저장경로 생성

MODEL_SAVE_FOLDER_PATH = './model/'

if not os.path.exists(MODEL_SAVE_FOLDER_PATH):

    os.mkdir(MODEL_SAVE_FOLDER_PATH)
def f1(y_true, y_pred):

    def recall(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall



    def precision(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision

    

    precision = precision(y_true, y_pred)

    recall = recall(y_true, y_pred)

    

    return 2*((precision*recall)/(precision+recall+K.epsilon()))
def get_model(model_name, iamge_size):

    base_model = model_name(weights='imagenet', input_shape=(iamge_size,iamge_size,3), include_top=False)

    #base_model.trainable = False

    model = models.Sequential()

    model.add(base_model)

    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Dense(1024, activation='relu'))

    model.add(layers.Dropout(0.25))

    model.add(layers.Dense(1024, activation='relu'))

    model.add(layers.Dropout(0.25))

 

    model.add(layers.Dense(196, activation='softmax'))

    #model.summary()

    

    optimizer = optimizers.Nadam(lr=0.0002)

    #optimizer = optimizers.RMSprop(lr=0.0001)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc', f1])



    return model
def get_model_eff():

    EfficientNet_model = efn.EfficientNetB3(weights='imagenet', include_top=False, 

                                                     input_shape=(299, 299, 3))



    model = Sequential()

    model.add(EfficientNet_model)

    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Dense(2048, activation='relu'))

    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(196, activation='softmax'))

    #model.summary()



    #compile

    optimizer = optimizers.Nadam(lr=0.0002)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc', f1])

    

    return model
xception_list = []

efficient_list = []

path = '../input/mytrain2/pretrain2/'

for i in range(5):

    xception_list.append(path+str(i)+'_xception.hdf5')

    efficient_list.append(path+str(i)+'_EfficientNet.hdf5')
TRAIN_CROPPED_PATH = '../input/kakr3rdcropped/train_crop/'

TEST_CROPPED_PATH = '../input/kakr3rdcropped/test_crop/'



test_generator = test_datagen.flow_from_dataframe(

    dataframe=df_test,

    directory=TEST_CROPPED_PATH,

    x_col='img_file',

    y_col=None,

    target_size= (IMAGE_SIZE, IMAGE_SIZE),

    color_mode='rgb',

    class_mode=None,

    batch_size=BATCH_SIZE,

    shuffle=False

)
xception_prediction = []

efficient_prediction = []



for i, name in enumerate(xception_list):

    model_xception = get_model(Xception, IMAGE_SIZE)

    model_xception.load_weights(name)

    test_generator.reset()

    pred = model_xception.predict_generator(

        generator=test_generator,

        steps = len(df_test)/BATCH_SIZE,

        verbose=1

    )

    xception_prediction.append(pred)



y_pred_xception = np.mean(xception_prediction, axis=0)
for i, name in enumerate(efficient_list):

    model_efficient = get_model_eff()

    model_efficient.load_weights(name)

    test_generator.reset()

    pred = model_efficient.predict_generator(

        generator=test_generator,

        steps = len(df_test)/BATCH_SIZE,

        verbose=1

    )

    efficient_prediction.append(pred)



y_pred_efficient = np.mean(efficient_prediction, axis=0)
train_generator = train_datagen.flow_from_dataframe(

        dataframe=df_train,

        directory=TRAIN_CROPPED_PATH,

        x_col='img_file',

        y_col='class',

        target_size= (IMAGE_SIZE, IMAGE_SIZE),

        color_mode='rgb',

        class_mode='categorical',

        batch_size=BATCH_SIZE,

        seed=seed,

        shuffle=True

        )
ens = np.argmax((0.5*y_pred_xception + 0.5*y_pred_efficient), axis=1)



labels = (train_generator.class_indices)

labels = dict((v,k) for k,v in labels.items())

ens_pred = [labels[k] for k in ens]
submission = pd.read_csv(os.path.join(DATA_PATH2, 'sample_submission.csv'))



sub = submission

sub["class"] = ens_pred

sub.to_csv("submission.csv", index=False)