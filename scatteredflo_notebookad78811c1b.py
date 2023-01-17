import shutil

from tqdm import tqdm

from tqdm import tqdm_notebook 

import warnings

warnings.filterwarnings(action='ignore')   # 에러코드 무시



import tensorflow as tf

from tensorflow import keras

import re

import PIL

from PIL import ImageOps, ImageFilter, ImageDraw, Image

from PIL import Image

from sklearn.model_selection import train_test_split

import random

import matplotlib.pyplot as plt



import numpy as np

import pandas as pd

from scipy import ndimage

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import os



from scipy import ndimage

import cv2

os.listdir('../input/kaggle-defect-dataset-v3')
train_path = '../input/kaggle-defect-dataset-v3/train'

valid_path = '../input/kaggle-defect-dataset-v3/valid'

test_path = '../input/kaggle-defect-dataset-v3/test'



#=== Check all the dataset ===#

train_list = []

valid_list = []



for file_path, file_list in [[train_path, train_list], [valid_path, valid_list]]:

    path0_list = os.listdir(file_path)

    for i in range(len(path0_list)):

        path1 = os.path.join(file_path, path0_list[i])

        path1_list = os.listdir(path1)

        [file_list.append(os.path.join(path1,i)) for i in path1_list] 

        

print(len(train_list),len(valid_list))

class_list = os.listdir(train_path)

class_list.sort()

class_list = [i for i in class_list if 'csv' not in i]

print('class_list length : ',len(class_list),'\n')

class_list
#=== 이미지를 출력해서 확인해보자 ===#

# Channel 이슈(Pretrained Model에서는 3Channel만 사용 가능) https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/63710

train_path = train_path

train_defect_path = os.path.join(train_path, os.listdir(train_path)[7])

img_one = os.path.join(train_defect_path, os.listdir(train_defect_path)[1])



image = Image.open(img_one)   # 이미지를 Open해줌

image = np.array(image)   # 이미지를 array로 바꿔줌



print(image.shape)        # 333(행), 500(열), 3(Channel RGB)

plt.imshow(image)

plt.show()
#=== Check all the dataset ===#

file_list = []



path0_list = os.listdir(train_path)

path0_list = [i for i in path0_list if 'csv' not in i]

path0_list = [i for i in path0_list if 'AUG' not in i]

path0_list = [i for i in path0_list if 'jpg' not in i]



for i in range(len(path0_list)):

    path1 = os.path.join(train_path, path0_list[i])

    path1_list = os.listdir(path1)

    print(i, ' ', path0_list[i], '-',len(path1_list))

    [file_list.append(i) for i in path1_list] 



print('='*30)

print('Total File : ', len(file_list))



# Bright, Insect의 개수가 현저히 적음
#=== Check all the dataset ===#

file_list = []



path0_list = os.listdir(valid_path)

path0_list = [i for i in path0_list if 'csv' not in i]

path0_list = [i for i in path0_list if 'AUG' not in i]

path0_list = [i for i in path0_list if 'jpg' not in i]



for i in range(len(path0_list)):

    path1 = os.path.join(valid_path, path0_list[i])

    path1_list = os.listdir(path1)

    print(i, ' ', path0_list[i], '-',len(path1_list))

    [file_list.append(i) for i in path1_list] 



print('='*30)

print('Total File : ', len(file_list))



# Bright, Insect의 개수가 현저히 적음
#=== Hyper Parameters ===#

n_train = 0



N_CLASS = len(class_list)

N_EPOCHS = 50                                      # Training 시킬 회수를 조정, 전체 데이터가 5920개면 5920개를 50번 반복한다

N_BATCH = 512                                       # 얼마나 많은 이미지를 모델에 동시에 넣을지 설정함(한번에 40개씩 학습함 -> 5920/40회 학습한다고 보면됨)



N_TRAIN = len(train_list)                      # Train 이미지 데이터의 개수

N_VAL = len(valid_list)                        # Valid 이미지 데이터의 개수



IMG_SIZE = 100                                     # 이미지를 Resize 시킬 사이즈 설정 (해당 사이즈로 가로 및 세로가 변형됨)

learning_rate = 0.0001                             # 학습 폭(?) 말그대로 learing rate

steps_per_epoch = N_TRAIN / N_BATCH                # 한 epoch에서 얼마나 움직일지(?) 스텝수를 정해줌

validation_steps = int(np.ceil(N_VAL / N_BATCH))   # Valid 개수가 506개인데, 40 Batch로 나누면, Step이 12.65개가 됨, 가급적 남은것도 다 쓰고자 올림(np.ceil)해줌
from keras.preprocessing.image import ImageDataGenerator



#=== Train 이미지를 변형시킬 내용을 ImageDataGenerator에 작성해줌 ===#

train_datagen = ImageDataGenerator(rescale=1. / 255,

                                #    rotation_range=40,

                                #    width_shift_range=0.2,     # 가로 방향으로 이동

                                #    height_shift_range=0.2,    # 세로 방향으로 이동

                                #    shear_range=0.2,           # 이미지 굴절

                                #    zoom_range=0.2,            # 이미지 확대

                                #    horizontal_flip=True,      # 횡방향으로 이미지 반전 50%확률

                                #    fill_mode='nearest'

                                  )      



#=== 위에서 만든 ImageDataGenerator(변형)을 사용해서 Batch 별로 출력될 수 있게 train_generator를 만듦 ===#

train_generator = train_datagen.flow_from_directory(train_path, 

                                                    batch_size=N_BATCH,

                                                    target_size=(IMG_SIZE, IMG_SIZE),

                                                    class_mode='categorical',     # binary / categorical

                                                    )
#=== valid 이미지를 변형시킬 내용을 ImageDataGenerator에 작성해줌, Valid 데이터는 가급적 변형없이 넣는게 좋음  ===#

valid_datagen = ImageDataGenerator(rescale=1. / 255)      



#=== 위에서 만든 ImageDataGenerator(변형)을 사용해서 Batch 별로 출력될 수 있게 valid_generator를 만듦 ===#

valid_generator = valid_datagen.flow_from_directory(valid_path, 

                                                    batch_size=N_BATCH,

                                                    target_size=(IMG_SIZE, IMG_SIZE),

                                                    class_mode='categorical',     # binary / categorical

                                                    )
#=== import Pretrained Model을 불러옴 ===#

from tensorflow.keras.applications.densenet import DenseNet121   # 좀더 좋은 모델로 Training 시켜보자

from tensorflow.keras import models

from tensorflow.keras.layers import Conv2D, ReLU, MaxPooling2D, Dense, BatchNormalization, GlobalAveragePooling2D



densenet = DenseNet121(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
#=== 모델을 설계함(함수로 만들어서) ===#

def create_dense_model():

    model = models.Sequential()

    model.add(densenet)                # 위에서 불러온 pretrained Densenet을 불러옴

    model.add(GlobalAveragePooling2D())

    model.add(Dense(256))

    model.add(BatchNormalization())

    model.add(ReLU())

    model.add(Dense(N_CLASS, activation='softmax'))

    return model
#=== 위에서 설계한 모델을 불러옴 모델을 만들어 줌 ===#

model = create_dense_model()
#=== Learning Rate를 제어하는 함수(저도 잘 모름) ===#

LR_INIT = 0.000001

LR_MAX = 0.0002

LR_MIN = LR_INIT

RAMPUP_EPOCH = 4

EXP_DECAY = 0.9



def lr_schedule_fn(epoch):

    if epoch < RAMPUP_EPOCH:

        lr = (LR_MAX - LR_MIN) / RAMPUP_EPOCH * epoch + LR_INIT

    else:

        lr = (LR_MAX - LR_MIN) * EXP_DECAY**(epoch - RAMPUP_EPOCH)

    return lr



lr_callback = keras.callbacks.LearningRateScheduler(lr_schedule_fn)
#=== 위에서 만든 모델에 Optimizer, loss, metric을 설정(compile)해 줌 ===#

model.compile(optimizer=tf.keras.optimizers.Adam(LR_INIT),

              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),

              metrics=['accuracy'])



#=== 만든 모델을 불러와봄 ===#

model.summary()
#=== Training the Model ===#

history = model.fit(train_generator,

                    epochs=N_EPOCHS,

                    steps_per_epoch=steps_per_epoch,

                    validation_data=valid_generator,

                    validation_steps=validation_steps,

                    callbacks=[lr_callback]

                    )
# 5. 모델 학습 과정 표시하기

%matplotlib inline

import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()



loss_ax.plot(history.history['loss'], 'y', label='train loss')

loss_ax.plot(history.history['val_loss'], 'r', label='val loss')



acc_ax.plot(history.history['accuracy'], 'b', label='train accuracy')

acc_ax.plot(history.history['val_accuracy'], 'g', label='val accuracy')



loss_ax.set_xlabel('epoch')

loss_ax.set_ylabel('loss')

acc_ax.set_ylabel('accuray')



loss_ax.legend(loc='upper left')

acc_ax.legend(loc='lower left')



plt.show()
#=== Save Trained-Model ===#

model.save('Densenet121_model_v1.0.h5')


#=== Load Trained Model ===#

new_model = keras.models.load_model('Densenet121_model_v1.0.h5')
#=== 외부 이미지를 가져다가 Inferencing ===#

test_folder_list = os.listdir(test_path) 

test_img_path_list = [os.path.join(test_path,i) for i in test_folder_list if '.jpg' in i]





pred_all = []



for img_path in tqdm_notebook(test_img_path_list):

    #=== Image upload 후 실행 ===#

    image = Image.open(img_path)       # Image Open

    image = image.resize((100, 100))   # Image를 100*100으로 resize

    image = np.array(image)            # Image를 Array로 변형

    image = image/255                  # Array의 각 cell을 255로 나눠서 데이터 크기를 줄여줌

#     print(image.shape)

    image_reshape = np.reshape(image, (1, 100, 100, 3))   # CNN은 4차원의 데이터만 읽을 수 있음, 현재는 3차원이니 앞에 1차원을 껴줌

    

    #=== Inferencing ===#

    prediction = new_model.predict(image_reshape)        # 예측값은 각 Class별로의 확률값으로 출력됨

    pred_class = np.argmax(prediction, axis=-1)          # argmax를 하면 앞에서 가장 확률이 높은 Class index가 출력됨

    pred_value = class_list[int(pred_class)]             # 앞에서 나온 Index를 class_list에 넣으면 class(이물) 값이 나옴

    pred_all.append(pred_value)



    print('='*50)

    

    plt.imshow(image)

    plt.show()



    print('모델 예측값 : ', pred_value)
