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
folder_path = '../input/kaggle-defect-dataset-v3/'
folder_list = os.listdir(folder_path)

train_path = os.path.join(folder_path, 'train')
valid_path = os.path.join(folder_path, 'valid')
test_path = os.path.join(folder_path, 'test')
train_folder_list = os.listdir(train_path)
class_list = train_folder_list

class_list.sort() ######## 추가된 내용, Class의 List를 오름차순으로 정리하면 이상없이 가동됨
image = Image.open('../input/kaggle-defect-dataset-v3/test/L41_RW_1_41021914_000032.jpg')
image = np.array(image)

plt.imshow(image)
plt.show()
#=== Hyper Parameters ===#
n_train = 0

N_CLASS = len(class_list)
N_EPOCHS = 30                                      # Training 시킬 회수를 조정, 전체 데이터가 5920개면 5920개를 50번 반복한다
N_BATCH = 512                                       # 얼마나 많은 이미지를 모델에 동시에 넣을지 설정함(한번에 40개씩 학습함 -> 5920/40회 학습한다고 보면됨)

N_TRAIN = 15849                      # Train 이미지 데이터의 개수
N_VAL = 1716                        # Valid 이미지 데이터의 개수

IMG_SIZE = 100                                     # 이미지를 Resize 시킬 사이즈 설정 (해당 사이즈로 가로 및 세로가 변형됨)
learning_rate = 0.0001                             # 학습 폭(?) 말그대로 learing rate
steps_per_epoch = N_TRAIN / N_BATCH                # 한 epoch에서 얼마나 움직일지(?) 스텝수를 정해줌
validation_steps = int(np.ceil(N_VAL / N_BATCH))   # Valid 개수가 506개인데, 40 Batch로 나누면, Step이 12.65개가 됨, 가급적 남은것도 다 쓰고자 올림(np.ceil)해줌
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1. / 255)      

train_generator = train_datagen.flow_from_directory(train_path, 
                                                    batch_size=N_BATCH,
                                                    target_size=(IMG_SIZE, IMG_SIZE),
                                                    class_mode='categorical',     # binary / categorical
                                                    )
valid_datagen = ImageDataGenerator(rescale=1. / 255)      

valid_generator = train_datagen.flow_from_directory(valid_path, 
                                                    batch_size=N_BATCH,
                                                    target_size=(IMG_SIZE, IMG_SIZE),
                                                    class_mode='categorical',     # binary / categorical
                                                    )
#=== import Pretrained Model을 불러옴 ===#
from tensorflow.keras.applications.densenet import DenseNet121   # 좀더 좋은 모델로 Training 시켜보자
from tensorflow.keras import models
from tensorflow.keras.layers import Conv2D, ReLU, MaxPooling2D, Dense, BatchNormalization, GlobalAveragePooling2D

densenet = DenseNet121(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
def create_dense_model():
    model = models.Sequential()
    model.add(densenet)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dense(N_CLASS, activation = 'softmax'))
    return model
    
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
#=== Model Compile ===#
model.compile(optimizer = tf.keras.optimizers.Adam(LR_INIT),
             loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing = 0.1),
             metrics = ['accuracy'])

model.summary()
#=== Training the Model ===#
history = model.fit(train_generator,
                    epochs=N_EPOCHS,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=valid_generator,
                    validation_steps=validation_steps,
                    callbacks=[lr_callback]
                    )
#=== Image Open and Preprocessing ===#
image = Image.open('../input/kaggle-defect-dataset-v3/test/L41_RW_1_41022026_000033.jpg')       # Image Open
image = image.resize((100, 100))                      # Image를 100*100으로 resize
image = np.array(image)                               # Image를 Array로 변형
image = image/255                                     # Array의 각 cell을 255로 나눠서 데이터 크기를 줄여줌
image_reshape = np.reshape(image, (1, 100, 100, 3))   # CNN은 4차원의 데이터만 읽을 수 있음, 현재는 3차원이니 앞에 1차원을 껴줌


#=== Inferencing ===#
prediction = model.predict(image_reshape)


#=== Interpretation of results ===#
plt.imshow(image)     
plt.show() # 이미지 출력

pred_class = np.argmax(prediction, axis=-1)           # 앞에 Inferencing을 하면, Softmax Probability 값으로 출력됨 -> Argmax를 하면 그중 가장 큰 Probabilirt Class Number를 출력함
pred_value = class_list[int(pred_class)]              # 앞에서 Argmax를 해서 얻은 가장 큰 Probability가 큰 Class Number를 Class List에 넣어서 해당 값을 뽑아줌
print(pred_value)
