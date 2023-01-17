# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import re
from PIL import Image
import shutil
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook

import numpy as np
import pandas as pd
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tqdm import tqdm
import os
import PIL
from PIL import ImageOps, ImageFilter, ImageDraw, Image

from tqdm import tqdm
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


print(tf.__version__)
print(keras.__version__)
os.listdir('/kaggle/input/kaggle-oxford-dataset-rgh/KAGGLE/train/basset_hound')
# Kaggle -> Input이라는 폴더에 들어있는 데이터 리스트를 출력
# 직접 하위 폴더까지 들어가보세요
#=== 이미지를 출력해서 확인해보자 ===#
image = Image.open('/kaggle/input/kaggle-oxford-dataset-rgh/KAGGLE/train/basset_hound/basset_hound_150.jpg')   # 이미지를 Open해줌
image = np.array(image)   # 이미지를 array로 바꿔줌
print(image.shape)        # 333(행), 500(열), 3(Channel RGB)
#=== 위에서 불러온 이미지 출력 ===#
plt.imshow(image)
plt.show()
# Train 폴더에 각 이미지 Class 별로 또다른 폴더가 있음, 그 폴더속에 이미지가 들어있음
# 전체 리스트를 얻기 위해서는 각 폴더의 리스트들을 불러서 하나의 리스트로 합쳐줘야 함
# Train 폴더 > Class 폴더 > 이미지

train_files_path = os.path.join('/kaggle/input/kaggle-oxford-dataset-rgh/KAGGLE/train')   # 폴더 경로를 입력
train_files = os.listdir(train_files_path)                                                # 해당 폴더 경로에 들어있는 데이터 리스트를 집어넣음

train_all_list = []                                          # 비어있는 List를 만듦

for train_file in train_files:                               # 위에 Train 폴더의 들어있는 Class 폴더 리스트를 For문으로 돌림
    temp_path = os.path.join(train_files_path,train_file)    # Train폴더 경로 + Class 폴더명으로 새로운 경로를 만듦
    for i in os.listdir(temp_path):                          # 새로운 Class 폴더 경로내에 폴더들의 리스트를 For문으로 돌림
        train_all_list.append(i)                             # Class 폴더내에 파일들을 리스트에 추가시킴 --> Class 폴더를 돌며 반복
        
train_all_list[:10]   # 각각의 Class에 있는 파일들을 하나의 리스트로 만듦
# 위와 같은 이유로 Valid 폴더내의 Class 폴더내의 이미지들의 리스트르 만듦
# Valid 폴더 > Class 폴더 > 이미지

valid_files_path = os.path.join('/kaggle/input/kaggle-oxford-dataset-rgh/KAGGLE/valid')
valid_files = os.listdir(valid_files_path)

valid_all_list = []

for valid_file in valid_files:
    temp_path = os.path.join(valid_files_path,valid_file)
    for i in os.listdir(temp_path):
        valid_all_list.append(i)
        
image_files = train_all_list + valid_all_list
print(len(train_all_list), len(valid_all_list), '-->',len(image_files))
image_files
# 앞에서 만든 실제 파일명 리스트에서 이름만 뽑아내서 List로 만들어줌

class_list = []   # 비어있는 리스트를 만듦

for image_file in image_files:
    file_name = os.path.splitext(image_file)[0]    # '.'을 기준으로 나누고 0번째, havanese_48.jpg -> havanese_48
    
#     print(class_name)   
#     raise RuntimeError                           # 고의로 에러를 발생시켜 For문을 멈추게 함, For문 내에 함수들이 어떤 Output을 내는지 확인할 때 유용

    class_name = re.sub('_\d+', '', file_name)     # 정규식 처리방법이라 함, '_'이후의 숫자들을 지워줌(쉽지 않음)  https://greeksharifa.github.io/%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D(re)/2018/08/04/regex-usage-05-intermediate/

    class_list.append(class_name)

class_list = list(set(class_list))                 # set함수를 사용하면, list들중 중복된 값을 지워줌(set형태로 변형됨) -> 다시 List로 바꿔줘야함 
class_list.sort()                                  # list를 오름차순으로 정렬시켜줌
class_list
class_list[3]    # Class List에 숫자를 넣어주면 
#=== Hyper Parameters ===#
n_train = 0

N_CLASS = len(class_list)
N_EPOCHS = 50                                      # Training 시킬 회수를 조정, 전체 데이터가 5920개면 5920개를 50번 반복한다
N_BATCH = 40                                       # 얼마나 많은 이미지를 모델에 동시에 넣을지 설정함(한번에 40개씩 학습함 -> 5920/40회 학습한다고 보면됨)

N_TRAIN = len(train_all_list)                      # Train 이미지 데이터의 개수
N_VAL = len(valid_all_list)                        # Valid 이미지 데이터의 개수

IMG_SIZE = 224                                     # 이미지를 Resize 시킬 사이즈 설정 (해당 사이즈로 가로 및 세로가 변형됨)
learning_rate = 0.0001                             # 학습 폭(?) 말그대로 learing rate
steps_per_epoch = N_TRAIN / N_BATCH                # 한 epoch에서 얼마나 움직일지(?) 스텝수를 정해줌
validation_steps = int(np.ceil(N_VAL / N_BATCH))   # Valid 개수가 506개인데, 40 Batch로 나누면, Step이 12.65개가 됨, 가급적 남은것도 다 쓰고자 올림(np.ceil)해줌
from keras.preprocessing.image import ImageDataGenerator

#=== Train 이미지를 변형시킬 내용을 ImageDataGenerator에 작성해줌 ===#
train_datagen = ImageDataGenerator(rescale=1. / 255,
#                                    rotation_range=40,
#                                    width_shift_range=0.2,     # 가로 방향으로 이동
#                                    height_shift_range=0.2,    # 세로 방향으로 이동
#                                    shear_range=0.2,           # 이미지 굴절
#                                    zoom_range=0.2,            # 이미지 확대
#                                    horizontal_flip=True,      # 횡방향으로 이미지 반전
#                                    fill_mode='nearest'
                                  )      

#=== 위에서 만든 ImageDataGenerator(변형)을 사용해서 Batch 별로 출력될 수 있게 train_generator를 만듦 ===#
train_generator = train_datagen.flow_from_directory(train_files_path, 
                                                    batch_size=N_BATCH,
                                                    target_size=(224, 224),
                                                    class_mode='categorical',     # binary / categorical
                                                    )

#=== valid 이미지를 변형시킬 내용을 ImageDataGenerator에 작성해줌, Valid 데이터는 가급적 변형없이 넣는게 좋음  ===#
valid_datagen = ImageDataGenerator(rescale=1. / 255)      

#=== 위에서 만든 ImageDataGenerator(변형)을 사용해서 Batch 별로 출력될 수 있게 valid_generator를 만듦 ===#
valid_generator = valid_datagen.flow_from_directory(valid_files_path, 
                                                    batch_size=N_BATCH,
                                                    target_size=(224, 224),
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
#=== Save Trained-Model ===#
model.save('Densenet121_model.h5')

#=== Load Trained Model ===#
new_model = keras.models.load_model('Densenet121_model.h5')
#=== 외부 이미지를 가져다가 Inferencing ===#
img_list = []
img_folder_list = [i for i in os.listdir('/kaggle/input/') if i != 'kaggle-oxford-dataset-rgh']

for img_folder in img_folder_list:
    img_folder_path = os.path.join('/kaggle/input/',img_folder)
    for img in os.listdir(img_folder_path):
        img_list.append(os.path.join('/kaggle/input/',img_folder,img))
    

for img_path in img_list:

    #=== Image upload 후 실행 ===#
    image = Image.open(img_path)   # 구글에서 다운받아서 raw directory 폴더에 넣고 돌리면 됨
    image = image.resize((224, 224))
    image = np.array(image)
    image = image/255.

    plt.imshow(image)
    plt.show()
    print(image.shape)

    image = np.reshape(image, (1, 224, 224, 3))   # CNN은 4차원의 데이터를 받으니까, 앞에 1차원도 껴줌


    #=== Predict ===#
    prediction = new_model.predict(image)
    pred_class = np.argmax(prediction, axis=-1)   # argmax를 하면 앞에서 OHE로 나온 확률에 대한 class가 나옴
    print(pred_class)
    print(class_list[int(pred_class)])    # Chihauhau!! 쏘리질러!!

