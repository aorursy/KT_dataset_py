import os, cv2, random

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

from matplotlib import ticker

import seaborn as sns



from keras.models import Sequential

from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation

from keras.optimizers import RMSprop

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

from keras.utils import np_utils
TRAIN_DIR = '../input/train/'

TEST_DIR = '../input/test/'

train_dogs =   [(TRAIN_DIR+i, 1) for i in os.listdir(TRAIN_DIR) if 'dog' in i]

train_cats =   [(TRAIN_DIR+i, 0) for i in os.listdir(TRAIN_DIR) if 'cat' in i]

#取测试集，这个地方需要提前占位，-1就是占位符。

test_images =  [(TEST_DIR+i, -1) for i in os.listdir(TEST_DIR)]

print(test_images[0])
TRAIN_NUM = 50

# 因为运行速度太慢，选取TRAIN_NUM个数据，并随机散列



train_images = train_dogs[:TRAIN_NUM]+ train_cats[:TRAIN_NUM]

random.shuffle(train_images)

# 取个TEST_NUM个做测试

test_images =  test_images
ROWS = 64

COLS = 64



#这里传入类似train_dogs的K-V格式list。将图片统一化

# 返回 统一化后的图片 + label    

def read_image(tuple_set):

    file_path = tuple_set[0]

    label = tuple_set[1]

    img = cv2.imread(file_path, cv2.IMREAD_COLOR)

    # 这里的参数，可以是彩色或者灰度(GRAYSCALE)

    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC), label

    # 这里，可以选择压缩图片的方式，zoom（cv2.INTER_CUBIC & cv2.INTER_LINEAR）还是shrink（cv2.INTER_AREA）
CHANNELS = 3

# 代表RGB三个颜色频道

# 将图片变为numpy数组

# 返回统一化后的numpy数组 + label  

def prep_data(images):

    no_images = len(images)

    data = np.ndarray((no_images, CHANNELS, ROWS, COLS), dtype=np.uint8)

    labels = []

    

    for i, image_file in enumerate(images):       

        image, label = read_image(image_file)

        data[i] = image.T

        labels.append(label)

    return data, labels



x_train, y_train = prep_data(train_images)

x_test, y_test = prep_data(test_images)
optimizer = RMSprop(lr=1e-4)      #学习方法，一步跨多大

objective = 'binary_crossentropy' #0-1差距熵，因为这里就是0-1



# 建造模型VGG-16构架方式    

model = Sequential()

#输入一个矩阵 32个特征，3*3 

#几套CNN后，特征要越来越多，从大处到细节看。如果从细节到大看。feature从大到小

model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3, ROWS, COLS), activation='relu'))

model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))

model.add(MaxPooling2D( pool_size=(2, 2),dim_ordering="th"))



model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))

model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering="th"))



model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))

model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering="th"))



model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))

model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering="th"))



model.add(Flatten()) ##变成一条

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(1))

model.add(Activation('sigmoid'))



model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])
#这一部分算CNN停止点



nb_epoch = 10

batch_size = 10

#没有减少反而增高了，我们给他3步耐心



early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')        

# 跑模型

model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,

          validation_split=0.2, verbose=0, shuffle=True, callbacks=[early_stopping])



predictions = model.predict(x_test, verbose=0)

print(predictions)
predictions[0]