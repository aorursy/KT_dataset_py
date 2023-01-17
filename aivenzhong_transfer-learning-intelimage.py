import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
# 更改根目录

import os

os.chdir('/kaggle/input/intel-image-classification')

os.getcwd()
# 训练集和测试集路径

train_dir='seg_train/seg_train/'

test_dir='seg_test/seg_test/'
import sys

from tqdm import tqdm

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.image as img

import numpy as np

from skimage import color

from skimage.transform import resize

from sklearn.utils import shuffle

from sklearn.decomposition import PCA

import tensorflow as tf

%matplotlib inline

img_path='buildings/1001.jpg'

image=img.imread(os.path.join(train_dir, img_path))

plt.imshow(image)

plt.show()
# 读数据

train_X, train_Y = [], []     # train_X(训练集数据集X)，train_Y(训练集目标值Y);

IMG_HEIGHT = IMG_WIDTH = 100  # 图像尺寸为IMG_HEIGHT，IMG_WIDTH



# 建立生成器取得批量数据

sys.stdout.flush()

train_folders = next(os.walk(train_dir))[1]



print('开始读取训练集图像文件...')

for folder in train_folders:

    PATH = os.path.join(train_dir, folder)

    train_imgs = os.listdir(PATH)

    print('读取文件夹' + folder)

    for n, id_ in tqdm(enumerate(train_imgs), total = len(train_imgs)):

        train_img_path=os.path.join(PATH, id_)

        I=img.imread(train_img_path)

        # 图像裁剪

        cropped_img= resize(I, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

        train_X.append(cropped_img)

        train_Y.append(folder)

    print('文件夹'+folder+'读取完成!\n')

print('训练集图像文件读取完成!!!\n')

# 打乱数据

train_X, train_Y = shuffle(train_X, train_Y)
# 独热编码转换

map_dic={'buildings':[1,0,0,0,0,0],'forest':[0,1,0,0,0,0],\

        'glacier':[0,0,1,0,0,0],'mountain':[0,0,0,1,0,0],\

        'sea':[0,0,0,0,1,0],'street':[0,0,0,0,0,1]}



train_Y = np.array([map_dic[e] for e in train_Y])

train_X = np.array(train_X)
import numpy as np

import pandas as pd

import os

import numpy

import glob

import cv2

from keras.applications.vgg16 import VGG16

from keras.applications.vgg19 import VGG19

from keras.models import Sequential

from keras.layers import Flatten, Dense, Dropout

weights = 'vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

# weights = 'vgg19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
model = Sequential()

# model.add(VGG16(include_top = False, weights = weights, input_shape = (150,150,3)))

# model.add(VGG19(include_top = False, input_shape = (100,100,3)))

model.add(VGG19(include_top = False, input_shape = (100,100,3), weights='imagenet'))

model.add(Flatten())

model.add(Dense(180, activation = 'relu'))

model.add(Dense(120, activation = 'relu'))

model.add(Dropout(rate=0.5))

model.add(Dense(50, activation = 'relu'))

model.add(Dropout(rate=0.5))

model.add(Dense(6, activation = 'softmax'))

model.compile(loss='categorical_crossentropy',

              optimizer='adagrad',

              metrics=['accuracy'])

# 优化器选项 rmsprop adagrad sgd

model.summary()
Fit = model.fit(train_X, train_Y, epochs = 40, validation_split = 0.30)