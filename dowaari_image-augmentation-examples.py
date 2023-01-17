# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import warnings

import matplotlib.pylab as plt

import PIL

warnings.filterwarnings('ignore')

image_size = 224
# 이미지 폴더 경로

DATA_PATH = '../input/'

TRAIN_IMG_PATH = os.path.join(DATA_PATH, 'train')



# CSV 파일 경로

df_train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))

df_train = df_train.iloc[:50] # 편의상 50개까지만 이용 

df_train.head()
def crop_boxing_img(img_name, margin=16, size=(image_size, image_size)):

    if img_name.split('_')[0] == 'train':

        PATH = TRAIN_IMG_PATH

        data = df_train

    else:

        PATH = TEST_IMG_PATH

        data = df_test



    img = PIL.Image.open(os.path.join(PATH, img_name))

    pos = data.loc[data["img_file"] == img_name, \

                   ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']].values.reshape(-1)



    width, height = img.size

    x1 = max(0, pos[0] - margin)

    y1 = max(0, pos[1] - margin)

    x2 = min(pos[2] + margin, width)

    y2 = min(pos[3] + margin, height)



    return img.crop((x1, y1, x2, y2)).resize(size)
TRAIN_CROP_PATH = "./train_crop"

!mkdir {TRAIN_CROP_PATH}
# train_set의 모든 이미지에 대해 적용할 경우 시간이 좀 걸립니다.

for i, row in df_train.iterrows():

    cropped = crop_boxing_img(row['img_file'])

    cropped.save(f"{TRAIN_CROP_PATH}/{row['img_file']}")
from sklearn.model_selection import train_test_split



df_train["class"] = df_train["class"].astype('str')

df_train = df_train[['img_file', 'class']]



X_val = df_train.copy()

# its = np.arange(df_train.shape[0])

# train_idx, val_idx = train_test_split(its, test_size = 0.8, shuffle= False)

# X_train = df_train.iloc[train_idx, :]

# X_val = df_train.iloc[val_idx, :]



# print(X_train.shape)

print(X_val.shape)
import keras 

from keras import models

from keras import layers

from keras import optimizers

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=20,

    width_shift_range=0.1,

    height_shift_range=0.1,

    zoom_range=0.1,

    horizontal_flip=True,

    fill_mode='nearest'

)



valid_datagen = ImageDataGenerator(rescale=1./255) 
img_size = (image_size, image_size)

batch_size = 10



train_generator = train_datagen.flow_from_dataframe(

    dataframe = X_val, 

    directory = TRAIN_CROP_PATH,

    x_col = 'img_file',

    y_col = 'class',

    target_size = img_size,

    color_mode ='rgb',

    class_mode ='categorical',

    batch_size =batch_size,

    shuffle =False

)



validation_generator = valid_datagen.flow_from_dataframe(

    dataframe = X_val, 

    directory = TRAIN_CROP_PATH,

    x_col ='img_file',

    y_col ='class',

    target_size = img_size,

    color_mode ='rgb',

    class_mode ='categorical',

    batch_size =batch_size,

    shuffle =False

)
# 아래와 같이 augmentation 처리된 이미지들을 확인할 수 있습니다.



for data1, label1 in train_generator:

    print('배치 데이터 크기:', data1.shape)

    print('배치 레이블 크기:', label1.shape)

    break

    

for data2, label2 in validation_generator:

    print('배치 데이터 크기:', data2.shape)

    print('배치 레이블 크기:', label2.shape)

    break    



       
# 왼쪽은 증식된(변형된)이미지, 오른쪽은 기존 이미지입니다.



import matplotlib.pyplot as plt

from keras.preprocessing import image



for i in range(4):   

    fig, axs = plt.subplots(ncols=2, figsize=(8,4), sharex=True, sharey=True)

    axs[0].imshow(image.array_to_img(data1[i]))

    axs[1].imshow(image.array_to_img(data2[i]))

    fig.tight_layout()

    plt.show()

# compute quantities required for featurewise normalization

# (std, mean, and principal components if ZCA whitening is applied)

# Only required if featurewise_center or featurewise_std_normalization or zca_whitening are set to True.

# train_datagen.fit(train_features)



train_datagen = ImageDataGenerator(

    rescale=1./255,

    featurewise_center= True,  # set input mean to 0 over the dataset

    samplewise_center=False,  # set each sample mean to 0

    featurewise_std_normalization= True,  # divide inputs by std of the dataset

    samplewise_std_normalization=False,  # divide each input by its std

    zca_whitening=False,  # apply ZCA whitening

    #rotation_range=40, # randomly rotate images in the range (degrees, 0 to 180)

    #zoom_range = 0.2, # Randomly zoom image 

    #shear_range=0.2,

    #width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)

    #height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)

    horizontal_flip=True)  # randomly flip images



# train_datagen.fit() # 이 부분에서 실행되지 않는다.



train_generator = train_datagen.flow_from_dataframe(

    dataframe = X_val, 

    directory = TRAIN_CROP_PATH,

    x_col = 'img_file',

    y_col = 'class',

    target_size = img_size,

    color_mode ='rgb',

    class_mode ='categorical',

    batch_size =batch_size,

    shuffle =False

)



train_generator.reset()

validation_generator.reset()

from skimage.transform import warp, AffineTransform, ProjectiveTransform

from skimage.exposure import equalize_adapthist, equalize_hist, rescale_intensity, adjust_gamma, adjust_log, adjust_sigmoid

from skimage.filters import gaussian

from skimage.util import random_noise

import random



def randRange(a, b):

    return np.random.rand() * (b - a) + a



def randomIntensity(im):

    # rescales the intesity of the image to random interval of image intensity distribution

    return rescale_intensity(im,

                             in_range=tuple(np.percentile(im, (randRange(0,10), randRange(90,100)))),

                             out_range=tuple(np.percentile(im, (randRange(0,10), randRange(90,100)))))



def randomGamma(im):

    # Gamma filter for contrast adjustment with random gamma value.

    return adjust_gamma(im, gamma=randRange(1, 2.5))



def randomGaussian(im):

    # Gaussian filter for bluring the image with random variance.

    return gaussian(im, sigma=randRange(0, 5))



def randomNoise(im):

    # random gaussian noise with random variance.

    var = randRange(0.005, 0.01)

    return random_noise(im, var=var)



# 위 4가지 함수를 넣어 random하게 적용시킨다.

def augment(im, Steps= [randomGamma, randomGaussian, randomNoise]):

    

    im /= 255. # 추가    

    i= np.random.randint(3)

    step = Steps[i]

    im = step(im)

    return im

# augment 함수 내부적으로 스케일링을 하기 때문에 rescale을 뺀다.

train_datagen = ImageDataGenerator(preprocessing_function=augment) 

valid_datagen = ImageDataGenerator(rescale=1./255) 



train_generator = train_datagen.flow_from_dataframe(

    dataframe = X_val, 

    directory = TRAIN_CROP_PATH,

    x_col = 'img_file',

    y_col = 'class',

    target_size = img_size,

    color_mode ='rgb',

    class_mode ='categorical',

    batch_size =batch_size,

    shuffle =False

)



train_generator.reset()

validation_generator.reset()

for data1, label1 in train_generator:

    print('배치 데이터 크기:', data1.shape)

    print('배치 레이블 크기:', label1.shape)

    break

    

for data2, label2 in validation_generator:

    print('배치 데이터 크기:', data2.shape)

    print('배치 레이블 크기:', label2.shape)

    break    
import matplotlib.pyplot as plt

from keras.preprocessing import image



for i in range(8):   

    fig, axs = plt.subplots(ncols=2, figsize=(8,4), sharex=True, sharey=True)

    axs[0].imshow(image.array_to_img(data1[i]))

    axs[1].imshow(image.array_to_img(data2[i]))

    fig.tight_layout()

    plt.show()



!rm -rf *_crop