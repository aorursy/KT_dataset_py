# 라이브러리(도서관), 함수(도서관의 책)  

# 우리가 저장한 주피터노트북의 한 파일을 저장함으로써 그 파일이 도서관이 되고 그 안에 있는 우리가 만든 여러 함수들이 책이 된다.

# 모든 라이브러리는 사람이 만들어 놓은 것이라고 생각하면 된다.  

# keras라는 도서관에서 케라스안에 내장된 어떠한(책) 기능을 가져오는것 from keras import 책



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from glob import glob # Finds the pathname matching a specific pattern  경로명 찾아주는 라이브러리

# glob 모듈은 유닉스 셸이 사용하는 규칙에 따라 지정된 패턴과 일치하는 모든 경로명을 찾습니다. 

# 하지만 결과는 임의의 순서로 반환됩니다. 물결표(tilde) 확장은 수행되지 않지만, *, ? 및 []로 표시되는 문자 범위는 올바르게 일치합니다. 

# 이는 서브 셸을 실제로 호출하지 않고 os.scandir() 과 fnmatch.fnmatch() 함수를 사용하여 수행됩니다. fnmatch.fnmatch()와 달리, 

# glob은 점(.)으로 시작하는 파일 이름을 특수한 경우로 취급합니다. (물결표와 셸 변수 확장은 os.path.expanduser() 와 os.path.expandvars()를 사용하십시오.)

import cv2 # For image manipulation

# 실시간 컴퓨터 비젼을 처리하는 목적으로 만들어진 라이브러리

# 이미지 파일 읽고 쓰기

# 카메라 영상 처리

# 카메라 영상 저장하기

import keras.backend as k

#  케라스는 거의 모든 종류의 딥러닝 모델을 간편하게 만들고 훈련시킬 수 있는 파이썬을 위한 딥러닝 프레임워크

#  딥러닝 모델을 만들기 위한 고수준의 구성 요소를 제공하는 모델 수준의 라이브러리

#  백엔드 엔진backend engine에서 제공하는 최적화되고 특화된 텐서 라이브러리를 사용

#  백엔드, 사용자 눈에 보이지 않는 뒤에서 이루어지는 작업 , 서버나 클라이언트 작업

import tensorflow as tf

# 입력된 데이터를 가지고 머신러닝 알고리즘을 만들고 모델을 훈련시키는 걸 도와주는 함수를 가지고 있는 라이브러리

# 딥러닝 오픈소스 패키지 . 실제 파이썬 개발자들도 텐서플로우를 이해하기 상당히 힘들다라고 나와있음.

import os

# 운영체제에서 제공되는 여러 기능을 파이썬에서 수행

# 예를 들어, 파이썬을 이용해 파일을 복사하거나 디렉터리를 생성하고 특정 디렉터리 내의 파일 목록을 구하고자 할 때 os 모듈을 사용

from tqdm import tqdm_notebook

# 작업 진행상태 표시바 생성

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder # For encoding labels into 0 to n classes (데이터를 그룹화하여 문자를 숫자로 인코딩)

from keras.utils import np_utils # To convert encoded labels to binary data

# (from)케라스에서 제공되는 유틸리티 라이브러리에서 제공되는 (import)넘피의 기능을 쓰겠다.
# Reading the images and labels

images_path = '../input/train/*/*.png'   # train셋은 이미지 데이터로 구성되어있는데 이미지 데이터를 불러오기 위한 경로를 설정

images = glob(images_path)               # 불러온 경로를 glob방식으로 변경하겠다.

train_images = []                        # 빈 리스트 생성

train_labels = []                        # 빈 리스트 생성

                                         # 반복문 수행 하는동안, train_images의 빈리스트에 cv2방식으로 resize하겠다,() c2방식으로 이미지데이터를 imread 한것을 (70,70)사이즈로

                                         #                  train_labels의 빈리스트에 이미지를 나누겠다 ('/')[-2] 방식으로

for img in tqdm_notebook(images):       

    train_images.append(cv2.resize(cv2.imread(img), (70, 70)))

    train_labels.append(img.split('/')[-2])

    

    

train_X = np.asarray(train_images)        # 반복문을 수행하면서 만들어진 리스트를 np.asarray 방식으로 train_X라는 변수에 담겠다.

                                          # np.asarray = 배열조작함수, 기존의 array 보다 옵션이 많아 배열의 구조나 타입을 조작하기 더 쉽다.

train_Y = pd.DataFrame(train_labels)      # 반복문을 수행하면서 만들어진 리스트를 pd.df 방식으로 train_Y라는 변수에 담겠다.
# Displaying an image

plt.title(train_Y[0][100])               # 그래프 제목을 train_Y[0][100](풀잎의 종류)으로 지정하겠다.  

_ = plt.imshow(train_X[100])             # 그래프에서 보여주는 것을 train_X[100](2차원 어레이리스트 형식으로 되어있다)으로 지정하겠다.



print("plt.title = ",train_Y[0][100])

# print("train_X",train_X[0][0])
# Converting labels to numbers

encoder = LabelEncoder()   #라벨 인코딩 함수

#train_Y[0]을 라벨 인코딩  train_Y = 기존새싹의 종류.ex(Small-flowered Cranesbill)

encoder.fit(train_Y[0])    

encoded_labels = encoder.transform(train_Y[0])  # 

# 기존의 라벨인코딩 방식에서 케라스의 np_utils 기능을 가져와서 카테고리별_라벨을 만들었다.

categorical_labels = np_utils.to_categorical(encoded_labels)

print(categorical_labels)
plt.title(str(categorical_labels[100])) # np_utils방식으로 라벨인코딩된 새싹의 종류(이름)를 그래프 타이틀로 지정.

_ = plt.imshow(train_X[100])            # 그래프 보여주기 train_X[100](train_X = 문제,풀잎의 특징) 
x_train,x_test,y_train,y_test=train_test_split(train_X,categorical_labels,test_size=0.25,random_state=7)

#print(x_train.shape,x_test.shape)

import keras

from keras import layers # 케라스 방식의 모델을 구성하기 위해 쓰이는 기능

# 케라스의 핵심 데이터 구조는 모델이고, 이 모델을 구성하는 것이 레이어

# https://tykimos.github.io/2017/01/27/MLP_Layer_Talk/ << 케라스의 기본적인 구조와 개념, 레이어에 대한 기본적인 정보들. 

# 케라스 모델의 구성이 인간의 신경망 구조와 비슷하다고 해서 인공신경망이라고 불리는데 케라스에서도 뉴럴네트워크라고 부른다.

from keras.layers import Input, Dense, Activation,ZeroPadding2D, BatchNormalization, Flatten, Conv2D  # 레이어 기법의 종류, 각 레이어마다 특징과 역할이 조금씩 다르다.

from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D # 레이어 기법의 종류, 각 레이어마다 특징과 역할이 조금씩 다르다.

from keras.models import Sequential   # 케라스 모델에서의 선형 모델 , Sequential 모델은 레이어를 선형으로 연결하여 구성

# 케라스 모델링의 예 

# model = Sequential([

#     Dense(32, input_shape=(784,)),  from keras.layers(Dense)  기존의 머신러닝의 하이퍼파라미터 대신에 레이어가 들어가 있다고 생각하면 될듯?

#     Activation('relu'),             from keras.layers(Activation)

#     Dense(10),                      from keras.layers(Dense)

#     Activation('softmax'),          from keras.layers(Activation)

# ])

from keras.preprocessing.image import ImageDataGenerator

# 데이터를 이리저리 변형시켜서 새로운 학습 데이터를 만들어줍니다. 변형의 예시는 회전, 이동 등등 매우 다양

# https://neurowhai.tistory.com/158

from keras.applications.vgg16 import VGG16

# VGG-16은 ImageNet 데이터베이스의 1백만 개가 넘는 이미지에 대해 훈련된 컨벌루션 신경망. 이 네트워크에는 16개의 계층이 있으며, 

# 이미지를 키보드, 마우스, 연필, 각종 동물 등 1,000가지 사물 범주로 분류할 수 있다. 그 결과 이 네트워크는 다양한 이미지를 대표하는 다양한 특징을 학습하게 된다.

# https://kr.mathworks.com/help/deeplearning/ref/vgg16.html
base_model = VGG16(include_top=False, weights='imagenet', input_shape=(70, 70, 3))
model = Sequential()

model.add(base_model)    # 모델에 모델을 추가하였다. 이미지 모델.

model.add(layers.Flatten()) # 1차원 자료로 변형. 모델학습을 하기위해?

model.add(layers.Dense(256, activation='relu'))  # 레이어 추가  레이어의 옵션

# activation : 활성화 함수 설정합니다.

# ‘linear’ : 디폴트 값, 입력뉴런과 가중치로 계산된 결과값이 그대로 출력으로 나옵니다.

# ‘relu’ : rectifier 함수, 은익층에 주로 쓰입니다.

# ‘sigmoid’ : 시그모이드 함수, 이진 분류 문제에서 출력층에 주로 쓰입니다.

# ‘softmax’ : 소프트맥스 함수, 다중 클래스 분류 문제에서 출력층에 주로 쓰입니다.

#  https://tykimos.github.io/2017/01/27/CNN_Layer_Talk/

model.add(layers.Dense(12, activation='sigmoid')) # 레이어 추가. 레이어의 옵션
opt = keras.optimizers.adam(lr=0.0001, decay=1e-6) # lr = lunning_rate

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# 모델의 정확도 지표



# 학습률 지정 , 기존의 모델의 eport,running_rate와 비슷한 역할로써 여기서는 옵티마이저라는 새로운 기능으로 쓰인다.



# https://forensics.tistory.com/28
datagen = ImageDataGenerator(

    featurewise_center=False,  # set input mean to 0 over the dataset, 데이터 세트에서 입력 평균을 0으로 설정

    samplewise_center=False,  # set each sample mean to 0 , 각 표본 평균을 0으로 설정

    featurewise_std_normalization=False,  # divide inputs by std of the dataset , 입력을 데이터 세트의 표준으로 나누기

    samplewise_std_normalization=False,  # divide each input by its std, 입력을 데이터 세트의 표준으로 나누기

    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180) ,각 입력을 표준으로 나눕니다. 범위 내에서 이미지를 무작위로 회전 (도, 0 ~ 180)

    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width), 이미지를 가로로 무작위 이동 (총 너비의 비율)

    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height) ,이미지를 세로로 무작위 이동 (전체 높이의 비율)

    horizontal_flip=True,  # randomly flip images , 무작위로 이미지 뒤집기

    vertical_flip=False)
datagen.fit(x_train) # 학습
model.fit_generator(datagen.flow(x_train, y_train, # 학습

                                    batch_size=50),

                    steps_per_epoch=x_train.shape[0],

                    epochs=1,

                    validation_data=(x_test, y_test),

                    verbose=1)     

[loss, accuracy] = model.evaluate(x_test, y_test) # 테스트데이터 오차율,정확도
print('Test Set Accuracy: '+str(accuracy*100)+"%");
test_images_path = '../input/test/*.png'  # 트레인데이터에 해주었던 전처리를 똑같이 테스트 데이터에서 반복

test_images = glob(test_images_path)

test_images_arr = []

test_files = []



for img in test_images:

    test_images_arr.append(cv2.resize(cv2.imread(img), (70, 70)))

    test_files.append(img.split('/')[-1])



test_X = np.asarray(test_images_arr)
_ = plt.imshow(test_X[100])    # 그래프로 확인
predictions = model.predict(test_X) #결과 저장
preds = np.argmax(predictions, axis=1) #결과를 어레이최대값의 인덱스 번호로 저장.

pred_str = encoder.classes_[preds] # preds값의 인코딩했던 숫자를 문자로 다시 디코딩해줌
final_predictions = {'file':test_files, 'species':pred_str}  # 확인한 파일 , 저장된 파일의 결과파일을 딕셔너리형태로

final_predictions = pd.DataFrame(final_predictions)          # 데이터프레임으로

final_predictions.to_csv("submission.csv", index=False)      # 저장   
final_predictions