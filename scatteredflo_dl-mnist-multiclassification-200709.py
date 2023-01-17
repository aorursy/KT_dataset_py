from IPython.display import Image

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf



from tensorflow.keras.layers import Dense, Flatten

from tensorflow.keras.models import Sequential

from tensorflow.keras.callbacks import ModelCheckpoint



from PIL import Image

mnist = tf.keras.datasets.mnist

(X_train, y_train), (X_valid, y_valid) = mnist.load_data()
X_train.shape, y_train.shape, X_valid.shape, y_valid.shape
#=== X_valid의 일부를 때어내서 Test 데이터로 사용하자(모델 학습이후 Predict 확인) ===#

X_test = X_valid[:100]

y_test = y_valid[:100]



y_valid = y_valid[100:]

X_valid = X_valid[100:]





X_valid.shape, X_test.shape, " /// ", y_valid.shape,  y_test.shape
X_train[0]

# 한게의 X_train의 Array를 출력해보자

# 뭔데 이게
import matplotlib as mpl

import matplotlib.pylab as plt

import seaborn as sns



NCOLS = 20



fig, ax = plt.subplots(ncols=NCOLS,figsize = (NCOLS*2,5))

                       

for i in range(NCOLS):

    sns.heatmap(X_train[i], vmax=.8, square=True, cmap="Blues",ax=ax[i])   # Array를 Heatmap으로 출력



print(y_train[:NCOLS])   # y_train 값을 출력



# heatmap으로 출력해보니 잘보임, y_train 값과 비교해보자
X_train = X_train/255

X_valid = X_valid/255



# 전체적으로 Normalization을 해주자 (0~255값을 그대로 넣어주면 모델이 힘들어함)
model = tf.keras.models.Sequential([

    #=== Flatten으로 shape 펼치기 ===#

    tf.keras.layers.Flatten(input_shape=(28, 28)),



    #=== Dense Layer ===#

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(256, activation='relu'),

    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.Dense(64, activation='relu'),

    tf.keras.layers.Dense(32, activation='relu'),



    #=== Classification을 위한 Softmax ===# 

    tf.keras.layers.Dense(10, activation='softmax'),     # 마지막 Softmax 함수층은 배출할 항목 개수로 지정 (0~10 출력)

    # tf.keras.layers.Dense(1, activation='sigmoid')     # 2진 분류(0,1) 일때는 출력값 1개 및 sigmoid로 지정

])
model.summary()

# 앞에서 만든 모델의 구조를 보여줌
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])



checkpoint_path = "checkpoint_valid.ckpt"



checkpoint = ModelCheckpoint(filepath=checkpoint_path, 

                            save_weights_only=True,     # 용량을 줄이기 위해 weight

                            save_best_only=True,        # 제일 Best만 저장(아니면 전부)

                            monitor='val_loss',         # 우리가 Monitoring할 loss는 Validation

                            verbose=1)



history = model.fit(X_train, y_train,

                    validation_data=(X_valid, y_valid),

                    epochs=20,     # 몇번 반복할지

                    callbacks=[checkpoint],     # 학습할때 Checkpoint가 저장됨, Overfit을 방지함

                )



# checkpoint 를 저장한 파일명을 입력합니다.

model.load_weights(checkpoint_path)
import pandas as pd

import numpy as np

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# kaggle dataset 확인
#=== Image upload 후 실행 ===#

image = Image.open('/kaggle/input/image-data/a.jpg')   # 내가 만든 이미지 불러오기



print(np.array(image).shape)

plt.imshow(image)                      # image 출력

plt.show()
image = image.convert('1')             # 3channel --> 1channel로 변경



image = image.resize((28, 28))         # 이미지 사이즈를 28,28로 변형시켜줌 (Training-data shape와 일치화)

image = np.array(image)                # image를 array로 변형



image = image/255.                     # Normalization



print(image.shape)

plt.imshow(image)                      # image 출력

plt.show()
image = image.reshape(1,28,28)         # shape를 (28,28) -> (1,28,28)로 변형 (Training-data shape와 일치화)

print(image.shape)
prediction = model.predict(image)             #모델 예측



pred_class = np.argmax(prediction, axis=-1)   # argmax를 하면 앞에서 OHE로 나온 확률에 대한 class가 나옴

pred_class