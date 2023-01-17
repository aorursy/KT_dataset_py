# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('../input/digit-recognizer/train.csv')
test = pd.read_csv('../input/digit-recognizer/test.csv')
print(train.shape , test.shape)
train.head()
# LOAD LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#넘파이, 판다스, MATPLOT은 기본으로 항상

from sklearn.model_selection import train_test_split
# train.test 셋을 쉽게 분리하기 위해서

from keras.utils.np_utils import to_categorical
# cnn을 통해 최종적으로 결과를 받으면 라벨수만큼의 각각의 확률값으로 반환된다. 결과값을 받기 편하게 하기위한 함수
from keras.models import Sequential
# 케라스 모델구성기본 함수
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
# 케라스에서 필요한 레이어들 간편하게 쓸수 있다.
from keras.preprocessing.image import ImageDataGenerator
# 이미지를 조금 변화해줌으로써 성능을 올릴수 있다. 그랜드 마스터 Chris Deotte 의 25 Million Images! [0.99757] MNIST 커널에서 참고했다.(그외에도 거의 많이 참고했다.)
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tqdm.keras import TqdmCallback
# 콜벡 모델이 어떤 기준으로 돌다가 멈추고 저장하고 하는것들을 설정해줄수 있다.
import warnings
warnings.filterwarnings('ignore')
# 지저분하게 워닝뜨는걸 막아준다.
# X의 라벨값을 CNN에 넣을수 없고 이따 예측비교시 쓸거니까 분리해준다.
Y_train = train["label"]
X_train = train.drop(labels = ["label"],axis = 1)

# 0~255 사이의 픽셀명암의 숫자를 계산하기 편하기 255로 나눠 비율은 유지하고 숫자는 작게
X_train = X_train / 255.0
X_final = test / 255.0
#이제 (28,28) 모양으로 RESHAPE
X_train = X_train.values.reshape(-1,28,28,1)
X_final = X_final.values.reshape(-1,28,28,1)
#아까 뺴놓은 라벨값도 CNN결과값이랑 비교할수 있는 형태로 
Y_train = to_categorical(Y_train, num_classes = 10)

# matplot으로 간단하게 시각화 해보면 라벨값에 맞는 숫자를 확인할수 있다.

fig = plt.figure(figsize=(10,10))

for i in range(10):
    i += 1
    plt.subplot(2,5,i)
    plt.title(train['label'][i])
    plt.imshow(X_train[i].reshape(28,28))
    plt.axis('off')
plt.show()
# 10도 정도 돌리고 10% 정도 줌하고, 왼쪽, 오른쪽 시프트를 해서 다양한 변화를 준 데이터를 추가해준다.(이따 케라스 모델 fit_generator 할때 사용예정)
datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,)
# 아까 사이킷런에서 가져온 split으로 train안에서 훈련분과 검증분을 나눈다.(보통 0.3 비중으로 검증사이즈를 잡지만 데이터가 충분해서 0.1로 한다)
X_train, X_test, y_train, y_test = train_test_split(X_train, Y_train, test_size = 0.1)
model = Sequential()

model.add(Conv2D(32,kernel_size=3,activation= 'relu', input_shape = (28,28,1) ))
model.add(BatchNormalization())
model.add(Conv2D(32,kernel_size=3,activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(32,kernel_size=5,activation = 'relu', padding='same',strides=2))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(64,kernel_size=3,activation= 'relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,kernel_size=3,activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,kernel_size=5,activation = 'relu', padding='same',strides=2))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(128,kernel_size=4,activation= 'relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(10,activation='softmax'))
# MODEL확인 : 참 편하다
model.summary()
model.compile(optimizer='adam', loss = "categorical_crossentropy", metrics=['accuracy'])

# 콜벡은 이렇게 선언해서 callbacks에 담아놓자
earlyStopping = EarlyStopping(patience=10, verbose=0)
reduce_lr_loss = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.000001, verbose=0)
tqdm = TqdmCallback(verbose=0) #진행율 표시해준다.(없으면 답답하다)
callbacks = [earlyStopping, reduce_lr_loss, tqdm]
history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=64),
                              epochs = 20,
                              steps_per_epoch = X_train.shape[0]//64,
                              validation_data = (X_test,y_test),
                              callbacks=callbacks,
                              verbose=0)
# 결과를 확인해보자(학습대상의 정확도, 검증대상의 정확도)
print('train_acc:{0:.5f} , val_acc:{1:.5f}'.format(max(history.history['accuracy']),max(history.history['val_accuracy'])))
# 그래프로 표시해보는 정확도 Accuracy, 훈련이 적당히 잘된거 같다. 더이상 훈련은 생략...

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy', fontsize=14)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Accuracy',fontsize=14)
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
#결과값을 담을 results를 0으로 testsize, 라벨개수의 행렬로 선언하고, 거기다가 결과값을 더해서 담는다
results = np.zeros( (X_final.shape[0],10) ) 
results = results+model.predict(X_final)
# 이런식으로 처음 데이터의 대한 10개 라벨의 예측값이 담겨있다
results[0]
# 각각의 확률값중에 가장높은값이 바로 예측값이니까 argmax를 이용해서 뽑아준다.
results = np.argmax(results, axis=1)
# 하나 뽑힌값을 pd.Series를 이용해 1차원으로 만들어 준다.순서대로 각 데이터의 예측 라벨이된다.
results = pd.Series(results,name='label')
#submission 양식에 맞춰야 하니까 0부터 시작이 아닌 1부터 시작하는 형태로 데이터 프레임을 만든다.
submission = pd.concat([pd.Series(range(1,28001), name='Imageid'),results],axis=1)
# 최종 데이터값 저장
submission.to_csv("submission.csv",index=False)
