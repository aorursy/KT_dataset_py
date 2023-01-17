# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing sShift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# import 해야할 것 하기.

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten, BatchNormalization
from keras.layers import Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
# 디렉토리 안의 파일 확인
print(os.listdir("../input/dogs-vs-cats/"))
from zipfile import ZipFile
zf = ZipFile('../input/dogs-vs-cats/train.zip', 'r')
zf.extractall('../kaggle/working/Temp')
zf.close()
print(os.listdir("../kaggle/working/Temp/train"))
# jpg그림 개와 고양이 분류카테고리 생성
filenames = os.listdir("../kaggle/working/Temp/train")
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append('dog')
    else:
        categories.append('cat')

df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})
df.head()
df['category'].value_counts()
from tensorflow.keras.preprocessing import image
img = image.load_img("../kaggle/working/Temp/train/"+filenames[0])
plt.imshow(img)
# 이미지 전처리
test_image = image.load_img("../kaggle/working/Temp/train/"+filenames[0], 
                            target_size=(128, 128))
test_image = image.img_to_array(test_image)
plt.imshow(test_image[:, :, 2])
#교육할 trainset 분리
from sklearn.model_selection import train_test_split

train_data, val_data = train_test_split(df, test_size=0.20, random_state=42)
train_data = train_data.reset_index(drop=True)
val_data   = val_data.reset_index(drop=True)
train_data.head()
val_data.head()
train_data['category'].value_counts()
val_data['category'].value_counts()
#모델 구성
model = Sequential([Convolution2D(filters=32, kernel_size=(3, 3), strides=(1, 1), input_shape=(128,128,3),
                            padding='valid', activation='relu'),
                         BatchNormalization(),
                         MaxPooling2D(pool_size=(2, 2)),
                         Dropout(0.2),
                         Convolution2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
                            padding='valid', activation='relu'),
                         BatchNormalization(),
                         MaxPooling2D(pool_size=(2, 2)),
                         Dropout(0.2),
                         Flatten(),
                         Dense(512, activation='relu'),
                         BatchNormalization(),
                         Dropout(0.25),
                         Dense(2, activation='softmax')])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
# 전처리
train_image = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

train_generator = train_image.flow_from_dataframe(
        train_data,
        "../kaggle/working/Temp/train/",
        x_col='filename',
        y_col='category',
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical')
val_datagen = ImageDataGenerator(rescale=1./255)

val_generator = val_datagen.flow_from_dataframe(
        val_data,
        "../kaggle/working/Temp/train/",
        x_col='filename',
        y_col='category',
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical')
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
# 학습이 일정 이상 좋아지지 않으면 중단.
early_stopp = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,restore_best_weights=True)
best_model =  tf.keras.callbacks.ModelCheckpoint('best_model.h5',moniter='val_accuracy',verbose=1,save_best_only=True)
# 모델 학습
hist=model.fit(train_generator,
                                epochs=10,
                                validation_data=val_generator,
                                callbacks=[early_stopp,best_model])
hist.history
# epoch에 의한 accuracy 변화율
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'], '')
plt.xlabel("Epochs")
plt.ylabel('Accuracy')
plt.title('Change of Accuracy over Epochs')
plt.legend(['accuracy', 'val_accuracy'])
plt.show()
# epoch에 의한 loss변화율
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'], '')
plt.xlabel("Epochs")
plt.ylabel('Loss')
plt.title('Change of Loss over Epochs')
plt.legend(['loss', 'val_loss'])
plt.show()
train_generator.class_indices
# 예측을 위한 test1 파일을 불러오기.
from zipfile import ZipFile
zf = ZipFile('../input/dogs-vs-cats/test1.zip', 'r')
zf.extractall('../kaggle/working/Temp')
zf.close()
filenames = os.listdir("../kaggle/working/Temp/test1")

test_data = pd.DataFrame({
    'filename': filenames
})
# 아까 model을 불러와서 예측준비.
from keras.models import load_model

pred = load_model('best_model.h5')
# dog = 1로 예측함.
img = image.load_img("../kaggle/working/Temp/test1/"+filenames[29])
                            
test_image = image.load_img("../kaggle/working/Temp/test1/"+filenames[29], 
                            target_size=(128, 128))
test_image = image.img_to_array(test_image)
plt.imshow(img)
test_image = np.expand_dims(test_image, axis=0)
result = pred.predict(test_image)
print(np.argmax(result, axis=1))
# cat 은 0으로 나와야하나 예측실패.
img = image.load_img("../kaggle/working/Temp/test1/"+filenames[55])
                            
test_image = image.load_img("../kaggle/working/Temp/test1/"+filenames[55], 
                            target_size=(128, 128))
test_image = image.img_to_array(test_image)
plt.imshow(img)
test_image = np.expand_dims(test_image, axis=0)
result = pred.predict(test_image)
print(np.argmax(result, axis=1))
