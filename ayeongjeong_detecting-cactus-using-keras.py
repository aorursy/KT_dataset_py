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
import matplotlib.pyplot as plt

from matplotlib.pyplot import imshow

import matplotlib.image as mpimg

from tensorflow.keras.preprocessing import image

from PIL import Image

import tensorflow as tf

from tensorflow.keras import layers
train_data = pd.read_csv('../input/aerial-cactus-identification/train.csv')

sub_data = pd.read_csv('../input/aerial-cactus-identification/sample_submission.csv')
train_data.head()
np.unique(train_data.has_cactus, return_counts=True)
has_cactus = train_data[train_data['has_cactus'] == 1].id.values

no_cactus = train_data[train_data['has_cactus'] == 0].id.values
# zip파일 풀기

!unzip ../input/aerial-cactus-identification/train.zip -d train

!unzip ../input/aerial-cactus-identification/test.zip -d test
print(os.listdir("../working"))
TRAIN_IMG_PATH = "../working/train/train/"

TEST_IMG_PATH = "../working/test/test/"
check_img = mpimg.imread(TRAIN_IMG_PATH + has_cactus[0])

imgplot = plt.imshow(check_img)
from tensorflow.keras.preprocessing.image import ImageDataGenerator



datagen=ImageDataGenerator(rescale=1./255)

batch_size=150

train_data.has_cactus=train_data.has_cactus.astype(str)



train_generator=datagen.flow_from_dataframe(dataframe=train_data[:15001],directory=TRAIN_IMG_PATH,x_col='id',

                                            y_col='has_cactus',class_mode='binary',batch_size=batch_size,

                                            target_size=(150,150))





validation_generator=datagen.flow_from_dataframe(dataframe=train_data[15000:],directory=TRAIN_IMG_PATH,x_col='id',

                                                y_col='has_cactus',class_mode='binary',batch_size=50,

                                                target_size=(150,150))
model = tf.keras.models.Sequential()

model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))

model.add(layers.MaxPool2D((2,2)))

model.add(layers.Conv2D(64,(3,3),activation='relu',input_shape=(150,150,3)))

model.add(layers.MaxPool2D((2,2)))

model.add(layers.Conv2D(128,(3,3),activation='relu',input_shape=(150,150,3)))

model.add(layers.MaxPool2D((2,2)))

model.add(layers.Conv2D(128,(3,3),activation='relu',input_shape=(150,150,3)))

model.add(layers.MaxPool2D((2,2)))

model.add(layers.Flatten())

model.add(layers.Dense(512,activation='relu'))

model.add(layers.Dense(1,activation='sigmoid'))



model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
hist = model.fit_generator(train_generator,steps_per_epoch=50,epochs=5,validation_data=validation_generator,validation_steps=50)
plt.plot(hist.history['accuracy'],'r',label='accuracy')

plt.plot(hist.history['val_accuracy'],'b',label='val_acc')

plt.legend()

plt.show()
plt.plot(hist.history['loss'],'r',label='loss')

plt.plot(hist.history['val_loss'],'b',label='val_loss')

plt.legend()

plt.show()
sub_data.has_cactus = sub_data.has_cactus.astype(str)



train_generator = datagen.flow_from_dataframe(dataframe=sub_data,directory=TEST_IMG_PATH,x_col='id',

                                              y_col='has_cactus',class_mode=None,batch_size=batch_size,target_size=(150,150))
datagen=ImageDataGenerator(rescale=1./255)

batch_size=150

sub_data.has_cactus=sub_data.has_cactus.astype(str)



test_generator=datagen.flow_from_dataframe(dataframe=sub_data,directory=TEST_IMG_PATH,x_col='id',

                                            y_col='has_cactus',class_mode=None,batch_size=batch_size,

                                            target_size=(150,150))
y_pred = model.predict(test_generator)



df=pd.DataFrame({'id':sub_data['id'] })



df['has_cactus']=y_pred

df.to_csv("/kaggle/working/submission.csv",index=False)
df