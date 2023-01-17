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
PATH = '../input/digit-recognizer'

df_train = pd.read_csv(os.path.join(PATH, 'train.csv'))

train_y = df_train['label'].values

train_x = df_train.drop(['label'], axis=1).values





df_test = pd.read_csv(os.path.join(PATH, 'test.csv'))

test_x = df_test.values





print(train_x.shape)

print(train_y.shape)

print(test_x.shape)
train_x[0][0].shape
#import numpy as np

#np.set_printoptions(linewidth=200)

#import matplotlib.pyplot as plt

#plt.imshow(train_x[0])

#print(train_label[0])

#print(train_x[0])
IMG_SIZE = 32
import cv2
def resize(img_array):

    tmp=np.empty((img_array.shape[0] ,IMG_SIZE,IMG_SIZE))

    

    for i in range(len(img_array)):

        img=img_array[i].reshape(28,28).astype('uint8')

        img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))

        img=img.astype('float32')/255

        tmp[i]=img

    return tmp
train_x_resize=resize(train_x)

test_x_resize=resize(test_x)
train_x_resize[0].shape
import numpy as np

np.set_printoptions(linewidth=200)

import matplotlib.pyplot as plt

plt.imshow(train_x_resize[9])

#print(train_label[0])

print(train_x_resize[9])
train_x_final = np.stack((train_x_resize,)*3, axis=-1)

test_x_final = np.stack((test_x_resize,)*3, axis=-1)

print(train_x_final.shape)

print(test_x_final.shape)
from keras.utils import to_categorical

train_y_final=to_categorical(train_y,num_classes=10)
train_y_final.shape
from keras.models import Sequential

from keras.layers import Dense,Flatten

from keras.applications import VGG19
#
vgg19=VGG19(weights='imagenet',

         include_top=False,

        input_shape=(IMG_SIZE,IMG_SIZE,3),

         )
vgg19.summary()
model=Sequential()

model.add(vgg19)

model.add(Flatten())

model.add(Dense(10,activation='softmax'))
model.compile(loss='categorical_crossentropy', 

              optimizer='sgd', 

              metrics=['accuracy'])
model.summary()
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(train_x_final, train_y_final, test_size=0.2, random_state=2019)

print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)
history = model.fit(x_train, y_train, 

                    epochs=10, 

                    batch_size=128, 

                    validation_data=(x_test, y_test),

                    )
preds = model.predict(test_x_final, batch_size=128)

preds.shape



results = np.argmax(preds, axis=-1)

results.shape
# submission

sub = pd.read_csv(os.path.join(PATH, 'sample_submission.csv'))

sub.head()

df = pd.DataFrame({'ImageId': sub['ImageId'], 'Label': results})

df.to_csv('submission.csv', index=False)

os.listdir('./')