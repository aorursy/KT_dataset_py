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
import numpy as np
import pandas as pd
import cv2
import keras
from PIL import Image as im

from keras import regularizers
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten
df = pd.read_csv('../input/nnfl-cnn-lab2/upload/train_set.csv')
df
image_name = []
for i in range(len(df)):
    image_name.append(df.iloc[i,0])
x_train = []
del_list = []

for i in range(0,len(image_name)):
    img = cv2.imread('../input/nnfl-resized-image/resized_train_images/'+str(image_name[i]),0)
    x_train.append(img)

X_train = np.array(x_train)
X_train.shape
from skimage import img_as_ubyte

y_train = []
for i in range(len(df)):
        y_train.append(df.iloc[i,1])

y_train = np.array(y_train)
y_train = img_as_ubyte(y_train)
y_train
y_train.shape
df1 = pd.read_csv('../input/nnfl-cnn-lab2/upload/sample_submission.csv')
df1
image_name1 = []
for i in range(len(df1)):
    image_name1.append(df1.iloc[i,0])
x_test = []

for i in range(0,len(image_name1)):
    img = cv2.imread('../input/nnfl-resized-image/resized_test_images/'+str(image_name1[i]),0)
    x_test.append(img)

X_test = np.array(x_test)
X_test.shape
X_train=X_train.reshape(14034,150,150,1).astype('float32') 
X_test=X_test.reshape(7301,150,150,1).astype('float32') 
# Input shape format: (28, 28, 1)
# If 128x128 RGB, (128,128,3)
X_train = X_train / 255
X_test = X_test / 255
print(X_train.shape)
print(X_test.shape)
y_train_onehot = np_utils.to_categorical(y_train, num_classes=6)
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(5,5), padding='same', input_shape=(150,150,1), activation='relu'))
prediction = model.predict_classes(X_train[:1,:])
import matplotlib.pyplot as plt
train_img = np.reshape(X_train[:1,:], (150, 150))
plt.matshow(train_img, cmap = plt.get_cmap('binary'))
plt.show()
import matplotlib.pyplot as plt
cov_img = np.reshape(prediction[:1,:], (150, 150))
plt.matshow(cov_img, cmap = plt.get_cmap('binary'))
plt.show()
model = Sequential()
model.add(Conv2D(filters=16,kernel_size=(5,5), padding='same', input_shape=(150,150,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
prediction = model.predict_classes(X_train[:1,:])
prediction[:1,:]
import matplotlib.pyplot as plt
max_pooling_img = np.reshape(prediction[:1, :], (75, 75))
plt.matshow(max_pooling_img, cmap = plt.get_cmap('binary'))
plt.show()
model = Sequential()

#模型不要設太複雜

# Conv + Max-pooling 1
model.add(Conv2D(filters=16,kernel_size=(3,3),padding='same', input_shape=(150,150,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=2))


# Conv + Max-pooling 2
model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=2))



# Conv + Max-pooling 3
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
model.add(Dropout(0.25))


# Conv + Max-pooling 4
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
model.add(Dropout(0.5))


# Conv + Max-pooling 5
model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
model.add(Dropout(0.5))

# Conv + Max-pooling 5
model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
model.add(Dropout(0.5))


# Flatten層: 壓成一維
# Dense 接在內層不用input_dim，其他參數先用預設值
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='normal'))
model.add(Dropout(0.5))





model.add(Dense(6, kernel_initializer = 'normal',activation='softmax'))
model.summary()
from keras.callbacks import EarlyStopping
earlyStopping=EarlyStopping(monitor='val_accuracy', patience=8) 

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
history=model.fit(X_train, y_train_onehot, validation_split=0.2, epochs=50, batch_size=50, verbose=2, callbacks=[earlyStopping])
import matplotlib.pyplot as plt
%matplotlib inline
def plot_train_history(history, train_metrics, val_metrics):
    plt.plot(history.history.get(train_metrics),'-o')
    plt.plot(history.history.get(val_metrics),'-o')
    plt.ylabel(train_metrics)
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'])
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plot_train_history(history, 'loss','val_loss')
plt.subplot(1,2,2)
plot_train_history(history, 'accuracy','val_accuracy')
test_label = []

for i in range(len(X_test)):
    mmm = model.predict_classes(X_test[i:i+1,:])
    test_label.append(mmm)
    
test_label
df_test_label = pd.DataFrame(test_label)
df_test_label.shape
df_test_label = df_test_label.rename(columns = {0:'label'})
df_test_label
submission_df = pd.concat([df1, df_test_label], axis=1)
submission_df
submission_df.to_csv('submission.csv', index = False)