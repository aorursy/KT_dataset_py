import cv2

import glob

import datetime

import numpy as np

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow import keras

from shutil import copyfile
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    #for filename in filenames:

        #print(os.path.join(dirname, filename))

    print(dirname)

# Any results you write to the current directory are saved as output.

#man_dir='D:\\Temp\\MAN_WOMAN\\man';

#woman_dir='D:\\Temp\\MAN_WOMAN\\woman';

man_dir='/kaggle/input/man-woman-faces/Man_Woman/man';

woman_dir='/kaggle/input/man-woman-faces/Man_Woman/woman';
man_files = [f for f in glob.glob(man_dir + "**/*.jpg", recursive=True)]

woman_files = [f for f in glob.glob(woman_dir + "**/*.jpg", recursive=True)]
def read_image(file):

    return cv2.imread(file,0)/255.0

x=[]

y=[]

for file in man_files:

    img_np = read_image(file)

    x.append(img_np)



i=0    

for file in woman_files:

    img_np = read_image(file)

    x.append(img_np)    



y=np.append(np.zeros(len(man_files)),np.ones(len(woman_files)))

x=np.array(x)

#y=np.array(y)

#woman_files
y.shape
x[0].shape
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
x_train.shape
class_names = ['Man', 'Woman']
x_train.shape
index=3

plt.figure()

plt.imshow(x_train[index])

plt.colorbar()

plt.grid(False)

plt.xlabel(class_names[int(y_train[index])])

plt.show()
plt.figure(figsize=(10,10))

for i in range(25):

    plt.subplot(5,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(x_train[i], cmap=plt.cm.binary)

    #print(y_train[i])    

    plt.xlabel(class_names[int(y_train[i])])

plt.show()
model = keras.Sequential([

    keras.layers.Conv2D(32,(3,3),input_shape=(150,150,1),activation='relu'),

    keras.layers.MaxPooling2D(3,3),

    keras.layers.Flatten(),    

    keras.layers.Dense(128, activation='relu'),  

    keras.layers.Dropout(0.4),

    keras.layers.Dense(16, activation='relu'),

    keras.layers.Dense(2, activation='softmax')

])
#log_dir="D:\\MAN_WOMAN_MODEL\\LOGS\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

#print(log_dir)

#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#print(tensorboard_callback)
model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy']

              #callbacks=[tensorboard_callback]

             )
y_train_categorical = keras.utils.to_categorical(y_train)

x_train_changed=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],x_train.shape[2],1))

#y_train_categorical
model.fit(x_train_changed, y_train_categorical, epochs=10)
y_test_categorical = keras.utils.to_categorical(y_test)

x_test_changed=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],x_test.shape[2],1))

test_loss, test_acc = model.evaluate(x_test_changed,  y_test_categorical, verbose=2)
predictions = model.predict(x_test_changed)

predictions_result=[]

for prediction in predictions:

    predictions_result.append(int(np.argmax(prediction)))

predictions_result=np.array(predictions_result)

predictions_result_categorical = keras.utils.to_categorical(predictions_result)

#np.sum(np.ab-y_test))

num_of_error_images=np.sum(np.abs(predictions_result-y_test))

#y_test[1]

print('num_of_error_images='+str(num_of_error_images))

weights_list = model.get_weights()

#print('weights_list='+str(weights_list))
#model.save_weights('D:\\MAN_WOMAN_MODEL\\man_woman_19.model')