# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv('../input/digit-recognizer/train.csv')
Test=pd.read_csv('../input/digit-recognizer/test.csv')
sub=pd.read_csv('../input/digit-recognizer/sample_submission.csv')
train.head()
x=train.drop('label',axis=1).values
y=train['label'].values
test=Test.values
x=x/255
test=test/255
y=pd.get_dummies(y)
y=y.values
y.shape
x.shape
test.shape
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=9)
x_train.shape,x_test.shape,y_train.shape,y_test.shape
x_train=x_train.reshape(-1,28,28,1)
x_test=x_test.reshape(-1,28,28,1)
test=test.reshape(-1,28,28,1)
x_train_plot=x_train.reshape(x_train.shape[0],28,28)
plt.imshow(x_train_plot[5],cmap='binary')
digit=np.argmax(y_train[5])
plt.title(f'The Value in image is {digit}')
x_train_plot=x_train.reshape(x_train.shape[0],28,28)
plt.imshow(x_train_plot[568],cmap='binary')
digit=np.argmax(y_train[568])
plt.title(f'The Value in image is {digit}')
#Data augmentation to prevent overfitting
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

#datagen.fit(X_train)
train_data=datagen.flow(x_train, y_train, batch_size=32)
test_data=datagen.flow(x_test, y_test, batch_size=32)
cnn=tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(input_shape=[28,28,1],filters=32,kernel_size=(3,3),activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(strides=2,pool_size=2))
cnn.add(BatchNormalization())
cnn.add(tf.keras.layers.Conv2D(input_shape=[28,28,1],filters=32,kernel_size=(3,3),activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(strides=2,pool_size=2))
cnn.add(BatchNormalization())
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=784,activation='relu'))
cnn.add(tf.keras.layers.Dense(units=10,activation='softmax'))
cnn.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
cnn.fit(x=train_data,validation_data=test_data,epochs=10)
from keras.utils.vis_utils import plot_model
plot_model(cnn, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
pred_y=cnn.predict_classes(x_test)
Y_test=np.argmax(y_test,1)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(pred_y,Y_test)
sns.heatmap(cm,annot=True)
predictions=cnn.predict_classes(test)
test_plot=test.reshape(test.shape[0],28,28)
plt.imshow(test_plot[5],cmap='binary')
plt.title(f'The Value in image is {predictions[5]}')
sub['Label']=predictions
sub.to_csv("MNIST_Kaggle.csv", index=False)
sub.head()
sub
