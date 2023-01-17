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
from IPython.display import Image
Image("../input/sign-language-mnist/amer_sign2.png")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
print(tf.__version__)
path_sign_mnist_train='../input/sign-language-mnist/sign_mnist_train.csv'
path_sign_mnist_test='../input/sign-language-mnist/sign_mnist_test.csv'
def get_data(filename):
    labels=[]
    images=[]
    arr=np.loadtxt(filename,delimiter=',',skiprows=1)
    labels = arr[:,0].astype('int')
    images = arr[:,1:]
    images = images.astype('float').reshape(images.shape[0], 28, 28)
    arr=None
    
    return images,labels
training_images,training_labels=get_data(path_sign_mnist_train)
testing_images,testing_labels=get_data(path_sign_mnist_test)

plt.figure(figsize=(12,6))
sns.countplot(x=training_labels)
training_images = np.expand_dims(training_images, axis=3)
testing_images = np.expand_dims(testing_images, axis=3)

from tensorflow.keras.preprocessing.image import ImageDataGenerator as idg
plt.imshow(training_images[9].reshape(28,28))
train_datagen= idg(rescale=1.0/255.0)
validation_datagen=idg(rescale=1.0/255.0)
print(training_images.shape)
print(testing_images.shape)

batch_size=16
num_classes=25
epochs=50
model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(28,28,1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(num_classes,activation='softmax')
])
model.summary()
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history=model.fit_generator(train_datagen.flow(training_images,training_labels,batch_size=batch_size),
                            epochs = epochs,
                              validation_data=validation_datagen.flow(testing_images,testing_labels,batch_size=batch_size))
model.evaluate(testing_images,testing_labels,verbose=0)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy")
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train','test'])

plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Loss")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','test'])

plt.show()
y_pred = model.evaluate(testing_images,testing_labels,verbose=0)
y_pred[1]
