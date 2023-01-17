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
import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow.keras import layers, models

from sklearn.model_selection import train_test_split



from tensorflow.keras.preprocessing import image

from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_data = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

train_label = train_data.pop('label')



test_data = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
train_label.value_counts().plot.bar()
print(train_data.shape)

train_data = train_data.values.reshape(train_data.shape[0], 28, 28)

train_data = train_data / 255.0



test_data = test_data.values.reshape(test_data.shape[0], 28,28)

test_data = test_data / 255.0
fig = plt.figure(figsize=(6,8))

for i in range(1, 10):

    fig.add_subplot(3,3,i)

    plt.imshow(train_data[i],cmap='gray')

    plt.grid(False)

    plt.title(train_label[i])
model = tf.keras.Sequential([

    tf.keras.layers.Flatten(input_shape=(28,28)),

    tf.keras.layers.Dense(128, activation = "relu"),

    tf.keras.layers.Dense(10, activation = "softmax")

])

model.compile(optimizer = "adam", loss = tf.losses.sparse_categorical_crossentropy, metrics = ['accuracy'])

hist = model.fit(train_data, train_label, epochs = 10)
def plot_training(history):

    fig, axs = plt.subplots(1,2,figsize=(16,5)) 

    axs[0].plot(history.history['accuracy'], 'c') 

    axs[0].set_title('Model Accuracy') 

    axs[0].set_ylabel('Accuracy') 

    axs[0].set_xlabel('Epoch') 

    axs[0].legend(['train', 'validate'], loc='upper left') 

    

    axs[1].plot(history.history['loss'], 'c') 

    axs[1].set_title('Model Loss') 

    axs[1].set_ylabel('Loss') 

    axs[1].set_xlabel('Epoch') 

    axs[1].legend(['train', 'validate'], loc='upper right') 

    plt.show()

print(hist.history)

plot_training(hist)
test_result = model.predict_classes(test_data)

fig = plt.figure(figsize=(6,8))

for i in range(1, 10):

    image = test_data[i+10]

    label = test_result[i+10]

    fig.add_subplot(3,3,i)

    plt.imshow(image,cmap='gray')

    plt.grid(False)

    plt.title(label)
train_data_cnn = train_data.reshape(-1, 28,28,1)

test_data_cnn = test_data.reshape(-1, 28,28,1)



train_x, eval_x, train_y, eval_y = train_test_split(train_data_cnn,train_label,test_size=0.3,random_state=0)
datagen = ImageDataGenerator(

        rotation_range=10,

        zoom_range=0.20,

        width_shift_range=0.1,

        height_shift_range=0.1,

        shear_range=0.1,

        horizontal_flip=True,

        fill_mode="nearest")





train_datagen = datagen.flow(train_x,train_y,batch_size=50)

eval_datagen = datagen.flow(eval_x,eval_y,batch_size=50) 
model = models.Sequential()



model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape = (28,28,1)))

model.add(layers.MaxPooling2D(2,2))



model.add(layers.Conv2D(64, (3,3), activation='relu'))

model.add(layers.MaxPooling2D(2,2))



model.add(layers.Conv2D(64, (3,3), activation='relu'))



model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(10, activation='softmax')) 
model.summary()
model.compile(optimizer = "adam", 

              loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True), metrics = ['accuracy'])

hist = model.fit(train_x, train_y, validation_data=(eval_x, eval_y), epochs=20)
print(hist.history)

plot_training(hist)
test_result = model.predict_classes(test_data_cnn)



labels = []

for i in range(28000):

    labels.append(test_result[i])

index = [i for i in range(1,28001)]

df = pd.DataFrame({'ImageId': index, 'Label': labels})

df.to_csv('/kaggle/working/answer.csv', index = False)