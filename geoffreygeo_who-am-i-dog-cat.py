# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

from PIL import Image

print(os.listdir("../input"))





from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pylab as plt

import seaborn as sns



import tensorflow as tf

tf.enable_eager_execution()



import tensorflow_hub as hub

from tensorflow.keras import layers

from tensorflow.keras.preprocessing.image import ImageDataGenerator



# Any results you write to the current directory are saved as output.
sample = pd.read_csv('../input/sampleSubmission.csv')
train = os.listdir("../input/train/train")

train_dir = "../input/train/train"
image = Image.open(r"../input/train/train/cat.11679.jpg")

display(image)

image = np.array(image)

display(image.shape)
categories = []



train[0].split('.')[0]



for data in train:

    category = data.split('.')[0]

    if(category == 'cat'):

        categories.append(0)

    elif(category == 'dog'):

        categories.append(1)



train_df = pd.DataFrame({

    'filename':train,

    'category':categories

})

train_df.head()
#plotting ham and spam data % in pie chart 

count_Class=pd.value_counts(train_df.category, sort= True)



# Data to plot

labels = 'dog', 'cat'

sizes = [count_Class[0], count_Class[1]]

colors = ['blue', 'lightskyblue'] # 'lightcoral', 'lightskyblue'

explode = (0, 0)  # explode 1st slice

 

# Plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')

plt.show()
print(tf.test.is_gpu_available())
BATCH_SIZE = 100

IMG_SHAPE = 224

train_generator = ImageDataGenerator(rescale=1./255)

test_generator = ImageDataGenerator(rescale = 1./255)
train_df.head()

train_df.category=train_df.category.astype(str)
train_data = train_generator.flow_from_dataframe(dataframe=train_df,directory=train_dir,x_col='filename',y_col='category',class_mode='binary',batch_size=BATCH_SIZE,

                                            target_size=(IMG_SHAPE,IMG_SHAPE))
#loading the state of art neural network 

URL = "https://tfhub.dev/google/imagenet/inception_resnet_v2/classification/2"



IMAGE_RES = 224



feature_extractor =  hub.Module(URL)
#Freezing so that the training modeifies only the final layer

feature_extractor.trainable = False
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),

    tf.keras.layers.MaxPooling2D(2, 2),



    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(2, activation='softmax')

])
model.compile(optimizer='adam',

             loss='sparse_categorical_crossentropy',

             metrics=['accuracy'])
%%time

EPOCHS=10

history = model.fit_generator(

    train_data,

    steps_per_epoch=20,

    epochs=EPOCHS,

)
history.history
acc = history.history['acc']



loss = history.history['loss']





epochs_range = range(EPOCHS)



plt.figure(figsize=(8, 8))

plt.subplot(1, 2, 1)

plt.plot(epochs_range, acc, label='Training Accuracy')

plt.legend(loc='lower right')

plt.title('Training  Accuracy')



plt.subplot(1, 2, 2)

plt.plot(epochs_range, loss, label='Training Loss')

plt.legend(loc='upper right')

plt.title('Training  Loss')

plt.savefig('./foo.png')

plt.show()