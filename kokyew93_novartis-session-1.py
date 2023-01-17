import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import gc

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



pal = sns.color_palette()



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls
df_train = pd.read_csv('../input/train_v2.csv')

df_train['deforest'] = df_train['tags'].str.contains("agriculture|habitation|road|cultivation|slash_burn|conventional_mine|bare_ground|artisinal_mine|selective_logging|blow_down")

df_train.head()
df_train['deforest'].describe()
import cv2



new_style = {'grid': False}

plt.rc('axes', **new_style)

_, ax = plt.subplots(3, 3, sharex='col', sharey='row', figsize=(20, 20))

i = 0

for f, l, j in df_train[:9].values:

    img = cv2.imread('../input/train-jpg/{}.jpg'.format(f))

    ax[i // 3, i % 3].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    ax[i // 3, i % 3].set_title('{} - {} - {}'.format(f, l, j))

    #ax[i // 4, i % 4].show()

    i += 1

    

plt.show()
## "deforest" need to be string

df_train.deforest = df_train.deforest.apply(str)

df_train["id"] = df_train["image_name"] + '.jpg'
from keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(

    rescale=1./255,

    shear_range=0.2,  

    zoom_range=0.2,        

    horizontal_flip=True,

    validation_split=0.2)  



train_generator = datagen.flow_from_dataframe(

    dataframe = df_train,

    x_col = 'id',

    y_col = 'deforest',

    directory = '../input/train-jpg',

    target_size = (150,150),

    batch_size=32,

    class_mode = 'binary',

    subset='training')



val_generator = datagen.flow_from_dataframe(

    dataframe = df_train,

    x_col = 'id',

    y_col = 'deforest',

    directory = '../input/train-jpg',

    target_size=(150,150),

    batch_size=32,

    class_mode='binary',

    subset='validation')
from keras import layers

from keras import models



model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))
from keras import optimizers



model.compile(loss='binary_crossentropy',

             optimizer=optimizers.RMSprop(lr=1e-4),

             metrics=['acc'])
history = model.fit_generator(

    train_generator, 

    steps_per_epoch  = 100, 

    validation_data  = val_generator,

    validation_steps = 50,

    epochs = 10)
acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(acc) + 1)

fig = plt.figure(figsize=(16,9))



plt.subplot(1, 2, 1)

plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()



plt.subplot(1, 2, 2)

plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()
model.save('amazon_forest.h5')
import cv2



new_style = {'grid': False}

plt.rc('axes', **new_style)



for f, l, j, k in df_train.iloc[[2521]].values:

    img = cv2.imread('../input/train-jpg/{}.jpg'.format(f))

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    plt.title(('{} - {}'.format(l, j)))

    

plt.show()
from keras.models import load_model

from keras.preprocessing import image



model = load_model('amazon_forest.h5')



test_image = image.load_img('../input/train-jpg/train_1.jpg', target_size=(150, 150))

test_image = image.img_to_array(test_image)

test_image = np.expand_dims(test_image, axis=0)

test_image = test_image.reshape(1, 150, 150, 3)   



result = np.array(model.predict(test_image))

classes = result.item(0)



if classes == 0:

    print ("Great! Forest is still well preserved!") 

elif classes == 1:

    print ("Oh no! Send some guards there!")