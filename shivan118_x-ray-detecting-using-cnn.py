# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array

from keras.models import Sequential, Model

from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D

from keras.layers import Activation, Dropout, BatchNormalization, Flatten, Dense, AvgPool2D,MaxPool2D

from keras.models import Sequential, Model

from keras.applications.vgg16 import VGG16, preprocess_input

from keras.optimizers import Adam, SGD, RMSprop

import tensorflow as tf



import os

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import glob

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

%matplotlib inline



import plotly.offline as py

import plotly.express as px

from fbprophet import Prophet

from fbprophet.plot import plot_plotly, add_changepoints_to_plot



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = '/kaggle/input/covid-19-x-ray-10000-images/dataset'
os.listdir(data)
normal_images = []

for img_path in glob.glob(data + '/normal/*'):

    normal_images.append(mpimg.imread(img_path))



fig = plt.figure()

fig.suptitle('Normal Image')

plt.imshow(normal_images[0], cmap='gray')
covid_images = []

for img_path in glob.glob(data + '/covid/*'):

    covid_images.append(mpimg.imread(img_path))



fig = plt.figure()

fig.suptitle('covid Image')

plt.imshow(covid_images[0], cmap='gray')
Image_Width = 150

Image_Height = 150

Cannels = 3



INPUT_SHAPE = (Image_Width, Image_Height, Cannels)

NB_CLASSES = 2

EPOCHS = 45

BATCH_SIZE = 6
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=INPUT_SHAPE))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))





model.add(Conv2D(64,(3,3)))

model.add(Activation("relu"))

model.add(Conv2D(250,(3,3)))

model.add(Activation("relu"))

  

model.add(Conv2D(128,(3,3)))

model.add(Activation("relu"))

model.add(AvgPool2D(2,2))

model.add(Conv2D(64,(3,3)))

model.add(Activation("relu"))

model.add(AvgPool2D(2,2))



model.add(Conv2D(256,(2,2)))

model.add(Activation("relu"))

model.add(MaxPool2D(2,2))

    

model.add(Flatten())

model.add(Dense(32))

model.add(Dropout(0.25))

model.add(Dense(1))

model.add(Activation("sigmoid"))
model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
model.summary()

from tensorflow.keras.utils import plot_model

plot_model(model, to_file='model1.png')
train_datagen = ImageDataGenerator(rescale=1./255,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    validation_split=0.25)



train_generator = train_datagen.flow_from_directory(

    data,

    target_size=(Image_Height, Image_Width),

    batch_size=BATCH_SIZE,

    class_mode='binary',

    subset='training')



validation_generator = train_datagen.flow_from_directory(

    data, 

    target_size=(Image_Height, Image_Width),

    batch_size=BATCH_SIZE,

    class_mode='binary',

    shuffle= False,

    subset='validation')



history = model.fit_generator(

    train_generator,

    steps_per_epoch = train_generator.samples // BATCH_SIZE,

    validation_data = validation_generator, 

    validation_steps = validation_generator.samples // BATCH_SIZE,

    epochs = EPOCHS)
fig , ax = plt.subplots(1,2, figsize=(14,5))

ax[0].plot(history.history['accuracy'])

ax[0].plot(history.history['val_accuracy'])

ax[0].set_title('model accuracy')

ax[0].set_ylabel('accuracy')

ax[0].set_xlabel('epoch')

ax[0].legend(['train', 'test'], loc='upper left')



ax[1].plot(history.history['loss'])

ax[1].plot(history.history['val_loss'])

ax[1].set_title('model loss')

ax[1].set_ylabel('loss')

ax[1].set_xlabel('epoch')

ax[1].legend(['train', 'test'], loc='upper left')

plt.show()
print("training_accuracy", history.history['accuracy'][-1])

print("validation_accuracy", history.history['val_accuracy'][-1])
label = validation_generator.classes

pred= model.predict(validation_generator)

predicted_class_indices=np.argmax(pred,axis=1)

labels = (validation_generator.class_indices)

labels2 = dict((v,k) for k,v in labels.items())

predictions = [labels2[k] for k in predicted_class_indices]

print(predicted_class_indices)

print (labels)

print (predictions)
plt.figure(figsize = (6,6))

from sklearn.metrics import confusion_matrix

cf = confusion_matrix(predicted_class_indices,label)

sns.heatmap(cf,cmap= "Blues", linecolor = 'black' , annot = True, fmt='')
correct = np.nonzero(predicted_class_indices == label)[0]

pred_class = predicted_class_indices.astype(int)
i = 0

for c in correct[:6]:

    plt.subplot(3,2,i+1)

    plt.imshow(validation_generator[0][0][c].reshape(150,150,3))

    plt.title("Predicted Class {},Actual Class {}".format(pred_class.reshape(1,-1)[0][c], label[c]))

    plt.tight_layout()

    i += 1
#### refrences

### https://www.kaggle.com/madz2000/x-ray-detection-using-cnn-100-accuracy