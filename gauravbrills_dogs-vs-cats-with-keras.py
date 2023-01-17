# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd 

import tensorflow as tf

print("Tensorflow version " + tf.__version__)

AUTO = tf.data.experimental.AUTOTUNE

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import tensorflow as tf

import shutil

from shutil import copyfile

import random

import os

import zipfile

print("What we've Got",os.listdir("../input/dogs-vs-cats"))

input = "../input/dogs-vs-cats/train"



### ADDING TPU's

# detect and init the TPU

tpu="TPU v3-8"

# Detect hardware, return appropriate distribution strategy

try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.



print("REPLICAS: ", strategy.num_replicas_in_sync)
## Unzip All

Dataset = "train"

with zipfile.ZipFile("../input/dogs-vs-cats/"+Dataset+".zip","r") as z:

    z.extractall(".")
def mk_categories(df,dtype):

    os.makedirs(dtype+"/dogs", exist_ok=True)

    os.makedirs(dtype+"/cats", exist_ok=True)

    for index, row in df.iterrows():

        filename= row['filename']

        category = row['filename'].split('.')[0]

        if category == 'dog':

            copyfile("train/"+filename, dtype+"/dogs/"+filename)

        else:

            copyfile("train/"+filename, dtype+"/cats/"+filename)



filenames = os.listdir("train")

categories = []

for filename in filenames:

    category = filename.split('.')[0]

    if category == 'dog':

        categories.append(1)

    else:

        categories.append(0)



df = pd.DataFrame({

    'filename': filenames,

    'category': categories

})

## Make some test data out of training data

train_df, test_df = train_test_split(df, test_size=0.20, random_state=42)

train_df = train_df.reset_index(drop=True)

test_df = test_df.reset_index(drop=True) 

mk_categories(train_df,"training")

mk_categories(test_df,"test")
train_df['category'].value_counts().plot.bar()
test_df['category'].value_counts().plot.bar()
sample = random.choice(filenames)

image = load_img("train/"+sample)

plt.imshow(image)
def plotSnap(who,folder="train/"): 

    # plot first few images

    for i in range(9):

         # define subplot

        plt.subplot(330 + 1 + i)

        # define filename

        filename = folder + who +'.' + str(i) + '.jpg'

        # load image pixels

        image = load_img(filename)

        # plot raw pixel data

        plt.imshow(image)
plotSnap("dog")

    

# show the figure

plt.show()
plotSnap("cat")

    

# show the figure

plt.show()
def plotPreview(who,folder="preview/"): 

    i=1

    # plot first few images

    for file in os.listdir(folder):

         # define subplot

        plt.subplot(8,8,i)

        i=i+1

        filename = folder + file

        image = load_img(filename)

        plt.imshow(image)

        if i>20:

            break



if not os.path.exists("preview"):

    os.mkdir("preview")

    

datagen = ImageDataGenerator(

        rotation_range=40,

        width_shift_range=0.2,

        height_shift_range=0.2,

        rescale=1./255,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True,

        fill_mode='nearest')

img = load_img('train/cat.0.jpg')  # this is a PIL image

x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)

x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)



# the .flow() command below generates batches of randomly transformed images

# and saves the results to the `preview/` directory

i = 0

for batch in datagen.flow(x, batch_size=1,save_to_dir='preview', save_prefix='cat', save_format='jpeg'):

    i += 1

    if i > 20:

        break  # otherwise the generator would loop indefinitely

        

plotPreview("cat")

plt.show()
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.optimizers import SGD

from keras.models import Model

from keras.applications.vgg16 import VGG16

from keras.layers import Activation, Dropout, Flatten, Dense

# create convnet model

def build_model(type="covnet"):

    with strategy.scope():

        if type == "covnet":

            model = Sequential()

            model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))

            model.add(MaxPooling2D((2, 2)))

            model.add(Dropout(0.2))

            model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

            model.add(MaxPooling2D((2, 2)))

            model.add(Dropout(0.2))

            model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

            model.add(MaxPooling2D((2, 2)))

            model.add(Dropout(0.2))

            model.add(Flatten())

            model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))

            model.add(Dropout(0.5))

            model.add(Dense(1, activation='sigmoid'))

            opt = SGD(lr=0.001, momentum=0.9)

            model.compile(loss='binary_crossentropy',

                          optimizer='rmsprop',

                          metrics=['accuracy'])

        if type == "VGG16":

            model = VGG16(include_top=False, input_shape=(224, 224, 3))

            for layer in model.layers:

                layer.trainable = False

            flat1 = Flatten()(model.layers[-1].output)

            class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)

            output = Dense(1, activation='sigmoid')(class1)

            model = Model(inputs=model.inputs, outputs=output)

            opt = SGD(lr=0.001, momentum=0.9)

            model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    return model





import tensorflow as tf

import keras.backend.tensorflow_backend as tfback

print("tf.__version__ is", tf.__version__)

print("tf.keras.__version__ is:", tf.keras.__version__)



def _get_available_gpus():

    """Get a list of available gpu devices (formatted as strings).



    # Returns

        A list of available GPU devices.

    """

    #global _LOCAL_DEVICES

    if tfback._LOCAL_DEVICES is None:

        devices = tf.config.list_logical_devices()

        tfback._LOCAL_DEVICES = [x.name for x in devices]

    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]



tfback._get_available_gpus = _get_available_gpus
## Unzip All test

Dataset = "test1"

with zipfile.ZipFile("../input/dogs-vs-cats/"+Dataset+".zip","r") as z:

    z.extractall(".")

def plot_summaries(model):

    plt.subplot(211)

    plt.title('Cross Entropy Loss')

    plt.plot(model.history['loss'], color='blue', label='train')

    plt.plot(model.history['val_loss'], color='orange', label='test')

    # plot accuracy

    plt.subplot(212)

    plt.title('Classification Accuracy')

    plt.plot(model.history['accuracy'], color='blue', label='train')

    plt.plot(model.history['val_accuracy'], color='orange', label='test')

    plt.show()

 

# Run Train and eval Model 

def run_train_eval(savepoint="savepoint.h5",type="covnet"):

    batch_size=16 * strategy.num_replicas_in_sync

    if type=="covnet":

        size = 200

    else:

        size = 224

    # this is the augmentation configuration we will use for training

    train_datagen = ImageDataGenerator(

            rescale=1./255,

            shear_range=0.2,

            zoom_range=0.2,

            horizontal_flip=True)



    # this is the augmentation configuration we will use for testing:

    # only rescaling

    test_datagen = ImageDataGenerator(rescale=1./255)



    # this is a generator that will read pictures found in

    # subfolers of 'data/train', and indefinitely generate

    # batches of augmented image data

    train_generator = train_datagen.flow_from_directory(

            '/kaggle/working/training/',  # this is the target directory

            target_size=(size, size),  # all images will be resized to 150x150

            batch_size=batch_size,

            class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels



    # this is a similar generator, for validation data

    validation_generator = test_datagen.flow_from_directory(

            '/kaggle/working/test/',

            target_size=(size, size),

            batch_size=batch_size,

            class_mode='binary')

    model = build_model(type=type)

    fit = model.fit_generator(

            train_generator,

            steps_per_epoch=2000// batch_size,

            epochs=50,

            validation_data=validation_generator,

            validation_steps=800 // batch_size

    )

    model.save_weights(savepoint)

    # evaluate model

    _, acc = model.evaluate_generator(validation_generator, steps=len(validation_generator), verbose=0)

    display('> %.3f' % (acc * 100.0))

    plot_summaries(fit)

    return (model,fit)
#model = run_train_eval()
modelVGG = run_train_eval(savepoint="vgg.h5",type="VGG16")
from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

from keras.models import load_model

 

# load and prepare the image

def load_image(filename,imgSize =224):

    img = load_img(filename, target_size=(imgSize, imgSize))

    img = img_to_array(img)

    img = img.reshape(1, imgSize, imgSize, 3)

    img = img.astype('float32')

    img = img - [123.68, 116.779, 103.939]

    return img

 

# load an image and predict the class

def predict_img(img):

    # load the image

    img = load_image(img)

    # load model

    model = build_model(type="VGG16")

    model.load_weights('vgg.h5')

    # predict the class

    result = model.predict(img)

    return result[0]



def predict_img(img):

    # load the image

    img = load_image(img)

    # load model

    model = build_model(type="VGG16")

    model.load_weights('vgg.h5')

    # predict the class

    result = model.predict(img)

    return result[0]
load_img("test1/5853.jpg") 
val = predict_img("test1/5853.jpg") 

print(val[0])

if val[0]== 1:

    print("Found A Dog !")

else:

    print("Found A Cat ! ")
filenames = os.listdir("test1") 

ids =[]

labels =[]

i =1 

plt.figure(figsize=(12, 24))

model = build_model(type="VGG16")

model.load_weights('vgg.h5')

for filename in filenames:

    id = filename.split('.')[0]

    ids.append(id)

    img="test1/"+filename

    truncImg = load_image(img)

    lbl = model.predict(truncImg) 

    labels.append(int(round(lbl[0][0])))

    # Plot few samples

    if i <= 5:

        plt.subplot(6, 3, i+1)

        plt.imshow(load_img("test1/"+filename))

        plt.xlabel('%s > %f' % (id,int(round(lbl[0][0]))))

        plt.show()

        #display('%s > %f' % (id,lbl))

    i = i+1



  
## remode dirs

def deldir(dirPath):

    try:

        print("Removing Directory",dirPath) 

        shutil.rmtree(dirPath)

    except:

        print('Error while deleting directory')

deldir("train")

deldir("training")

deldir("test1")

deldir("test")

deldir("preview")



## create submission file

submission_df = pd.DataFrame({

    'id': ids,

    'label': labels

})

display(submission_df.head())

submission_df.to_csv('submission.csv', index=False)  