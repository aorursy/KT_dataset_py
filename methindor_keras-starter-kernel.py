# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from IPython.core.debugger import set_trace

import cv2

from tqdm import tqdm

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator

from keras.applications.inception_v3 import InceptionV3

from keras.layers.core import Activation

from keras.layers.core import Flatten

from keras.layers.core import Dense

from keras.layers import Dropout

from keras.models import  Model

from keras import optimizers

from os import makedirs

from os.path import join, exists, expanduser

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
!ls ../input/dog-breed-identification

cache_dir = expanduser(join('~', '.keras'))

if not exists(cache_dir):

    makedirs(cache_dir)

models_dir = join(cache_dir, 'models')

if not exists(models_dir):

    makedirs(models_dir)
!ls ../input/inceptionv3
!cp  ../input/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 ~/.keras/models/

!ls ~/.keras/models/
df_train = pd.read_csv("../input/dog-breed-identification/labels.csv")

df_train.head(10)
ax=pd.value_counts(df_train['breed'],ascending=True).plot(kind='barh',

                                                       fontsize="40",

                                                       title="Class Distribution",

                                                       figsize=(50,100))

ax.set(xlabel="Images per class", ylabel="Classes")

ax.xaxis.label.set_size(40)

ax.yaxis.label.set_size(40)

ax.title.set_size(60)



# take a subset of the trainigsdata with die ten most frequently classes

NUM_CLASSES=10

print("Dataset shape before: {0}".format(df_train.shape))

selected_breed_list = list(df_train.groupby('breed').count().sort_values(by='id', ascending=False).head(NUM_CLASSES).index)

df_sub_train=df_train[df_train['breed'].isin(selected_breed_list)]

print("Dataset shape after: {0}".format(df_sub_train.shape))



# plot the distribution of this subset

ax=pd.value_counts(df_sub_train['breed'],ascending=True).plot(kind='barh',

                                                       fontsize="40",

                                                       title="Class Distribution",

                                                       figsize=(50,20))

ax.set(xlabel="Images per class", ylabel="Classes")

ax.xaxis.label.set_size(40)

ax.yaxis.label.set_size(40)

ax.title.set_size(60)

IMG_WIDTH=250

IMG_Height=250

images=[]

classes=[]

targets_series = pd.Series(df_sub_train['breed'])

one_hot = pd.get_dummies(targets_series, sparse=True)

one_hot_labels = np.asarray(one_hot)

i = 0

#load training images

for f, breed in tqdm(df_sub_train.values):

    img = cv2.imread('../input/dog-breed-identification/train/{}.jpg'.format(f))

    images.append(cv2.resize(img, (IMG_WIDTH, IMG_Height)))   

    label = one_hot_labels[i]

    classes.append(label)

    

    
classes_raw = np.array(classes, np.uint8)

images_raw = np.array(images, np.float32)



x_train, x_valid, y_train, y_valid = train_test_split(images_raw, classes_raw, test_size=0.2, random_state=1)



print("Trainigsdata shape : {0}".format(x_train.shape))

print("Trainingslabel shape : {0}".format(y_train.shape))

print("Validdata shape : {0}".format(x_valid.shape))

print("Validlabel shape : {0}".format(y_valid.shape))

train_datagen = ImageDataGenerator(

        rescale=1. / 255,   

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True)

       



valid_datagen = ImageDataGenerator(rescale=1. / 255)



train_generator = train_datagen.flow(x_train, y_train, batch_size=32)

valid_generator = valid_datagen.flow(x_valid, y_valid, batch_size=32)
base_model=InceptionV3(include_top=False, weights='imagenet', 

                        input_shape=(IMG_WIDTH, IMG_Height, 3),

                        classes=NUM_CLASSES)

# Adding custom Layers

model = base_model.output

model = Flatten()(model)

model = Dense(1024, activation="relu")(model)

model = Dropout(0.5)(model)

model = Dense(1024, activation="relu")(model)

predictions = Dense(NUM_CLASSES, activation="softmax")(model)

# creating the final model

model_final = Model(input=base_model.input, output=predictions)

# compile the model

model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),

                           metrics=["accuracy"])

#print the model

model_final.summary()
# Train the model

history=model_final.fit_generator(

        train_generator,

        samples_per_epoch=912,

        epochs=100,

        validation_data=valid_generator,

        nb_val_samples=229)
# summarize history for accuracy

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()