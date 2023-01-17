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

from keras.preprocessing.image import ImageDataGenerator, load_img

from keras.models import Model

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import random

import os

from matplotlib import pyplot

from matplotlib.image import imread

import sys

from matplotlib import pyplot

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

from keras.optimizers import SGD

from keras.preprocessing.image import ImageDataGenerator

from shutil import copyfile

from keras.utils import to_categorical

from sklearn.metrics import classification_report

import seaborn as sns

from keras.applications.vgg16 import VGG16

from rl.agents.dqn import DQNAgent

from rl.policy import EpsGreedyQPolicy

from rl.memory import SequentialMemory

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

print(os.listdir("../input/butterfly-dataset/leedsbutterfly/images/"))
FAST_RUN = False

IMAGE_WIDTH=128

IMAGE_HEIGHT=128

IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)

IMAGE_CHANNELS=3
filenames = os.listdir("../input/butterfly-dataset/leedsbutterfly/images/")

categories = []

for filename in filenames:

    category = filename.split('.')[0]

    categories.append(category[0:3])



df = pd.DataFrame({

    'filename': filenames,

    'category': categories

})



df.head()

df.shape
df['category'].value_counts().plot.bar()
# define location of dataset

folder = '../input/butterfly-dataset/leedsbutterfly/images/'

# plot first few images

for i in range(9):

    # define subplot

    pyplot.subplot(330 + 1 + i)

    # define filename

    filename = folder + '0010009.png'

    # load image pixels

    image = imread(filename)

    # plot raw pixel data

    pyplot.imshow(image)

# show the figure

pyplot.show()
model = VGG16(include_top=False, input_shape=(128, 128, 3))



# mark loaded layers as not trainable

for layer in model.layers:

    layer.trainable = False



# add new classifier layers

flat1 = Flatten()(model.layers[-1].output)

class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)

output = Dense(10, activation='sigmoid')(class1)

# define new model

model = Model(inputs=model.inputs, outputs=output)

# compile model

opt = SGD(lr=0.001, momentum=0.9)

model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

earlystop = EarlyStopping(patience=10)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=2, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)

callbacks = [earlystop, learning_rate_reduction]
df["category"] = df["category"].replace({'001': 'Danaus_plexippus', '002': 'Heliconius_charitonius', '003': 'Heliconius_erato', '004': 'Junonia_coenia', '005': 'Lycaena_phlaeas', '006': 'Nymphalis_antiopa', '007': 'Papilio_cresphontes', '008': 'Pieris_rapae', '009': 'Vanessa_atalanta', '010': 'Vanessa_cardui'}) 

train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)

train_df = train_df.reset_index(drop=True)

validate_df = validate_df.reset_index(drop=True)
train_df['category'].value_counts().plot.bar()

validate_df['category'].value_counts().plot.bar()
total_train = train_df.shape[0]

total_validate = validate_df.shape[0]

batch_size=15
train_datagen = ImageDataGenerator(

    rotation_range=15,

    rescale=1./255,

    shear_range=0.1,

    zoom_range=0.2,

    horizontal_flip=True,

    width_shift_range=0.1,

    height_shift_range=0.1

)



train_generator = train_datagen.flow_from_dataframe(

    train_df, 

    "../input/butterfly-dataset/leedsbutterfly/images/", 

    x_col='filename',

    y_col='category',

    target_size=IMAGE_SIZE,

    class_mode='categorical',

    batch_size=batch_size

)
validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_dataframe(

    validate_df, 

    "../input/butterfly-dataset/leedsbutterfly/images/", 

    x_col='filename',

    y_col='category',

    target_size=IMAGE_SIZE,

    class_mode='categorical',

    batch_size=batch_size

)
example_df = train_df.sample(n=1).reset_index(drop=True)

example_generator = train_datagen.flow_from_dataframe(

    example_df, 

    "../input/butterfly-dataset/leedsbutterfly/images/", 

    x_col='filename',

    y_col='category',

    target_size=IMAGE_SIZE,

    class_mode='categorical'

)
plt.figure(figsize=(12, 12))

for i in range(0, 15):

    plt.subplot(5, 3, i+1)

    for X_batch, Y_batch in example_generator:

        image = X_batch[0]

        plt.imshow(image)

        break

plt.tight_layout()

plt.show()
epochs=3 if FAST_RUN else 50

history = model.fit_generator(

    train_generator, 

    epochs=1,

    validation_data=validation_generator,

    validation_steps=total_validate//batch_size,

    steps_per_epoch=total_train//batch_size,

    callbacks=callbacks

)
model.save_weights("model.h5")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

ax1.plot(history.history['loss'], color='b', label="Training loss")

ax1.plot(history.history['val_loss'], color='r', label="validation loss")

ax1.set_xticks(np.arange(1, epochs, 1))

ax1.set_yticks(np.arange(0, 1, 0.1))



ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")

ax2.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")

ax2.set_xticks(np.arange(1, epochs, 1))



legend = plt.legend(loc='best', shadow=True)

plt.tight_layout()

plt.show()
test_filenames = os.listdir("../input/butterfly-dataset/leedsbutterfly/images/")

test_df = pd.DataFrame({

    'filename': test_filenames

})

nb_samples = test_df.shape[0]
test_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_gen.flow_from_dataframe(

    test_df, 

    "../input/butterfly-dataset/leedsbutterfly/images/", 

    x_col='filename',

    y_col=None,

    class_mode=None,

    target_size=IMAGE_SIZE,

    batch_size=batch_size,

    shuffle=False

)
predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))

test_df['category'] = np.argmax(predict, axis=-1)

label_map = dict((v,k) for k,v in train_generator.class_indices.items())

test_df['category'] = test_df['category'].replace(label_map)

test_df['category'] = test_df['category'].replace({'001': 'Danaus_plexippus', '002': 'Heliconius_charitonius', '003': 'Heliconius_erato', '004': 'Junonia_coenia', '005': 'Lycaena_phlaeas', '006': 'Nymphalis_antiopa', '007': 'Papilio_cresphontes', '008': 'Pieris_rapae', '009': 'Vanessa_atalanta', '010': 'Vanessa_cardui'})

test_df['category'].value_counts().plot.bar()
sample_test = test_df.head(18)

sample_test.head()

plt.figure(figsize=(12, 24))

for index, row in sample_test.iterrows():

    filename = row['filename']

    category = row['category']

    img = load_img("../input/butterfly-dataset/leedsbutterfly/images/"+filename, target_size=IMAGE_SIZE)

    plt.subplot(6, 3, index+1)

    plt.imshow(img)

    plt.xlabel(filename + '(' + "{}".format(category) + ')' )

plt.tight_layout()

plt.show()
# Prediction accuracy on train data

score = model.evaluate_generator(train_generator, verbose=1)

print("Prediction accuracy on train data =", score[1])



# Prediction accuracy on CV data

score = model.evaluate_generator(validation_generator, verbose=1)

print("Prediction accuracy on CV data =", score[1])
def showClassficationReport_Generator(model, valid_generator, STEP_SIZE_VALID):

    # Loop on each generator batch and predict

    y_pred, y_true = [], []

    for i in range(STEP_SIZE_VALID):

        (X,y) = next(valid_generator)

        y_pred.append(model.predict(X))

        y_true.append(y)

    

    # Create a flat list for y_true and y_pred

    y_pred = [subresult for result in y_pred for subresult in result]

    y_true = [subresult for result in y_true for subresult in result]

    

    # Update Truth vector based on argmax

    y_true = np.argmax(y_true, axis=1)

    y_true = np.asarray(y_true).ravel()

    

    # Update Prediction vector based on argmax

    y_pred = np.argmax(y_pred, axis=1)

    y_pred = np.asarray(y_pred).ravel()

    

    print('Classification Report:')

    print(classification_report(y_true, y_pred))



showClassficationReport_Generator(model, validation_generator, batch_size)