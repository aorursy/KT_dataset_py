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
from PIL import Image

import numpy as np

import os

import shutil

#from tqdm import tqdm

from tqdm.notebook import trange, tqdm

import imageio

import matplotlib.pyplot as plt

import pandas as pd

from random import shuffle

import random



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



from tensorflow.keras.applications.vgg16 import VGG16

from tensorflow.keras.applications.inception_v3 import InceptionV3



"""

import keras

import tensorflow as tf

from keras.models import Sequential

from keras.utils import to_categorical

from keras.regularizers import l2

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from keras import optimizers



from keras.layers import Activation, Dense, Dropout, Flatten

from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import Conv2D # to add convolutional layers

from keras.layers.convolutional import MaxPooling2D # to add pooling layers

from keras.layers.advanced_activations import LeakyReLU,ThresholdedReLU



"""

import tensorflow as tf

#tf.logging.set_verbosity(tf.logging.ERROR)

import cv2

def plot_loss_accuracy(history):

    historydf = pd.DataFrame(history.history, index = history.epoch)

    plt.figure(figsize=(8, 6))

    historydf.plot(ylim=(0, max(1, historydf.values.max())))

    loss = history.history['val_loss'][-1]

    acc = history.history['val_accuracy'][-1]

    plt.title('Validation Loss: %.3f, Validation Accuracy: %.3f' % (loss, acc))
train_categories = pd.read_csv('/kaggle/input/train.csv')

train_categories['Image'] =  train_categories['Image'].apply(lambda x: int(x.split('.')[0]))

train_images = train_categories.Image.values.tolist()

location = '/kaggle/input/identify_dance_form/'

trainLabels = {}

f = open("/kaggle/input/train.csv", "r")

dances = f.read()

dances = dances.split('\n')



for i in tqdm(range(len(dances) - 1)):

    dances[i] = dances[i].split(',')

    trainLabels[dances[i][0]] = dances[i][1]

del trainLabels['Image']



test_images = pd.read_csv('/kaggle/input/test.csv')

testImages = test_images.Image.values.tolist()
Dances = trainLabels.values()

trainSet = set(Dances)

itr_set = {}

for i in trainSet:

    itr_set[i] = 0
if not os.path.exists(location + str('/train_labelled')):

    os.makedirs(location + str('/train_labelled'))

    os.makedirs(location + str('/test_labelled'))

    

    # Combine labels and images and move to labelled train folder

    for img in tqdm(os.listdir(location + '/train')):

        if not int(img.split('.')[0]) in train_images:

            continue

        imgName = img.split('.')[0]

        label = trainLabels[str(imgName) + '.jpg']

        itr_set[label] += 1

        path = os.path.join(location + '/train/', img)

        saveName = location + '/train_labelled/' + label + '-' + str(itr_set[label]) + '.jpg'

        image_data = np.array(Image.open(path))

        imageio.imwrite(saveName, image_data)

        

    # Move 20% of labelled data to validation folder for testing

    validation_data = os.listdir(location + '/train_labelled')

    random.Random(22).shuffle(validation_data)

    for i in itr_set:

        itr_set[i] = int(itr_set[i]*0.2)

    for i in tqdm(itr_set):

        for j in validation_data:

            if j.split('-')[0] == i:

                if itr_set[i] > 0:

                    shutil.move(location + '/train_labelled/' + str(j), location + str('/test_labelled'))

                    itr_set[i] -= 1



# Move unlabelled data for classification to test folder

if not os.path.exists(location + str('/test_images')):

    os.makedirs(location + str('/test_images'))

    for image in tqdm(testImages):

        shutil.move(location + '/test/' + str(image), location + str('/test_images'))
def label_img(name):

    word_label = name.split('-')[0]

    if word_label == 'kathak' : return np.array([1,0,0,0,0,0,0,0])

    elif word_label == 'mohiniyattam' : return np.array([0,1,0,0,0,0,0,0])

    elif word_label == 'kuchipudi' : return np.array([0,0,1,0,0,0,0,0])

    elif word_label == 'kathakali' : return np.array([0,0,0,1,0,0,0,0])

    elif word_label == 'bharatanatyam' : return np.array([0,0,0,0,1,0,0,0])

    elif word_label == 'odissi' : return np.array([0,0,0,0,0,1,0,0])

    elif word_label == 'sattriya' : return np.array([0,0,0,0,0,0,1,0])

    elif word_label == 'manipuri' : return np.array([0,0,0,0,0,0,0,1])
def get_size_statistics(DIR):

    heights = []

    widths = []

    for img in tqdm(os.listdir(DIR)): 

        path = os.path.join(DIR, img)

        data = np.array(Image.open(path)) #PIL Image library

        heights.append(data.shape[0])

        widths.append(data.shape[1])

    avg_height = sum(heights) / len(heights)

    avg_width = sum(widths) / len(widths)

    print("Average Height: " + str(avg_height))

    print("Max Height: " + str(max(heights)))

    print("Min Height: " + str(min(heights)))

    print("Average Width: " + str(avg_width))

    print("Max Width: " + str(max(widths)))

    print("Min Width: " + str(min(widths)))
get_size_statistics(location + '/train_labelled')
IMG_SIZE_H = 300

IMG_SIZE_W = 300



def load_training_data(DIR):

    train_data = []

    for img in tqdm(os.listdir(DIR)):

        label = label_img(img)

        path = os.path.join(DIR, img)

        img = cv2.imread(path)

        img = cv2.resize(img, (IMG_SIZE_W, IMG_SIZE_H), interpolation = cv2.INTER_CUBIC)

        train_data.append([np.array(img), label])

    shuffle(train_data)

    return train_data



def load_validation_data(DIR):

    val_data = []

    for img in tqdm(os.listdir(DIR)):

        label = label_img(img)

        path = os.path.join(DIR, img)

        img = cv2.imread(path)

        img = cv2.resize(img, (IMG_SIZE_W, IMG_SIZE_H), interpolation = cv2.INTER_CUBIC)

        val_data.append([np.array(img), label])

    shuffle(val_data)

    return val_data



def load_testing_data(DIR):

    test_data = []

    for Img in tqdm(os.listdir(DIR)):

        path = os.path.join(DIR, Img)

        img = cv2.imread(path)

        img = cv2.resize(img, (IMG_SIZE_W, IMG_SIZE_W), interpolation = cv2.INTER_CUBIC)

        test_data.append([np.array(img), Img])

    return test_data
train_data = load_training_data(location + '/train_labelled')

val_data = load_validation_data(location + '/test_labelled')

X_train = np.array([i[0] for i in train_data])#.reshape(-1, IMG_SIZE_H, IMG_SIZE_W, 3)

X_train = X_train / 255 # normalize training data

y_train = np.array([i[1] for i in train_data])

#y_train = y_train / 255 # normalize training data

X_test = np.array([i[0] for i in val_data])#.reshape(-1, IMG_SIZE_H, IMG_SIZE_W, 3)

X_test = X_test / 255 # normalize test data

y_test = np.array([i[1] for i in val_data])

#y_test = y_test / 255 # normalize training data
plt.imshow(train_data[120][0], #cmap = 'gist_gray'

          )

plt.show()
vggmodel = VGG16(weights = 'imagenet', include_top = False, input_shape = (IMG_SIZE_H, IMG_SIZE_W, 3), pooling = 'max')
 # Print the model summary

vggmodel.summary()
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten, GlobalAveragePooling2D, GlobalMaxPooling2D, BatchNormalization

from tensorflow.keras import optimizers

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.models import Sequential



ADAMAX = optimizers.Adamax(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999)



vggmodel.trainable = False

model = tf.keras.Sequential([vggmodel, 

                    Dense(1024, activation = 'relu'), Dropout(0.15),

                    Dense(256, activation = 'relu'), Dropout(0.15),

                    Dense(8, activation = 'softmax'),

                   ])

model.compile(optimizer = ADAMAX, loss = 'categorical_crossentropy',  metrics = ['accuracy'])
# Use annelar to gradually decrese the learning rate to improve generalization



reduce_lr = ReduceLROnPlateau(monitor = 'loss', patience = 5, verbose = 1, factor = 0.2, min_lr = 0.00002,

                                            mode = 'auto', cooldown = 1)
gen = ImageDataGenerator(rotation_range = 40, width_shift_range = 0.1,

                         height_shift_range = 0.1, zoom_range = 0.20, horizontal_flip = True,

                         vertical_flip = False, featurewise_center = False,

                         samplewise_center = False, featurewise_std_normalization = False,

                         samplewise_std_normalization = False)

test_gen = ImageDataGenerator()



# Create batches to  train models faster

train_generator = gen.flow(X_train, y_train, batch_size = 16)

test_generator = test_gen.flow(X_test, y_test, batch_size = 16)
epochs = 100



history = model.fit_generator(train_generator, steps_per_epoch = 16, epochs = epochs, 

                              validation_data = test_generator, validation_steps = 16, verbose = 1,

                              callbacks=[reduce_lr])



# evaluate the model



scores = model.evaluate(X_test, y_test, verbose = 1)

print("Accuracy: {} \n Error: {}".format(scores[1], 100 - scores[1]*100))
plot_loss_accuracy(history)
test_data = load_testing_data(location + '/test_images')

test = np.array([i[0] for i in test_data])#.reshape(-1, IMG_SIZE_H, IMG_SIZE_W, 1)

test_labels = np.array([i[1] for i in test_data])

test = test / 255 # normalize test data

Y_pred = np.round(model.predict(test), 0)

Y_pred = np.argmax(Y_pred, axis = 1)

Y_pred = pd.Series(Y_pred, name = "label")
def dance_label(word_label):

    if word_label == 0: return 'kathak'

    elif word_label == 1: return 'mohiniyattam'

    elif word_label == 2: return 'kuchipudi'

    elif word_label == 3: return 'kathakali'

    elif word_label == 4: return 'bharatanatyam'

    elif word_label == 5: return 'odissi'

    elif word_label == 6: return 'sattriya'

    elif word_label == 7: return 'manipuri'
submission_df = pd.DataFrame({

                  "Image": pd.Series(test_labels),

                  "target": pd.Series(Y_pred)})

submission_df['target'] = submission_df['target'].apply(lambda x: dance_label(x))

submission_df.to_csv('submission_vgg_v1.csv', index = False)
LR = 0.002

model_name = 'classify_dances-{}-{}.model'.format(LR, 'vgg_v1')

model.save(model_name)
inception_model = InceptionV3(weights = 'imagenet', include_top = False, input_shape = (IMG_SIZE_H, IMG_SIZE_W, 3))
inception_model.trainable = False
model = tf.keras.Sequential([inception_model, GlobalAveragePooling2D(),

                    Dense(1024, activation = 'relu'), Dropout(0.10),

                    Dense(512, activation = 'relu'), Dropout(0.10),

                    Dense(8, activation = 'softmax'),

                   ])

model.compile(optimizer = ADAMAX, loss = 'categorical_crossentropy',  metrics = ['accuracy'])

model.summary()
epochs = 100



history = model.fit_generator(train_generator, steps_per_epoch = 16, epochs = epochs, 

                              validation_data = test_generator, validation_steps = 16, verbose = 1,

                              callbacks=[reduce_lr])



# evaluate the model



scores = model.evaluate(X_test, y_test, verbose = 1)

print("Accuracy: {} \n Error: {}".format(scores[1], 100 - scores[1]*100))
Y_pred = np.round(model.predict(test), 0)

Y_pred = np.argmax(Y_pred, axis = 1)

Y_pred = pd.Series(Y_pred, name = "label")
submission_df = pd.DataFrame({

                  "Image": pd.Series(test_labels),

                  "target": pd.Series(Y_pred)})

submission_df['target'] = submission_df['target'].apply(lambda x: dance_label(x))

submission_df.to_csv('submission_IV3_v1.csv', index = False)
LR = 0.002

model_name = 'classify_dances-{}-{}.model'.format(LR, 'inception_v1')

model.save(model_name)