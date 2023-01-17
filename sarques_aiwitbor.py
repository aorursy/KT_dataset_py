# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

base_add = os.path.join('..', 'input')

# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt

from glob import glob

%matplotlib inline

from PIL import Image

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import label_binarize

from sklearn.metrics import confusion_matrix



import keras

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras import backend as K

import itertools

from keras.optimizers import Adam

from keras.layers.normalization import BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau
image_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(base_add, '*', '*.jpg'))}

lesion_type_dict = {'nv': 'Melanocytic nevi',

    'mel': 'Melanoma',

    'bkl': 'Benign keratosis-like lesions ',

    'bcc': 'Basal cell carcinoma',

    'akiec': 'Actinic keratoses',

    'vasc': 'Vascular lesions',

    'df': 'Dermatofibroma'}

df = pd.read_csv(os.path.join(base_add, 'HAM10000_metadata.csv'))

print(image_path_dict)
df.sample(10)

df.info()
df['cell_type'] = df['dx'].map(lesion_type_dict.get)

df['path'] = df['image_id'].map(image_path_dict.get)

df['dx_code'] = pd.Categorical(df['dx']).codes

df.sample(3)
df['age'].fillna(df['age'].mean(), inplace = True)

df.info()
df['cell_type'].value_counts().plot(kind = 'bar')
df['age'].hist(bins = 20)
df['image'] = df['path'].map(lambda x: np.asarray(Image.open(x).resize((100, 75))))

df
df['image'].map(lambda x: x.shape).value_counts()
feats = df.drop(['dx_code'], axis = 1)

target = df['dx_code']
x_train, x_test, y_train, y_test = train_test_split(feats, target, test_size = 0.2)
# Normalization

x_train = np.asarray(x_train['image'].tolist())

x_test = np.asarray(x_test['image'].tolist())



x_train_mean = np.mean(x_train)

x_test_mean = np.mean(x_test)



x_train_std = np.std(x_train)

x_test_std = np.std(x_test)



x_train = (x_train - x_train_mean)/x_train_std

x_test = (x_test - x_test_mean)/x_test_std
y_train.value_counts()
# Label Encoding

# y_train = to_categorical(y_train, num_classes = 7)

# y_test = to_categorical(y_test, num_classes = 7)

print(y_train[13])

x_train.shape
# Reshape images in 3 dimensions

x_train = x_train.reshape(x_train.shape[0], *(75, 100, 3))

x_test = x_test.reshape(x_test.shape[0], *(75, 100, 3))

x_train.shape
# Set CNN model

# Our system of layers => [[Conv2D -> relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out

input_shape = (75, 100, 3)

num_classes = 7



model = Sequential()

model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu', padding = 'Same', input_shape = input_shape))

model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu', padding = 'Same'))

model.add(MaxPool2D(pool_size = (2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu', padding = 'Same'))

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu', padding = 'Same'))

model.add(MaxPool2D(pool_size = (2, 2)))

model.add(Dropout(0.40))



model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))

model.summary()
# Define the optimizer

optimizer = Adam(lr = .001, beta_1 = .9, beta_2 = .999, epsilon = None, decay = .0, amsgrad = False)
# Compile the model

model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
# Set learning rate annealer

learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_acc', patience = 3, verbose = 1, 

                                           factor = 0.5, min_lr = 0.00001)
# Data Augmentation

datagen = ImageDataGenerator(featurewise_center = False, samplewise_center = False,

                            featurewise_std_normalization = False, samplewise_std_normalization = False, 

                            zca_whitening = False, rotation_range = 10, zoom_range = 0.1, 

                            width_shift_range = 0.1, height_shift_range = 0.1, horizontal_flip = False, 

                            vertical_flip = False)

datagen.fit(x_train)
# Fit the model

# epochs = 50 # Accuracy of 0.7682

epochs = 25

batch_size = 10

history = model.fit_generator(datagen.flow(x_train, y_train, batch_size = batch_size), 

                             epochs = epochs, verbose = 1, steps_per_epoch = x_train.shape[0] // batch_size,

                             callbacks = [learning_rate_reduction])
from matplotlib import figure
def plot_weight_image(layer, x, y):

    weights = model.layers[layer].get_weights()

    fig = plt.figure()

    for j in range(len(weights[0])):

        ax = fig.add_subplot(y, x, j+1)

        ax.matshow(weights[0][j][0], cmap = plt.cm.binary)

        plt.xticks(np.array([]))

        plt.yticks(np.array([]))

    plt.tight_layout()

    return plt
a = model.layers[0].get_weights()

print(a[0][:, :, :, 0])

print("*"*40)

print(a[0][:, :, :, 1])

a[0].shape
np.random.seed(12345)

def save_image(layer):

    a = model.layers[layer].get_weights()

    for i in range(100):

        for j in range(100):

            try:

                grid = a[0][:, :, i, j]

                img = plt.imshow(grid, interpolation = 'spline16', cmap = 'plasma')

                plt.axis('off')

                plt.tight_layout()

                plt.savefig("test{} {}-{}.jpg".format(layer, i , j))

            except:

                break
# We are creating these images for weights of only 1st covolutional layer now, 

# all of them would take a lot more time.



layers = [0]

for layer in layers:

    save_image(layer)