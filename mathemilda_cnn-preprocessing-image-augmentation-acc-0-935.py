import time

from time import perf_counter as timer

start = timer()
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.pyplot import imshow as imshow

import seaborn as sns

import warnings

from sklearn.preprocessing import LabelBinarizer

from skimage import exposure
train = pd.read_csv('../input/fashion-mnist_train.csv')

test = pd.read_csv('../input/fashion-mnist_test.csv')

train.head(8)
train.shape
labels = train['label'].values

labels[0:10]
unique_val = np.array(labels)

np.unique(unique_val)
plt.figure(figsize = (18,8))

sns.countplot(x =labels)
label_binrizer = LabelBinarizer()

labels = label_binrizer.fit_transform(labels)

labels
train.drop('label', axis = 1, inplace = True)
images = train.values

print(images.dtype, np.round(images.min(), 4), np.round(images.max(), 4), images.shape)
def plot_10(imgs):

    plt.style.use('grayscale')

    fig, axs = plt.subplots(2, 5, figsize=(15, 6), sharey=True)

    for i in range(2): 

        for j in range(5):

            axs[i,j].imshow((225-images[5*i+j]).reshape(28,28))

    fig.suptitle('Grayscale images:\n a Pullover,   an Ankle boot,   a Shirt,   a T-shirt/top,   a Dress,\n' +

            'a Coat,   a Coat,    a Sandal,    a Coat,   a Bag')



plot_10(images)
images = images/255
new_images = np.zeros((10, 784))

for i in range(10):

        new_images[i,:]= exposure.rescale_intensity(images[i, :], in_range=(0.045, 0.955))

        

plot_10(new_images)
warnings.simplefilter('once')

for i in range(10):

        new_images[i,:]= exposure.equalize_hist(images[i, :].reshape(28, 28), nbins=100).flatten() #, clip_limit=0.03, nbins=200



plot_10(new_images)
new_images = np.zeros((10, 784))

for i in range(10):

        new_images[i,:]= exposure.adjust_gamma(images[i, :])

        

plot_10(new_images)
def my_prep(x):

    x = exposure.adjust_gamma(x)

    return x



images = np.apply_along_axis(my_prep, 1, images)

images.shape
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(images, labels, stratify = labels, test_size = 0.2)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
import keras

from keras.models import Sequential

from keras.layers import Dense, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dropout

from keras.preprocessing.image import ImageDataGenerator
num_classes = 10

batch_size = 500

epochs = 100
img_rows, img_cols = 28, 28



model = Sequential()

model.add(Conv2D(160, kernel_size=(6, 6),

                 padding = "same",

                 activation='relu',

                 kernel_initializer='he_normal',

                 input_shape=(img_rows, img_cols ,1)))

model.add(BatchNormalization())

model.add(MaxPooling2D((2, 2)))

model.add(Dropout(0.5))

model.add(Conv2D(128, (4, 4), padding = "same", activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.5))

model.add(Conv2D(128, (3,3), padding = "same", activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.4))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adamax(),#keras.optimizers.Nadam()

              metrics=['accuracy'])
train_datagen = ImageDataGenerator(shear_range = 0.1,

                                   zoom_range = [.95, 1.0],

                                   rotation_range = 10,

                                   horizontal_flip = True,

                                   fill_mode = 'constant', cval = 0,

                                   brightness_range = [.6, 1],

                                   width_shift_range = [ -2, -1, 0, +1, +2],

                                   height_shift_range = [-1, 0, +1])

test_datagen = ImageDataGenerator()
history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=epochs, verbose=2, batch_size=batch_size)
plt.style.use('tableau-colorblind10')

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.ylim(0.85, 0.98)

plt.title("Accuracy")

plt.xlabel('epoch')

plt.ylabel('accuracy')

plt.legend(['train','test'])

plt.show()
fig = plt.hist(history.history['val_acc'][60:], bins=8)

# when you want to run this code quickly with fewer epochs

#plt.hist(history.history['val_acc'], bins=8)

fig = plt.figure()

fig.savefig('plot.png')
test_labels = test.iloc[:, 0]

test.drop('label', axis = 1, inplace = True)

test_images = test.values/255

test_images = np.apply_along_axis(my_prep, 1, test_images)

labels_as_array = label_binrizer.fit_transform(test_labels)

test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

test_images.shape
y_pred = model.predict(test_images).round()

from sklearn.metrics import accuracy_score

accuracy_score(labels_as_array, y_pred)
from sklearn.metrics import confusion_matrix

class_names = ['T-shirt/top',  'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

cm = pd.DataFrame(confusion_matrix(test_labels.values, label_binrizer.inverse_transform(y_pred)))

cm.assign(Classes = class_names)
end = timer()

elapsed_time = time.gmtime(end - start)

print("Elapsed time:")

print("{0} minutes {1} seconds.".format(elapsed_time.tm_min, elapsed_time.tm_sec))