%matplotlib notebook

import numpy as np # linear algebra
import matplotlib.pyplot as plt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob, os 

from skimage import io, transform
from sklearn.model_selection import train_test_split


from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Conv2D, Dropout, MaxPool2D
path="../input/pollendataset/PollenDataset/images/"
imlist= glob.glob(os.path.join(path, '*.jpg'))

def dataset(file_list,size=(300,180)):
    # Returns the image and the label 
    # The label is obtained from the first character in the filename (P: Carries pollen, NP: Does not carry pollen)
    data = []
    for i, file in enumerate(file_list):
        image = io.imread(file)
        image = transform.resize(image, size, mode='constant')
        data.append(image)

    labels = [1 if f.split("/")[-1][0] == 'P' else 0 for f in file_list]

    return np.array(data), np.array(labels)
# Load the dataset (may take a few seconds)
images, labels = dataset(imlist)
image_dim = images.shape[1:]
# X has the following structure: X[imageid, y,x,channel]
print("Dimensions")
print('Image: ',images.shape)  # data
print('y: ',labels.shape)  # target
print("")
print("Class Distribution")
print('Class 0: ',sum(labels==0))
print('Class 1: ',sum(labels==1))
print('Total  : ',len(labels))

fig, axes = plt.subplots(1,2)
k=1
plt.sca(axes[0])
plt.imshow(images[k])
plt.title('img {} - class {}'.format(k, labels[k]))

k=400
plt.sca(axes[1])
plt.imshow(images[k])
plt.title('img {} - class {}'.format(k, labels[k]));
# Obtain the mean and standard deviation of the images in each channel
mean_image = np.mean(images, axis = 0)
std_image = np.std(images, axis = 0)

x = (images[:] - mean_image)/std_image

print("Check the mean and std values")
print(mean_image.shape)
print(std_image.shape)

fig, axes = plt.subplots(1,2)
plt.sca(axes[0])
plt.imshow(mean_image)
plt.sca(axes[1])
plt.imshow(std_image)
test_sample_size = 100 # I want to have at least 100 images for the test predictions.
validation_sample_size = 124 # I want to split the train/validation data in 80/20 proportion

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = test_sample_size)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size = validation_sample_size)

print('Train samples: ', x_train.shape[0])
print('Validation samples: ', x_validation.shape[0])
print('Test samples: ', x_test.shape[0])
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation = "relu", data_format = "channels_last" ,input_shape = image_dim))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(64, kernel_size=(2,2), activation = "relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
model.fit(x_train, y_train, batch_size = 16, epochs= 10, validation_data = (x_validation, y_validation))
model.evaluate(x_test,y_test, batch_size=16)
from keras.preprocessing import image

image_gen = image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation = "relu", data_format = "channels_last" ,input_shape = image_dim))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(64, kernel_size=(2,2), activation = "relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])

model.fit_generator( image_gen.flow(x_train, y_train, batch_size=16), epochs = 10, steps_per_epoch=40, validation_data = image_gen.flow(x_validation, y_validation, batch_size = 12))
model.evaluate_generator(image_gen.flow(x_test, y_test, batch_size=10))
model.evaluate(x_test,y_test, batch_size=10)
