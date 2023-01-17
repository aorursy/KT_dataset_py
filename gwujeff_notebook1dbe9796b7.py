import numpy as np

import pandas as pd



%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib.cm as cm



import tensorflow as tf



# settings

LEARNING_RATE = 1e-4

# set to 20000 on local environment to get 0.99 accuracy

TRAINING_ITERATIONS = 2500        

    

DROPOUT = 0.5

BATCH_SIZE = 50



# set to 0 to train on all available data

VALIDATION_SIZE = 2000



# image number to output

IMAGE_TO_DISPLAY = 10




# read training data from CSV file 

data = pd.read_csv('../input/train.csv')



print('data({0[0]},{0[1]})'.format(data.shape))

print (data.head())







images = data.iloc[:,1:].values

images = images.astype(np.float)



# convert from [0:255] => [0.0:1.0]

images = np.multiply(images, 1.0 / 255.0)



print('images({0[0]},{0[1]})'.format(images.shape))



image_size = images.shape[1]

print ('image_size => {0}'.format(image_size))



# in this case all images are square

image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)



print ('image_width => {0}\nimage_height => {1}'.format(image_width,image_height))




# display image

def display(img):

    

    # (784) => (28,28)

    one_image = img.reshape(image_width,image_height)

    

    plt.axis('off')

    plt.imshow(one_image, cmap=cm.binary)



# output image     

display(images[IMAGE_TO_DISPLAY])



labels_flat = data[[0]].values.ravel()



print('labels_flat({0})'.format(len(labels_flat)))

print ('labels_flat[{0}] => {1}'.format(IMAGE_TO_DISPLAY,labels_flat[IMAGE_TO_DISPLAY]))







import numpy as np # linear algebra

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix







from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import LearningRateScheduler







train_file = "../input/train.csv"

test_file = "../input/test.csv"

output_file = "submission.csv"



raw_data = np.loadtxt(train_file, skiprows=1, dtype='int', delimiter=',')

x_train, x_val, y_train, y_val = train_test_split(

    raw_data[:,1:], raw_data[:,0], test_size=0.1)
raw_data = np.loadtxt(train_file, skiprows=1, dtype='int', delimiter=',')

x_train, x_val, y_train, y_val = train_test_split(

    raw_data[:,1:], raw_data[:,0], test_size=0.1)

fig, ax = plt.subplots(2, 1, figsize=(12,6))

ax[0].plot(x_train[0])

ax[0].set_title('784x1 data')

ax[1].imshow(x_train[0].reshape(28,28), cmap='gray')

ax[1].set_title('28x28 data')
x_train = x_train.reshape(-1, 28, 28, 1)

x_val = x_val.reshape(-1, 28, 28, 1)
x_train = x_train.astype("float32")/255.

x_val = x_val.astype("float32")/255.
y_train = to_categorical(y_train)

y_val = to_categorical(y_val)

#example:

print(y_train[0])
model = Sequential()



model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu',

                 input_shape = (28, 28, 1)))

model.add(BatchNormalization())

model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu'))

model.add(BatchNormalization())

#model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu'))

#model.add(BatchNormalization())

model.add(MaxPool2D(strides=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))

model.add(BatchNormalization())

#model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))

#model.add(BatchNormalization())

model.add(MaxPool2D(strides=(2,2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))
datagen = ImageDataGenerator(zoom_range = 0.1,

                            height_shift_range = 0.1,

                            width_shift_range = 0.1,

                            rotation_range = 10)
model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=1e-4), metrics=["accuracy"])
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)
hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size=16),

                           steps_per_epoch=500,

                           epochs=20, #Increase this when not on Kaggle kernel

                           verbose=2,  #1 for ETA, 0 for silent

                           validation_data=(x_val[:400,:], y_val[:400,:]), #For speed

                           callbacks=[annealer])
final_loss, final_acc = model.evaluate(x_val, y_val, verbose=0)

print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss, final_acc))
plt.plot(hist.history['loss'], color='b')

plt.plot(hist.history['val_loss'], color='r')

plt.show()

plt.plot(hist.history['acc'], color='b')

plt.plot(hist.history['val_acc'], color='r')

plt.show()
y_hat = model.predict(x_val)

y_pred = np.argmax(y_hat, axis=1)

y_true = np.argmax(y_val, axis=1)

cm = confusion_matrix(y_true, y_pred)

print(cm)
mnist_testset = np.loadtxt(test_file, skiprows=1, dtype='int', delimiter=',')

x_test = mnist_testset.astype("float32")

x_test = x_test.reshape(-1, 28, 28, 1)/255.
y_hat = model.predict(x_test, batch_size=64)
y_pred = np.argmax(y_hat,axis=1)
with open(output_file, 'w') as f :

    f.write('ImageId,Label\n')

    for i in range(len(y_pred)) :

        f.write("".join([str(i+1),',',str(y_pred[i]),'\n']))