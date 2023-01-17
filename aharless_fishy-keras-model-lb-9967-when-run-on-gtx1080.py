# Parameters



RUNNING_AS_KERAS_KERNEL = True



RANDOM_SEED = 0



if RUNNING_AS_KERAS_KERNEL:

   LEARNING_RATE = 1e-2

   N_EPOCHS = 1

   OUTPUT_FILE = "mnistGrenholm-nocv-1ep01seed" + str(RANDOM_SEED) + ".csv"

else:

   LEARNING_RATE = 2e-4

   N_EPOCHS = 72

   OUTPUT_FILE = "mnistGrenholm-nocv-72ep0002seed" + str(RANDOM_SEED) + ".csv"

    

import keras
import tensorflow as tf

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
import numpy as np # linear algebra



keras.__version__

from keras import backend as K

K.set_image_dim_ordering('tf')

from keras.utils import np_utils

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D

from keras.optimizers import Adam

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator
train_file = "../input/train.csv"

test_file = "../input/test.csv"

output_file = OUTPUT_FILE
mnist_dataset = np.loadtxt(train_file, skiprows=1, dtype='int', delimiter=',')
n_train = mnist_dataset.shape[0]



np.random.seed(RANDOM_SEED)

np.random.shuffle(mnist_dataset)

x_train = mnist_dataset[:,1:]

y_train = mnist_dataset[:,0]



x_train = x_train.astype("float32")/255.0

y_train = np_utils.to_categorical(y_train)



n_classes = y_train.shape[1]

x_train = x_train.reshape(n_train, 28, 28, 1)
model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (3, 3), input_shape = (28, 28, 1), padding='same'))

model.add(Activation('relu'))

model.add(Conv2D(filters = 32, kernel_size = (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 64, kernel_size = (3, 3)))

model.add(Activation('relu'))

model.add(Conv2D(filters = 64, kernel_size = (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(128))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(10))

model.add(Activation('softmax'))
datagen = ImageDataGenerator(zoom_range = 0.1,

                            height_shift_range = 0.1,

                            width_shift_range = 0.1,

                            rotation_range = 20)
model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=LEARNING_RATE), metrics = ["accuracy"])
hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size = 64),

                           steps_per_epoch = n_train/20,

                           epochs = N_EPOCHS, 

                           verbose = 2  #verbose=1 outputs ETA, but doesn't work well in the cloud

                          )
mnist_testset = np.loadtxt(test_file, skiprows=1, dtype='int', delimiter=',')

x_test = mnist_testset.astype("float32")/255.0

n_test = x_test.shape[0]

x_test = x_test.reshape(n_test, 28, 28, 1)
y_test = model.predict(x_test, batch_size=64)
y_index = np.argmax(y_test,axis=1)
with open(output_file, 'w') as f :

    f.write('ImageId,Label\n')

    for i in range(0,n_test) :

        f.write("".join([str(i+1),',',str(y_index[i]),'\n']))