import numpy as np # linear algebra



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
from keras.utils import np_utils

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D

from keras.optimizers import Adam

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator
train_file = "../input/train.csv"

test_file = "../input/test.csv"

output_file = "submission.csv"
mnist_dataset = np.loadtxt(train_file, skiprows=1, dtype='int', delimiter=',')
val_split = 0.125

n_raw = mnist_dataset.shape[0]

n_val = int(n_raw * val_split + 0.5)

n_train = n_raw - n_val



np.random.shuffle(mnist_dataset)

x_val, x_train = mnist_dataset[:n_val,1:], mnist_dataset[n_val:,1:]

y_val, y_train = mnist_dataset[:n_val,0], mnist_dataset[n_val:,0]



x_train = x_train.astype("float32")/255.0

x_val = x_val.astype("float32")/255.0

y_train = np_utils.to_categorical(y_train)

y_val = np_utils.to_categorical(y_val)



n_classes = y_train.shape[1]

x_train = x_train.reshape(n_train, 28, 28, 1)

x_val = x_val.reshape(n_val, 28, 28, 1)
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
model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=1e-3), metrics = ["accuracy"])
callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=2, mode='auto'),

            ModelCheckpoint('mnist.h5', monitor='val_loss', save_best_only=True, verbose=0)]
hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size = 64),

                           steps_per_epoch = n_train/100, #Take away 100 when not on Kaggle kernel

                           epochs = 5, #Increase this when not on Kaggle kernel

                           verbose = 2,  #verbose=1 outputs ETA, but doesn't work well in the cloud

                           validation_data = (x_val, y_val),

                           callbacks = callbacks)
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