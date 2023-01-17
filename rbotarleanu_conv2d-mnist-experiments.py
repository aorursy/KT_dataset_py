import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
from sklearn.model_selection import train_test_split
train, test = train_test_split(train_df,

                               test_size=.2,

                               random_state=42)
pixel_columns = ['pixel%d'%i for i in range(784)]
train_x = pd.DataFrame.as_matrix(train[pixel_columns])

train_y = pd.DataFrame.as_matrix(train['label'])



test_x = pd.DataFrame.as_matrix(test[pixel_columns])

test_y = pd.DataFrame.as_matrix(test['label'])
train_x = train_x / 255.

test_x = test_x / 255.
from keras.utils.np_utils import to_categorical

train_y = to_categorical(train_y)

test_y = to_categorical(test_y)

NUM_CLASSES = train_y.shape[1]

NUM_CLASSES
IMAGE_ROWS = 28

IMAGE_COLS = 28

CHANNELS = 1

INPUT_SHAPE = (IMAGE_ROWS, IMAGE_COLS, CHANNELS) # (rows, cols, number_of_color_channels)
train_x = train_x.reshape(train_x.shape[0], IMAGE_ROWS, IMAGE_COLS, CHANNELS)

test_x = test_x.reshape(test_x.shape[0], IMAGE_ROWS, IMAGE_COLS, CHANNELS)
train_x.shape
from keras.models import Sequential

from keras.layers import Dense, Dropout, Conv2D, Flatten

from keras.layers.pooling import MaxPooling2D
model = Sequential()

model.add(Conv2D(32,

                 kernel_size=(4, 4),

                 strides=(1,1),

                 activation='relu',

                 input_shape=INPUT_SHAPE))

model.add(Dropout(.5))

model.add(MaxPooling2D(pool_size=(2,2),

                       strides=(2,2)))

model.add(Conv2D(24,

                 kernel_size=(8, 8),

                 strides=(1, 1)))

model.add(Flatten())

model.add(Dropout(.5))

model.add(Dense(128, activation='relu')) # the FC layer is to further learn the flattened filtered images

model.add(Dense(NUM_CLASSES, activation='softmax')) # output layer
model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['mae', 'accuracy'])
print(train_x.shape)

print(train_y.shape)
callbacks_history = model.fit(train_x,

                              train_y,

                              batch_size=64,

                              validation_split=.2,

                              epochs=10)
import matplotlib.pyplot as plt

%matplotlib inline
image = train_x[100]
plt.matshow(image.reshape(IMAGE_ROWS, IMAGE_COLS))

plt.show()
first_conv = model.layers[0]

first_conv_weights = first_conv.get_weights()
filter_weights = first_conv_weights[0]
filter_weights.shape
filter_weights[:, :, 0, 2]
for i in range(32):

    print(i)

    plt.imshow(filter_weights[:, :, 0, i], cmap='viridis', interpolation='None')

    plt.show()
model_first_conv = Sequential()

model_first_conv.add(model.layers[0])

model_first_conv.compile(optimizer='adam', loss='categorical_crossentropy')
first_layer_out = model_first_conv.predict_proba(image.reshape(1, IMAGE_ROWS, IMAGE_COLS, 1))
first_layer_out.shape
weights_orig = model.layers[0].get_weights()

weights_new = model_first_conv.layers[0].get_weights()
first_layer_out[0, :, :, 0].shape
for i in range(32):

    plt.matshow(first_layer_out[0, :, :, i])

    plt.show()
model_second_conv = Sequential()

# model.add(Conv2D(32,

#                  kernel_size=(4, 4),

#                  strides=(1,1),

#                  activation='relu',

#                  input_shape=INPUT_SHAPE))

# model.add(Dropout(.5))

# model.add(MaxPooling2D(pool_size=(2,2),

#                        strides=(2,2)))

# model.add(Conv2D(24,

#                  kernel_size=(8, 8),

#                  strides=(1, 1)))

model_second_conv.add(model.layers[0])

model_second_conv.add(model.layers[1])

model_second_conv.add(model.layers[2])

model_second_conv.add(model.layers[3])

model_second_conv.compile(optimizer='adam', loss='categorical_crossentropy')
second_conv_out = model_second_conv.predict_proba(image.reshape(1, IMAGE_ROWS, IMAGE_COLS, 1))
second_conv_out.shape
for i in range(24):

    plt.matshow(second_conv_out[0, :, :, i])

    plt.show()