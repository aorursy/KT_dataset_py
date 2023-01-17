import numpy as np

import pandas as pd

import os

print(os.listdir("../input"))
# load data from csv files

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/sample_submission.csv')

print(train_df.shape, test_df.shape)
train_df['has_cactus'].value_counts()
import matplotlib.pyplot as plt

from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array



train_path = '../input/train/train/'

test_path = '../input/test/test/'
# look at some of the pics from train_df with cactus

has_cactus = train_df[train_df['has_cactus']==1]

plt.figure(figsize=(15,7))

for i in range(40):  

    plt.subplot(4, 10, i+1)

    plt.imshow(load_img(train_path+has_cactus.iloc[i]['id']))

    plt.title("label=%d" % has_cactus.iloc[i]['has_cactus'], y=1)

    plt.axis('off')

plt.subplots_adjust(wspace=0.3, hspace=-0.1)

plt.show()
# look at some of the pics from train_df with no cactus

no_cactus = train_df[train_df['has_cactus']==0]

plt.figure(figsize=(15,7))

for i in range(40):  

    plt.subplot(4, 10, i+1)

    plt.imshow(load_img(train_path+no_cactus.iloc[i]['id']))

    plt.title("label=%d" % no_cactus.iloc[i]['has_cactus'], y=1)

    plt.axis('off')

plt.subplots_adjust(wspace=0.3, hspace=-0.1)

plt.show()
def prep_cnn_data(df, n_x, n_c, path):

    """

    This function loads the image jpg data into tensors

    """

    # initialize tensors

    tensors = np.zeros((df.shape[0], n_x, n_x, n_c))

    # load image as arrays into tensors

    for i in range(df.shape[0]):

        pic = load_img(path+df.iloc[i]['id'])

        pic_array = img_to_array(pic)

        tensors[i,:] = pic_array

    # standardize the values by dividing by 255

    tensors = tensors / 255.

    return tensors
# prepare the train data for CNN

train_pic_array = prep_cnn_data(train_df, 32, 3, path='../input/train/train/')

train_Y = train_df['has_cactus'].values
# prepare the test data for prediction later on

test_pic_array = prep_cnn_data(test_df, 32, 3, path='../input/test/test/')
print(train_pic_array.shape, train_Y.shape)

print(test_pic_array.shape)
# use Keras data generator to augment the training set

from keras_preprocessing.image import ImageDataGenerator

data_augment = ImageDataGenerator(zoom_range=0.1, 

                                  width_shift_range=0.1, height_shift_range=0.1,

                                  horizontal_flip=True, vertical_flip=True)
# build the CNN from keras

from keras import models

from keras import layers



model = models.Sequential()

model.add(layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=(32, 32, 3)))

model.add(layers.Conv2D(32, kernel_size=3, padding='valid', activation='relu'))

model.add(layers.MaxPooling2D(pool_size=(2,2), strides=2))

model.add(layers.Dropout(rate=0.4))

model.add(layers.Conv2D(64, kernel_size=5, padding='same', activation='relu'))

model.add(layers.Conv2D(64, kernel_size=5, padding='valid', activation='relu'))

model.add(layers.MaxPooling2D(pool_size=(2,2), strides=2))

model.add(layers.Dropout(rate=0.4))

model.add(layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))

model.add(layers.Conv2D(128, kernel_size=3, padding='valid', activation='relu'))

model.add(layers.Flatten())

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dropout(rate=0.4))

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))



model.summary()
# compile the model

model.compile(optimizer='adam', loss='binary_crossentropy', 

              metrics=['accuracy'])
# here I use 3500 examples from the training data

X_dev = train_pic_array[:3500]

rem_X_train = train_pic_array[3500:]

print(X_dev.shape, rem_X_train.shape)



Y_dev = train_Y[:3500]

rem_Y_train = train_Y[3500:]

print(Y_dev.shape, rem_Y_train.shape)
# Train and validate the model

epochs = 150

batch_size = 1024

history = model.fit_generator(data_augment.flow(rem_X_train, rem_Y_train, batch_size=batch_size), 

                              epochs=epochs, steps_per_epoch=rem_X_train.shape[0]//batch_size, 

                              validation_data=(X_dev, Y_dev))
# plot and visualise the training and validation losses

loss = history.history['loss']

dev_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)



from matplotlib import pyplot as plt

plt.figure(figsize=(15,10))

plt.plot(epochs, loss, 'bo', label='training loss')

plt.plot(epochs, dev_loss, 'b', label='validation loss')

plt.title('Training and Validation Loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
# do error analysis on the predictions for X_dev

pred_dev = model.predict(X_dev)

pred_dev = (pred_dev > 0.5).astype(int)
# look at those that were classified wrongly in X_dev

result = pd.DataFrame(train_Y[:3500], columns=['Y_dev'])

result['Y_pred'] = pred_dev

result['correct'] = result['Y_dev'] - result['Y_pred']

errors = result[result['correct'] != 0]

error_list = errors.index

print('Number of errors is ', len(errors))

print('The indices are ', error_list)
# plot the image of the wrong in predictions for X_dev

plt.figure(figsize=(15,8))

for i in range(len(error_list)):

    plt.subplot(4, 10, i+1)

    plt.imshow(load_img(train_path+train_df.iloc[error_list[i]]['id']))

    plt.title("true={}\npredict={}".format(train_Y[error_list[i]], 

                                           pred_dev[error_list[i]]), y=1)

    plt.axis('off')

plt.subplots_adjust(wspace=0.3, hspace=-0.1)

plt.show()
# predict on test set

predictions = model.predict(test_pic_array)

print(predictions.shape)
test_df['has_cactus'] = predictions

test_df.head()
# generate submission file in csv format

test_df.to_csv('submission.csv', index=False)