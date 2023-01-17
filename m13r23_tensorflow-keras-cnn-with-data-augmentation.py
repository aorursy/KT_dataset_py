import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import tensorflow as tf

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout

from tensorflow.keras.preprocessing.image import ImageDataGenerator
print(tf.__version__)
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
print (train_data.shape)

print (test_data.shape)
# Convert from pandas dataframe to numpy

# .to_numpy() -- not available in the numpy running in Kaggle, use deprecated .values member.

train_data = train_data.values

test_data = test_data.values
# Shuffle the training data before separating lables and images

np.random.shuffle(train_data)
## Separate the labels from the pixels in the training data, and reshape images to 28x28 instead of 784x1

train_labels = train_data[:,0]

train_digits = train_data[:,1:].reshape(-1, 28, 28, 1)



## Convert test digits to numpy, and reshape.  

test_digits = test_data.reshape(-1, 28, 28, 1)



print(train_labels.shape, train_digits.shape, test_digits.shape)



## don't need these any longer.

del(train_data)

del(test_data)
split = len(train_labels)//11



val_labels, train_labels = train_labels[:split], train_labels[split:]

val_digits, train_digits = train_digits[:split], train_digits[split:]



print(val_labels.shape, val_digits.shape, train_labels.shape, train_digits.shape)

plt.bar(range(10), [ np.sum(val_labels == r) for r in range(10) ])
train_digits = train_digits / 255.0

val_digits = val_digits / 255.0

test_digits = test_digits / 255.0
i = 1

plt.figure()

plt.imshow(train_digits[i,:,:,0])

plt.title('Digit: ' + str(train_labels[i]))

plt.colorbar()

plt.grid(False)
X0 = Input(shape = (28,28,1))



X = Conv2D(filters = 32, kernel_size = 3, padding = 'Same', activation ='relu')(X0)

X = Conv2D(filters = 32, kernel_size = 3, padding = 'Same', activation ='relu')(X)

X = MaxPooling2D(pool_size = 2, strides = 2)(X)



X = Dropout(0.25)(X)



X = Conv2D(filters = 64, kernel_size = 3, padding = 'Same', activation ='relu')(X)

X = Conv2D(filters = 64, kernel_size = 3, padding = 'Same', activation ='relu')(X)

X = MaxPooling2D(pool_size = 2, strides = 2)(X)



X = Dropout(0.25)(X)



X = Flatten()(X)

X = Dense(512, activation = "relu")(X)

X = Dropout(0.50)(X)



Out = Dense(10, activation = "softmax")(X)
model = Model(inputs=X0, outputs=Out)

model.compile(optimizer = 'adam', 

              loss = 'sparse_categorical_crossentropy',

              metrics=['accuracy'])

## Establish a checkpoint with the model in its initial state.  We can go back to this 

## state later and train the same model differently to see different results. 

initial_model_weights = model.get_weights()
model.summary()
history = model.fit(train_digits, train_labels, 

                    epochs=10, 

                    verbose=2,

                    validation_data=(val_digits,val_labels))
def plot_history(hist) :

    fig, ax = plt.subplots(nrows=2)

    fig.set_size_inches(8,8)



    ax[0].plot(hist['acc'][:], marker='.', color='red', linewidth=1, alpha=0.5)

    if ('val_acc' in hist) :

        ax[0].plot(hist['val_acc'][:], marker='.', color='blue', linewidth=1, alpha=0.5)



    ax[1].plot(hist['loss'][:], marker='.', color='red', linewidth=1, alpha=0.2)

    if ('val_loss' in hist) :

        ax[1].plot(hist['val_loss'][:], marker='.', color='blue', linewidth=1, alpha=0.2)

    ax[0].set_title('Accuracy')

    ax[0].set_ylabel('accuracy')

    ax[1].set_title('Loss')

    ax[1].set_ylabel('loss')

    ax[1].set_xlabel('epochs')
plot_history(history.history)
## Save off the results for later analysis. 

model.save("cnn1.h5")
predictions = model.predict(test_digits)

predictions = np.argmax(predictions,axis = 1)

predictions = pd.Series(predictions,name = 'Label')

ids = pd.Series(range(1,28001),name = 'ImageId')

predictions = pd.concat([predictions,ids],axis = 1)

predictions.to_csv('pred1.csv',index = False)



del(predictions)

del(ids)
## Return the model to its initial state.

model.set_weights(initial_model_weights)
datagen = ImageDataGenerator(

    featurewise_center=False,  # set input mean to 0 over the dataset

    samplewise_center=False,  # set each sample mean to 0

    featurewise_std_normalization=False,  # divide inputs by std of the dataset

    samplewise_std_normalization=False,  # divide each input by its std

    zca_whitening=False,  # apply ZCA whitening

    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

    # randomly shift images horizontally (fraction of total width)

    width_shift_range=0.1,

    # randomly shift images vertically (fraction of total height)

    height_shift_range=0.1,

    shear_range=0.1,  # set range for random shear

    zoom_range=0.1,  # set range for random zoom

    channel_shift_range=0.,  # set range for random channel shifts

    # set mode for filling points outside the input boundaries

    fill_mode='nearest',

    cval=0.,  # value used for fill_mode = "constant"

    horizontal_flip=False,  # randomly flip images

    vertical_flip=False,  # randomly flip images

    # set rescaling factor (applied before any other transformation)

    rescale=None,

    # set function that will be applied on each input

    preprocessing_function=None,

    # image data format, either "channels_first" or "channels_last"

    data_format=None,

    # fraction of images reserved for validation (strictly between 0 and 1)

    validation_split=0.0)



# Compute quantities required for feature-wise normalization

# (std, mean, and principal components if ZCA whitening is applied).

datagen.fit(train_digits)
# Fit the model on the batches generated by datagen.flow().

batch_size = 64

gen_history = model.fit_generator(datagen.flow(train_digits, train_labels, batch_size=batch_size),

                    epochs=10,

                    verbose=2,

                    steps_per_epoch=train_digits.shape[0]/batch_size,

                    validation_data=(val_digits,val_labels))

plot_history(gen_history.history)
predictions = model.predict(val_digits)

predictions = np.argmax(predictions,axis = 1)

c = tf.confusion_matrix(labels=val_labels, predictions=predictions, num_classes=10)

sess = tf.Session()

r = sess.run(c)

del(c)

del(sess)

print(r)



plt.figure(figsize = (10,8))

sns.heatmap(r, annot=True, label='small')
predictions = model.predict(test_digits)

predictions = np.argmax(predictions,axis = 1)

predictions = pd.Series(predictions,name = 'Label')

ids = pd.Series(range(1,28001),name = 'ImageId')

predictions = pd.concat([predictions,ids],axis = 1)

predictions.to_csv('pred2.csv',index = False)

del(predictions)

del(ids)



## Save off the results for later analysis. 

model.save("cnn2.h5")