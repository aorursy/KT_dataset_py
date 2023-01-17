import tensorflow as tf

print(tf.test.gpu_device_name())



import os

from tensorflow.keras.optimizers import SGD, Adam

from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator

from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.datasets import mnist

from tensorflow.keras import backend as K



from sklearn.preprocessing import LabelBinarizer

from sklearn.metrics import classification_report

from sklearn.feature_extraction.image import extract_patches_2d

from sklearn.model_selection import train_test_split



import matplotlib

import matplotlib.pyplot as plt

matplotlib.use('Agg')     

%matplotlib inline



import pandas as pd

import numpy as np
(X_train, y_train), (X_ttest, y_ttest) = mnist.load_data()
'''

Let's log the training set shape.



The first column indicates that we have 60 000 training samples

and the two last columns indicate that the dimension of each sample is (h, w) <=> (28, 28).

'''



X_train.shape
'''

As we can see, we've got 60000 samples in our training set and 10000 samples in our testing set.

Each image, or sample, or datapoint has a dimension of (28, 28) and belongs

to the gray-scale color-space, so their depth equals 1.

'''



plt.imshow(X_train[0])

plt.title(label='Image at position 0')

plt.show()
'''

As you can see, images in the MNIST dataset are heavily pre-processed.

That's why this is used as a benchmark dataset and getting very high accuracy is common.

In a real-world problem, we'll need to do some image pre-processing to enchance images

and extract meaningful features.

'''



fig, ax = plt.subplots(

    nrows=6,

    ncols=5,

    figsize=[6, 8]

)



for index, axi in enumerate(ax.flat):

    axi.imshow(X_train[index])

    axi.set_title(f'Image #{index}')



plt.tight_layout(True)

plt.show()
'''

Normalization of pixel intensities is adjusting values measured on different scales to a notionally common scale.

That's a best practice you have to follow because weights reach optimum values faster.

Therefore, the network converges faster.



So, instead of having pixel intensities in the range [0, 255] in the gray-scale color-space,

we're going to scale them into the range [0, 1].

There are many normalization techniques and this is one of them.

'''



(X_train, X_ttest) = (child.astype('float32') / 255.0 for child in [X_train, X_ttest])
'''

Reshape the both training and testing set that way 

the number of samples is the first entry in the matrix,

the single channel as the second entry,

followed by the number of rows and columns.



(num_samples, rows, columns, channel)

'''



X_train = X_train.reshape(-1, 28, 28, 1)

X_ttest = X_ttest.reshape(-1, 28, 28, 1)



# Let's log the new shape.

X_train.shape
'''

There are 60000 integers labels for the training set,

each one corresponding to one single datapoint.

That means, for a given Xi datapoint we got a Yi label in the range [0, 9].

For instance, the datapoint at the index 59995 of our training set has the label 8.

'''



pd.DataFrame(y_train)
'''

Now, our previous integer labels are converted to vector labels.

This process is called one-hot encoding and most of the machine learning algorithms

benefit from this label representation. 2 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],

8 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0].



We could also use the to_categorical() function from Keras which yields the exact same values.

'''



lb = LabelBinarizer()

(y_train, y_ttest) = (lb.fit_transform(labels) for labels in [y_train, y_ttest])

pd.DataFrame(y_train)
'''

Let's apply some data augmentation.



Data augmentation is a set of techniques used to generate new training samples from the original ones

by applying jitters and perturbations such that the classes labels are not changed.

In the context of computer vision, these random transformations can be translating,

rotating, scaling, shearing, flipping etc.



Data augmentation is a form of regularization because the training algorithm is being

constantly presented with new training samples,

allowing it to learn more robust and discriminative patterns

and reducing overfitting.

'''



daug = ImageDataGenerator(

    featurewise_center=False,

    samplewise_center=False,

    featurewise_std_normalization=False,

    samplewise_std_normalization=False,

    zca_whitening=False,

    rotation_range=10,

    zoom_range = 0.1, 

    width_shift_range=0.1,

    height_shift_range=0.1,

    horizontal_flip=False,

    vertical_flip=False

)
'''

Here, we've imported our CNN.



That's a small VGG-like net, with two stacks of (Conv => ReLU => BN) * 2 => POOL => DO

and a Fully-Connected layer at the end.

Pay attention to Batch Normalization and Dropout layers

which help to reduce overfitting.

'''



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense



class CustomNet(object):

    @staticmethod

    def build(width, height, num_classes, depth=3):

        model = Sequential()

        input_shape = (height, width, depth)

        chan_dim = -1

        

        # (Conv => ReLU => BN) * 3 => POOL

        model.add(Conv2D(64, (3, 3), padding='same', input_shape=input_shape))

        model.add(Activation('relu'))

        model.add(BatchNormalization(axis=chan_dim))

        model.add(Conv2D(64, (3, 3), padding='same'))

        model.add(Activation('relu'))

        model.add(BatchNormalization(axis=chan_dim))

        model.add(Conv2D(64, (3, 3), padding='same'))

        model.add(Activation('relu'))

        model.add(BatchNormalization(axis=chan_dim))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        

         # (Conv => ReLU => BN) * 3 => POOL => DO

        model.add(Conv2D(128, (3, 3), padding='same'))

        model.add(Activation('relu'))

        model.add(BatchNormalization(axis=chan_dim))

        model.add(Conv2D(128, (3, 3), padding='same'))

        model.add(Activation('relu'))

        model.add(BatchNormalization(axis=chan_dim))

        model.add(Conv2D(128, (3, 3), padding='same'))

        model.add(Activation('relu'))

        model.add(BatchNormalization(axis=chan_dim))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Dropout(0.25))

        

        # FC => ReLU => BN => DO

        model.add(Flatten())

        model.add(Dense(256))

        model.add(Activation('relu'))

        model.add(BatchNormalization())

        model.add(Dropout(0.5))

        

        # Softmax

        model.add(Dense(num_classes))

        model.add(Activation('softmax'))

        

        print(model.summary())

        

        return model
net = CustomNet()



model = net.build(

    width=28,

    height=28,

    num_classes=10,

    depth=1)
'''

When the model has seen all of your training samples, we say that one epoch has passed.

We're going to train the model for 50 epochs.

'''



num_epochs = 50
'''

Let's use an optimization method.



Optimization algorithms are the engines that power neural networks and

enable them to learn patterns from data by tweaking and seeking for optimal weights values.

Most common one is the (Stochastic) Gradient Descent, but I'll use Adam here.



As you can see, the first param is lr, or learning rate.

This is one of the most important hyperparameters we have to tune.

A learning rate is the step your optimization algorithm is going to make toward

the direction that leads to a lower loss function (and a higher accuracy).



If the learning rate is too small, the algorithm is going to make tiny steps slowing down the process.

But on the other hand, if the learning rate is too high,

the algorithm risks to bounce around the loss landscape and not actually “learn” any patterns from your data.

'''



# Initial learning rate

init_lr = 0.001



adam_opt = Adam(

    lr=init_lr,

    beta_1=0.9,

    beta_2=0.999,

    epsilon=1e-08,

    decay=0.0

)



'''

Let's now define a learn-rate scheduler.



The decay is used to slowly reduce the learning rate over time.

Decaying the learning rate is helpful in reducing overfitting 

and obtaining higher classification accuracy – the smaller the learning rate is, 

the smaller the weight updates will be. 

We're going to use a polynomial decay. 

Although there are many way to do that.

'''



def polynomial_decay(epoch):

    max_epochs = num_epochs

    base_lr = init_lr

    power = 2.0

    

    return base_lr * (1 - (epoch / float(max_epochs))) ** power
# Let's plot it.



x = np.linspace(0, num_epochs)

fx = [init_lr * (1 - (i / float(num_epochs))) ** 2.0 for i in range(len(x))]

plt.plot(x, fx)

plt.title(label='Polynomial decay, power 2')

plt.show()
'''

Here, we define two callbacks.



Callbacks are functions executed at the end of an epoch.

The first one save our model (checkpoint) whenever the loss decreases (therefore our accuracy improves).

That way we keep the best model. The last one is our learning rate scheduler using the polynomial decay.

'''



'''

checkpointHandler = ModelCheckpoint(

    os.path.join(base_dir, 'best_c10_weights.hdf5'),

    monitor='val_loss',

    save_best_only=True,

    verbose=1

)

'''



callbacks = [

    LearningRateScheduler(polynomial_decay)

    # checkpointHandler

]
batch_size = 128



print('# Compiling the model...')

model.compile(

    loss='categorical_crossentropy',

    optimizer=adam_opt,

    metrics=['accuracy']

)



print('# Training the network...')

h = model.fit_generator(

    daug.flow(X_train, y_train, batch_size=batch_size),

    validation_data=(X_ttest, y_ttest),

    epochs=num_epochs,

    steps_per_epoch=len(X_train) // batch_size,

    callbacks=callbacks,

    verbose=1

)
label_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']



print('Confusion matrix:')

preds = model.predict(X_ttest, batch_size=batch_size)

print(classification_report(y_ttest.argmax(axis=1),

preds.argmax(axis=1), target_names=label_names))
# Loss

plt.figure(figsize=(8, 5))

plt.plot(np.arange(0, num_epochs), h.history['loss'], label='train_loss')

plt.plot(np.arange(0, num_epochs), h.history['val_loss'], label='val_loss')

plt.title('Loss')

plt.legend()

plt.show()



# Accuracy

plt.figure(figsize=(8, 5))

plt.plot(np.arange(0, num_epochs), h.history['accuracy'], label='train_acc')

plt.plot(np.arange(0, num_epochs), h.history['val_accuracy'], label='val_acc')

plt.title('Accuracy')

plt.legend()

plt.show()
'''

Pushing a submission to Kaggle.



First of, load the test set, normalize it and reshape it.

Then, make predictions via the trained model and build the

submission csv file.

'''



sub_X_test = pd.read_csv('../input/digit-recognizer/test.csv')   # Load CSV

sub_X_test = sub_X_test.iloc[:,:].values                         # Get raw pixel intensities

sub_X_test = sub_X_test.reshape(sub_X_test.shape[0], 28, 28, 1)  # Reshape to meet Keras requirements

sub_X_test = sub_X_test / 255.0                                  # Normalize to range [0, 1]

# Get predictions

preds = model.predict_classes(sub_X_test, batch_size=batch_size)

# Generate a submission file

id_col = np.arange(1, preds.shape[0] + 1)

submission = pd.DataFrame({'ImageId': id_col, 'Label': preds})                                         # Shift index

print(pd.DataFrame(submission))

submission.to_csv('submission.csv', index = False)