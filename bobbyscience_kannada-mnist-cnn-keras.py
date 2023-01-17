import tensorflow as tf



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

import cv2
base_dir = '../input/Kannada-MNIST/'
L = 10 # labels

S = 28 # dimensions (H, W)

C = 1  # depth
# Load the Kannada MNIST training dataset downloaded from Kaggle

# into a Pandas DataFrame

X_df = pd.read_csv(os.path.join(base_dir, 'train.csv'))
# Plot pixel intensities

X_df.head()
# Extract dataset samples pixel intensities

X = X_df.iloc[:, 1:]

X = X.values 



# Extract labels

y = X_df.iloc[:, 0]

y = y.values



# Plot labels

pd.DataFrame(y)



del X_df
# Reshape dataset samples to (num_samples, height[rows], width[cols])

# to do some plotting

X = X.reshape(X.shape[0], S, S)

X.shape
# Plot a sample of the dataset

plt.imshow(X[420])

plt.title(label='Image at position 420 (blaze it)')

plt.show()
# Plot more samples



fig, ax = plt.subplots(

    nrows=6,

    ncols=5,

    figsize=[6, 8]

)



for index, axi in enumerate(ax.flat):

    axi.imshow(X[index])

    axi.set_title(f'Image #{index}')



plt.tight_layout(True)

plt.show()
# Normalize pixel intensities to range [0, 1]

X = X.astype('float32') / 255.0
# Reshape the samples from the dataset

# to meet Keras requirements (num_samples, height, width, num_channels)

X = X.reshape(-1, 28, 28, 1)

X.shape
# One-hot encode labels from integers to vectors

y = to_categorical(y, num_classes=L)

pd.DataFrame(y)
# Split the dataset into 80/20 train/test sets

(X_train, X_ttest, y_train, y_ttest) = train_test_split(X, y, test_size = 0.2, random_state=42)

print(X_train.shape) # <- num of train samples

print(X_ttest.shape) # <- num of test  samples



del X
# Data Augmentation: Set the image generator



daug = ImageDataGenerator(

    featurewise_center=False,

    samplewise_center=False,

    featurewise_std_normalization=False,

    samplewise_std_normalization=False,

    zca_whitening=False,

    rotation_range=12,

    zoom_range = 0.3, 

    width_shift_range=0.2,

    height_shift_range=0.2,

    horizontal_flip=False,

    vertical_flip=False

)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense, LeakyReLU



class CustomNet(object):

    @staticmethod

    def build(width, height, num_classes, depth=3):

        model = Sequential()

        input_shape = (height, width, depth)

        chan_dim = -1

        

        # (Conv => LReLU => BN) * 3 => POOL => DO

        model.add(Conv2D(64, (3, 3), padding='same', input_shape=input_shape))

        model.add(LeakyReLU(alpha=0.1))

        model.add(BatchNormalization(axis=chan_dim))

        model.add(Conv2D(64, (3, 3), padding='same'))

        model.add(LeakyReLU(alpha=0.1))

        model.add(BatchNormalization(axis=chan_dim))

        model.add(Conv2D(64, (3, 3), padding='same'))

        model.add(LeakyReLU(alpha=0.1))

        model.add(BatchNormalization(axis=chan_dim))

        

        # (Conv => LReLU => BN) * 3 => POOL => DO

        model.add(Conv2D(96, (3, 3), padding='same'))

        model.add(LeakyReLU(alpha=0.1))

        model.add(BatchNormalization(axis=chan_dim))

        model.add(Conv2D(96, (3, 3), padding='same'))

        model.add(LeakyReLU(alpha=0.1))

        model.add(BatchNormalization(axis=chan_dim))

        model.add(Conv2D(96, (3, 3), padding='same'))

        model.add(LeakyReLU(alpha=0.1))

        model.add(BatchNormalization(axis=chan_dim))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Dropout(0.2))

        

        # (Conv => LReLU => BN) * 3 => POOL => DO

        model.add(Conv2D(128, (3, 3), padding='same'))

        model.add(LeakyReLU(alpha=0.1))

        model.add(BatchNormalization(axis=chan_dim))

        model.add(Conv2D(128, (3, 3), padding='same'))

        model.add(LeakyReLU(alpha=0.1))

        model.add(BatchNormalization(axis=chan_dim))

        model.add(Conv2D(128, (3, 3), padding='same'))

        model.add(LeakyReLU(alpha=0.1))

        model.add(BatchNormalization(axis=chan_dim))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Dropout(0.2))

        

        # (Conv => LReLU => BN) * 3 => POOL => DO

        model.add(Conv2D(256, (3, 3), padding='same'))

        model.add(LeakyReLU(alpha=0.1))

        model.add(BatchNormalization(axis=chan_dim))

        model.add(Conv2D(256, (3, 3), padding='same'))

        model.add(LeakyReLU(alpha=0.1))

        model.add(BatchNormalization(axis=chan_dim))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Dropout(0.2))

        

        # (FC => LReLU => BN) * 2 => DO

        model.add(Flatten())

        model.add(Dense(256))

        model.add(LeakyReLU(alpha=0.1))

        model.add(BatchNormalization())

        model.add(Dense(128))

        model.add(LeakyReLU(alpha=0.1))

        model.add(BatchNormalization())

        model.add(Dropout(0.5))

        

        # Softmax

        model.add(Dense(num_classes))

        model.add(Activation('softmax'))

        

        print(model.summary())

        

        return model
net = CustomNet()



model = net.build(

    width=S,

    height=S,

    num_classes=L,

    depth=C)
num_epochs = 80
init_lr = 0.002

power = 2.0



adam_opt = Adam(

    lr=init_lr,

    beta_1=0.9,

    beta_2=0.999,

    epsilon=1e-08,

    decay=0.0

)





def polynomial_decay(epoch):

    max_epochs = num_epochs

    

    return init_lr * (1 - (epoch / float(max_epochs))) ** power
# Plot polynomial decay

x = np.linspace(0, num_epochs)

fx = [init_lr * (1 - (i / float(num_epochs))) ** power for i in range(len(x))]

plt.plot(x, fx)

plt.title(label=f'Polynomial decay, power {power}')

plt.show()
# Ignore checkpointing on Kaggle kernel



'''

checkpointHandler = ModelCheckpoint(

    os.path.join(base_dir, 'best_weights.hdf5'),

    monitor='val_loss',

    save_best_only=True,

    verbose=1

)

'''



callbacks = [

    LearningRateScheduler(polynomial_decay),

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
plt.style.use('ggplot')

plt.figure(figsize=(10, 7))

plt.plot(np.arange(0, num_epochs), h.history['loss'], label='train_loss')

plt.plot(np.arange(0, num_epochs), h.history['val_loss'], label='val_loss')

plt.plot(np.arange(0, num_epochs), h.history['accuracy'], label='train_accuracy')

plt.plot(np.arange(0, num_epochs), h.history['val_accuracy'], label='val_accuracy')



plt.title('Training Loss and Accuracy')

plt.ylabel('Loss/Accuracy')

plt.xlabel('Epoch #')

plt.legend()



plt.show()
'''

Pushing a submission to Kaggle.



First of, load the test set, normalize it and reshape it.

Then, make predictions via the trained model and build the

submission csv file.

'''



sub_X_test = pd.read_csv(os.path.join(base_dir, 'test.csv'))     # Load CSV

sub_X_test = sub_X_test.drop('id', axis=1)                       # Drop the ID column

sub_X_test = sub_X_test.iloc[:,:].values                         # Get raw pixel intensities

sub_X_test = sub_X_test.reshape(sub_X_test.shape[0], S, S, C)    # Reshape to meet Keras requirements

sub_X_test = sub_X_test / 255.0                                  # Normalize to range [0, 1]



preds = model.predict_classes(sub_X_test, batch_size=batch_size)        # Make predictions and build the submission file



# https://www.kaggle.com/c/Kannada-MNIST/discussion/110586

id_col = np.arange(preds.shape[0])

submission = pd.DataFrame({'id': id_col, 'label': preds})

print(pd.DataFrame(submission))

submission.to_csv('submission.csv', index = False)