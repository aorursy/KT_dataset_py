# Linear algebra

import numpy as np 

# Data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd



# Keras 

from keras.utils.np_utils import to_categorical # For convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense,MaxPooling2D,Flatten,Dropout,MaxPool2D

from keras.layers.convolutional import Conv2D

from keras import backend

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import SGD

from keras.optimizers import RMSprop

from keras.losses import categorical_crossentropy

from keras.callbacks import ReduceLROnPlateau



# Seaborn

from seaborn import countplot
# Globals constants

input_neurons=784  # 28 * 28 = 784

output_neurons=10
# Fix random seed for reproducibility

np.random.seed(2)
# Load the dataset

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
# Apply a reshape and load to the x and y

xtrain_df=train_df.drop(['label'], axis=1)

ytrain_df=train_df.label

# Load data for test, just the variables without the prediction

xtest_df = test_df
# Free some space

del train_df

del test_df
# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)

xtrain_df = xtrain_df.values.reshape(-1, 28, 28, 1)

xtest_df=xtest_df.values.reshape(-1, 28, 28, 1)
ytrain_df.value_counts()
# Chart with the quantity of fields

_=countplot(ytrain_df)
# Generate categorical labels

ytrain_df = to_categorical(ytrain_df, output_neurons)
# Function for normalize the data

def normalize(df):

    df.astype(np.float32)

    df=df/255.0

    return df

    

xtrain_df=normalize(xtrain_df)

xtest_df=normalize(xtest_df)
model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(output_neurons, activation = "softmax"))
# Set a learning rate annealer

learning_rate_reduction = ReduceLROnPlateau(monitor='acc',patience=3,verbose=1,factor=0.5, min_lr=0.00001)
# Define the optimizer

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# optimizer = SGD(lr=0.01) # 0.9822, model 1

# optimizer='adam' # 0.994, model 2
model.compile(

    loss=categorical_crossentropy,

    optimizer=optimizer,

    metrics=['accuracy']

)
datagen = ImageDataGenerator(

        featurewise_center=False,  # Set input mean to 0 over the dataset

        samplewise_center=False,  # Set each sample mean to 0

        featurewise_std_normalization=False,  # Divide inputs by std of the dataset

        samplewise_std_normalization=False,  # Divide each input by its std

        zca_whitening=False,  # Apply ZCA whitening

        rotation_range=13,  # Randomly rotate images in the range (degrees, 0 to 180) # 10

        zoom_range = 0.13, # Randomly zoom image # 0.1

        width_shift_range=0.1,  # Randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # Randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # Randomly flip images

        vertical_flip=False)  # Randomly flip images



datagen.fit(xtrain_df)
model.fit_generator(

    datagen.flow(xtrain_df, ytrain_df, batch_size=86),

    steps_per_epoch=len(xtrain_df) // 86,

    epochs=50,

    callbacks=[learning_rate_reduction]

)
# model.fit(xtrain_df, ytrain_df, epochs=30, batch_size=86,verbose=1)
scores = model.evaluate(xtrain_df, ytrain_df)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
predictions = model.predict(xtest_df,verbose=0)

results=np.argmax(predictions,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("digit_recognizer_datagen.csv",index=False)

# For clear the tensorflow session

backend.clear_session()