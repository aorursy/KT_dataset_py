# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline



np.random.seed(2)



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

# Load the data

# Input data files are available in the "../input/" directory.

train_images = pd.read_csv("../input/digit-recognizer/train.csv")

test_images = pd.read_csv("../input/digit-recognizer/test.csv")

print("train_images shape", train_images.shape)

print("test_images shape", test_images.shape)
train_images.head()
# Prepare the data for train and test

y_train = train_images["label"]



# Drop 'label' column

X_train = train_images.drop(labels = ["label"],axis = 1) 



# free some space

del train_images
g = sns.countplot(y_train)



y_train.value_counts()
# Check the data

X_train.isnull().any().describe()
test_images.isnull().any().describe()
# Normalize the data

X_train = X_train / 255.0

test_images = test_images / 255.0
# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)

X_train = X_train.values.reshape(-1,28,28,1)

X_test = test_images.values.reshape(-1,28,28,1)
print("X_train shape", X_train.shape)

print("X_test shape", X_test.shape)
# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])

y_train = to_categorical(y_train, num_classes = 10)
# Set the random seed

random_seed = 2



# Split the train and the validation set for the fitting

X_train, X_val, Y_train, Y_val = train_test_split(X_train, y_train, 

                                                  test_size = 0.1, 

                                                  random_state=random_seed)
# Let's visualising one image and looking at the label.

# Some examples

plt.imshow(X_train[0][:,:,0])
# Set the CNN model 

# my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out



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

model.add(Dense(10, activation = "softmax"))
# model.summary() # 887,530
# Define the optimizer

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# Compile the model

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
# Set a learning rate annealer

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
model.fit(X_train, Y_train, epochs=30, batch_size=86, validation_data = (X_val,Y_val))
#arr = arr.astype('float64')

predictions = model.predict(X_test)

# argmax will choose the value having max probability

model_res = np.argmax(predictions, axis=1)
#Print our predicitons as number labels for the first 4 images

print(model_res[:4])
# create sample submission file and submit

pred = pd.DataFrame(model_res)

sub_df = pd.read_csv('../input/digit-recognizer/sample_submission.csv')

datasets = pd.concat([sub_df['ImageId'], pred], axis =1)

datasets.columns = ['ImageId', 'Label']

datasets.to_csv("sample_submission.csv", index = False)
datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images





datagen.fit(X_train)
batch_size = 86

# Fit the model

history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),

                              epochs = 50, validation_data = (X_val,Y_val),

                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size, 

                              callbacks=[learning_rate_reduction])
#arr = arr.astype('float64')

predictions = model.predict(X_test)

# argmax will choose the value having max probability

model_res = np.argmax(predictions, axis=1)
# create sample submission file and submit

pred = pd.DataFrame(model_res)

sub_df = pd.read_csv('../input/digit-recognizer/sample_submission.csv')

datasets = pd.concat([sub_df['ImageId'], pred], axis =1)

datasets.columns = ['ImageId', 'Label']

datasets.to_csv("sample_submission_datagen.csv", index = False)