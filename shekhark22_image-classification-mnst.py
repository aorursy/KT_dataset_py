import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns



%matplotlib inline



np.random.seed(2)



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical # convert to ont_hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

train= pd.read_csv("../input/digit-recognizer/train.csv")

test=pd.read_csv("../input/digit-recognizer/test.csv")
train.head()
print("The shape of train dataset is {}".format(train.shape))

print("The shape of test dataset is {}".format(test.shape))
Y_train = train["label"]



# Drop 'label' column

X_train = train.drop(["label"], axis=1)





del train



g= sns.countplot(Y_train)



Y_train.value_counts()
X_train.isnull().any().describe()
test.isnull().any().describe()
# Normalize the data

X_train = X_train/255.0

test = test / 255.0
# Reshape image in 3 dimensions (height= 28px, width= 28px, canal=1)



X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
# Encoding labels to one hot vectors 



Y_train = to_categorical(Y_train, num_classes=10)
Y_train[0]
# Split the train and the validation set for the fitting



X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size= 0.1, random_state= 2)
g= plt.imshow(X_train[0][:, :, 0])
# Set the CNN model

# The CNN architechture is IN >[COnv2D -> relu]*2 ->MaxPool2D -> Dropout]*2

# -> Flatten -> Dense -> Dropout -> Out
model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'Same',

                activation = 'relu', input_shape= (28, 28, 1)))



model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = "Same", 

                activation = 'relu'))



model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters= 64, kernel_size=(5,5), padding = "Same",

                activation= 'relu'))



model.add(Conv2D(filters= 63, kernel_size=(5,5), padding = "Same",

                activation='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(256, activation="relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
# set a learning rate annealer



learning_rate_reduction = ReduceLROnPlateau(monitor= 'val_acc',

                                           patience= 3,

                                           verbose= 1,

                                           factor = 0,

                                           min_lr=0.00001)
epochs = 2

batch_size = 86
datagen = ImageDataGenerator(

        featurewise_center= False,

        samplewise_center=False,

        featurewise_std_normalization=False,

        samplewise_std_normalization=False,

        zca_whitening=False,

        rotation_range=10,

        zoom_range=0.1,

        width_shift_range=0.1,

        height_shift_range=0.1,

        horizontal_flip=False,

        vertical_flip=False)



datagen.fit(X_train)
# Fit the model



history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),

                             epochs=epochs, validation_data=(X_val, Y_val),

                             verbose=2, steps_per_epoch=X_train.shape[0]// batch_size,

                             callbacks=[learning_rate_reduction])
# predict results

results = model.predict(test)



# select the index with the maximum probability

results = np.argmax(results, axis=1)



results = pd.Series(results, name="Label")
submission = pd.concat([pd.Series(range(1, 28001), name= "ImageId"),

                       results], axis = 1)



submission.to_csv("cnn_mnist_datagen.csv", index=False)