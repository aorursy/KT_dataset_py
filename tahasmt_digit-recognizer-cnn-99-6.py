import pandas as pd

import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau
data_path = '../input/'



# read data

train = pd.read_csv(data_path + 'train.csv')

test = pd.read_csv(data_path + 'test.csv')

print('train_data shape is :' + str(train.shape))

print('test_data shape is :' + str(test.shape))
# form train.csv => split data to train and validation using stratified random shuffling

strat_train_set = []

strat_val_set = []

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=76)

for train_index, val_index in split.split(train, train['label']):

    strat_train_set = train.loc[train_index]

    strat_val_set = train.loc[val_index]
X_train = strat_train_set.drop(labels=['label'], axis=1)

X_val = strat_val_set.drop(labels=['label'], axis=1)



Y_train = strat_train_set['label']

Y_val = strat_val_set['label']



# Free some memmory

del strat_val_set

del strat_train_set
# one-hot encoding

Y_train = to_categorical(Y_train, num_classes = 10)

Y_val = to_categorical(Y_val, num_classes = 10)

# reshape

X_train = X_train.values.reshape(-1,28,28,1).astype(np.float32)

X_val = X_val.values.reshape(-1,28,28,1).astype(np.float32)

test = np.array(test).reshape(-1,28,28,1).astype(np.float32)

# Preprocess test,validation and train data

test/=255.0

X_train/=255.0

X_val/=255.0
# define model

model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='elu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='elu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))





model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='elu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='elu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(256, activation = "elu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))



# Compile the model

model.compile(optimizer = 'adam' , loss = "categorical_crossentropy", metrics=["accuracy"])
epochs = 32

batch_size = 64

# Set a learning rate annealer

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)

# data augmentation



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
# train the model

hist = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_val,Y_val),

                              verbose = 1, steps_per_epoch=X_train.shape[0] // batch_size

                              , callbacks=[learning_rate_reduction])
# ready prediction on test set for submitting...

preds = model.predict(test)

index = [i for i in range(1,len(preds)+1)]

result = []

for i in range(len(preds)):

    result.append(np.argmax(preds[i]).astype(np.int))



out = pd.DataFrame({'ImageId':index,'Label':result})



out.to_csv("cnn_mnist_submission.csv",index=False)