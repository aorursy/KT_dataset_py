import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns



%matplotlib inline

sns.set(style='white', context='notebook', palette='deep')



np.random.seed(2)



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau
# importing training and testing dataset



train = pd.read_csv("../input/Kannada-MNIST/train.csv")

test = pd.read_csv("../input/Kannada-MNIST/test.csv")
# training dataset

train.head()
# test dataset

test.head()
# train.isna().sum().sum()

# test.isna().sum().sum()
# train.info()

# train.describe()

train.shape
# test.info()

# test.describe()

test.shape
# Store training features in X_train and training targets in Y_train



X = train.drop(["label"], axis=1)

y = train['label']
print(X.shape)

print(y.shape)
# sns.countplot(y)
X_reshape = X.to_numpy().reshape(-1,28,28)

X_reshape.shape
plt.figure(figsize=[10,5])

plt.subplot(111)

plt.imshow(X_reshape[0])

plt.title(y[0])

plt.show()
X_reshape = X.to_numpy().reshape(-1,28,28,1)

X_reshape.shape
y_enc = to_categorical(y)
print(X.shape)

print(y.shape)

print(X_reshape.shape)

print(y_enc.shape)
X_train, X_test, y_train, y_test = train_test_split(X_reshape, y_enc, test_size = 0.1)

print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.5)

print(X_train.shape)

print(y_train.shape)

print(X_valid.shape)

print(y_valid.shape)
model = Sequential()



model.add(Conv2D(32, (5,5), padding = 'Same', activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(32, (5,5), padding = 'Same', activation ='relu'))

model.add(MaxPool2D((2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3,3), padding = 'Same', activation ='relu'))

model.add(Conv2D(64, (3,3), padding = 'Same', activation ='relu'))

model.add(MaxPool2D((2,2), strides=(2,2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))
model.compile(optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0), 

              loss = "categorical_crossentropy", 

              metrics = ["accuracy"])
datagen = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset

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
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
epochs = 10

batch_size = 86
history = model.fit_generator(datagen.flow(X_train, y_train, batch_size = batch_size),

                              epochs = epochs, 

                              validation_data = (X_valid, y_valid),

                              verbose = 2, 

                              steps_per_epoch = X_train.shape[0] // batch_size, 

                              callbacks = [learning_rate_reduction])
# Plot the loss and accuracy curves for training and validation 



fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)





ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
print(model.summary())
id = test['id']

test_data = test.drop('id', axis=1)

test_data = test_data.to_numpy().reshape(-1, 28, 28, 1)
results = model.predict(test_data)

results = np.argmax(results,axis = 1) # select the index with the maximum probability
sim = pd.read_csv('/kaggle/input/Kannada-MNIST/sample_submission.csv')

print(sim.head())
id = np.arange(0, results.shape[0])
save = pd.DataFrame({'id':id,

                     'label':results})

print(save.head())
save.to_csv('submission.csv', index=False)