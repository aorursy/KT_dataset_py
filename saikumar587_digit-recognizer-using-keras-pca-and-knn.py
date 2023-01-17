import pandas as pd

import numpy as np 

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.model_selection import StratifiedKFold,KFold

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, accuracy_score
# Load train data

train = pd.read_csv("../input/train.csv")

print(train.shape)

train.head()
# test= pd.read_csv("../input/test.csv")

# print(test.shape)

# test.head()
# change data type to float

X_train_origi = (train.iloc[:,1:].values).astype('float32') 

y_train_origi = train.iloc[:,0].values.astype('int32') 

#X_test_origi = test.values.astype('float32')
# visualizing the first 10 images in the dataset and their labels

plt.figure(figsize=(10, 1))

for i in range(10):

    plt.subplot(1, 10, i+1)

    plt.imshow(X_train_origi[i].reshape(28, 28), cmap="gray")

    plt.axis('off')

plt.show()

print('label for each of the above image: %s' % (y_train_origi[0:10]))
from sklearn.decomposition import PCA



pca = PCA(n_components=2)

pca_result = pca.fit_transform(X_train_origi)



print(pca.explained_variance_ratio_)
print(X_train_origi.shape)

print(pca_result.shape)
plt.scatter(pca_result[:4000, 0], pca_result[:4000, 1], c=y_train_origi[:4000], cmap = "nipy_spectral", edgecolor = "None", s=5)

plt.colorbar()
pca = PCA(n_components=50)

X_train_transformed = pca.fit_transform(X_train_origi)
i=1

kf = KFold(n_splits=5,random_state=1,shuffle=True)

for train_index,test_index in kf.split(X_train_transformed):

    print('\n{} of kfold {}'.format(i,kf.n_splits))

    xtr,xvl = X_train_transformed[train_index],X_train_transformed[test_index]

    ytr,yvl = y_train_origi[train_index],y_train_origi[test_index]

    model_kc = KNeighborsClassifier(n_neighbors=7)

    model_kc.fit(xtr, ytr)

    pred_test = model_kc.predict(xvl)

    score = accuracy_score(yvl,pred_test)

    print('accuracy_score',score)

    i+=1
from sklearn.model_selection import train_test_split

from keras.models import Sequential,Model

from keras.utils.np_utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator



from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten,BatchNormalization,Input,Conv2D, MaxPool2D

from keras.layers import Conv2D, MaxPooling2D

from keras.optimizers import RMSprop

from keras.callbacks import ReduceLROnPlateau

from keras import backend as K

from keras.layers import Activation, Dense

from keras import optimizers

from keras.layers import Flatten

import tensorflow as tf
tf.compat.v1.get_default_graph

tf.compat.v1.placeholder

tf.compat.v1.train.Optimizer

tf.compat.v1.assign_add
# Normalize the data

X_train_norm = X_train_origi / 255.0

#X_test_norm = X_test_origi / 255.0
# reshape

X_train_rs = X_train_norm.reshape(-1,28,28,1)
# converting y data into categorical (one-hot encoding)

y_train_en = to_categorical(y_train_origi)
print(X_train_rs.shape, y_train_en.shape)
# Split the train and the validation set for the fitting

X_train, X_val, Y_train, Y_val = train_test_split(X_train_rs, y_train_en, test_size = 0.1, random_state=2)
# data augmentation can prevent overfitting



image_gen = ImageDataGenerator(

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

    vertical_flip=False)



#training the image preprocessing

image_gen.fit(X_train, augment=True)
for X_batch, y_batch in image_gen.flow(X_train, Y_train, batch_size=9):

    # create a grid of 3x3 images

    for i in range(0, 9):

        plt.subplot(330 + 1 + i)

        plt.imshow(X_batch[i].reshape(28, 28), cmap=plt.get_cmap('gray'))

    # show the plot

    plt.show()

    break
model = Sequential()

model.add(Conv2D(40, kernel_size=5, padding="same",input_shape=(28, 28, 1), activation = 'relu'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(70, kernel_size=3, padding="same", activation = 'relu'))

model.add(Conv2D(200, kernel_size=3, padding="same", activation = 'relu'))

model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))



model.add(Conv2D(512, kernel_size=3, padding="valid", activation = 'relu'))

model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))

model.add(Flatten())

model.add(Dense(units=100, activation='relu'  ))

model.add(Dropout(0.3))



model.add(Dense(10))

model.add(Activation("softmax"))
# Compile the model

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
model_history = model.fit_generator(image_gen.flow(X_train, Y_train, batch_size=86),validation_data = (X_val,Y_val),

                    steps_per_epoch=len(X_train) / 86, epochs = 10,verbose = 1)
# Plot the loss and accuracy curves for training and validation 

fig, ax = plt.subplots(2,1)

ax[0].plot(model_history.history['loss'], color='b', label="Training loss")

ax[0].plot(model_history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(model_history.history['acc'], color='b', label="Training accuracy")

ax[1].plot(model_history.history['val_acc'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)