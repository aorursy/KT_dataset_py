import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
from keras.models import Model
from keras.layers import Input,Dense, Dropout ,Lambda, Flatten, Conv2D, MaxPool2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam ,RMSprop
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
for dirname, _, filenames in os.walk('/kaggle/input/digit-recognizer'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
train.head()
print(train.shape)
test= pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
test.head()
print(test.shape)
y_train = train['label'].values.astype('int32')
y_train
X_train = train.drop(['label'],axis = 1).values.astype('float32')
X_train.shape
X_train
X_test = test.values.astype('float32')
X_test.shape
pd.Series(y_train).value_counts().plot(kind='bar')
plt.show()
## Reshape the data into image format 
n_images = X_train.shape[0]
pixel_size = 28

X_train = X_train.reshape((n_images,pixel_size,pixel_size))

## Plot random images
for i in range(100,105):
    plt.subplot(550 + (i+1))
    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
    plt.title(y_train[i]);
## Add one more dimension for color 
X_train = X_train.reshape((n_images,pixel_size,pixel_size,1))
X_test = X_test.reshape((X_test.shape[0],pixel_size,pixel_size,1))
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
X_train  = X_train/255
X_test = X_test/255
y_train= to_categorical(y_train)
print(f"Number of classes {y_train.shape[1]}")

X = X_train
y = y_train
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=42)
def CNN(lr = 0.075):
    optimizer = Adam(lr=lr)
    inputs = Input((28,28,1))
    X = Conv2D(32,(5,5), activation='relu')(inputs)
    X = BatchNormalization()(X)
    X = Conv2D(32,(5,5), activation='relu')(X)
    X = MaxPool2D()(X)
    X = Dropout(0.25)(X)
    X = BatchNormalization()(X)
    X = Conv2D(64,(3,3), activation='relu')(X)
    X = BatchNormalization()(X)
    X = Conv2D(64,(3,3), activation='relu')(X)
    X = MaxPool2D()(X)
    X = Dropout(0.25)(X)
    X = Flatten()(X)
    X = BatchNormalization()(X)
    X = Dense(512, activation='relu')(X)
    X = Dropout(0.5)(X)
    X = BatchNormalization()(X)
    X = Dense(10, activation='softmax')(X)
    model = Model(inputs = inputs, outputs = X)
    model.compile(optimizer = optimizer, loss='categorical_crossentropy',metrics=['accuracy'])
    return model
batch_size = 128
epochs = 20
model = CNN()
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_val,y_val))
score = model.evaluate(X_val,y_val, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save("model.h5")
datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=10,
    zoom_range = 0.1,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=False,
    vertical_flip=False)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(X_train)

# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(X_train, y_train, batch_size = batch_size),
                    steps_per_epoch=len(y_train) / batch_size, 
                    epochs = epochs, 
                    verbose = 2,
                    validation_data = (X_val,y_val))
model.save("model_aug.h5")
predictions = model.predict(X_test, verbose=0)
print(predictions)

y_classes = predictions.argmax(axis=-1)
print(y_classes)
submission = pd.DataFrame({"ImageId": list(range(1,len(y_classes)+1)),
                         "Label": y_classes})
submission.to_csv("submission_augumented.csv", index=False)
submission.shape
