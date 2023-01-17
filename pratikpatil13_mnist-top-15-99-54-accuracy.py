# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
train.isnull().sum()
X=train.iloc[:,:-1].values
X
Y=train['label'].values
g = sns.countplot(Y)
X=X/255.0
test=test/255.0
X = X.reshape(-1,28,28,1)
test=test.values.reshape(-1,28,28,1)
Y=Y.reshape(-1,1)
Y
from keras.utils.np_utils import to_categorical
Y = to_categorical(Y, num_classes = 10)
from sklearn.model_selection import train_test_split
# Split the train and the validation set for the fitting
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.05,random_state=90)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.05, random_state=90)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.optimizers import SGD, Adam
from keras.regularizers import l2

model = Sequential()


model.add(Convolution2D(32, (3, 3), input_shape=(28,28,1),activation='relu',padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(Convolution2D(32, (3, 3), activation='relu',padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))   # reduces to 16x16x3xnum_filters
model.add(Dropout(0.2))

model.add(Convolution2D(2*48, (3, 3), activation='relu',padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(Convolution2D(2*48, (3, 3), activation='relu',padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))   # reduces to 4x4x3x(4*num_filters)
model.add(Dropout(0.2))


model.add(Convolution2D(64, (3, 3), activation='relu',padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(Convolution2D(64, (3, 3), activation='relu',padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))   # reduces to 4x4x3x(4*num_filters)
model.add(Dropout(0.2))


model.add(Convolution2D(128, (3, 3), activation='relu',padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(Convolution2D(128, (3, 3), activation='relu',padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))   # reduces to 4x4x3x(4*num_filters)
model.add(Dropout(0.2))

model.add(Convolution2D(256, (3, 3), activation='relu',padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(Convolution2D(256, (3, 3), activation='relu',padding='same'))
model.add(BatchNormalization(axis=-1))
# reduces to 4x4x3x(4*num_filters)
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(units=256,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(10, activation = "softmax"))
model.summary()



# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten
# from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
# from keras.optimizers import SGD, Adam
# from keras.regularizers import l2

# model = Sequential()


# model.add(Convolution2D(32, (3, 3), input_shape=(28,28,1),activation='relu',padding='same'))
# model.add(BatchNormalization(axis=-1))
# model.add(Convolution2D(32, (3, 3), activation='relu',padding='same'))
# model.add(BatchNormalization(axis=-1))
# model.add(MaxPooling2D(pool_size=(2, 2)))   # reduces to 16x16x3xnum_filters
# model.add(Dropout(0.2))

# model.add(Convolution2D(2*32, (3, 3), activation='relu',padding='same'))
# model.add(BatchNormalization(axis=-1))
# model.add(Convolution2D(2*32, (3, 3), activation='relu',padding='same'))
# model.add(BatchNormalization(axis=-1))
# model.add(MaxPooling2D(pool_size=(2, 2)))   # reduces to 4x4x3x(4*num_filters)
# model.add(Dropout(0.2))


# model.add(Convolution2D(4*32, (3, 3), activation='relu',padding='same'))
# model.add(BatchNormalization(axis=-1))
# model.add(Convolution2D(4*32, (3, 3), activation='relu',padding='same'))
# model.add(BatchNormalization(axis=-1))
# model.add(MaxPooling2D(pool_size=(2, 2)))   # reduces to 4x4x3x(4*num_filters)
# model.add(Dropout(0.2))


# model.add(Convolution2D(6*32, (3, 3), activation='relu',padding='same'))
# model.add(BatchNormalization(axis=-1))
# model.add(Convolution2D(6*32, (3, 3), activation='relu',padding='same'))
# model.add(BatchNormalization(axis=-1))
# model.add(MaxPooling2D(pool_size=(2, 2)))   # reduces to 4x4x3x(4*num_filters)
# model.add(Dropout(0.2))

# model.add(Convolution2D(8*32, (3, 3), activation='relu',padding='same'))
# model.add(BatchNormalization(axis=-1))
#   # reduces to 4x4x3x(4*num_filters)
# model.add(Dropout(0.2))

# model.add(Flatten())
# model.add(Dense(units=256,activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.3))
# model.add(Dense(10, activation = "softmax"))
# model.summary()

model.compile(optimizer = 'adam' , loss = "categorical_crossentropy", metrics=["accuracy"])
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


datagen.fit(x_train)
# Fit the model
history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=32),
                              epochs = 50, validation_data = (x_val,y_val),
                              verbose = 1)
val_loss = history.history['val_loss']
loss = history.history['loss']

plt.plot(val_loss)
plt.plot(loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Val error','Train error'], loc='upper right')
plt.savefig('plot_error.png')
plt.show()
val_accuracy = history.history['val_accuracy']
accuracy = history.history['accuracy']

plt.plot(val_accuracy)
plt.plot(accuracy)
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend(['Val accuracy','Train accuracy'], loc='upper right')
plt.savefig( 'plot_accuracy.png')
plt.show()
score=model.evaluate(x_test,y_test,verbose=1)               #evaluates the model
accuracy=100*score[1]                                       
print('Test accuracy is %.4f%%' % accuracy)                 #returns the accuracy of the model 
score=model.evaluate(x_val,y_val,verbose=1)               #evaluates the model
accuracy=100*score[1]                                       
print('Test accuracy is %.4f%%' % accuracy)                 #returns the accuracy of the model 
pred = model.predict(test)
pred = pd.DataFrame(pred)
pred['Label'] = pred.idxmax(axis=1)
pred.head(5)
pred['index'] = list(range(1,len(pred)+1))
pred.head()
submission = pred[['index','Label']]
submission.head()
submission.rename(columns={'index':'ImageId'},inplace = True)
submission.head()
submission.to_csv('submission.csv',index=False)
pd.read_csv('submission.csv')

