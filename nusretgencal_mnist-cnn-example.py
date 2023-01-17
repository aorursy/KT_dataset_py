# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import warnings

warnings.filterwarnings("ignore")



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from keras.layers import Conv2D, LeakyReLU, Dense, Flatten, Dropout, MaxPool2D # Layers

from keras.models import Sequential # Sequential Model

from keras.optimizers import Adam,RMSprop # optimizer

from keras.preprocessing.image import ImageDataGenerator # data generator

from keras.callbacks import ReduceLROnPlateau 
train = pd.read_csv("../input/mnist-in-csv/mnist_train.csv")

test = pd.read_csv("../input/mnist-in-csv/mnist_test.csv")

print("\n",train.info())

print(test.info())
train.head()
print(train.shape)

train.describe()
x_train = train.drop(['label'], axis=1) # x_train'e label dışındaki tüm px değerlerimi alıyorum

y_train = train.label # label değeri yani sayının değerini 

x_test = test.drop(['label'], axis=1) # x_train ile aynı şekilde 

y_test = test.label # y_train ile aynı şekilde 
print("x_train shape before reshape:", x_train.shape)

print("x_test shape before reshape:", x_test.shape)

print("y_train shape before reshape:", y_train.shape)

print("y_test shape before reshape:", y_test.shape)



x_train = np.array(x_train).reshape(-1, 28, 28, 1)

x_test = np.array(x_test).reshape(-1, 28, 28, 1)
from keras.utils.np_utils import to_categorical # one-hot-encoding'a çevirmek için 

y_train = to_categorical(y_train, num_classes = 10) # label encoding 

y_test = to_categorical(y_test, num_classes = 10)
print("x_train shape after reshape:", x_train.shape)

print("x_test shape after reshape:", x_test.shape)

print("y_train shape after reshape:", y_train.shape)

print("y_test shape after reshape:", y_test.shape)
x_train = x_train/255.0

x_test = x_test/255.0
epochs = 30

batch_size = 250

model = Sequential()



# Block 1

model.add(Conv2D(32,3, padding  ="same", input_shape=(28,28,1)))

model.add(LeakyReLU())

model.add(Conv2D(32,3, padding  ="same"))

model.add(LeakyReLU())

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



#Block 2

model.add(Conv2D(64,3, padding  ="same"))

model.add(LeakyReLU())

model.add(Conv2D(64,3, padding  ="same"))

model.add(LeakyReLU())

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



#Flatten Block

model.add(Flatten())



#Output Block

model.add(Dense(256,activation='relu'))

model.add(Dropout(rate=0.25))

model.add(Dense(32,activation='relu'))

model.add(Dropout(rate=0.25))

model.add(Dense(10,activation="softmax"))

model.summary()
# kullanılabilir optimizer featureları

"""

sgd = keras.optimizers.SGD(lr=1e-4, momentum=0.9)

rms_prop = keras.optimizers.RMSprop(lr=1e-4)

adam = keras.optimizers.adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08) 

adamax = keras.optimizers.Adamax(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)

adadelta = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)



learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.0001) """



adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

#rms_prop = RMSprop(lr=0.001)

learning_rate_reduce = ReduceLROnPlateau(monitor='val_accuracy', 

                                            patience=2, 

                                            verbose=1, 

                                            factor=0.4, 

                                            min_lr=0.00001)

loss = "categorical_crossentropy"
model.compile( optimizer= adam, loss=loss ,metrics=['accuracy'])
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
history = model.fit_generator(datagen.flow(x_train,y_train, batch_size = batch_size),

                              epochs = epochs, validation_data = (x_test, y_test),

                              steps_per_epoch = x_train.shape[0] // batch_size,

                              callbacks=[learning_rate_reduce])

result = model.evaluate(x = x_train, y = y_train)

print('Accuracy:', result[1])
plt.plot(history.history['loss'])

plt.title("Loss Plot", fontsize = 15)

plt.xlabel("Epochs", fontsize = 12)

plt.ylabel("Loss", fontsize = 12)

plt.grid(alpha=0.3)

plt.legend(["Train", "Test"])

plt.show()



plt.plot(history.history["accuracy"])

plt.title("Accuracy Plot")

plt.xlabel("Epochs")

plt.ylabel("Accuracy", fontsize = 12)

plt.grid(alpha=0.3)

plt.legend(["Train","Test"])

plt.show()
import seaborn as sns

from sklearn.metrics import confusion_matrix



y_pred = model.predict(x_test)



# Convert predictions classes to one hot vectors 

y_pred_classes = np.argmax(y_pred,axis = 1) 



# Convert validation observations to one hot vectors

y_true = np.argmax(y_test,axis = 1) 



# compute the confusion matrix

cm = confusion_matrix(y_true, y_pred_classes) 



# plot the confusion matrix

f,ax = plt.subplots(figsize=(8, 8))



sns.heatmap(cm, annot = True, linewidths = 0.01, cmap = "Greens", linecolor = "gray", fmt = '.1f',ax = ax)



plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.title("Confusion Matrix")



plt.show()
test_y = np.argmax(model.predict(x_test),axis =1)
df_submission = pd.DataFrame([test.index+1,test_y],["ImageId","Label"]).transpose()

df_submission.to_csv("submission.csv",index=False)