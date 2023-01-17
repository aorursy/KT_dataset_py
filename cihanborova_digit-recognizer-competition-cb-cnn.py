import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")

train = pd.read_csv("../input/digit-recognizer/train.csv")
train.head()
# put labels into y_train vaiable

y_train = train["label"]

# Drop "label" column

x_train = train.drop(labels = ["label"] , axis = 1)

#visualize number of digits classes

plt.figure(figsize=(20,10))

sns.countplot(y_train,palette="icefire")

plt.title("Number of digit classes")

#y_train.value_counts()
# plot some samples

img = x_train.iloc[2].as_matrix()

img = img.reshape((28,28))

plt.imshow(img,cmap="gray")

plt.title(train.iloc[2,0])

plt.axis("off")

plt.show()
x_train = x_train / 255.0

test = test / 255.0

print("x_train shape: " ,x_train.shape)

print("test shape: " , test.shape)
x_train = x_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)

print("x_train shape: " ,x_train.shape)

print("test shape: " , test.shape)


from keras.utils.np_utils import to_categorical # Convert to one-hot-encoding

y_train = to_categorical(y_train , num_classes=10)

from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(x_train,y_train,test_size = 0.15,random_state = 1)
print("x_train shape: " ,x_train.shape)

print("x_test shape: " ,x_test.shape)

print("y_train shape: " ,y_train.shape)

print("y_test shape: " ,y_test.shape)
from sklearn.metrics import confusion_matrix

import itertools

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D

from keras.optimizers import RMSprop,Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau
model = Sequential()
model.add(Conv2D(filters = 8 ,kernel_size = (5,5),padding = "Same"

                ,activation = "relu",input_shape = (28,28,1)))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Conv2D(filters = 16,kernel_size=(3,3),padding="Same"

                ,activation="relu"))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256,activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10,activation="softmax"))
optimizer = Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999)
model.compile(optimizer=optimizer, loss="categorical_crossentropy",metrics=["accuracy"])
epochs = 20 # for better result incrrease the epochs

batch_size = 250
datagen = ImageDataGenerator(

featurewise_center=False # set input mean to 0 over the dataset

,samplewise_center=False # set each sample mean to 0

,featurewise_std_normalization=False # divide inputs by std of the dataset

,samplewise_std_normalization=False  # divide each input by its std

,zca_whitening=False # dimension reduction

,rotation_range=0.5 # randomly rotate images in the range 5 degrees

,zoom_range=0.5 # Randomly zoom image %15

,width_shift_range=0.5 #randomly shiftimages horizontally %5

,height_shift_range=0.5 #randomly shiftimages vertically %5

,horizontal_flip=False # randomly flip images

,vertical_flip=False) # randomly flip images
datagen.fit(x_train)
history = model.fit_generator(datagen.flow(x_train,y_train,batch_size=batch_size),

                             epochs=epochs,validation_data=(x_test,y_test),

                             steps_per_epoch=x_train.shape[0] // batch_size)
plt.plot(history.history["val_loss"],color = "b" , label = "Validation Loss")

plt.title("Test Loss")

plt.xlabel("Number of Epochs")

plt.ylabel("Loss")

plt.legend()

plt.show()