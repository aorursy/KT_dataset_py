import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.optimizers import RMSprop
train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")
sns.countplot(train['label'])
x_train = train.drop('label',axis=1)

y_train = train['label']
train.isnull().sum()
x_train = x_train / 255.0

test = test / 255.0
x_train = x_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
y_train = to_categorical(y_train, num_classes = 10)
X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size = 0.1, random_state=42)


g = plt.imshow(X_train[0][:,:,0])
from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.callbacks import ReduceLROnPlateau

from keras.preprocessing.image import ImageDataGenerator
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu'))

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
model.compile(optimizer =  RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0) , loss = "categorical_crossentropy", metrics=["accuracy"])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
epochs = 1

batch_size = 86
datagen = ImageDataGenerator(

        featurewise_center=False,

        samplewise_center=False,

        featurewise_std_normalization=False,

        samplewise_std_normalization=False,

        zca_whitening=False,

        rotation_range=10,

        zoom_range = 0.1,

        width_shift_range=0.1,

        height_shift_range=0.1,

        horizontal_flip=False,

        vertical_flip=False)



datagen.fit(X_train)
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_test,Y_test),

                              verbose = 2, steps_per_epoch=X_train.shape[0]

                              , callbacks=[learning_rate_reduction])
results = model.predict(test)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_mnist_datagen.csv",index=False)