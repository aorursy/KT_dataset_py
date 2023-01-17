import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn import model_selection as ms
%pylab inline
# Import train data

x = pd.read_csv('../input/train.csv')
y = x.label
x.drop('label', axis=1, inplace=True)
print(x.shape)
x.head()
# Save 20% of data for validation. 
# We can't use simple cross-validation because train set will be used for generating new images

x_train, x_test, y_train, y_test = ms.train_test_split(x.values, y, test_size=0.2)
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

# Convert targets to categorical type
from keras import utils
y_train = utils.to_categorical(y_train, num_classes=10)
y_test  = utils.to_categorical(y_test , num_classes=10)
# Generator of new images by randomly shifting original ones

from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1)
datagen.fit(x_train)
# Create Keras CNN model
# Commented rows are of original model that require to match time for training on keras

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, Activation, MaxPool2D, Flatten, Dense
from keras.optimizers import RMSprop

model = Sequential()

model.add(Conv2D(filters = 8, kernel_size = (5,5), padding = 'Same', activation ='relu', input_shape = (28,28,1)))
#model.add(Conv2D(filters = 16, kernel_size = (5,5),padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same', activation ='relu'))
#model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(200, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

model.compile(optimizer = 'adam' , loss = "categorical_crossentropy", metrics=["accuracy"])
# Train the model

model.fit_generator(datagen.flow(x_train, y_train,
                        batch_size = 128),
                        epochs = 3,
                        samples_per_epoch = x_train.shape[0],
                        validation_data = (x_test, y_test),
                        verbose = 1,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=1)])
# Import test data

x_submit = pd.read_csv('../input/test.csv')
x_submit.head()
# Normalize data and predict test

x_submit = x_submit.values/255
x_submit = x_submit.reshape(-1,28,28,1)
y_submit = model.predict_classes(x_submit)
submit_df = pd.DataFrame({'ImageId':range(1, y_submit.shape[0]+1), 'Label':y_submit})
submit_df.to_csv('submit_data.csv',index=False)