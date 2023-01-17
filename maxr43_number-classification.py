import numpy as np 

import pandas as pd 

from sklearn.model_selection import train_test_split



from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import Adam
training_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')



images = training_data.drop('label', axis=1)
model = Sequential()



model.add(Conv2D(filters = 128, kernel_size = (2,2),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 128, kernel_size = (2,2),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))





model.add(Conv2D(filters = 256, kernel_size = (1,1),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 256, kernel_size = (1,1),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.125))





model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.25))

model.add(Dense(10, activation = "softmax"))



optimizer = Adam(lr=0.001)



model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
x_train = []

for _, image in images.iterrows():

    pixels_array = image.to_numpy()

    three_dimensional_image = np.reshape(pixels_array, (28,28,1))

    x_train.append(three_dimensional_image)

    

x_train = np.array(x_train)

y_train = np.array(training_data['label'])

y_train = to_categorical(y_train, num_classes=10)



y_train.shape
model.fit(x_train, y_train, epochs=1, verbose=1, validation_split=0.2)
testing_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

tests = []

for _, image in testing_data.iterrows():

    pixels_array = image.to_numpy()

    three_dimensional_image = np.reshape(pixels_array, (28,28,1))

    tests.append(three_dimensional_image)

    

tests = np.array(tests)



tests.shape
results = model.predict(tests)

results = np.argmax(results, axis=1).astype(int)

results = pd.Series(results, name='Label')

image_id = pd.Series(range(1,280001), name='ImageId')

submission = pd.concat([image_id, results], axis=1).astype(int)

submission.to_csv('competiton_submission.csv', index=False)