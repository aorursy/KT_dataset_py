# import csv

# import numpy as np



# def transfer_to_array(file_name, with_label=True):

#     print('Start reading {0}.'.format(file_name))

#     # Check row count.

#     with open(file_name) as file:

#         cnt = sum(1 for line in file)



#     # Initialize variables.

#     labels = np.zeros((cnt-1))

#     data = np.zeros((cnt-1, 28, 28, 1))

#     if with_label:

#         label_bit = 1

#     else:

#         label_bit = 0

#     # Transfer to array. (cnt, 28, 28, 1)

#     with open(file_name, newline='') as csv_file:

#         rows = csv.reader(csv_file)

#         is_first_row = True

#         i = 0

#         for row in rows:

#             if not is_first_row:

#                 if with_label:

#                     labels[i] = np.int(row[0])

#                 tmp = np.array(row[label_bit:784+label_bit], dtype=np.int).reshape(28, 28)

#                 tmp = np.expand_dims(tmp, axis=3)

#                 data[i] = tmp

#                 i += 1

#             else:

#                 is_first_row = False



#     print('Reading completed.')



#     # Show image.

#     # img = image.array_to_img(data[3])

#     # img.show()



#     return data, labels





# x_train, y_train = transfer_to_array('07.Kaggle_train.csv', with_label=True)

# x_test, y_test = transfer_to_array('07.Kaggle_test.csv', with_label=False)  # y_test will not be used.

# print('Saving...')

# np.savez_compressed('07.Kaggle_digit-recognizer.npz', x_train=x_train, y_train=y_train, x_test=x_test)

# print('File: 07.Kaggle_digit-recognizer.npz Saved completed.')
import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, AveragePooling2D

from keras.utils import np_utils

from sklearn.model_selection import train_test_split

import numpy as np

import csv



## Settings in training phase.

batch_size = 256

num_classes = 10

epochs = 100



## Loads MNIST datasets(Kaggle).

MNIST_dataset = np.load('/kaggle/input/kaggle-digitrecognizer/07.Kaggle_digit-recognizer.npz')

mnist_x, mnist_y = MNIST_dataset['x_train'], MNIST_dataset['y_train']



## Split the MNIST dataset into training set and developmnet set.

x_train, x_dev, y_train, y_dev = train_test_split(mnist_x, mnist_y, test_size=0.05, random_state=1)

print('Size of training set:  {0}'.format(x_train.shape[0]))

print('Size of dev. set:      {0}'.format(x_dev.shape[0]))



## Size of input images.

img_x, img_y = 28, 28

input_shape = (28, 28, 1)



## Transfer y into one-hot vectors.

y_train = np_utils.to_categorical(y_train, num_classes=num_classes)

y_dev = np_utils.to_categorical(y_dev, num_classes=num_classes)



## Normalize the input images' value.

x_train = x_train.astype('float32')

x_train /= 255
## Initialize model.

model = Sequential()

## LeNet-5

 # Note: The shape of input images were changed to be (28x28), instead of (32x32).

model.add(Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=input_shape))

model.add(AveragePooling2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))

model.add(AveragePooling2D(pool_size=(2, 2), strides=2))



model.add(Flatten())

model.add(Dropout(0.2))

model.add(Dense(units=120, activation='relu'))

model.add(Dropout(0.1))

model.add(Dense(units=84, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,

                optimizer=keras.optimizers.Adadelta(),

                metrics=['accuracy'])



## Show model summary.

print(model.summary())
## Start training model.

train_history = model.fit(x_train, y_train, epochs=epochs,

                            batch_size=batch_size, verbose=2, validation_data=(x_dev, y_dev))
## Read the test set.

x_test = MNIST_dataset['x_test']



## Predict the x_test.

predictions = model.predict_classes(x_test)

print('Prediction completed.')
## Save as CSV for submission.

with open('07.Kaggle_submission.csv', 'w', newline='') as csv_file:

    print('Saving file...')

    csv_writer = csv.writer(csv_file, delimiter=',')



    # Define column name.

    csv_writer.writerow(['ImageId', 'Label'])

    for i in range(len(predictions)):

        csv_writer.writerow([i+1, predictions[i]])



    print('File: 07.Kaggle_submission.csv Saved completed.')