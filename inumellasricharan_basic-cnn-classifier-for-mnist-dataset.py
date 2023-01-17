import tensorflow as tf

import pandas as pd



#separating the pixel columns from the labels

train_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv', delimiter = ',', header = 0, usecols = [x for x in range(0, 785)])



labels_df = train_data[['label']].copy()



train_data.drop(columns = ['label'], inplace = True)



test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv', delimiter = ',', header = 0, usecols = [x for x in range(0, 784)])



print("-----------------------------------------------PIXEL_DATA----------------------------------------------")



print(train_data.head())



print("-----------------------------------------------LABELS----------------------------------------------")



print(labels_df.head())
#extract the numpy array out of the respective dataframes and perform the necessary transformation

#each image is present as a single 1d array of 784 values, we need to transform it to an array of shape (28, 28, 1) which corresponds to (height, width, channels)



train_dataset = train_data.values

test_dataset = test_data.values



train_dataset = train_dataset.reshape(-1, 28, 28, 1)

test_dataset = test_dataset.reshape(-1, 28, 28, 1)



labels = labels_df.values



#normalizing the images is crucial in order to avoid exploding and vanishing gradient problems, it also helps in faster convergence



train_dataset = train_dataset/255.0

test_dataset = test_dataset/255.0
model = tf.keras.models.Sequential()





#28X28X1---->input dim

model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), strides = (1, 1), padding = 'valid'))

model.add(tf.keras.layers.Activation('relu'))



#(28-3)/1 + 1 = 26X26X32

model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), strides = (1, 1), padding = 'valid'))

model.add(tf.keras.layers.Activation('relu'))



#(26-3)/1 + 1 = 24X24X32

model.add(tf.keras.layers.MaxPool2D(pool_size = (2, 2)))

#model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = (2, 2), strides = (2, 2), padding = 'valid'))

model.add(tf.keras.layers.Activation('relu'))



#(24-2)/2 + 1 = 12X12X32

model.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'valid'))

model.add(tf.keras.layers.Activation('relu'))



#(12-3)/1 + 1 = 10X10X64

model.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'valid'))

model.add(tf.keras.layers.Activation('relu'))



#(10-3)/1 + 1 = 8X8X64

model.add(tf.keras.layers.MaxPool2D(pool_size = (2, 2)))

#model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = (2, 2), strides = (2, 2), padding = 'valid'))

model.add(tf.keras.layers.Activation('relu'))



#(8-2)/2 + 1 = 4X4X64

model.add(tf.keras.layers.Flatten())



#4X4X64 ----> 1024

model.add(tf.keras.layers.Dense(units = 512, activation = 'relu', kernel_initializer = 'glorot_uniform'))

model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Dense(units = 10, activation = 'softmax', kernel_initializer = 'glorot_uniform'))



model.build(input_shape = (None, 28, 28, 1))#None is used in place of batch_size since its a variable



model.summary()
from sklearn.model_selection import train_test_split



es_callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 15, restore_best_weights = True)

#note: es callback is optional, it is used to prevent overfitting. You can still eyeball the epoch where it will start overfitting and stop it there.



model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])



x_train, x_valid, y_train, y_valid = train_test_split(train_dataset, labels, train_size = 0.8, test_size = 0.2)



model.fit(x = x_train, y = y_train, validation_data = (x_valid, y_valid), batch_size = 32, validation_batch_size = 32, epochs = 30, callbacks = [es_callback])
predictions = model.predict(test_dataset)
import numpy as np



predictions = np.argmax(predictions, axis = 1)



predictions = predictions.reshape(-1, 1)



image_ids = np.array([i for i in range(1, 28001)])



image_ids = image_ids.reshape(-1, 1)









csv_content = np.concatenate((image_ids, predictions), axis = 1)



csv_data = pd.DataFrame(data = csv_content, columns = ['ImageId', 'Label'])



csv_data.to_csv(path_or_buf = 'submission.csv', sep = ',', index = False)