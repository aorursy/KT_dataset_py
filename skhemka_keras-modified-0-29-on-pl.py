import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
train_df = pd.read_json("../input/train.json")

test_df = pd.read_json("../input/test.json")
# Train data

x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train_df["band_1"]])

x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train_df["band_2"]])

X_train = np.concatenate([x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis]], axis=-1)

y_train = np.array(train_df["is_iceberg"])

print("Xtrain:", X_train.shape)

print("Ytrain:", y_train.shape)

# Test data

x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test_df["band_1"]])

x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test_df["band_2"]])

X_test = np.concatenate([x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis]], axis=-1)

print("Xtest:", X_test.shape)
from keras.models import Sequential

from keras.layers import Convolution2D, GlobalAveragePooling2D, Dense, Dropout, MaxPooling2D
from keras.models import Sequential

from keras.layers import Conv2D, BatchNormalization, Dropout, MaxPooling2D, GlobalMaxPooling2D, Dense

simple_cnn = Sequential()

simple_cnn.add(BatchNormalization(input_shape = (75, 75, 2)))

for i in range(4):

    simple_cnn.add(Conv2D(8*2**i, kernel_size = (3,3)))

    simple_cnn.add(MaxPooling2D((2,2)))

simple_cnn.add(GlobalMaxPooling2D())

simple_cnn.add(Dropout(0.2))

simple_cnn.add(Dense(64))

simple_cnn.add(Dropout(0.2))

simple_cnn.add(Dense(32))

simple_cnn.add(Dense(1, activation = 'sigmoid'))

simple_cnn.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

simple_cnn.summary()
simple_cnn.fit(X_train, y_train, validation_split=0.2, epochs = 10)

# Make predictions

prediction = simple_cnn.predict(X_test, verbose=1)
submit_df = pd.DataFrame({'id': test_df["id"], 'is_iceberg': prediction.flatten()})

submit_df.to_csv("./naive_submission.csv", index=False)