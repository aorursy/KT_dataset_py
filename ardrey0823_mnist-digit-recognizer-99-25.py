import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, BatchNormalization

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.losses import categorical_crossentropy

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import EarlyStopping
train_path = '../input/digit-recognizer/train.csv'

test_path = '../input/digit-recognizer/test.csv'
train_data = pd.read_csv(train_path)

test_data = pd.read_csv(test_path)
train_data.head()
img_rows, img_cols = 28, 28

num_classes = train_data.label.nunique()
def data_prep(raw_data, img_row, img_col, num_classes):

    # One hot encode the label data

    out_y = to_categorical(raw_data.label, num_classes)

    num_images = raw_data.shape[0]

    x_as_array = raw_data.values[:, 1:]

    x_as_matrix = x_as_array.reshape(num_images, img_row, img_col, 1)

    

    # Scale down the features between 0 and 1

    out_x = x_as_matrix/255.0

    

    return out_x, out_y
X, y = data_prep(train_data, img_rows, img_cols, num_classes)
# Creating the model

model = Sequential()
# Adding layers to the model



model.add(Conv2D(filters=40, kernel_size=(3,3), padding='Same', activation='relu', input_shape=(img_rows, img_cols, 1)))

model.add(MaxPool2D(2,2))

model.add(BatchNormalization(axis=1))



model.add(Conv2D(filters=60, kernel_size=(3,3), padding='Same', activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(BatchNormalization(axis=1))

model.add(Dropout(0.25))



model.add(Conv2D(filters=80, kernel_size=(3,3), padding='Same', activation='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(BatchNormalization(axis=1))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(BatchNormalization(axis=1))



model.add(Dense(256, activation='relu'))

model.add(BatchNormalization(axis=1))



model.add(Dropout(0.50))

model.add(BatchNormalization(axis=1))

model.add(Dense(10, activation='softmax'))

model.summary()
# Specifying model compilation parameters



model.compile(loss=categorical_crossentropy,

              optimizer=Adam(0.001),

              metrics=['accuracy'])
# Create an EarlyStopping object to monitor the validation loss

# Stops the training of model if the validation loss stopped decreasing



#early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode='auto')
# Training the model



history = model.fit(X, y, epochs=20, validation_split=0.2)
# Prepare test data

test_num_img = test_data.shape[0]

test_data_array = test_data.values

test_data_matrix = test_data_array.reshape(test_num_img, 28,28,1)

test_data_matrix = test_data_matrix/255
preds = model.predict_classes(test_data_matrix)
preds[0]
plt.imshow(test_data_matrix[0].reshape(28,28))

plt.title('Predicted value = {}'.format(preds[0]))
output = pd.DataFrame({'ImageId': list(range(1, len(preds)+1)),

                       'Label':preds})

output.to_csv('output2.csv', index=False)