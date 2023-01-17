import numpy as np
import pandas as pd
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout
img_rows, img_cols = 28, 28
num_classes = 10
digit_train_file = "../input/train.csv"
raw_train_data = pd.read_csv(digit_train_file)
digit_test_file = "../input/test.csv"
raw_test_data = pd.read_csv(digit_test_file)
def data_prep(raw):
    out_y = keras.utils.to_categorical(raw.label, num_classes)

    num_images = raw.shape[0]
    x_as_array = raw.values[:,1:]
    x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1)
    out_x = x_shaped_array / 255
    return out_x, out_y

train_x, train_y = data_prep(raw_train_data)
digit_model_1 = Sequential()
digit_model_1.add(Conv2D(30, kernel_size=(3, 3), 
                         strides=2, 
                         activation='relu', 
                         input_shape=(img_rows, img_cols, 1)))
digit_model_1.add(Conv2D(30, kernel_size=(3, 3), activation='relu'))
digit_model_1.add(Conv2D(30, kernel_size=(3, 3), activation='relu'))
digit_model_1.add(Conv2D(30, kernel_size=(3, 3), activation='relu'))
digit_model_1.add(Flatten())
digit_model_1.add(Dense(128, activation='relu'))
digit_model_1.add(Dense(num_classes, activation='softmax'))
digit_model_1.compile(loss=keras.losses.categorical_crossentropy, 
                      optimizer='adam', 
                      metrics=['accuracy'])

digit_model_1.fit(train_x, train_y, 
                  batch_size=128, 
                  epochs=4, 
                  validation_split = 0.2)
def test_prep(raw):
    num_images = raw.shape[0]
    x_as_array = raw.values[:,:]
    x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1)
    out_x = x_shaped_array / 255
    return out_x

test_array = test_prep(raw_test_data)
predictions = digit_model_1.predict(test_array)

predicted_labels = np.argmax(predictions, axis=1)

submit_df = pd.DataFrame()
submit_df['ImageId'] = np.arange(1,predicted_labels.shape[0]+1)
submit_df['Label'] = predicted_labels
# submit_df.to_csv('../input/submission.csv',index=False)