from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

from keras.utils import plot_model

from keras.models import Model

from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, Input

from keras.layers.merge import concatenate

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



TRAIN_DIR = '../input/digit-recognizer/train.csv'

TEST_DIR = '../input/digit-recognizer/test.csv'

IMG_SIZE = 28

CHANNELS = 1
train_data = pd.read_csv(TRAIN_DIR)

test_data = pd.read_csv(TEST_DIR)
X_train = train_data.drop('label', axis=1)

Y_train = train_data['label'].copy()



X_train = X_train / 255.0

test_data = test_data / 255.0



# Need to reshape images to be 2D array of 28 x 28, not flat 728

X_train = X_train.values.reshape(-1, IMG_SIZE, IMG_SIZE, CHANNELS)

test_data = test_data.values.reshape(-1, IMG_SIZE, IMG_SIZE, CHANNELS)



Y_train = to_categorical(Y_train, num_classes=10)



# Create validation data to test our model

x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.2)
# Define input layer

visible = Input(shape=x_train.shape[1:])



# FIRST INTERPRETATION MODEL

# 1 hidden layer

# 28x28

hidden_conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(visible)

# Resize to 14x14

hidden_maxpool1 = MaxPool2D(pool_size=(2, 2))(hidden_conv1)

# Drop 40%

hidden_drop1 = Dropout(0.4)(hidden_maxpool1)

hidden_flat1 = Flatten()(hidden_drop1)



# SECOND INTERPRETATION MDOEL

# 3 hidden layers

# 28x28

hidden_conv2 = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(visible)

# Resize to 14x14

hidden_maxpool2 = MaxPool2D(pool_size=(2, 2))(hidden_conv2)

# Drop 40%

hidden_drop5 = Dropout(0.4)(hidden_maxpool2)

# 14x14

hidden_conv4 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(hidden_drop5)

# Drop 40%

hidden_drop2 = Dropout(0.4)(hidden_conv4)

hidden_flat2 = Flatten()(hidden_drop2)



# Merge input models

hidden_merge = concatenate([hidden_flat1, hidden_flat2])



hidden_dense1 = Dense(512, activation='relu')(hidden_merge)

hidden_drop3 = Dropout(0.4)(hidden_dense1)



hidden_dense2 = Dense(512, activation='relu')(hidden_drop3)

hidden_drop4 = Dropout(0.4)(hidden_dense2)



# Output layer

output = Dense(10, activation='softmax')(hidden_drop4)



model = Model(inputs=visible, outputs=output)

model.compile(optimizer ='adam', loss = "categorical_crossentropy", metrics=["accuracy"])
print(model.summary())
plot_model(model)
history = model.fit(x_train, y_train, batch_size=32, epochs=8, validation_data=(x_val, y_val))
# Vizualize history

fig, ax = plt.subplots(2,1)



ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="Validation loss")

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_accuracy'], color='r', label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
predictions = model.predict(test_data)



submission=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": list(np.argmax(prediction) for prediction in predictions)})



submission.to_csv("submission.csv", index=False, header=True)