import numpy as np

import pandas as pd

import keras

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
data = pd.read_csv("../input/digit-recognizer/train.csv")

y_train = data["label"]

X_train = data.drop("label",axis=1)



X_train = X_train.values.reshape(-1,28,28,1)

y_train = to_categorical(y_train)



print(y_train.shape,X_train.shape)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.1, random_state=41)
model = keras.Sequential()



model.add(keras.layers.Conv2D(64, kernel_size=5, activation='relu', input_shape=(28,28,1)))

model.add(keras.layers.MaxPooling2D())

model.add(keras.layers.Conv2D(64, kernel_size=5, activation='relu',padding="same"))

model.add(keras.layers.MaxPooling2D())

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Conv2D(32, kernel_size=5, activation='relu'))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(256, activation="relu"))

model.add(keras.layers.Dense(10, activation='softmax'))



#compile model 

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])





#train model

model.fit(X_train, y_train,validation_data=(X_test, y_test), epochs=2)
X_submission = pd.read_csv("../input/digit-recognizer/test.csv")

X_submission = X_submission.values.reshape(-1,28,28,1)



softmax_output = model.predict(X_submission)

final_results = np.argmax(softmax_output, axis=1)

final_results = pd.Series(final_results,name="Label")



final_submission = pd.concat([pd.Series(range(1,len(final_results)+1),name = "ImageId"),final_results],axis = 1)



final_submission.to_csv("Digit_Recognizer.csv", index=False)