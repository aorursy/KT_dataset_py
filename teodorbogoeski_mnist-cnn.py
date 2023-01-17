from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
# import the files

train_data = pd.read_csv('../input/digit-recognizer/train.csv')

test_data = pd.read_csv('../input/digit-recognizer/test.csv')
# set the train data and the labels

X = train_data

y = train_data.label

X.drop(['label'], axis=1, inplace=True) # drop

X = X / 255 # normalize

X = X.values.reshape(-1,28,28,1) # reshape 28x28



X_test = test_data 

X_test = X_test / 255 # normalize

X_test = X_test.values.reshape(-1,28,28,1) # reshape
# split the train data

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=0)
print(X_train.shape)

print(X_valid.shape)
# show 28x28 image

plt.imshow(X_train[0][:,:,0])
# set up the model



model = Sequential()



model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=(28,28,1)))

model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=(28,28,1)))



model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(256, activation="relu"))

model.add(Dropout(0.3))

model.add(Dense(10, activation="softmax"))



model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=["accuracy"])
# fit and evaluate

model.fit(X_train, y_train, epochs=10)

model.evaluate(X_valid, y_valid)
# predict and submit

preds = model.predict(X_test)

final_preds = np.argmax(preds, axis=1)



output = pd.DataFrame({'ImageId': range(1,len(X_test)+1), 'Label': final_preds})

output.to_csv("submission.csv", index=False)