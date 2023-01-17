from sklearn.model_selection import train_test_split
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
img = np.squeeze(X[1])
plt.imshow(img, cmap='gray')
plt.show()
print(y[1])
# split the train data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=0)
from tensorflow.keras.utils import to_categorical

X_train_re = X_train.reshape(-1, 784)
X_valid_re = X_valid.reshape(-1, 784)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

model0 = Sequential()

model0.add(Dense(units=256, input_shape=(784,)))
model0.add(Activation('relu'))
model0.add(Dense(units=100))
model0.add(Activation('relu'))
model0.add(Dense(units=10))
model0.add(Activation('softmax'))

model0.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model0.summary()
print(X_train_re.shape)
print(y_train.shape)
print(X_valid_re.shape)
print(y_valid.shape)
y_train = to_categorical(y_train, 10)
y_valid = to_categorical(y_valid, 10)
model0.fit(X_train_re, y_train, batch_size=1000, epochs=10, verbose=1)
model0.evaluate(X_valid_re, y_valid)
# predict and submit
X_test_re = X_test.reshape(-1, 784)
preds = model0.predict(X_test_re)
final_preds = np.argmax(preds, axis=1)

output = pd.DataFrame({'ImageId': range(1,len(X_test)+1), 'Label': final_preds})
output.to_csv("submission.csv", index=False)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
# set up the model

model1 = Sequential()

model1.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=(28,28,1)))
model1.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=(28,28,1)))

model1.add(MaxPool2D(pool_size=(2,2)))
model1.add(Dropout(0.3))
model1.add(Flatten())
model1.add(Dense(256, activation="relu"))
model1.add(Dropout(0.3))
model1.add(Dense(10, activation="softmax"))

model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=["accuracy"])
model1.summary()
# split the train data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=0)
print(X_train.shape)
print(X_valid.shape)
print(y_train.shape)
print(y_valid.shape)
# fit and evaluate
model1.fit(X_train, y_train, epochs=10)
model1.evaluate(X_valid, y_valid)
# predict and submit
preds = model1.predict(X_test)
final_preds = np.argmax(preds, axis=1)

output = pd.DataFrame({'ImageId': range(1,len(X_test)+1), 'Label': final_preds})
output.to_csv("submission.csv", index=False)
