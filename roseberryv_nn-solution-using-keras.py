import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib inline

from keras.models import Sequential
from keras.layers import Dense, Lambda, Flatten
from keras.optimizers import RMSprop
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

print ("train shape: %s" % str(train.shape))
print ("test shape: %s" % str(test.shape))
x_train = train.iloc[:,1:].values.astype('float32')
y_train = to_categorical(train.iloc[:,0].astype('int32'))

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

print("Splitting input in train and validation sets...")
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.10)
print(x_train.shape)
print(y_train.shape)
print(x_validation.shape)
print(y_validation.shape)
x_test = test.values.astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
print(x_test.shape)
digit_index = 8
plt.imshow(x_train[digit_index].reshape(28,28), cmap=plt.get_cmap('gray'))
plt.title(y_train[digit_index])
mean_px = x_train.mean().astype(np.float32)
std_px = x_train.std().astype(np.float32)

def standardize(x):
    return (x-mean_px)/std_px
model = Sequential()

model.add(Lambda(standardize,input_shape=(28,28,1)))
model.add(Flatten())
model.add(Dense(units=50, activation='tanh'))
model.add(Dense(units=50, activation='tanh'))
model.add(Dense(units=10, activation='softmax'))

print("input shape ", model.input_shape)
print("output shape ", model.output_shape)

model.compile(
	loss='categorical_crossentropy',
	optimizer=RMSprop(lr=0.001),
	metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=32)
metrics = model.evaluate(x_validation, y_validation)
print(model.metrics_names)
print(metrics)
predictions = model.predict_classes(x_test)
submission = pd.DataFrame({
	"ImageId": list(range(1, len(predictions) + 1)),
	"Label": predictions
})
submission.to_csv("submission.csv", index=False, header=True)