import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from collections import Counter

from sklearn.metrics import confusion_matrix

import itertools

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

%matplotlib inline
#loading the dataset.......(Train)

train = pd.read_csv("../input/train.csv")

#print(train.shape)

#train.head()
#z_train = Counter(train['label'])

#z_train
#loading the dataset.......(Test)

test= pd.read_csv("../input/test.csv")

#print(test.shape)

#test.head()
x_train = (train.iloc[:,1:].values).astype('float32') # all pixel values

y_train = train.iloc[:,0].values.astype('int32') # only labels i.e targets digits

x_test = test.values.astype('float32')
X_train = x_train.reshape(42000,784).astype('float32')

X_test = x_test.reshape(28000,784).astype('float32')
X_train = X_train/255.0

X_test = X_test/255.0
print('x_train shape:', x_train.shape)

print(x_train.shape[0], 'train samples')

print(x_test.shape[0], 'test samples')
# X_train = x_train.reshape(x_train.shape[0], 28, 28,1)

# X_test = x_test.reshape(x_test.shape[0], 28, 28,1)
import keras

from keras.models import Sequential

from keras.layers import Dense

from sklearn.model_selection import train_test_split

batch_size = 64

num_classes = 10

epochs = 10

input_shape = (28, 28, 1)
# convert class vectors to binary class matrices One Hot Encoding

y_train = keras.utils.to_categorical(y_train, num_classes)

X_train, X_val, Y_train, Y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state=42)
model = Sequential()

model.add(Dense(1000, input_dim=784, activation='relu'))

model.add(Dense(700, input_dim=784, activation='relu'))

model.add(Dense(500, input_dim=784, activation='relu'))

model.add(Dense(10, kernel_initializer='normal', activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=20, batch_size=200, verbose=2)

final_loss, final_acc = model.evaluate(X_val, Y_val, verbose=0)

print("Final loss: {0:.6f}, final accuracy: {1:.6f}".format(final_loss, final_acc))
#get the predictions for the test data

predicted_classes = model.predict_classes(X_test)
submissions=pd.DataFrame({"ImageId": list(range(1,len(predicted_classes)+1)),

                         "Label": predicted_classes})

submissions.to_csv("MNIST_byShaymaa1.csv", index=False, header=True)
model.save('my_model_1.h5')

json_string = model.to_json()