import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split



from tensorflow import keras

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

training_dataset = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test_dataset = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')



print(training_dataset.shape)

print(test_dataset.shape)
train_labels = training_dataset["label"] 

training_dataset.drop(["label"], axis = 1, inplace = True)
train = training_dataset.values.reshape(-1,28,28,1)

test = test_dataset.values.reshape(-1,28,28,1)
train, test = train/255.0, test/255.0
X_train, X_test, Y_train, Y_test = train_test_split(train, train_labels, test_size = 0.2, shuffle = True)
CNNmodel = Sequential()

CNNmodel.add(Conv2D(32, (3,3), activation = 'relu', input_shape = (28,28,1)))

CNNmodel.add(MaxPooling2D((2,2)))

CNNmodel.add(Conv2D(64, (3,3), activation = 'relu'))

CNNmodel.add(MaxPooling2D((2,2)))

CNNmodel.add(Conv2D(64, (3,3), activation = 'relu'))

CNNmodel.add(Flatten())

CNNmodel.add(Dense(64, activation = 'relu'))

CNNmodel.add(Dense(10, activation = 'softmax'))



CNNmodel.summary()
CNNmodel.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
history = CNNmodel.fit(X_train, Y_train, validation_data = (X_test, Y_test), epochs = 10)
plt.plot(history.history['accuracy'], label='accuracy')

plt.plot(history.history['val_accuracy'], label = 'val_accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.ylim([0.5, 1])

plt.legend(loc='lower right')
test_loss, test_acc = CNNmodel.evaluate(X_test,  Y_test, verbose=2)

print("Test Accuracy:", test_acc)
predictions = np.argmax(CNNmodel.predict(test), axis = 1)
submission_dataframe = pd.DataFrame({"ImageId" : range(1, 28001), "Label" : predictions})
submission_dataframe.to_csv("submission.csv", index = False)