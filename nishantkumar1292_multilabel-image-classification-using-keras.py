import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

#keras imports

from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Dense, Dropout, Flatten
train_data_file = "/kaggle/input/fashionmnist/fashion-mnist_train.csv"

test_data_file = "/kaggle/input/fashionmnist/fashion-mnist_test.csv"
train_data = pd.read_csv(train_data_file)

test_data = pd.read_csv(test_data_file)

print("train data shape is {}".format(train_data.shape))

print("test data shape is {}".format(test_data.shape))
def get_xy(data):

    y = to_categorical(data['label'].values)

    x = data.drop(columns=['label']).to_numpy().reshape((data.shape[0], int(np.sqrt(data.shape[1])), int(np.sqrt(data.shape[1])), 1))/255

    return x, y
#getting train data

X, y = get_xy(train_data)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
print("X_train shape {}".format(X_train.shape))

print("X_test shape {}".format(X_test.shape))

print("y_train shape {}".format(y_train.shape))

print("y_test shape {}".format(y_test.shape))
#Build model

model = Sequential()

model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
img_test, label_test = get_xy(test_data)
scores = model.evaluate(img_test, label_test, verbose=0)

print("Accuracy: %.2f%%" % (scores[1]*100))
predictions = model.predict_classes(img_test)
image_label_dict = {'0': 'T-shirt/top','1': 'Trouser', '2': 'Pullover', '3': 'Dress', '4': 'Coat', '5': 'Sandal', '6': 'Shirt', '7': 'Sneaker', 

                    '8': 'Bag', '9': 'Ankle boot'}
#Visualise a random image from test dataset, its predicted label and actual label

image_id = np.random.randint(0, img_test.shape[0])

print("Image Id: {}".format(image_id))

plt.imshow(img_test[image_id].reshape((28,28)), interpolation='nearest')

plt.title("Actual Label: {}, Predicted Label: {}({}%)".format(image_label_dict[str(label_test[image_id].argmax())], 

                                                              image_label_dict[str(predictions[image_id])], 

                                                np.round(model.predict(img_test[image_id].reshape((1, 28, 28, 1)))[0, predictions[image_id]]*100, 1)))

plt.show()