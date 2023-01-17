# This is going to be a minimalistic kernel, I'ma cut to the chase

# Import necessary modules
import numpy as np
import keras
import pandas as pd
from sklearn.model_selection import train_test_split
np.random.seed(123)  # for reproducibility
import tensorflow as tf
tf.set_random_seed(123)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

batch_size = 128
# We can afford to keep it this big because the images are quite small (28 x 28), and lets face it, 
# the Kaggle GPU's can handle them memory

df = pd.read_csv("../input/train.csv")
y = df.label.values
X = df.drop("label",axis=1).values
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
# So now we've imported the test and train datasets and split them into train_test for validation
# So now we gotta convert them into numpy arrays, because they're images and best represented as numpy arrays
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_train = np.array(Y_train)

# So, once they've been converted to numpy format, they also need to be reshaped into the required 28 x 28 dimension.
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

# And now, since the max values of each cell in the matrix is 255, we can divide by it to normalize to 1, which makes
# it oh so easier for the CNN to converge!
X_train /= 255
X_test /= 255
# You do realize we got 10 classes (10 numbers), so gotta convert them into categorical labels!
Y_train = np_utils.to_categorical(Y_train, 10)
Y_test = np_utils.to_categorical(Y_test, 10)
# Now for the fun and most skilful part.. the architecture.
model = Sequential()

model.add(Convolution2D(32, (6, 6), activation='relu', input_shape=(28,28,1)))
model.add(Convolution2D(64, (6, 6), activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Dropout(0.25))
# So I really cannot justify the values I've used here, I've just found them to be quite good by trial and error. 

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))
# Here too. Trial and error. I'm sure you could improve upon it! :)
# And now, to compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# And.... moment of truth. Training!
model.fit(X_train, Y_train,
          batch_size=batch_size, epochs=40, verbose=1,
          validation_data=(X_test, Y_test))
model.save('model_new.h5')
# To evaluate the model on test data
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
test_data = pd.read_csv("../input/test.csv")
import matplotlib.pyplot as plt
ans = []
for i in range(len(test_data.as_matrix())):
	img = test_data.as_matrix()[i]
	img = img / 255
	img = np.array(img).reshape((28, 28, 1))
	img = np.expand_dims(img, axis=0)
	img_class = model.predict_classes(img)
	ans.append(img_class)
ids = [i+1 for i in range(len(ans))]
df_1 = pd.DataFrame({"ImageId" : ids, "Label" : ans})
df_1.to_csv("1.csv", index = False)


#print(classes[0:10])
print(img_class)
#prediction = img_class[0]
classname = img_class[0]
print("Predicted number is: ",classname)

img = img.reshape((28,28))
plt.imshow(img)
plt.title(classname)
plt.show()
