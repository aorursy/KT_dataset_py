import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from scipy.misc import toimage

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
df_train = pd.read_csv("../input/train.csv", encoding = 'ISO-8859-1')
df_subm =  pd.read_csv("../input/test.csv", encoding = 'ISO-8859-1')
df_train.isnull().sum().sum() , df_subm.isnull().sum().sum()
df_train.head()
df_subm.head()
X_train = df_train[df_train.columns[1:]]
y_train = df_train['label']
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
for i in range(0, 9):
    plt.subplot(330 + 1 + i)
    plt.imshow((X_train.iloc[i].values.reshape(28,28)))
plt.show()
X_train = X_train.values.reshape(X_train.shape[0],28,28,1)
X_test = X_test.values.reshape(X_test.shape[0],28,28,1)
# X_train = X_train.values.reshape(df_train.shape[0],28,28)
X_train.max() - X_train.min() 
X_train = X_train/255
X_test = X_test/255
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
print(model.summary())
# model.fit(X_train, y_train, batch_size=128, epochs=10)
# score = model.evaluate(X_test, y_test, batch_size=128)
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    )
datagen.fit(X_train)
model.fit_generator(datagen.flow(X_train, y_train, batch_size=128),
                    steps_per_epoch=int(len(X_train) / 128), epochs=30)
# model.evaluate(X_test, y_test, batch_size=128)
# confusion_matrix(np.argmax(y_test,axis=1,out=None),np.argmax(model.predict(X_test), axis=1, out=None))
# accuracy_score(np.argmax(y_test,axis=1,out=None),np.argmax(model.predict(X_test), axis=1, out=None))
#We put the values of the submission data in a matrix
test_data = df_subm.values
#We reshape the matrix as a 4 dimensional tensor
test_data = test_data.reshape(test_data.shape[0],28,28,1)
#We normalize the data with the same factor as we did for the train data
test_data = test_data/255
#We use the model to generate the predictions
predictions = model.predict(test_data)
#We get the labels of the predictions, recalling that we used a one-hot-encoding 
#to train the model
predictions = np.argmax(predictions, axis=1, out=None)
#We generate a csv file with the predictions in the required format
with open("resultCNNwithPrepros.csv", "wb") as f:
    f.write(b'ImageId,Label\n')
    np.savetxt(f, np.hstack([(np.array(range(28000))+1).reshape(-1,1), predictions.astype(int).reshape(-1,1)]), fmt='%i', delimiter=",")
