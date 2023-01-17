import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline
from sklearn.model_selection import train_test_split

import seaborn as sns
sns.set(style='white', context='notebook', palette='deep')
from PIL import Image

from keras.utils.np_utils import to_categorical 
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D




train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

Y_train = train["label"]
X_train = train.drop(labels = ["label"],axis = 1).values
X_test = test.values
print('X_train\'s shape: ' + str(X_train.shape))
print('Y_train\'s shape: ' + str(Y_train.shape))
plt.figure(figsize=(20,5))
for i in range(10):
    img = plt.subplot(2, 5, i + 1)
    img.set_title('label:' + str(Y_train[i]))
    plt.imshow(X_train[i].reshape((28, 28)))


#Getting the frequency of each label    
print(Y_train.value_counts())
sns.countplot(Y_train)
model = Sequential()
model.add(Conv2D(128, (5,5), padding='same', input_shape=(28,28,1), data_format='channels_last', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (5,5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(256, (3,3), padding='valid', activation='relu'))
model.add(Dropout(0.2))

# Output layer
model.add(Flatten())
model.add(Dense(units=10, activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

print('The model is successfully created.')
from keras.utils import to_categorical

Y_train_categorical = to_categorical(Y_train, num_classes=10)

X = X_train.reshape(X_train.shape[0],28,28,1) 
X = X / 255 #normalization


print(X.shape)
print(Y_train_categorical.shape)

model.fit(x=X, y=Y_train_categorical, batch_size=1000, epochs=32, verbose=1, validation_split=0.2)
(loss_train, accuracy_train) = model.evaluate(X, Y_train_categorical)
print('Traing Performance:')
print('Accuracy : ' + str(accuracy_train*100) )
print('Loss : ' + str(loss_train))
result = model.predict(X_test.reshape(X_test.shape[0],28,28,1) )
result = np.argmax(result,axis = 1)
result = pd.Series(result,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),result],axis = 1)
submission.to_csv("mnist_predictions_cnn_model.csv",index=False)
print(submission.head(15))
