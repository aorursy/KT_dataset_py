import numpy as np

import pandas as pd



import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,MaxPooling2D

from keras.layers.normalization import BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
print(train.shape) 

print(test.shape)
train.head()

test.head()
X_train = train.iloc[:,1:].values.astype("float32")

y_train = train.iloc[:,0].values.astype("int32")

X_test = test.values.astype('float32')

print(X_test.shape)
#Normalize image vectors

X_train = X_train/255.0

X_test = X_test/255.0

y_train
X_train = X_train.reshape(X_train.shape[0],28,28,1)

X_test = X_test.reshape(X_test.shape[0],28,28,1)
num_classes = 10

y_train = keras.utils.to_categorical(y_train, num_classes)

# print(y_train[:5])

from keras.layers import Dense,Activation

from keras.layers import Dropout

model = Sequential()



# CONV -> CONV -> MAX -> DP 

model.add(Conv2D(32, (3, 3),input_shape=(28,28,1),activation = 'relu', kernel_initializer='he_normal'))

model.add(Conv2D(32, (3, 3), activation = 'relu', kernel_initializer='he_normal'))

model.add(MaxPooling2D((2, 2)))

model.add(Dropout(0.25))



# CONV -> CONV -> MAX -> DP  

model.add(Conv2D(64, (3, 3),activation = 'relu', kernel_initializer='he_normal'))

model.add(Conv2D(64, (3, 3),activation = 'relu', kernel_initializer='he_normal'))

model.add(MaxPooling2D((2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(128,(3, 3),activation = 'relu', padding="same", kernel_initializer='he_normal'))

model.add(MaxPooling2D((2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(264, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='sigmoid', name='fc'))

# keras.losses.categorical_crossentropy

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

X_train, X_test1, y_train, y_test= train_test_split(X_train, y_train, test_size = 0.1, random_state = 42)
y_train.shape
epochs = 30

drop=0.2

optimizer = 'adam'

batch_size = 32

model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs)
preds = model.evaluate(x = X_test1  , y = y_test )

print()

print ("Loss = " + str(preds[0]))

print ("Test Accuracy = " + str(preds[1]))
y_pred = model.predict_classes(X_test)
y_pred = np.squeeze(y_pred)

print(y_pred[:5])
sample_sub = pd.read_csv("../input/sample_submission.csv")
Id = sample_sub["ImageId"].values


sub = {'ImageId':Id,'Label':y_pred}

data = pd.DataFrame(sub)
data.set_index("ImageId").to_csv("final_sub_2.csv")