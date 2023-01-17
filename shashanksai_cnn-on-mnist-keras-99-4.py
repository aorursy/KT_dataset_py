import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
Y_train = train['label']
Y_train.shape
X_train = train.drop(labels=['label'], axis =1)

del train             # Free RAM
import seaborn as sns

cnt = sns.countplot(Y_train)
# Checking for null Values

X_train.isnull().any().describe()
test.isnull().any().describe()
# Normalizing the data

X_train = X_train/255.0
test = test/255.0
# Converting 1D vectors into 2D vectors RESHAPING

# Reshape image in 3 dimensions (height = 28px, width = 28px , channel = 1)
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
im = plt.imshow(X_train[3][:,:,0])
# One hot encoding class labels
Y_train = to_categorical(Y_train , num_classes = 10)
# Split to get Val data
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state = 42)
# CNN Model


model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'same', activation = 'relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'same', activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'same', activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
epochs = 30
batch_size = 86
# Data Augmentation
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rotation_range = 10,
                            zoom_range = 0.1,
                            width_shift_range = 0.1,
                            height_shift_range = 0.1
                            )

datagen.fit(X_train)

history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size = batch_size), epochs = epochs, validation_data = (X_val, Y_val), steps_per_epoch=X_train.shape[0] // batch_size)
results = model.predict(test)
results = np.argmax(results,axis = 1)

results = pd.Series(results, name = 'Label')
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_datagen.csv",index=False)
