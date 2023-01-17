import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

from keras import backend as K
K.set_image_dim_ordering('th')

from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
%matplotlib inline
# Load the data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
y_train = train["label"]
X_train = train.drop(labels = ["label"],axis = 1)

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

#split the data to train and validation data
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state=seed)

n_train = len(X_train)
n_train_y = len(y_train)
n_test = len(test)

counter_train = Counter(y_train)
counter_valid = Counter(y_val)

n_classes = len(counter_train.keys())

print("Number of training examples =", n_train)
print("Number of labels in training examples =", n_train_y)
print("Number of testing examples =", n_test)
print("Number of classes =", n_classes)
X_train.isnull().any().describe()
test.isnull().any().describe()
frequency = sns.countplot(y_train)
X_train = X_train.values.reshape(-1,1,28,28).astype('float32')
X_val = X_val.values.reshape(-1,1,28,28).astype('float32')
test = test.values.reshape(-1,1,28,28).astype('float32')
### Data exploration visualization
X_train, y_train = shuffle(X_train, y_train)

fig = plt.figure()
fig.suptitle('Example images of the German Traffic Signs', fontsize=18)

for i in range(50):
    image = X_train[i].squeeze()
    plt.axis("off")
    plt.subplot(5,10,i+1)
    plt.imshow(image,cmap='gray')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_val = X_val / 255
test = test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_val = np_utils.to_categorical(y_val)
# define the model
def model():
# create model
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
# build the model
model = model()
# Fit the model

epochs=38
batch_size=200

# Fit the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=0)

scores = model.evaluate(X_val, y_val, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
# predict results
results = model.predict(test)
# select the indix with the maximum probability
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("digits.csv",index=False)