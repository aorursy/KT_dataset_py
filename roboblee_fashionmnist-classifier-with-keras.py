import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



from sklearn.metrics import classification_report



from keras.models import Sequential

from keras.utils import to_categorical

from keras.layers import Dense, Dropout, Activation, Flatten, MaxPool2D

from keras.layers import Conv2D, MaxPooling2D

from keras.callbacks import EarlyStopping



os.environ['KERAS_BAKEND'] = 'tensorflow'

%matplotlib inline
# Import train and test data.

trainFile = '../input/fashion-mnist_train.csv'

testFile = '../input/fashion-mnist_test.csv'



# Create dataframes and X(feature) matrix and target vector

df_train = pd.read_csv(trainFile)

df_test = pd.read_csv(testFile)



X_train = df_train.drop('label', axis=1)

y_train = df_train['label']



X_test = df_test.drop('label', axis=1)

y_test = df_test['label']
# What the training data looks like.

df_train.head(5)
# Training set has 60,000 images with 784 pixels (28 x 28).

print(X_train.shape)



pixel_num = X_train.shape[1]
# 10 distinct labels (categories).

print(set(df_train['label']))



label_num = len(set(df_train['label']))
def printImg(index):

    '''This function takes an index and prints the first image in the "X_train" dataframe with that index.'''

    for ind, row in enumerate(df_train['label']):

        if index == row:

            plt.imshow(X_train.iloc[ind].values.reshape(28,28), cmap=plt.get_cmap('gray'))

            plt.show()

            break



# Print an image from each label.

for i in range(10):

    print('Category: ' + str(i))

    printImg(i)
# Training set has 6,000 images of each class.

df_train.groupby('label').count().iloc[:,0]
# Normalize data into 0 - 1 scale.

X_train = X_train / 255

X_test = X_test / 255
# Transform pixel dataframe into arrays.

X_train_arr = X_train.values

X_test_arr = X_test.values
# One hot encode target vectors.

y_train = to_categorical(y_train)

y_test = to_categorical(y_test)
# Instantiate Sequential.

model = Sequential()



# Hidden Layers

model.add(Dense(300, activation='relu', input_dim=pixel_num))



# Output layer

model.add(Dense(10, activation='softmax'))



# Compile model.

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit Model

early_stopping_monitor = EarlyStopping(patience=3)

model.fit(X_train_arr, y_train, validation_split=0.3, epochs=20, batch_size=200, 

          callbacks=[early_stopping_monitor])
model.summary()
# Test data score

test_accuracy = model.evaluate(X_test_arr, y_test)

print('test loss:', test_accuracy[0])

print('test accuracy:', test_accuracy[1])
# Build classification report

predict_test = model.predict_classes(X_test_arr)



class_dct = {0:'T-shirt/top',

             1:'Trouser',

             2:'Pullover',

             3:'Dress',

             4:'Coat',

             5:'Sandal',

             6:'Shirt',

             7:'Sneaker',

             8:'Bag',

             9:'Ankle boot'

}

target_names = []

for k, v in class_dct.items():

    target_names.append(v)



print(classification_report(np.array(df_test['label']), predict_test, target_names=target_names))
# Reshape pixel matrix to 3 dimensional matrix

X_train_cnn = X_train.values.reshape(X_train.shape[0], 28, 28, 1)

X_test_cnn = X_train.values.reshape(X_train.shape[0], 28, 28, 1)
# Instantiate and compile model

model = Sequential()



model.add(Conv2D(10, (5, 5), input_shape=(28, 28, 1), activation='relu')) 

model.add(MaxPool2D((2, 2)))

model.add(Conv2D(10, (5, 5), activation='relu'))

model.add(MaxPool2D((2, 2)))

model.add(Flatten())

model.add(Dense(50, activation='relu'))

model.add(Dense(50, activation='relu'))

model.add(Dense(10, activation='softmax')) 



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_cnn, y_train, validation_split=0.3, epochs=10)
model.summary()
printImg(0), printImg(2)

# Both look like shirts!!