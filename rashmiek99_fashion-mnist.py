import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D

from keras.models import Sequential

from keras.layers.normalization import BatchNormalization

from keras.layers.advanced_activations import LeakyReLU

from keras.losses import categorical_crossentropy

from keras.optimizers import Adam



import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_train.csv')

test = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_test.csv')
train.shape,test.shape
train.label.nunique()
train = np.array(train,dtype='float32')

test = np.array(test,dtype='float32')

                 

train.shape, test.shape
train_X = train[:,1:] / 255

test_X =  test[:,1:] / 255



train_X = train_X.reshape(train_X.shape[0], 28,28)

test_X = test_X.reshape(test_X.shape[0], 28,28)



train_y = train[:,0]

test_y = test[:,0]

train_X = train_X.reshape(-1, 28,28, 1)

test_X = test_X.reshape(-1, 28,28, 1)



train_Y_one_hot = to_categorical(train_y)

test_Y_one_hot = to_categorical(test_y)



train_X.shape, test_X.shape
X_train,X_valid,y_train,y_valid = train_test_split(train_X,train_Y_one_hot,test_size=0.3)

X_train.shape,X_valid.shape,y_train.shape,y_valid.shape
plt.imshow( X_train[5850,:].reshape((28,28)))

plt.show()
batch_size = 64

epochs = 20

num_classes = 10
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(28,28,1)))

model.add(LeakyReLU(alpha=0.1))

model.add(MaxPooling2D((2, 2),padding='same'))

model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))

model.add(LeakyReLU(alpha=0.1))

model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))

model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))

model.add(LeakyReLU(alpha=0.1))                  

model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))

model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(128, activation='linear'))

model.add(LeakyReLU(alpha=0.1))           

model.add(Dropout(0.3))

model.add(Dense(num_classes, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
model_dropout = model.fit(X_train,y_train, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(X_valid, y_valid))
test_eval = model.evaluate(test_X, test_Y_one_hot, verbose=1)
print("Loss=",test_eval[0])

print("Accuracy=",test_eval[1])
predicted_classes = model.predict(test_X)

predicted_classes = np.argmax(np.round(predicted_classes),axis=1)

predicted_classes.shape, test_y.shape
correct = np.where(predicted_classes==test_y)[0]

print( "Found %d correct labels" % len(correct))

for i, correct in enumerate(correct[:9]):

    plt.subplot(3,3,i+1)

    plt.imshow(test_X[correct].reshape(28,28), cmap='gray', interpolation='none')

    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], test_y[correct]))

    plt.tight_layout()
incorrect = np.where(predicted_classes!=test_y)[0]

print( "Found %d incorrect labels" % len(incorrect))

for i, incorrect in enumerate(incorrect[:9]):

    plt.subplot(3,3,i+1)

    plt.imshow(test_X[incorrect].reshape(28,28), cmap='gray', interpolation='none')

    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], test_y[incorrect]))

    plt.tight_layout()