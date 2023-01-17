import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import pandas as pd

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from keras.utils import to_categorical

from keras import optimizers

from keras import backend as K



from sklearn.model_selection import train_test_split
train_data = pd.read_csv("/kaggle/input/Kannada-MNIST/train.csv")

test_data = pd.read_csv("/kaggle/input/Kannada-MNIST/test.csv")
num_classes = len(train_data['label'].unique())
X,y = train_data.iloc[:,1:].values,train_data.iloc[:,0].values
y = to_categorical(y,num_classes)
X = X.astype('float32')

X = X/255

X = X.reshape(X.shape[0], 28, 28, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=100)
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=X_train.shape[1:]))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)



model.compile(loss="categorical_crossentropy",optimizer=sgd,metrics=['accuracy'])

hist = model.fit(X_train, y_train,

          batch_size=50,

          epochs=70,

          verbose=1,

          validation_data=(X_test, y_test))
plt.plot(hist.history['loss'])

plt.plot(hist.history['val_loss'])

plt.legend(["traning data","validation data"])

plt.title("Loss")

plt.xlabel("epoch")
plt.plot(hist.history['accuracy'])

plt.plot(hist.history['val_accuracy'])

plt.legend(["traning data","validation data"])

plt.title("Accuracy")

plt.xlabel("epoch")
score = model.evaluate(X_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
valid_data = test_data.iloc[:,1:].values

valid_data = valid_data/255

valid_data = valid_data.reshape(valid_data.shape[0], 28, 28, 1)
y_pred = model.predict(valid_data)
y_pred = np.argmax(y_pred, axis=1)
test_data.head()
sub_df = pd.DataFrame({'id':test_data['id'].values,'label':y_pred})
sub_df.to_csv('submission.csv', index=False)