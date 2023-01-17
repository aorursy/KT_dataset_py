import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout

from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D

from sklearn.model_selection import train_test_split

import os

from tensorflow.keras.layers import BatchNormalization
print(os.listdir("../input/Kannada-MNIST/"))
df_train = pd.read_csv("../input/Kannada-MNIST/train.csv")

df_test = pd.read_csv("../input/Kannada-MNIST/test.csv")
print(df_train.shape)

print(df_test.shape)
df_train.head()
df_test.head()
X = df_train.iloc[:,1:]

y = df_train.iloc[:,0]

X_test_actual = df_test.iloc[:,1:]
X = X.to_numpy().reshape(len(X), 28, 28,1).astype('float32')

X_test_actual = X_test_actual.to_numpy().reshape(len(X_test_actual), 28, 28, 1).astype('float32')
X = X/255

X_test_actual = X_test_actual/255
n_classes=10

y = to_categorical(y, n_classes)
X_train, X_test, y_train,  y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = Sequential()



model.add(Conv2D(32, 3, activation='relu', padding='same', input_shape=(28, 28, 1)))

model.add(Conv2D(64, 3, activation='relu', padding='same'))

model.add(MaxPooling2D(2, 2))

model.add(BatchNormalization())



model.add(Conv2D(128, 3, activation='relu', padding='same'))

model.add(Conv2D(128, 3, activation='relu', padding='same'))

model.add(MaxPooling2D(2, 2))

model.add(BatchNormalization())



model.add(Conv2D(256, 3, activation='relu', padding='same'))

model.add(Conv2D(256, 3, activation='relu', padding='same'))

model.add(Conv2D(256, 3, activation='relu', padding='same'))

model.add(MaxPooling2D(2, 2))

model.add(BatchNormalization())



model.add(Conv2D(512, 3, activation='relu', padding='same'))

model.add(Conv2D(512, 3, activation='relu', padding='same'))

model.add(Conv2D(512, 3, activation='relu', padding='same'))

model.add(MaxPooling2D(2, 1)) # default stride is 2

model.add(BatchNormalization())



model.add(Conv2D(512, 3, activation='relu', padding='same'))

model.add(Conv2D(512, 3, activation='relu', padding='same'))

model.add(Conv2D(512, 3, activation='relu', padding='same'))

model.add(MaxPooling2D(2, 1)) # default stride is 2

model.add(BatchNormalization())



model.add(Flatten())

model.add(Dense(4096, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(4096, activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(10, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',

             optimizer='nadam',

             metrics=['accuracy'])
history = model.fit(X_train, 

                    y_train, 

                    batch_size=128, 

                    epochs=50,

                    verbose=1,

                    validation_data=(X_test, y_test)

                   )
import matplotlib.pyplot as plt



loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(loss) + 1)



plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.show()


acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()



plt.show()
data_submission = pd.read_csv("../input/Kannada-MNIST/sample_submission.csv")
y_pre=model.predict(X_test_actual)     ##making prediction

y_pre=np.argmax(y_pre,axis=1) ##changing the prediction intro labels
data_submission['label']=y_pre

data_submission.to_csv('submission.csv',index=False)
data_submission.head()