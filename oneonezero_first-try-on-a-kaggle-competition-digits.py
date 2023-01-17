import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelBinarizer 



from keras.models import Sequential

from keras.layers import Dense
X = pd.read_csv("../input/train.csv")

X_test_final = pd.read_csv("../input/test.csv")

y = X["label"].copy()

X.drop("label", axis = 1, inplace = True)
lb = LabelBinarizer()

y = lb.fit_transform(y)

pd.DataFrame(y[:3])
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)

X_train.shape
model = Sequential()

model.add(Dense(128, activation='relu', input_dim=784))

model.add(Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
no_epochs = 2
history = model.fit(np.array(X_train), np.array(y_train), batch_size=32, epochs=no_epochs, validation_data = (np.array(X_test), np.array(y_test)))
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
erg = pd.DataFrame(data = {'ImageId': range(1,28001),

                           'Label': model.predict_classes(np.array(X_test_final))})

erg.to_csv("predictions.csv", index = False)
X_train_r = pd.DataFrame(X_train).values.reshape(-1,28,28,1)

X_test_r = pd.DataFrame(X_test).values.reshape(-1,28,28,1)

X_test_final_r = pd.DataFrame(X_test_final).values.reshape(-1,28,28,1)
from keras.layers import Dropout, Flatten, Conv2D, MaxPool2D
model2 = Sequential()



model2.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model2.add(MaxPool2D(pool_size=(2,2)))

model2.add(Dropout(0.4))



model2.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model2.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model2.add(Dropout(0.4))



model2.add(Flatten())

model2.add(Dense(128, activation = "relu"))

model2.add(Dropout(0.5))

model2.add(Dense(10, activation = "softmax"))



model2.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])



history2 = model2.fit(np.array(X_train_r), np.array(y_train),

                      batch_size=32, epochs=no_epochs,

                      validation_data = (np.array(X_test_r), np.array(y_test)))
plt.plot(history2.history['loss'])

plt.plot(history2.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
erg = pd.DataFrame(data = {'ImageId': range(1,28001),

                           'Label': model2.predict_classes(np.array(X_test_final_r))})

erg.to_csv("predictions.csv", index = False)
X_train_r = X_train_r / 255

X_test_r = X_test_r /255

X_test_final_r = X_test_final_r /255



from keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(

        featurewise_center=True,

        featurewise_std_normalization=True,

        zca_whitening=True,

        fill_mode='nearest')



datagen.fit(np.array(X_train_r))



for i in range(len(X_test_r)):

    X_test_r[i] = datagen.standardize(X_test_r[i])

    

for i in range(len(X_test_final_r)):

    X_test_final_r[i] = datagen.standardize(X_test_final_r[i])
history3 = model2.fit_generator(datagen.flow(X_train_r, y_train, batch_size=32),

                    steps_per_epoch=len(X_train_r) / 32, epochs=no_epochs,

                    validation_data = (np.array(X_test_r), np.array(y_test)))
plt.plot(history3.history['loss'])

plt.plot(history3.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
erg = pd.DataFrame(data = {'ImageId': range(1,28001),

                           'Label': model2.predict_classes(np.array(X_test_final_r))})

erg.to_csv("predictions.csv", index = False)
model3 = Sequential()



model3.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model3.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model3.add(MaxPool2D(pool_size=(2,2)))

model3.add(Dropout(0.2))



model3.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model3.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model3.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model3.add(Dropout(0.2))



model3.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model3.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model3.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model3.add(Dropout(0.2))



model3.add(Flatten())

model3.add(Dense(512, activation = "relu"))

model3.add(Dropout(0.25))

model3.add(Dense(10, activation = "softmax"))



model3.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history4 = model3.fit_generator(datagen.flow(X_train_r, y_train, batch_size=64),

                    steps_per_epoch=len(X_train_r) / 64, epochs=no_epochs,

                    validation_data = (np.array(X_test_r), np.array(y_test)))
plt.plot(history4.history['loss'])

plt.plot(history4.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
erg = pd.DataFrame(data = {'ImageId': range(1,28001),

                           'Label': model3.predict_classes(np.array(X_test_final_r))})

erg.to_csv("predictions.csv", index = False)