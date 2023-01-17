import numpy as np

import pandas as pd

from time import time

from matplotlib import pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.preprocessing.image import ImageDataGenerator
data_train = pd.read_csv('../input/digit-recognizer/train.csv')

data_test = pd.read_csv('../input/digit-recognizer/test.csv')
target_train = data_train.iloc[:, 0].values

image_train = data_train.iloc[:, 1:].values

image_test = data_test.values
X = image_train.reshape(-1,28,28,1).astype('float32')

X_test = image_test.reshape(-1,28,28,1).astype('float32')

y = to_categorical(target_train.reshape(-1,1), num_classes=10)



X /= 255

X_test /= 255

plt.imshow(image_train[0].reshape(28,28))
datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1)

datagen.fit(X)
num_models = 15

model = [None] * num_models



for i in range(num_models):

    model[i] = Sequential()



    model[i].add(Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1)))

    model[i].add(BatchNormalization())

    model[i].add(Conv2D(32, (3, 3), activation='relu'))

    model[i].add(BatchNormalization())

    model[i].add(MaxPooling2D((2, 2), strides=2))

    model[i].add(BatchNormalization())



    model[i].add(Conv2D(64, (3, 3), activation='relu'))

    model[i].add(BatchNormalization())

    model[i].add(Conv2D(64, (3, 3), activation='relu'))

    model[i].add(BatchNormalization())

    model[i].add(MaxPooling2D((2, 2), strides=2))



    model[i].add(Flatten())

    model[i].add(BatchNormalization())



    model[i].add(Dense(256, activation='relu'))

    model[i].add(BatchNormalization())

    model[i].add(Dropout(0.5))

    model[i].add(Dense(10, activation='softmax'))



    model[i].compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # print(model.summary())



for i in range(num_models):

    t = time()

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=i)

    history = model[i].fit_generator(datagen.flow(X_train, y_train, batch_size=512), epochs=50, verbose=0, 

                                  validation_data=(X_val, y_val))

    print(history.history['val_accuracy'][-1], time()-t)
y_pred = np.zeros((X_test.shape[0], 10))

for i in range(num_models):

    y_pred += model[i].predict(X_test, batch_size=512, verbose=1)

y_pred = np.argmax(y_pred, axis=1)

    

submissions = pd.DataFrame({'ImageId': list(range(1,len(y_pred)+1)), 

                            'Label': y_pred})

submissions.to_csv('submission.csv', index=False, header=True)
y_pred_val = np.zeros((X_val.shape[0], 10))

for i in range(num_models):

    y_pred_val += model[i].predict(X_val, batch_size=512, verbose=1)

y_pred_val = np.argmax(y_pred_val, axis=1)

y_true_val = np.argmax(y_val, axis=1)

print(classification_report(y_true_val, y_pred_val, digits=4))
