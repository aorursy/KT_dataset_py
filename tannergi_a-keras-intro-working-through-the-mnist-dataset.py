import numpy as np # linear algebra libary
import pandas as pd # data processing libary
import matplotlib.pyplot as plt # visualization libary
import os # So we can see if we already saved a model

# Deep Learning Libary
from keras.models import Sequential, model_from_json
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau
from keras.utils import to_categorical
training_dataset = pd.read_csv('../input/train.csv')
testing_dataset = pd.read_csv('../input/test.csv')
training_dataset.head()
X_train = np.array(training_dataset.drop(['label'], axis=1))
y_train = np.array(training_dataset['label'])
X_test = np.array(testing_dataset)
def visualize_digits(data, n, true_labels, predicted_labels=[]):
    fig = plt.figure()
    plt.gray()
    for i in range(n):
        ax = plt.subplot(2, n, i+1)
        plt.imshow(data[i].reshape(28, 28))
        # disable axis
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if len(predicted_labels)!=0:
            ax.set_title('True: ' + str(true_labels[i]) + ' Predicted: ' + str(np.argmax(predicted_labels[i])))
        else:
            ax.set_title('True: ' + str(true_labels[i]))
    fig.set_size_inches(np.array(fig.get_size_inches()) * n)
    plt.show()
visualize_digits(X_train, 10, y_train)
X_train = X_train / 255
X_test = X_test / 255
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
y_train = to_categorical(y_train, num_classes=10)
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(10, activation='softmax'))
optimizer = RMSprop(lr=0.001)
learning_rate_reduction = ReduceLROnPlateau(monitor='loss', patience=2, verbose=1, factor=0.3, min_lr=0.00001)
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=64), epochs=10,
                              verbose=2, steps_per_epoch=X_train.shape[0]//64, 
                              callbacks=[learning_rate_reduction])
plt.plot(history.history['loss'])
plt.plot(history.history['acc'])
scores = model.evaluate(X_train, y_train)
scores
predictions = model.predict(X_train)

visualize_digits(X_train, 10, training_dataset['label'], predictions)
predictions = model.predict(X_test)
predictions = [np.argmax(x) for x in predictions]
image_id = range(len(predictions))
solution = pd.DataFrame({'ImageId':image_id, 'Label':predictions})
solution.head()