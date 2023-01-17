import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib inline
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
train.head()
print(train.shape)
print(test.shape)
from sklearn.model_selection import train_test_split

X = train.drop('label', axis=1)
y = train.label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train = X_train.values.reshape(-1,28, 28)
X_test = X_test.values.reshape(-1, 28, 28)

X_test = np.array(X_test)
y_test = np.array(y_test)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
plt.figure()
plt.imshow(X_train[0])
plt.colorbar()
plt.grid(False)
plt.show()
X_train = X_train / 255.0
X_test = X_test / 255.0
y_train.iloc[1]
plt.figure(figsize=(10, 15))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.grid(False)
    plt.imshow(X_train[i], cmap=plt.cm.binary)
    plt.xlabel(y_train.iloc[i])

plt.show()
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam

model = Sequential([Flatten(input_shape=(28, 28)), 
                    Dense(128, activation='relu'), 
                    Dropout(0.2),
                    Dense(10, activation='softmax')])
model.compile(optimizer=Adam(0.001), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
r = model.fit(X_train, y_train, 
              validation_data=(X_test, y_test),
              epochs=20)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc}')
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(r.history['accuracy'], label='accuracy')
plt.plot(r.history['val_accuracy'], label='val_accuracy')
plt.legend()
predictions = model.predict(X_test)
predictions[0]
np.argmax(predictions[0])
y_test[0]
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel(f"{predicted_label} {100*np.max(predictions_array):2.0f}% ({true_label})", 
               color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], y_test, X_test)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  y_test)
plt.show()
i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], y_test, X_test)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  y_test)
plt.show()
# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions[i], y_test, X_test)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], y_test)
plt.tight_layout()
plt.show()
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)
from keras.layers import Conv2D, MaxPool2D
from keras.callbacks import EarlyStopping
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same', 
                 input_shape=X_train[0].shape))
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', 
                 input_shape=X_train[0].shape))
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', 
              optimizer=Adam(0.0001), 
              metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=2)
r = model.fit(X_train, y_train, epochs=10, 
              validation_data=(X_test, y_test), 
              callbacks=[early_stop])
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc}')
r = model.fit(X_test, y_test, epochs=10)
test = test / 255
test = test.values.reshape(-1, 28, 28, 1)
test.shape
predictions = model.predict(test)
predictions = np.argmax(predictions, axis=1)
predictions
submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
submission['Label'] = predictions
submission.to_csv('submission.csv', index=False)