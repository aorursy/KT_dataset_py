import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, ZeroPadding2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
train = pd.read_csv('../input/fashion-mnist_train.csv')
test = pd.read_csv('../input/fashion-mnist_test.csv')

print(train.shape)
print(test.shape)
train.head()
test.head()
X_train = train.iloc[:, 1:].values.astype('float32') / 255
y_train = train.iloc[:, :1].values.astype('int32')  # 1st column is 'label' for images
X_test = test.iloc[:, 1:].values.astype('float32') / 255
y_test = test.iloc[:, :1].values.astype('int32')  # 1st column is 'label' for images
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_train
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print(X_train.shape)
print(X_val.shape)
print(y_train.shape)
print(y_val.shape)
im_rows = 28
im_cols = 28
im_shape = (im_rows, im_cols, 1)

X_train = X_train.reshape(X_train.shape[0], *im_shape)
X_val = X_val.reshape(X_val.shape[0], *im_shape)
X_test = X_test.reshape(X_test.shape[0], *im_shape)

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)
f, ax = plt.subplots(1,5)
f.set_size_inches(80, 40)

for i in range(5):
    ax[i].imshow(X_train[i].reshape(28, 28))
plt.show()
model = Sequential([
    ZeroPadding2D((1, 1)),
    Conv2D(32, (3, 3), activation='relu', input_shape=im_shape),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    ZeroPadding2D((1, 1)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    ZeroPadding2D((1, 1)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
model.compile(optimizer=Adam(lr=0.001),
             loss='categorical_crossentropy',
             metrics=['accuracy'])
history = model.fit(X_train, y_train,
          batch_size=240, epochs=50, verbose=1,
          validation_data=(X_val, y_val))
model.summary()
score = model.evaluate(X_test, y_test, verbose=0)
print(score)

print('Loss :', score[0])
print('Accuracy : ' + str(score[1] * 100) + '%')
accuracy = history.history['acc']
val_accuracy = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Loss')
plt.legend()
plt.show()