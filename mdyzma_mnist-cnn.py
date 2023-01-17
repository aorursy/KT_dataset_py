import numpy as np
import pandas as pd
import keras
img_rows, img_cols = 28, 28
num_classes = 10
num_epochs=12

def data_prep(raw):
    out_y = keras.utils.to_categorical(raw.label, num_classes)

    num_images = raw.shape[0]
    x_as_array = raw.values[:,1:]
    x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1)
    out_x = x_shaped_array / 255
    return out_x, out_y
raw_data = pd.read_csv('../input/mnist_train.csv')
x, y = data_prep(raw_data)
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D





model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), strides=2, activation='relu'))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3, 3), strides=2, activation='relu'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])
history = model.fit(x, y,
          batch_size=128,
          epochs=num_epochs,
          validation_split = 0.2)
model.summary()
import matplotlib.pyplot as plt
%matplotlib inline

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('acc.png')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss.png')
plt.show()
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D


model_pool = Sequential()

model_pool.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))

model_pool.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model_pool.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model_pool.add(Dropout(0.25))


model_pool.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model_pool.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model_pool.add(Dropout(0.25))

model_pool.add(Flatten())
model_pool.add(Dense(256, activation='relu'))
model_pool.add(Dropout(0.25))
model_pool.add(Dense(num_classes, activation='softmax'))

model_pool.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])
history_pool = model_pool.fit(x, y,
                              batch_size=128,
                              epochs=6,
                              validation_split = 0.2)
model_pool.summary()
#save model
model_json = model_pool.to_json()
open('mnist_architecture.json', 'w').write(model_json)
# And the weights learned by our deep network on the training set
model_pool.save_weights('mnist_weights.h5', overwrite=True)

# model_pool.save('MNIST_pool.h5')
print(history_pool.history.keys())
import matplotlib.pyplot as plt
%matplotlib inline

# summarize history for accuracy
plt.plot(history_pool.history['acc'])
plt.plot(history_pool.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('acc.png')
plt.show()

# summarize history for loss
plt.plot(history_pool.history['loss'])
plt.plot(history_pool.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss.png')
plt.show()
raw_x = pd.read_csv('../input/mnist_test.csv');raw_x.head()
num_images = raw_x.shape[0]
x_test_array = raw_x.values[:,1:]
x_test_shaped_array = x_test_array.reshape(num_images, img_rows, img_cols, 1)
x_test = x_test_shaped_array / 255

# not-pulled model
y_test = model.predict(x_test)

# pooled model
# y_testp = model_pool.predict(x_test)
score = model.evaluate(x_test, y_test, batch_size=16)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# score = model_pool.evaluate(x_test, y_test, batch_size=16)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
y_cat = y_test.argmax(axis=-1)
# y_catp = y_testp.argmax(axis=-1)
len(y_cat)
submission = pd.DataFrame(y_cat, columns=['Label'], index=range(1, y_cat.shape[0]+1))
# submissionp = pd.DataFrame(y_cat, columns=['Label'], index=range(1, y_catp.shape[0]+1))
submission.reset_index(inplace=True)
submission.columns = ['ImageId', 'Label']
submission.to_csv('MNIST_submisison.csv', header=True, index=False)