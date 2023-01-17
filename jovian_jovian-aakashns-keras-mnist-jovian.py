from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images[0].shape, train_labels[0]
%matplotlib inline

import matplotlib.pyplot as plt



grid_size = 6

f, axarr = plt.subplots(grid_size, grid_size)

for i in range(grid_size):

    for j in range(grid_size):

        ax = axarr[i, j]

        ax.get_xaxis().set_visible(False)

        ax.get_yaxis().set_visible(False)

        ax.imshow(train_images[i * grid_size + j], cmap='gray')
train_images = train_images.reshape((60000, 28, 28, 1))

train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))

test_images = test_images.astype('float32') / 255



from keras.utils import to_categorical



partial_train_images = train_images[:45000]

partial_train_labels = train_labels[:45000]



validation_images = train_images[45000:]

validation_labels = train_labels[45000:]



partial_train_labels = to_categorical(partial_train_labels)

validation_labels = to_categorical(validation_labels)

test_labels = to_categorical(test_labels)
input_shape = (28,28,1)

num_classes = 10
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout



model = Sequential()

model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))

model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(32, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))



model.summary()
from keras import optimizers



model.compile(optimizers.Adam(lr=0.005), loss='categorical_crossentropy', metrics=['accuracy'])
import jovian 



jovian.log_hyperparams({

    'arch': 'Conv(16+16)+Dense(32)',

    'epochs': 2,

    'optimizer': 'Adam',

    'lr': 0.005

})
history = model.fit(

    partial_train_images, 

    partial_train_labels, 

    epochs=2, 

    batch_size=128, 

    validation_data=(validation_images, validation_labels))
jovian.log_metrics({

    'loss': 0.0558,

    'acc': 0.9826,

    'val_loss': 0.0683,

    'val_acc': 0.9788

})
from utils import plot_history



plot_history(history)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test loss:', test_loss)

print('Test acc:', test_acc)
model.save('mnist-cnn.h5')
import jovian
jovian.commit(files=['utils.py'], artifacts=['mnist-cnn.h5'])