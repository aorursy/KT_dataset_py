!tar xzvf ../input/cifar-10-python.tar.gz

def load_data():
    """Loads CIFAR10 dataset.
    Returns:
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    import os
    import sys
    from six.moves import cPickle
    import numpy as np
    
    def load_batch(fpath):
        with open(fpath, 'rb') as f:
            d = cPickle.load(f, encoding='bytes')  
        data = d[b'data']
        labels = d[b'labels']
        data = data.reshape(data.shape[0], 3, 32, 32)
        return data, labels
    
    path = 'cifar-10-batches-py'
    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000:i * 10000, :, :, :],
         y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)
    
    x_test, y_test = load_batch(os.path.join(path, 'test_batch'))

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)

(X_train, y_train), (X_test, y_test) = load_data()
X_train.shape, y_train.shape, X_test.shape, y_test.shape
import numpy as np
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=10**4, random_state=42)
print(X_train.shape, X_val.shape)
class_names = np.array(['airplane','automobile ','bird ','cat ','deer ','dog ','frog ','horse ','ship ','truck'])

from imgaug import augmenters as iaa
seq = iaa.Sequential([
    iaa.Crop(px=(0, 2)), # crop images from each side by 0 to 16px (randomly chosen)
    iaa.Fliplr(0.5), # horizontally flip 50% of the images
    iaa.GaussianBlur(sigma=(0, 1.0)) # blur images with a sigma of 0 to 3.0
])


images_aug = seq.augment_images(X_train)
X_train = np.concatenate((X_train, images_aug))
y_train = np.concatenate((y_train, y_train))

X_train = X_train / 255
X_val = X_val / 255
X_test = X_test / 255

X_train.shape, y_train.shape

import keras, keras.layers as L
from keras.models import Sequential

model = Sequential()
model.add(L.InputLayer(input_shape=X_train.shape[1:]))
model.add(L.Conv2D(filters=32, kernel_size=[3,3]))
model.add(L.Activation('elu'))
model.add(L.BatchNormalization())
model.add(L.Conv2D(filters=32, kernel_size=[3,3]))
model.add(L.Activation('elu'))
model.add(L.BatchNormalization())

model.add(L.MaxPool2D(pool_size=(2,2)))
model.add(L.Dropout(0.2))
 
model.add(L.Conv2D(filters=64, kernel_size=[3,3]))
model.add(L.Activation('relu'))
model.add(L.BatchNormalization())
model.add(L.Conv2D(filters=64, kernel_size=[3,3]))
model.add(L.Activation('relu'))
model.add(L.BatchNormalization())
model.add(L.MaxPooling2D(pool_size=(2,2)))
model.add(L.Dropout(0.3))
 
model.add(L.Conv2D(filters=128, kernel_size=[3,3]))
model.add(L.Activation('relu'))
model.add(L.BatchNormalization())
model.add(L.Conv2D(filters=128, kernel_size=[3,3]))
model.add(L.Activation('relu'))
model.add(L.BatchNormalization())
#print(model.summary())
#model.add(L.MaxPool2D(pool_size=(2,2)))
#model.add(L.Dropout(0.4))
 
model.add(L.Flatten())
model.add(L.Dense(10, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=[X_val, y_val], epochs=100, batch_size=64, verbose=1)

# без нормальзации 0.8752 0.7901
from sklearn.metrics import accuracy_score
test_acc = accuracy_score(y_test, model.predict_classes(X_test))
print("\n Test_acc =", test_acc)
if test_acc > 0.8:
    print("Это победа!")