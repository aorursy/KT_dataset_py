import tables
import numpy as np
from random import shuffle
from math import ceil
import matplotlib.pyplot as plt
hdf5_path = '../input/dataset.hdf5'  # address to where you want to save the hdf5 file
subtract_mean = True
batch_size = 50
nb_class = 2
hdf5_file = tables.open_file(hdf5_path, mode='r')
# subtract the training mean
if subtract_mean:
    mm = hdf5_file.root.train_mean[0]
    mm = mm[np.newaxis, ...]

# Total number of samples
train_data = np.array(hdf5_file.root.train_img)
train_label = np.array(hdf5_file.root.train_labels)

test_data = np.array(hdf5_file.root.test_img)
test_label = np.array(hdf5_file.root.test_labels)

val_data = np.array(hdf5_file.root.val_img)
val_label = np.array(hdf5_file.root.val_labels)

print('train data:',train_data.shape,' train_label',train_label.shape)
print('test_data:',test_data.shape,' test_label:',test_label.shape)
print('val_data:',val_data.shape,' val_label:',val_label.shape)

# create list of batches to shuffle the data
#     batches_list = list(range(int(ceil(float(data_num) / batch_size))))
#     shuffle(batches_list)

#     # loop over batches
#     for n, i in enumerate(batches_list):
#         i_s = i * batch_size  # index of the first image in this batch
#         i_e = min([(i + 1) * batch_size, data_num])  # index of the last image in this batch
#         print('i_s:',i_s,' i_e:',i_e)

#         # read batch images and remove training mean
#         images = hdf5_file.root.train_img[i_s:i_e]
#         print('len:',len(images))
#         if subtract_mean:
#             images -= mm

#         # read labels and convert to one hot encoding
#         labels = hdf5_file.root.train_labels[i_s:i_e]
#         labels_one_hot = np.zeros((len(images), nb_class))
#         labels_one_hot[np.arange(len(images)), labels] = 1

#         print(n+1, '/', len(batches_list))

#         print (labels[0], labels_one_hot[0, :])
#         plt.imshow(images[0])
#         plt.show()

#         if n == 5:  # break after 5 batches
#             break
from keras.utils import np_utils

# one-hot encode the labels
num_classes = len(np.unique(train_label))
train_label = np_utils.to_categorical(train_label, num_classes)
test_label = np_utils.to_categorical(test_label, num_classes)
val_label = np_utils.to_categorical(val_label, num_classes)

# print shape of training set
print('num_classes:', num_classes)

# print number of training, validation, and test images
print(train_label.shape, 'train samples')
print(test_label.shape, 'test samples')
print(val_label.shape, 'validation samples')
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', 
                        input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='tanh'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='tanh'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', 
                  metrics=['accuracy'])
from keras.callbacks import ModelCheckpoint   

# train the model
checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=1, 
                               save_best_only=True)
hist = model.fit(train_data, train_label, batch_size=None, epochs=20,
          validation_data=(val_data, val_label),callbacks=[checkpointer], 
          verbose=1, shuffle=True)
model.load_weights('model.weights.best.hdf5')
score = model.evaluate(test_data, test_label, verbose=0)
print('\n', 'Test accuracy:', score[1])
