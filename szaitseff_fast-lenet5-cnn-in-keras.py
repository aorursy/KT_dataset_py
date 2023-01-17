# Load necessary libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # data visualization
%matplotlib inline

from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical # to convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Activation, Flatten, Dense, Dropout, Conv2D, MaxPool2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.regularizers import l2
# Load datasets
train, test = pd.read_csv("../input/train.csv"), pd.read_csv("../input/test.csv")
# Review data
print(f'train data shape = {train.shape}', '/', f'test data shape = {test.shape}')
train.head()
# let's check the count of different labels in the dataset (~balanced)
train['label'].value_counts()
# Numpy representation of the train and test data:
train_pixels, test_pixels = train.iloc[:,1:].values.astype('float32'), test.values.astype('float32') # all pixel values
train_labels = train.iloc[:,0].values.astype('int32') # only labels i.e targets digits
train_labels = train_labels.reshape(-1, 1) # ensure proper shape of the array

print(f'train_pixels shape = {train_pixels.shape}')
print(f'test_pixels shape = {test_pixels.shape}')
print(f'train_labels shape = {train_labels.shape}')
# Reshape input data to fit Keras model (height=28px, width=28px, channels=1):
train_pixels, test_pixels = train_pixels.reshape(-1, 28, 28, 1), test_pixels.reshape(-1, 28, 28, 1)
print(f'train_pixels shape = {train_pixels.shape}')
print(f'test_pixels shape = {test_pixels.shape}')
# Visualize some images from the dataset:
nrows, ncols = 3, 5  # number of rows and colums in subplots
fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(8,5))
for row in range(nrows):
    for col in range(ncols):
        i = np.random.randint(0, 30000)  # pick up arbitrary examples
        ax[row, col].imshow(train_pixels[i,:,:,0], cmap='Greys')
        ax[row, col].set_title(f'<{train.label[i]}>');
# Input data are greyscale pixels of intensity [0:255]. Let's normalize to [0:1]:
train_pixels, test_pixels = train_pixels / 255.0, test_pixels / 255.0
# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
train_labels = to_categorical(train_labels, num_classes = 10)
print(f'train_labels shape = {train_labels.shape}')
train_labels
# Split training and validation set for the fitting
train_pixels, val_pixels, train_labels, val_labels = train_test_split(train_pixels, train_labels, test_size = 0.1, random_state=None)

train_pixels.shape, train_labels.shape, val_pixels.shape, val_labels.shape, test_pixels.shape
# let's fix the important numbers for further modeling:
m_train = train_pixels.shape[0]   # number of examples in the training set
m_val = val_pixels.shape[0]       # number of examples in the validation set
m_test = test_pixels.shape[0]     # number of examples in the test set
n_x = test.shape[1]               # input size, number of pixels in the image
n_y = train_labels.shape[1]       # output size, number of label classes
print(f" m_train = {m_train} / m_val = {m_val} / m_test = {m_test} / n_x = {n_x} / n_y = {n_y}")
# Let's also define ImageDataGenerator for data augmentation to prevent overfitting:
datagen = ImageDataGenerator(
        rotation_range = 10,  # randomly rotate images in the range [0-180 degrees]
        zoom_range = 0.1, # randomly zoom image: [lower, upper] = [1-zoom_range, 1+zoom_range]
        shear_range = 0.1, # random distortion
        width_shift_range = 0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range = 0.1)  # randomly shift images vertically (fraction of total height)
# create an instance of a neural network:
model = Sequential()
# Layer 1:
model.add(Conv2D(filters=6, kernel_size=5, padding='Same', 
                 input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=2, strides=2))
# Layer 2:
model.add(Conv2D(filters=16, kernel_size=5))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=2, strides=2))
# Layer 3:
model.add(Flatten())
model.add(Dense(120))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
# Layer 4:
model.add(Dense(84))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
# Output layer 5:
model.add(Dense(10))
model.add(BatchNormalization())
model.add(Activation('softmax'))

# Compile the model w/Adam optimizer:
model.compile(optimizer=Adam(lr=1e-3),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# Define a learning rate decay method:
lr_decay = ReduceLROnPlateau(monitor='loss', 
                             patience=1, verbose=1, 
                             factor=0.5, min_lr=1e-7)

# Fit the model with the original dataset --> val_accuracy 0.992:
History = model.fit(train_pixels, train_labels, epochs=40,
                    validation_data=(val_pixels, val_labels),
                    callbacks=[lr_decay], verbose=1)
"""

# Fit the model on batches with real-time data augmentation:
History = model.fit_generator(datagen.flow(train_pixels, train_labels), epochs=40,
                              steps_per_epoch=m_train/32, callbacks=[lr_decay], verbose=1)
"""

# Evaluate the model:
train_loss, train_acc = model.evaluate(train_pixels, train_labels)
val_loss, val_acc = model.evaluate(val_pixels, val_labels)
print(f'model: train accuracy = {round(train_acc * 100, 4)}%')
print(f'model: val accuracy = {round(val_acc * 100, 4)}%')
print(f'model: val error = {round((1 - val_acc) * m_val)} examples')

# Plot the loss and accuracy curves over epochs:
fig, ax = plt.subplots()
ax.plot(History.history['loss'], color='b', label='Training loss')
ax.plot(History.history['val_loss'], color='r', label='Validation loss')
ax.set_title("Loss curves")
legend = ax.legend(loc='best', shadow=True)
predictions = model.predict_classes(test_pixels)

submission = pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                           "Label"  : predictions})
submission.to_csv("submission.csv", index=False, header=True)
submission.head()