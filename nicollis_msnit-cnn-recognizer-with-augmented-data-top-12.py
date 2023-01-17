import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline
import os

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Lambda, BatchNormalization
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

train_data = np.load('../input/augmented-kaggle-mnsit-dataset/aug_data_obj.npy')
train_label = np.load('../input/augmented-kaggle-mnsit-dataset/aug_labels_obj.npy')
test_data = pd.read_csv('../input/digit-recognizer/test.csv')
train_data = pd.DataFrame(train_data)
train_label = pd.DataFrame(train_label)
train_label.hist()
train_data = train_data.values.astype('float32')
train_label = train_label.values.astype('int32')
test_data = test_data.values.astype('float32')
train_data = train_data.reshape(-1,28,28,1)
test_data = test_data.reshape(-1,28,28,1)
mean_px = train_data.mean().astype(np.float32)
std_px = train_data.std().astype(np.float32)

def standardize(data):
    return (data-mean_px)/std_px
train_label = to_categorical(train_label, num_classes=10)
s_train_data, s_val_data, s_train_label, s_val_label = train_test_split(train_data, train_label, test_size=0.002, stratify=train_label)
train_data=None
from matplotlib import cm

plt.figure(figsize=(15,6))
for i in range(40):  
    plt.subplot(4, 10, i+1)
    plt.imshow(s_train_data[i][:,:,0], cmap=cm.binary)
    plt.title("predict=%d" % np.argmax(s_train_label[i]),y=0.9)
    plt.axis('off')
plt.subplots_adjust(wspace=0.3, hspace=-0.1)
plt.show()

model = Sequential()

model.add(Lambda(standardize, input_shape=(28,28,1), output_shape=(28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),
                     activation ='relu'))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(BatchNormalization())


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(256, activation = "relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

# Define the optimizer
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# Compile the model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

epochs = 30
batch_size = 512
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
datagen.fit(s_train_data)
import scipy.ndimage
history = model.fit_generator(datagen.flow(s_train_data,s_train_label, batch_size=batch_size),
                              epochs = epochs, validation_data = (s_val_data,s_val_label),
                              verbose = 2, steps_per_epoch=s_train_data.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])

results = model.predict(test_data)
results = np.argmax(results,axis = 1)
# Preview predictions
plt.figure(figsize=(15,6))
for i in range(40):  
    plt.subplot(4, 10, i+1)
    plt.imshow(test_data[i][:,:,0])
    plt.title("predict=%d" % results[i],y=0.9)
    plt.axis('off')
plt.subplots_adjust(wspace=0.3, hspace=-0.1)
plt.show()
# Exporting our results for Kaggle
df = pd.DataFrame(results)
df.index.name='ImageId'
df.index+=1
df.columns=['Label']
df.to_csv('submission.csv', header=True)
