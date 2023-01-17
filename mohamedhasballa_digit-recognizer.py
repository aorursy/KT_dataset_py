# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Import The Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
sns.set(style='white', context='notebook', palette='deep')
# Import The Dataset

train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
submissoion = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
train
test
# Split The Train_Set

y_train = train['label']
X_train = train.drop('label', axis = 1)
g = sns.countplot(y_train.value_counts())
del train
# Check For Null Values

X_train.isnull().any().describe()
test.isnull().any().describe()

X_train = X_train / 255.0
test = test / 255.0
# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)

X_train = X_train.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1, 28, 28, 1)
# Label Encoding
# # Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])

y_train = to_categorical(y_train, num_classes = 10)
# Split The Data
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size= 0.1, random_state = 2)
g = plt.imshow(X_train[0][:, :, 0])
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from keras.optimizers import RMSprop

model = Sequential()
model.add(Conv2D(32, (5,5), activation = 'relu', padding = 'Same', input_shape = (28, 28, 1)))
model.add(Conv2D(32, (5,5), activation = 'relu', padding = 'Same', input_shape = (28, 28, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (5,5), activation = 'relu', padding = 'Same'))
model.add(Conv2D(64, (5,5), activation = 'relu', padding = 'Same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10, activation = 'softmax'))
model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()
#Early Stop :- To prevent over fitting we will stop the learning after 10 epochs and val_loss value not decreased

earlystop = EarlyStopping(patience=10)

#Learning Rate Reduction :-  We will reduce the learning rate when the accuracy not increase for 2 steps

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
callbacks = [earlystop, learning_rate_reduction]
batch_size = 86
epochs = 2
# Data augmentation

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

datagen.fit(X_train)
# With epochs = 10 i get 99% accuracy, here i will do just 2 epoch
history = model.fit_generator(datagen.flow(X_train, y_train, batch_size = batch_size),
                            epochs = epochs, 
                            validation_data = (X_test, y_test),
                            verbose = 2,
                            steps_per_epoch=X_train.shape[0] // batch_size,
                            callbacks = callbacks
)
# Save The Model 
model_yaml = model.to_yaml()
with open("model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
# Predict
result = model.predict(test)
# select the indix with the maximum probability
result = np.argmax(result, axis = 1)

result = pd.Series(result, name="Label")
submission = pd.concat([pd.Series(range(1,28001), name = "ImageId"), result],axis = 1)
submission.to_csv("cnn_mnist_datagen.csv", index=False)
