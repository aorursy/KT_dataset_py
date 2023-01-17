import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

from PIL import Image

sns.set(style='white', context='notebook', palette='deep')
# Load the data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
# Get labels
Y_train = train["label"]

# Drop 'label' column to get only the images
X_train = train.drop(labels = ["label"],axis = 1).values

# Get test subset values
X_test = test.values

# free some space
del train
del test
print('X_train\'s shape: ' + str(X_train.shape))
print('Y_train\'s shape: ' + str(Y_train.shape))
print()
print('X_test\'s shape: ' + str(X_test.shape))
# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)

print('X_train\'s shape: ' + str(X_train.shape))
print('X_test\'s shape: ' + str(X_test.shape))
indices = np.random.randint(0, 42000, 16)

plt.figure(figsize=(20,5))
i = 0
for index in indices:
    ax = plt.subplot(2, 8, i + 1)
    ax.set_title('Label: ' + str(Y_train[index]))
    ax.imshow(X_train[index][:,:,0])
    i += 1
print(Y_train.value_counts())

sns.countplot(Y_train)
# Normalizing the data
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
Y_train = to_categorical(Y_train, num_classes = 10)

print('Y_train\'s shape: ' + str(Y_train.shape))
# Split the train and the validation set for the fitting
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=0)
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.3))
model.add(Dense(10, activation = "softmax"))
# Compile the model
model.compile(optimizer='RMSprop' , loss='categorical_crossentropy', metrics=['accuracy'])
# learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_acc', 
    patience=3, 
    verbose=1, 
    factor=0.5, 
    min_lr=0.00001)
# Data augmentation to prevent overfitting

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
epochs = 30
batch_size = 100

model.fit_generator(
    datagen.flow(X_train,Y_train, batch_size=batch_size),
    epochs = epochs,
    validation_data = (X_val,Y_val),
    verbose = 1,
    callbacks=[learning_rate_reduction])
(loss_train, accuracy_train) = model.evaluate(X_train, Y_train)
print('Performance on train data:')
print('Loss: ' + str(loss_train))
print('Accuracy: ' + str(accuracy_train*100) + '%')

(loss_val, accuracy_val) = model.evaluate(X_val, Y_val)
print('Performance on validation data:')
print('Loss: ' + str(loss_val))
print('Accuracy: ' + str(accuracy_val*100) + '%')
# predict results
results = model.predict(X_test)

# select the index with the maximum probability
results = np.argmax(results,axis = 1)
# insert the predictions into a pandas series
results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("mnist_predictions_cnn_datagen.csv",index=False)