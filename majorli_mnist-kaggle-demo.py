import numpy as np
import pandas as pd
import sklearn
import keras

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from keras.utils import to_categorical

import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))
df = pd.read_csv('../input/train.csv')
print('shape',df.shape)
df.head()
from sklearn.model_selection import train_test_split
seed = 66
np.random.seed(seed)
X = df.iloc[:,1:]
Y = df.iloc[:,0]
x_train , x_test , y_train , y_test = train_test_split(X, Y , test_size=0.1, random_state=seed)

# minor preprocessing
x_train = x_train.values.reshape(-1, 28, 28, 1)
x_test = x_test.values.reshape(-1, 28, 28, 1)
x_train = x_train.astype("float32")/255
x_test = x_test.astype("float32")/255
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
plt.setp(((ax1, ax2), (ax3, ax4)), xticks=[], yticks=[])
ax1.imshow(x_train[0,:,:,0],cmap='gray')
ax2.imshow(x_train[1,:,:,0],cmap='gray')
ax3.imshow(x_train[2,:,:,0],cmap='gray')
ax4.imshow(x_train[3,:,:,0],cmap='gray')
plt.show()
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
datagen.fit(x_train)
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', strides=1, padding='same', data_format='channels_last',
                 input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', strides=1, padding='same', data_format='channels_last'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid' ))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))

# Optimizer
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999 )

# Compiling the model
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# Summary
model.summary()
# Training parameters
reduce_lr = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)
batch_size = 256
epochs = 20
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size = batch_size), epochs = epochs, 
                              validation_data = (x_test, y_test), verbose=1, 
                              steps_per_epoch=x_train.shape[0] // batch_size,
                              callbacks = [reduce_lr])
# validation accuracy
model.evaluate(x_test, y_test)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Model Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'])
plt.show()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train','Test'])
plt.show()
test_df = pd.read_csv('../input/test.csv')
test_np = test_df.values
test_np = test_np.astype("float32")/255
test_np = test_np.reshape(-1,28,28,1)
predictions = model.predict(test_np)
predictions = np.argmax(predictions,axis=1) #convert value to 0-9
ImageId = list(range(1,28001))
dict_ans = {'ImageId':ImageId,'Label':predictions}
my_submission = pd.DataFrame(dict_ans)
# print out some rows to check you are doing it correctly
my_submission.head()
# Convert dataframe to csv
my_submission.to_csv('my_submission_demo.csv',index=False)
