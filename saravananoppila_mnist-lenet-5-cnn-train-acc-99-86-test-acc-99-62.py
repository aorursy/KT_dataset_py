# Install Emnist packages to load data #
!pip install emnist
# Import required packages #
import matplotlib.pyplot as plt,seaborn as sns,pandas as pd,numpy as np
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D,MaxPool2D,Flatten,BatchNormalization
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from emnist import extract_training_samples
from emnist import extract_test_samples
from keras.optimizers import Adam
# extract train and test data #
x_train, y_train = extract_training_samples('digits')
x_test, y_test = extract_test_samples('digits')
# Plot target class #
sns.countplot(y_train)
# Plot train input images #
fig = plt.figure()
for i in range(9):
  plt.subplot(3,3,i+1)
  plt.tight_layout()
  plt.imshow(x_train[i], cmap='gray', interpolation='none')
  plt.title("Digit: {}".format(y_train[i]))
  plt.xticks([])
  plt.yticks([])
# let's print the shape before we reshape and normalize
print("X_train shape", x_train.shape)
print("y_train shape", y_train.shape)
print("X_test shape", x_test.shape)
print("y_test shape", y_test.shape)
# load input train data #
in_train_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
ex_y_train = in_train_data["label"]
# Drop 'label' column
ex_x_train = in_train_data.drop(labels = ["label"],axis = 1) 
# building the input vector from the 28x28 pixels
X_train = x_train.reshape(240000, 28, 28,1)
X_test = x_test.reshape(40000, 28, 28,1)
ex_x_train = ex_x_train.values.reshape(42000,28,28,1)
X_train = np.vstack((X_train, ex_x_train))
print(X_train.shape)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# normalizing the data to help with the training
X_train /= 255
X_test /= 255
# print the final input shape ready for training
print("Train matrix shape", X_train.shape)
print("Test matrix shape", X_test.shape)
y_train = np.concatenate([y_train,ex_y_train.values])
print(y_train.shape)
# one-hot encoding using keras' numpy-related utilities
n_classes = 10
print("Shape before one-hot encoding: ", y_train.shape)
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)
# Build Convolution neural network #
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPool2D(pool_size = 2,strides=2))
model.add(Conv2D(filters=48, kernel_size=(5,5), padding='valid', activation='relu'))
model.add(MaxPool2D(pool_size = 2,strides=2))
model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(10, activation='softmax'))

adam = Adam(lr=5e-4)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)
model.summary()
# Set a learning rate annealer
reduce_lr = ReduceLROnPlateau(monitor='val_acc', 
                                patience=3, 
                                verbose=1, 
                                factor=0.2, 
                                min_lr=1e-6)
# Data Augmentation
datagen = ImageDataGenerator(
            rotation_range=10, 
            width_shift_range=0.1, 
            height_shift_range=0.1, 
            zoom_range=0.1)
history = datagen.fit(X_train)
history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=100), steps_per_epoch=len(X_train)/100, 
                    epochs=20, validation_data=(X_test, Y_test), callbacks=[reduce_lr])
# Fit the model #
#history = model.fit(X_train, Y_train,
#          batch_size=64,
#          epochs=15,
#          verbose=2,
#          validation_split=0.1)
# Evaluate the model with test data
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# Load test data #
test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
test_data = test_data.values
test_data = test_data.reshape(28000, 28, 28,1)
test_data = test_data.astype('float32')
test_data /= 255
print("Test data matrix shape", test_data.shape)
# predict test data #
y_pred = model.predict_classes(test_data, verbose=0)
print(y_pred)
# Plot Accuracy and Loss graph #
f = plt.figure(figsize=(20,7))
f.add_subplot(121)
plt.plot(history.epoch,history.history['accuracy'],label = "accuracy")
plt.plot(history.epoch,history.history['val_accuracy'],label = "val_accuracy")
plt.title("Accuracy Curve",fontsize=18)
plt.xlabel("Epochs",fontsize=15)
plt.ylabel("Accuracy",fontsize=15)
plt.grid(alpha=0.3)
plt.legend()


f.add_subplot(122)
plt.plot(history.epoch,history.history['loss'],label="loss") 
plt.plot(history.epoch,history.history['val_loss'],label="val_loss")
plt.title("Loss Curve",fontsize=18)
plt.xlabel("Epochs",fontsize=15)
plt.ylabel("Loss",fontsize=15)
plt.grid(alpha=0.3)
plt.legend()

plt.show()
# Predict indivdual input image #
i = 9713
predicted_value = np.argmax(model.predict(X_test[i].reshape(1,28, 28,1)))
print('predicted value:',predicted_value)
plt.imshow(X_test[i].reshape([28, 28]), cmap='Greys_r')
submissions=pd.DataFrame({"ImageId": list(range(1,len(y_pred)+1)),
                         "Label": y_pred})
submissions.to_csv("LeNet_CNN.csv", index=False)