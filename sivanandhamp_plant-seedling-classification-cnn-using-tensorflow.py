# Import necessary libraries.
import cv2
import math
import numpy as np
import pandas as pd
from glob import glob
import seaborn as sns
from matplotlib import pyplot as plt
%matplotlib inline

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, 
    Dropout, 
    Flatten, 
    Conv2D, 
    MaxPooling2D, 
    MaxPool2D,
    GlobalMaxPooling2D,
    BatchNormalization
)
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import RMSprop, Adam
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
# Set the path to the dataset folder. (The dataset contains image folder: "train")
base_path = '/content/drive/My Drive/'
train_path = base_path+"data/plant-seedlings-classification.zip"
extract_path = base_path+'data/Extracted/' # To extract the above seeding classification zip
save_extracted = base_path+'data/Save/'
from google.colab import drive
drive.mount('/content/drive')
# Make different folders for train and test data in the current directory of Google Colab notebook. (using mkdir)
!mkdir extract_path
# Extract the files from dataset to temp_train and temp_test folders (as the dataset is a zip file.)
from zipfile import ZipFile
with ZipFile(train_path, 'r') as zip:
  zip.extractall(extract_path)
# Extract Image and Label
def get_data(path):
  files = glob(path)

  trainImg = []                                              # Initialize empty list to store the image data as numbers.
  trainLabel = []                                            # Initialize empty list to store the labels of images
  j = 1
  num = len(files)
  print("Total #:",num)
  # Obtain images and resizing, obtain labels
  for img in files:
      '''
      Append the image data to trainImg list.
      Append the labels to trainLabel list.
      '''
      print(str(j) + "/" + str(num))
      trainImg.append(cv2.resize(cv2.imread(img), (128, 128)))  # Get image (with resizing to 128x128)
      trainLabel.append(img.split('/')[-2])  # Get image label (folder name contains the class to which the image belong)
      j += 1

  trainImg = np.asarray(trainImg)  # Train images set
  trainLabel = pd.DataFrame(trainLabel)  # Train labels set
  return (trainImg, trainLabel)

# Extract Train dataset
path = extract_path+"train/*/*.png"    # The path to all images in training set. (* means include all folders and files.)
print('reading data from:',path)
trainImg, trainLabel = get_data(path)

print(f"Training image array shape:{trainImg.shape}")
print(f"Training target labels:{trainLabel.shape}")
# Save data to file 
np.save(save_extracted+'trainImg.npy', trainImg)
np.save(save_extracted+'trainLabel.npy', trainLabel)
# Load data to file
trainImg = np.load(save_extracted+'trainImg.npy')
trainLabel = np.load(save_extracted+'trainLabel.npy', False, True)

trainImg.shape, trainLabel.shape
# Check Images
i = 0
img = trainImg[i]
label = trainLabel[0][i]
print(f'Image name:{label}')
plt.imshow(img)
i = 500
img = trainImg[i]
label = trainLabel[0][i]
print(f'Image name:{label}')
plt.imshow(img)
i = 1000
img = trainImg[i]
label = trainLabel[0][i]
print(f'Image name:{label}')
plt.imshow(img)
sobel = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=5)
plt.imshow(sobel)
trainImg = trainImg.astype('float32')
trainImg /= 255
# Check the nomalized data
print(f'Shape of the Train array:{trainImg.shape}')
print(f'Minimum value in the Train Array:{trainImg.min()}')
print(f'Maximum value in the Train Array:{trainImg.max()}')

# Step#1: Split train and test set
X_train, X_test, y_train, y_test = train_test_split(trainImg, trainLabel, test_size=0.3, random_state=42)
X_train.shape, X_test.shape
# Step#2: Split validation from test set
X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
X_test.shape, X_validation.shape
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)
y_validation = encoder.fit_transform(y_validation)
# Display target variable
y_train[0]
# Preview the image before Gaussian Blur
plt.imshow(X_train[1], cmap='gray')
plt.imshow(cv2.GaussianBlur(X_train[1], (15,15), 0))
# Now we apply the gaussian blur to each 128x128 pixels array (image) to reduce the noise in the image
for idx, img in enumerate(X_train):
  X_train[idx] = cv2.GaussianBlur(img, (5, 5), 0)
# Preview the image after Gaussian Blur
plt.imshow(X_train[0], cmap='gray')
# Gaussian Blue to Test and Validation sets
for idx, img in enumerate(X_test):
  X_test[idx] = cv2.GaussianBlur(img, (5, 5), 0)

for idx, img in enumerate(X_validation):
  X_validation[idx] = cv2.GaussianBlur(img, (5, 5), 0)
def create_model(input_shape, num_classes):
  # Initialize CNN Classified
  model = Sequential()

  # Add convolution layer with 32 filters and 3 kernels
  model.add(Conv2D(32, (3,3), input_shape=input_shape, padding='same', activation=tf.nn.relu))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(rate=0.25))

  # Add convolution layer with 32 filters and 3 kernels
  model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation=tf.nn.relu))
  model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(rate=0.25))

  # Add convolution layer with 32 filters and 3 kernels
  model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation=tf.nn.relu))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(rate=0.25))

  # Flatten the 2D array to 1D array
  model.add(Flatten())

  # Create fully connected layers with 512 units
  model.add(Dense(512, activation=tf.nn.relu))
  model.add(Dropout(0.5))


  # Adding a fully connected layer with 128 neurons
  model.add(Dense(units = 128, activation = tf.nn.relu))
  model.add(Dropout(0.5))

  # The final output layer with 12 neurons to predict the categorical classifcation
  model.add(Dense(units = num_classes, activation = tf.nn.softmax))
  return model
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.95):
      print("\nReached 95% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()

es = EarlyStopping(monitor='val_accuracy', mode='min', verbose=1, patience=10)
input_shape = X_train.shape[1:] # Input shape of X_train
num_classes = y_train.shape[1] # Target column size

model = create_model(input_shape, num_classes)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) # Optimizer
# optimizer = tf.keras.optimizers.SGD(lr=1 * 1e-1, momentum=0.9, nesterov=True)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), epochs=30, batch_size=100, callbacks=[callbacks])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('Epoch', fontsize=18)
plt.ylabel(r'Loss', fontsize=18)
plt.legend(('loss train','loss validation'), loc=0)

# Print accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('Epoch', fontsize=18)
plt.ylabel(r'Loss', fontsize=18)
plt.legend(('accuracy train','accuracy validation'), loc=0)
loss, accuracy = model.evaluate(X_test, y_test)
print('Test loss: {:.2f} \n Test accuracy: {:.2f}'.format(loss, accuracy))

loss, accuracy = model.evaluate(X_train, y_train)
print('Train loss: {:.2f} \n Train accuracy: {:.2f}'.format(loss, accuracy))
model1 = create_model(input_shape, num_classes)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) # Optimizer

model1.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model1.summary()
history = model1.fit(X_train, y_train, validation_data=(X_validation, y_validation), epochs=18, callbacks=[callbacks])
loss, accuracy = model1.evaluate(X_test, y_test)
print('Test loss: {:.2f} \n Test accuracy: {:.2f}'.format(loss, accuracy))

loss, accuracy = model1.evaluate(X_train, y_train)
print('Train loss: {:.2f} \n Train accuracy: {:.2f}'.format(loss, accuracy))
from keras.models import load_model
model.save(save_extracted+'final_model.h5')
model.load_weights(save_extracted+'final_model.h5')
y_pred = model1.predict(X_test)
y_pred = (y_pred > 0.5) 
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1)))
df_cm = pd.DataFrame(cm, index = [i for i in range(0,12)],
                     columns = [i for i in range(0,12)])
plt.figure(figsize = (10,7))
sns.heatmap(df_cm, annot=True, fmt='d')
print("=== Classification Report ===")
print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))
y_pred = encoder.inverse_transform(y_pred)

index = 2
plt.imshow(X_test[index], cmap='gray')
print("Predicted label:", y_pred[index])
index = 3
plt.imshow(X_test[index], cmap='gray')
print("Predicted label:", y_pred[index])
index = 33
plt.imshow(X_test[index], cmap='gray')
print("Predicted label:", y_pred[index])
index = 36
plt.imshow(X_test[index], cmap='gray')
print("Predicted label:", y_pred[index])
index = 59
plt.imshow(X_test[index], cmap='gray')
print("Predicted label:", y_pred[index])
# Extract Train dataset
path = extract_path+"test/*.png"  # The path to all images in test set.
# predictionImg, predictionLabel = get_data(path)
filenames = []
for img in glob(path):
  img_split = img.split("/")
  filenames.append(img_split[len(img_split)-1])

print(predictionImg.shape)
print(predictionLabel.shape)
# Saving the data to file for future use [avoiding the extraction]
np.save(save_extracted+'predictionImg.npy', predictionImg)
np.save(save_extracted+'predictionLabel.npy', predictionLabel)
print('saved')
# Load prediction data from directory
predictionImg = np.load(save_extracted+'predictionImg.npy')
predictionLabel = np.load(save_extracted+'predictionLabel.npy', False, True)
predictionImg.shape, predictionLabel.shape
print('------------------------')
predictionImg = predictionImg.astype('float32')
predictionImg /= 255
print(f'Shape of the Prediction array:{predictionImg.shape}')
print(f'Minimum value in the Prediction Array:{predictionImg.min()}')
print(f'Maximum value in the Prediction Array:{predictionImg.max()}')
for idx, img in enumerate(predictionImg):
  predictionImg[idx] = cv2.GaussianBlur(img, (5, 5), 0)
y_pred = model1.predict(predictionImg)
print('Prediction:', y_pred)
y_pred = encoder.inverse_transform(y_pred)
(unique, counts) = np.unique(y_pred, return_counts=True)
print('Unique:',unique, 'Counts:',counts)
y_pred
index = 2
plt.imshow(predictionImg[index], cmap='gray')
print("Predicted label:", y_pred[index])

df = pd.DataFrame(filenames, columns=['file'])
df['species'] = y_pred
df.head()
df.to_csv(save_extracted+"result.csv", index=False)
