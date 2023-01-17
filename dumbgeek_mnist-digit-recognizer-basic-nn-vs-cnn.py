# Importing packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras import regularizers
from keras.datasets import mnist

# loading train and test csv files
df_train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
df_test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

# Check training data structure
df_train.head()
# shape of train and test data
print("Dimension of Train Data: {}".format(df_train.shape))
print("Dimension of Test Data : {}".format(df_test.shape))
df_train.info()
# Split train data into pixels and labels
train_images1 = df_train.drop("label", axis=1)
train_labels1 = pd.DataFrame(df_train["label"])

# test data only contains pixels no labels
test_images1 = df_test

# Check the dimension of train images and test images dataset
print("Dimension train images1:{} ".format(train_images1.shape))
print("Dimension train labels1:{} ".format(train_labels1.shape)+'\n')
print("Dimension test images1:{} ".format(test_images1.shape))
# Now loading MNIST data from Keras package as well
(train_images2, train_labels2), (test_images2, test_labels2) = mnist.load_data()
#Check dimension of loaded data from mnist
print("Shape of train images2:{}".format(train_images2.shape))
print("Shape of train labels2:{}".format(train_labels2.shape)+'\n')
print("Shape of test images2:{}".format(test_images2.shape))
print("Shape of test labels2:{}".format(test_labels2.shape))
## Data manipulation on Keras mnist dataset
train_images2 = train_images2.reshape((60000, 28 * 28))
test_images2 = test_images2.reshape((10000, 28 * 28))

test_cnn = df_test.to_numpy()
test_cnn = test_cnn.reshape((28000,28,28,1))

print("Shape of train images2:{}".format(train_images2.shape))
print("Shape of test images2:{}".format(test_images2.shape))
#converting train_images2 & test_images2 into DataFrame
train_images2 = pd.DataFrame(train_images2, columns= train_images1.columns)
train_labels2 = pd.DataFrame(train_labels2, columns= ["label"])

test_images2  = pd.DataFrame(test_images2, columns= train_images1.columns)
test_labels2  = pd.DataFrame(test_labels2, columns= ["label"])

# Appending train data and train labels with MNIST dataset loaded using keras 
train_images = train_images2.append(test_images2, ignore_index = True)
train_labels = train_labels2.append(test_labels2, ignore_index =True)

train_images = train_images.append(train_images1, ignore_index = True)
train_labels = train_labels.append(train_labels1, ignore_index =True)

print("Shape of train images: {}".format(train_images.shape))
print("Shape of train labels: {}".format(train_labels.shape)+'\n')
# Normalizing train and test data
train_images = train_images/255
test_images1 = test_images1/255

# Categorical encoding for labels
from keras.utils import to_categorical
train_labels = to_categorical(train_labels)

# Creating Neural network type and adding layers
model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
model.add(layers.Dense(10, activation='softmax'))

#Compile model
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
#Train your neural network model
history = model.fit(train_images, train_labels, 
                    epochs=20, 
                    batch_size=128, 
                    validation_split=0.05)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# Now predicting on test data in sample_submission format
sq_pred_labels = model.predict(df_test)

# This step will decode one_hot_encode and give prediction by digits
sq_pred_labels = np.argmax(sq_pred_labels, axis = 1)
sq_pred_labels = pd.DataFrame(sq_pred_labels,columns=["Label"])

# Index column to start from 1
sq_pred_labels.index = np.arange(1, len(sq_pred_labels)+1)

# Assign Index column name
sq_pred_labels.index.name = "ImageId"
# Getting model prediction into csv file
sq_pred_labels.to_csv('/kaggle/working/sq_pred.csv')
#Since convolutional neural network takes images in 2D channels form. So need to reshape train data again
#Train Data reshaping for CNN
train_images_cnn = train_images.to_numpy()
train_images_cnn = train_images_cnn.reshape((112000,28,28,1))
# Creating convolutional nueral network architecture
CNN_model = models.Sequential()

CNN_model.add(layers.Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(28,28,1),padding='same'))
CNN_model.add(layers.MaxPooling2D((2, 2),padding='same'))

CNN_model.add(layers.Conv2D(64, (3, 3), activation='linear',padding='same'))
CNN_model.add(layers.MaxPooling2D(pool_size=(2, 2),padding='same'))

CNN_model.add(layers.Conv2D(64, (3, 3), activation='linear',padding='same'))
CNN_model.add(layers.MaxPooling2D(pool_size=(2, 2),padding='same'))

CNN_model.add(layers.Flatten())
CNN_model.add(layers.Dense(512, kernel_regularizer=regularizers.l1_l2(l1 = 0.001, l2=0.01), activation='relu'))
CNN_model.add(layers.Dropout(rate = 0.4))
CNN_model.add(layers.Dense(10, kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.01), activation='softmax'))
from keras.optimizers import RMSprop

opt = RMSprop(learning_rate=0.0001)

# Compile model
CNN_model.compile(optimizer= opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
#Train your neural network model
history_cnn = CNN_model.fit(train_images_cnn, train_labels, 
                            epochs=50, 
                            batch_size=64, 
                            validation_split=0.1)
# list all data in history
print(history_cnn.history.keys())
# summarize history for accuracy
import matplotlib.pyplot as plt
plt.plot(history_cnn.history['accuracy'])
plt.plot(history_cnn.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history_cnn.history['loss'])
plt.plot(history_cnn.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
test_cnn = df_test.to_numpy()
test_cnn = test_cnn.reshape((28000,28,28,1))
# Now predicting on test data in sample_submission format
cnn_pred_labels = CNN_model.predict(test_cnn)

# This step will decode one_hot_encode and give prediction by digits
cnn_pred_labels = np.argmax(cnn_pred_labels, axis = 1)
cnn_pred_labels = pd.DataFrame(cnn_pred_labels,columns=["Label"])

# Index column to start from 1
cnn_pred_labels.index = np.arange(1, len(cnn_pred_labels)+1)

# Assign Index column name
cnn_pred_labels.index.name = "ImageId"
# Getting CNN preiction into csv file
cnn_pred_labels.to_csv('/kaggle/working/cnn_pred.csv')