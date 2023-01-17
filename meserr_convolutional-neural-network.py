import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
#read train data
train = pd.read_csv("../input/train.csv")
print(train.shape)
train.head()
#read test data
test = pd.read_csv("../input/test.csv")
print(test.shape)
test.head()
#select column from train data
Y_train = train['label']
X_train = train.loc[:, train.columns != 'label'].values
Y_train.shape
#label counts digits
plt.figure(figsize=(15,7))
g = sns.countplot(Y_train, palette="icefire")
plt.title("Number of Digit Classes")
plt.show()
print(Y_train.value_counts())
#figure example
img = X_train[0].reshape(28,28)
plt.axis("off")
plt.imshow(img,cmap='gray')
plt.title(Y_train[0])
plt.show()
#figure example
img = X_train[3].reshape(28,28)
plt.axis("off")
plt.imshow(img,cmap='gray')
plt.title(Y_train[3])
plt.show()
#normalization
X_train = X_train / 255.0
test = test / 255.0
print(X_train.shape)
print(test.shape)
#reshaping data
X_train = X_train.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
print(X_train.shape)
print(test.shape)
#label encoding
from keras.utils.np_utils import to_categorical
Y_train = to_categorical(Y_train, num_classes=10)
Y_train.shape
# Split the train and the validation set for the fitting
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)
print("x_train shape",X_train.shape)
print("x_test shape",X_val.shape)
print("y_train shape",Y_train.shape)
print("y_test shape",Y_val.shape)
#create model
model = Sequential()

# convolotion => max pool => Dropout 1
model.add(Conv2D(filters=16, kernel_size=(5,5), padding='Same', activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# convolotion => max pool => Dropout 2
model.add(Conv2D(filters=16, kernel_size=(3,3), padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

#hidden layer
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))
#Define optimizer adam optimizer : changing learning rate
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
#compile the model
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
epochs = 100
batch_size = 250
#Data Augmentation
datagen = ImageDataGenerator(featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False,
                            samplewise_std_normalization=False, zca_whitening=False, rotation_range=0.5, zoom_range=0.5,
                             width_shift_range=0.5, height_shift_range=0.5, horizontal_flip=False, vertical_flip=False)
datagen.fit(X_train)
history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size), 
                              epochs=epochs, validation_data=(X_val, Y_val),
                              steps_per_epoch=X_train.shape[0] / batch_size)
#figure loss
plt.plot(history.history['val_loss'], color='b', label="validation loss")
plt.title("Test Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
#figure of confusion matrix
import seaborn as sns
from sklearn.metrics import confusion_matrix
# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
