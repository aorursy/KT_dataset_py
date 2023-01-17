import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#making random numbers predictable
np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

#one-hot encoding of y
from keras.utils.np_utils import to_categorical

#make it pretty
sns.set(context = 'notebook', palette='cubehelix', style = 'dark')
#get data

train = pd.read_csv("../input/digit-recognizer/train.csv")
test = pd.read_csv("../input/digit-recognizer/train.csv")


Y_train = train['label']
X_train = train.drop(labels= ['label'], axis = 1)
#Check if dropped
X_train.shape
test.shape
Y_test = test['label']
X_test = test.drop(labels= ['label'], axis = 1)
#Test test
X_test.shape
#See our training label distribution
plt.figure(figsize = (8,4))
sns.countplot(Y_train)
plt.show()
Y_train.value_counts()
#check missing values
X_train.isnull().any().describe()
X_test.isnull().any().describe()
#To reduce effect of illumination, we normalise to greyscale
X_train = X_train / 255 
X_test = X_test/255

img_width = 28
img_height = 28
channels = 1
#-1 in the shape makes it compatible with the original shape. 
#We want the new data to have shape (width, height, channels)

X_train = X_train.values.reshape(-1,28,28,1)
X_test = X_test.values.reshape(-1,28,28,1)

X_train.shape

#endoding y as a one hot vector of 10

Y_train = to_categorical(Y_train, num_classes = 10)
#using a 90-10 ratio
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.10, shuffle = True)

#alt: use random seed instead of shuffle
#random_seed = 2
#X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)

print(X_train.shape, Y_train.shape, X_val.shape, Y_val.shape)
#Build

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (img_width,img_height,channels)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))
#Optimiser

#using categorical crossentropy for categorical loss
#using RMSProp with default values
#Gradient Stochastic Descent is slower

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
#Compile
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
epochs = 30
batch_size = 64

#train to see accuracy

history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, validation_data = (X_val, Y_val), 
                    verbose = 2)
#Annealer
#Reduce learning by half if accuracy not improved in 3 epochs
#We use an annealer to decrease the learning rate so it converges to global minima and not local minima
#verbose=1 to give update messages
#min_lr: lower bound on learning rate

annealer = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
#using annealer
history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, validation_data = (X_val, Y_val), 
                    verbose = 2, callbacks= [annealer])
#Augmenting the data by rondomly rotating images by 10 degrees

imggen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        vertical_flip=False) 

imggen.fit(X_train)
#fit model to check accuracy with the augmented

history = model.fit_generator(imggen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size,
                              callbacks=[annealer])


#Accuracy

plt.plot(history.history['accuracy'], color='navy', label="Training Accuracy")
plt.plot(history.history['val_accuracy'], color='purple', label="Validation Accuracy")
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='best')
plt.show()
#Loss

plt.plot(history.history['loss'], color='navy', label="Training Loss")
plt.plot(history.history['val_loss'], color='purple', label="Validation Loss")
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='best')
plt.show()
#Confusion Matrix

Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val,axis = 1) 
cm = confusion_matrix(Y_true, Y_pred_classes)

print(cm)
results = model.predict(X_test)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("DRSubmission.csv",index=False)
