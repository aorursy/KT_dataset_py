#Libraries

import numpy as np 

import pandas as pd 

import os



np.random.seed(2)



import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline



from sklearn.model_selection import train_test_split



from tensorflow.keras import backend as K

from tensorflow.keras.utils import to_categorical # convert to one-hot-encoding

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout

from tensorflow.keras.optimizers import Adam





from keras.callbacks import ReduceLROnPlateau



sns.set(style='white', context='notebook', palette='deep')
#data

for dirname, _, filenames in os.walk('/kaggle/input'):

   for filename in filenames:

       print(os.path.join(dirname, filename))



path = "/kaggle/input/digit-recognizer"



train_data = pd.read_csv(path+"/train.csv")

test_data = pd.read_csv(path+"/test.csv")

Y_train = train_data['label']



#Drop 'label' column

X_train = train_data.drop(labels = ['label'], axis = 1)



#del train_data for space

#del train_data



g = sns.countplot(Y_train)



#Y_train.value_counts()
X_train.isnull().any().describe()
test_data.isnull().any().describe()
#Normalize data

X_train = X_train / 255.0

test_data = test_data / 255.0
#Constants

NUM_CLASSES = 10

IMG_HEIGHT = 28

IMG_WIDTH = 28

CHANNELS = 1
#reshape image into 3 dimensions height/width/channels

X_train = X_train.values.reshape(-1, IMG_HEIGHT, IMG_WIDTH, CHANNELS)

X_test = test_data.values.reshape(-1, IMG_HEIGHT, IMG_WIDTH, CHANNELS)

print("x_train shape: ",X_train.shape)

print("test shape: ",test_data.shape)
# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])

Y_train = to_categorical(Y_train, num_classes = NUM_CLASSES)
# Set the random seed

random_seed = 2



#Hyperparameters

TEST_SIZE = 0.1

BATCH_SIZE = 128

EPOCHS = 10
#split the train and validation set 

X_train, X_val, y_train, y_val = train_test_split(X_train, Y_train, 

                                                  test_size = TEST_SIZE, 

                                                  random_state = random_seed)
#Model

model = Sequential([

    Conv2D(filters = 32, kernel_size = (3, 3), activation = K.relu, input_shape = (IMG_WIDTH, IMG_HEIGHT, 1)),

    MaxPooling2D(pool_size = (2, 2)),

    Conv2D(filters = 64, kernel_size = (3, 3), activation = K.relu),

    MaxPooling2D(pool_size = (2, 2)),

    Flatten(),

    Dense(units = 64, activation = K.relu),

    Dropout(0.2),

    Dense(units = 32, activation = K.relu),

    Dropout(0.1),

    Dense(units = NUM_CLASSES, activation = K.softmax)

])
#model

model.summary()
#Compiler

model.compile(loss = K.categorical_crossentropy,

              optimizer = Adam(), 

              metrics = ['accuracy'])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
#Fit

history = model.fit(X_train, y_train,

          batch_size = BATCH_SIZE, 

          epochs = EPOCHS, 

          verbose = 2, 

          validation_data = (X_val, y_val))
# Plot the loss and accuracy curves for training and validation 

fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
predicted_classes = model.predict_classes(X_test)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predicted_classes)+1)),

                         "Label": predicted_classes})

submissions.to_csv("mnist_cnn.csv", index=False, header=True)
model.save('model1.h5')

json_string = model.to_json()