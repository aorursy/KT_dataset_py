#Imports

import numpy as np

import pandas as pd



#Visualizations

import matplotlib.pyplot as plt

import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

sns.set_style("whitegrid")



#Modeling

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score



#Keras

import keras

from keras.models import Sequential

from keras.layers import Dense, MaxPool2D, Conv2D, Flatten, Dropout, BatchNormalization

from keras.losses import categorical_crossentropy

from keras.utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau
#Data import

train = pd.read_csv("/kaggle/input/Kannada-MNIST/train.csv")

test = pd.read_csv("/kaggle/input/Kannada-MNIST/test.csv")

validation = pd.read_csv("/kaggle/input/Kannada-MNIST/Dig-MNIST.csv")

train.head()
sns.countplot(train["label"])

plt.title("Distribution of Digit Samples in the Training Set")
train.groupby(train["label"]).size()

validation.head()
sns.countplot(validation["label"])

plt.title("Distribution of Digit Samples in Validation Set")
validation.groupby(validation["label"]).size()
test.head()
#Spliting off labels/Ids

train_labels = to_categorical(train.iloc[:,0])

train = train.iloc[:, 1:].values



X_validation = validation.iloc[:, 1:].values

y_validation = to_categorical(validation.iloc[:,0])



test_id = test.iloc[:, 0]

test = test.iloc[:, 1:].values
#Normalizing the data

train = train/255

X_validation = X_validation/255



test = test/255
#Reshaping data

train = train.reshape(train.shape[0], 28, 28, 1)



X_validation = X_validation.reshape(validation.shape[0], 28, 28, 1)



test = test.reshape(test.shape[0], 28, 28, 1)
#Visualizing the Training Data

fig, ax = plt.subplots(5, 10)

for i in range(5):

    for j in range(10):

        ax[i][j].imshow(train[np.random.randint(0, train.shape[0]), :, :, 0], cmap = plt.cm.binary)

        ax[i][j].axis("off")

plt.subplots_adjust(wspace=0, hspace=0)

fig.set_figwidth(15)

fig.set_figheight(7)

plt.show()
#Visualizing the Validation Data

fig, ax = plt.subplots(5, 10)

for i in range(5):

    for j in range(10):

        ax[i][j].imshow(X_validation[np.random.randint(0, X_validation.shape[0]), :, :, 0], cmap = plt.cm.binary)

        ax[i][j].axis("off")

plt.subplots_adjust(wspace=0, hspace=0)

fig.set_figwidth(15)

fig.set_figheight(7)
#Augmenting data

train_datagen = ImageDataGenerator(

    rotation_range=12,

    width_shift_range=0.25,

    height_shift_range=0.25,

    shear_range=12,

    zoom_range=0.25

)



valid_datagen = ImageDataGenerator(    

    rotation_range=12,

    width_shift_range=0.25,

    height_shift_range=0.25,

    shear_range=12,

    zoom_range=0.25)



valid_datagen_simple = ImageDataGenerator()
#Splitting train/test sets

X_train, X_test, y_train, y_test = train_test_split(train, train_labels, test_size = 0.2, random_state = 84)

def build_model():

    model = Sequential()

    

    #First set of covolutional layers

    #Each with 32 unit output

    model.add(Conv2D(32, (3,3), activation = "relu", input_shape = (28,28,1), padding = "same"))

    model.add(BatchNormalization())

    model.add(Conv2D(32, (5,5), strides = (2,2) ,activation = "relu", padding = "same"))

    model.add(BatchNormalization())

    model.add(Dropout(0.2))

    

    #Second set of covolution layers

    #Each with a 64 unit output

    model.add(Conv2D(64, (3,3), activation = "relu", input_shape = (28,28,1), padding = "same"))

    model.add(BatchNormalization())

    model.add(Conv2D(64, (5,5), strides = (2,2) ,activation = "relu", padding = "same"))

    model.add(BatchNormalization())

    model.add(Dropout(0.2))

    

    #Third set of covolution layers

    #Each with a 128 unit output    

    model.add(Conv2D(128, (3,3), activation = "relu", padding = "same"))

    model.add(BatchNormalization())

    model.add(Conv2D(128, (5,5), strides = (2,2), activation = "relu", padding = "same"))

    model.add(BatchNormalization())

    model.add(Dropout(0.25))

    

    #Pooling Layer

    model.add(MaxPool2D(2,2))

    model.add(Dropout(0.2))

    

    #Fourth and final covolution layer

    #Output of 256 units

    model.add(Conv2D(256, (3,3), activation = "relu", padding = "same"))

    model.add(BatchNormalization())

    model.add(Dropout(0.2))

    

    #Flatteing model to pass to dense layer

    model.add(Flatten())

    

    #First Dense Layer    

    model.add(Dense(128, activation = "relu"))

    model.add(BatchNormalization())

    model.add(Dropout(0.2))

    

    #Final Dense layer

    #Used to predict the ten label classes    

    model.add(Dense(10, activation = "softmax"))

    

    #Compile the model

    model.compile(optimizer = "adam", loss = categorical_crossentropy, metrics = ["accuracy"])

    

    return(model)

#model

model = build_model()
#fitting model on training split

history = model.fit_generator(

    train_datagen.flow(X_train, y_train, batch_size = 1024), 

    epochs = 50,

    steps_per_epoch=50,

    #shuffle = True, this only has effect when steps_per_epoch = "None" 

    validation_data = (X_test, y_test))
pd.DataFrame(history.history).describe().iloc[1:,:]
y_test_labels = []

for i in y_test:

    for j, val in enumerate(i):

        if val == 0.:

            pass

        else:

            y_test_labels.append(j)
#predicting on the test split

preds = history.model.predict_classes(X_test)
#accuracy of the predictions

accuracy_score(preds, np.array(y_test_labels))
plt.figure(figsize=(9,7))

plt.plot(history.history["accuracy"])

plt.plot(history.history["val_accuracy"])

plt.title("Accuracy on Training Data vs. Accuracy on Validation Data\nTraining Step 1")

plt.legend(["Train", "Validation"], loc = "lower right")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.savefig("Accuracy_TrainS1.png")
plt.figure(figsize = (9,7))

plt.plot(history.history["loss"])

plt.plot(history.history["val_loss"])

plt.title("Loss on Training Data vs. Loss on Validation Data\nTraining Step 1")

plt.legend(["Train", "Validation"], loc = "upper right")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.savefig("Loss_TrainS1.png")
#reinitialize model

model = build_model()
#Fitting on full training set

#validation set used for validation

history_full = model.fit_generator(

    train_datagen.flow(train, train_labels, batch_size = 1024), 

    epochs = 50,

    steps_per_epoch=train.shape[0]//1024,

    validation_data = valid_datagen.flow(X_validation, y_validation))
pd.DataFrame(history_full.history).describe().iloc[1:,:]#

preds = history_full.model.predict_classes(X_validation)

accuracy_score(preds, np.argmax(y_validation, axis = 1))

pd.crosstab(preds, validation.iloc[:,0])

plt.figure(figsize=(9,7))

plt.plot(history_full.history["accuracy"])

plt.plot(history_full.history["val_accuracy"])

plt.title("Accuracy on Training Data vs. Accuracy on Validation Data\nTraining Step 2")

plt.legend(["Train", "Validation"], loc = "lower right")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.savefig("Accuracy_TrainS2.png")
plt.figure(figsize = (9,7))

plt.plot(history_full.history["loss"])

plt.plot(history_full.history["val_loss"])

plt.title("Loss on Training Data vs. Loss on Validation Data\nTraining Step 2")

plt.legend(["Train", "Validation"], loc = "upper right")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.savefig("Loss_TrainS2.png")
#concatenating data

X_train_total = np.concatenate((train, X_validation))

y_train_total = np.concatenate((train_labels, y_validation))

X_train_total.shape

y_train_total.shape

#Reinitializing model

model = build_model()
history_final = model.fit_generator(train_datagen.flow(

    X_train_total, y_train_total, batch_size=1024),

    epochs = 50, 

    steps_per_epoch = X_train_total.shape[0]//1024,

    validation_data = valid_datagen.flow(X_validation, y_validation))


plt.figure(figsize=(9,7))

plt.plot(history_final.history["accuracy"])

plt.plot(history_final.history["val_accuracy"])

plt.title("Accuracy on Training Data vs. Accuracy on Validation Data\nTraining Step 3")

plt.legend(["Train", "Validation"], loc = "lower right")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.savefig("Accuracy_TrainS3.png")
plt.figure(figsize = (9,7))

plt.plot(history_final.history["loss"])

plt.plot(history_final.history["val_loss"])

plt.title("Loss on Training Data vs. Loss on Validation Data\nTraining Step 3")

plt.legend(["Train", "Validation"], loc = "upper right")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.savefig("Loss_TrainS3.png")
preds = history_final.model.predict_classes(test)

preds[:10]
submission = pd.DataFrame(data = [pd.Series(test_id, name = "id"), pd.Series(preds, name = "label")], ).T
submission.to_csv("submission.csv", index = False)
