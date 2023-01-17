# Data handling, processing and visualisations

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline



# Couple of sklearn operations

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



# Deep learning tools from the keras library

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras import layers

from keras.layers.normalization import BatchNormalization

from keras import models

from keras.utils.np_utils import to_categorical

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau
# Read in data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
# Set up y_train

y_train = train["label"]



# Plot graph of labels

g = sns.countplot(y_train)



# Get value counts

y_train.value_counts()



# Drop 'label' column

X_train = train.drop(labels = ["label"],axis = 1) 



# Free up memory by deleting the train file

del train
X_train = X_train / 255.0

test = test / 255.0
# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)

X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])

y_train = to_categorical(y_train, num_classes = 10)
# Set the random seed

random_seed = 2



# Split the train and the validation set for the fitting

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.15, random_state=random_seed)
# Set up sequential model

model = models.Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(256, activation = "relu"))
model.add(BatchNormalization())
model.add(Dense(10, activation = "softmax"))
my_optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=my_optimizer,

              loss='categorical_crossentropy',

              metrics=['accuracy'])
datagen = ImageDataGenerator(rotation_range=10,

                             zoom_range=0.1,

                             width_shift_range=0.1,

                             height_shift_range=0.1,

                             fill_mode='nearest')  



datagen.fit(X_train)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=86),

                              epochs=100, validation_data = (X_val, y_val),

                              verbose = 2, steps_per_epoch=X_train.shape[0] // 86,

                              callbacks=[learning_rate_reduction]) 
# Cheking the model's performance

test_loss, test_acc = model.evaluate(X_val, y_val)



test_acc
def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

### End of function definition

    

# Predict the values from the validation dataset

y_pred = model.predict(X_val)



# Convert predictions classes to one hot vectors 

y_pred_classes = np.argmax(y_pred,axis = 1) 

# Convert validation observations to one hot vectors

y_true = np.argmax(y_val,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(y_true, y_pred_classes) 

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(10))
# Plotting accuracy and loss curves

acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()



plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()
# predict results

results = model.predict(test)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
# Concat prediction labels with the Image ID

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



# Create final csv file for submission

submission.to_csv("cnn_mnist_datagen.csv",index=False)