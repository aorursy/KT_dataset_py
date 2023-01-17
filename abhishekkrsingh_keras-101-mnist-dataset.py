import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras

import matplotlib.pyplot as plt

import seaborn as sns



import os

print(os.listdir("../input"))

%matplotlib inline

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
x_train = train.drop(columns=['label'])

y_train = train['label']

g = sns.countplot(y_train)
x_train = x_train / 255.0

test = test / 255.0

x_train.describe()
# Creating the network



model = keras.models.Sequential() # Using the Sequentioal feed-forward model

model.add(keras.layers.Dense(256, activation='relu', input_shape=(x_train.shape[1],))) # 1st hidden layer with 128 neurons

#model.add(keras.layers.Dense(128, activation='relu')) # 2nd hidden layer

model.add(keras.layers.Dropout(0.5)) # Randomly deactivates some neurons. (for 0.5, deactivates 50% neurons) Prevents overfitting.

model.add(keras.layers.Dense(128, activation='relu')) # 3rd hidden layer

model.add(keras.layers.Dense(10, activation='softmax')) # Output layer (using softmax activation function as we require categorical values)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',

             metrics=['accuracy'])

from sklearn.model_selection import train_test_split



X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state=np.random.seed(2))
x_train = X_train

y_train = Y_train

history = model.fit(x_train, y_train, epochs=30, validation_data = (X_val,Y_val)) # 30 epochs gave around 98.05% accuracy
# Plot the loss and accuracy curves for training and validation 

fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['acc'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
from sklearn.metrics import confusion_matrix

import itertools





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



# Predict the values from the validation dataset

Y_pred = model.predict(x_train)

# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

# Convert validation observations to one hot vectors

Y_true = y_train

# compute the confusion matrix

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(10)) 
predictions = model.predict([test])
predictions
val_loss, val_acc = model.evaluate(X_val, Y_val)

print(val_loss, val_acc)
results = np.argmax(predictions,axis = 1)  # Since each prediction is a 1-hot array



results = pd.Series(results,name="Label")

#results
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

#print(submission)
submission.to_csv('submission.csv', index=False)