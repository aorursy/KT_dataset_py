#Data Processing
import pandas as pd

#1-loading the data
train = pd.read_csv("../input/train.csv")
X_test = pd.read_csv("../input/test.csv")
Y_train = train['label']
X_train = train.drop(labels = ['label'], axis = 1)
del train

#2- see if the data is equally distributed or not
Y_train.value_counts()

#3- find if null value present
X_train.isnull().any().describe()
X_test.isnull().any().describe()

#4- normalize the data
X_train = X_train/255.0
X_test = X_test/255.0

#Remeber these are the dataframe so reshape it into ndarray
X_train = X_train.values.reshape(-1,28,28,1)
X_test = X_test.values.reshape(-1,28,28,1)
#encode Y value to 0-1 vector of size 10
from keras.utils.np_utils import to_categorical
Y_train = to_categorical(Y_train, num_classes = 10)

#splitting train and validation set
random_seed = 3

from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size= 0.2, random_state = random_seed) 

#data processing done
#develop CNN model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'Same', activation = 'relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'Same', activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', activation = 'relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation = 'softmax'))
#Define optimizer and compile the whole model
from keras.optimizers import RMSprop
optimizer = RMSprop(lr = 0.001, rho = 0.9, epsilon = 1e-08, decay = 0.0)
model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics = ["accuracy"])

#learning rate annealer
from keras.callbacks import ReduceLROnPlateau
learning_rate = ReduceLROnPlateau(monitor = 'val_acc', patience = 3, verbose = 1, factor = 0.5, min_lr = 0.00001)

epochs = 1
batch_size = 86
#without data augumentation
do_train = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, validation_data = (X_val, Y_val), verbose = 2)
#plot loss and accuracy graph of training model
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2,1)
ax[0].plot(do_train.history['loss'], color = 'b', label = "Training Loss")
ax[0].plot(do_train.history['val_loss'], color = 'r', label = "Validation Loss", axes = ax[0])
legend = ax[0].legend(loc = 'best', shadow = True)

ax[1].plot(do_train.history['acc'], color = 'b', label = "Training Accuracy")
ax[1].plot(do_train.history['val_acc'], color = 'r', label = "Validation Accuracy", axes = ax[1])
legend = ax[1].legend(loc = 'best', shadow = True)
#confusion matrix
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(cm, classes, normalize = False, title = 'Confusion_Matrix', cmap = plt.cm.Blues):
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float')/cm.sum(axis = 1)[:,np.newaxis]
    
    thresh = cm.max()/2
    
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i,j], horizontalalignment = "center", color = "White" if cm[i,j] > thresh else "Black")
    plt.tight_layout()
    plt.xlabel('Predicted label')
    plt.ylabel('True label')

Y_pred = model.predict(X_val)
Y_pred_classes = np.argmax(Y_pred, axis = 1)

Y_true = np.argmax(Y_val, axis = 1)
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)

plot_confusion_matrix(confusion_mtx, classes = range(10))
#predict the results
results = model.predict(X_test)
results = np.argmax(results, axis = 1)
results = pd.Series(results, name = 'Label')

final_result = pd.concat([pd.Series(range(1, 28001), name = 'ImageId'), results], axis = 1)
final_result.to_csv("predict_digit_cnn.csv", index = False)
