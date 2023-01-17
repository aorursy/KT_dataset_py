import pandas as pd
import numpy as np
np.random.seed(0)

from os.path import join
import random

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
%matplotlib inline
import itertools

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
sns.set(style = 'white' , context = 'notebook' , palette = 'deep')
rcParams['figure.figsize'] = 10,6
data_path = "../input/digit-recognizer/"
train  = pd.read_csv(join(data_path,"train.csv"))
test  = pd.read_csv(join(data_path,"test.csv"))
Y_train = train["label"]
X_train = train.drop(labels = ["label"], axis = 1)
del train

p = sns.countplot(Y_train)
def process_data(data):
    data = data/255.0
    data = data.values.reshape(-1,28,28,1)
    return data

X_train = process_data(X_train)
Y_train = to_categorical(Y_train,num_classes = 10)
def plot_image(size,images):
    if len(images)!= size[0]*size[1]:
        raise Exception("number of images doesn't match the size of plot")
    fig, ax = plt.subplots(size[0],size[1],figsize=(10,10))
    for i in range(size[0]):
        for j in range(size[1]):
            ax[i][j].imshow(images[i*size[1]+j][:,:,0], cmap = "gray_r")

images = random.sample(list(X_train),9)
plot_image((3,3),images)
num = 5 #make it 10
model = [0]*num
for i in  range(num):
    model[i] = Sequential()
    model[i].add(Conv2D(filters = 32, kernel_size = (5,5), padding = "same", activation = "relu", input_shape = (28,28,1)))
    model[i].add(BatchNormalization())
    model[i].add(Conv2D(filters = 32, kernel_size = (5,5), padding = "same", activation = "relu"))
    model[i].add(BatchNormalization())
    model[i].add(MaxPool2D(pool_size = (2,2)))
    model[i].add(Dropout(0.25))
    model[i].add(Conv2D(filters = 64, kernel_size = (3,3), padding = "same", activation = "relu"))
    model[i].add(BatchNormalization())
    model[i].add(Conv2D(filters = 64, kernel_size = (3,3), padding = "same", activation = "relu"))
    model[i].add(BatchNormalization())
    model[i].add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model[i].add(Dropout(0.25))
    model[i].add(Conv2D(filters = 64, kernel_size = (3,3), padding = "same", activation = "relu"))
    model[i].add(BatchNormalization())
    model[i].add(Dropout(0.25))
    model[i].add(Flatten())
    model[i].add(Dense(1024, activation = "relu"))
    model[i].add(BatchNormalization())
    model[i].add(Dropout(0.25))
    model[i].add(Dense(10, activation = "softmax"))
    model[i].compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
model[0].summary()
datagen = ImageDataGenerator(featurewise_center = False,
                             samplewise_center = False,
                             featurewise_std_normalization = False,
                             samplewise_std_normalization = False,
                             zca_whitening = False,
                             rotation_range = 10,
                             zoom_range = 0.1,
                             width_shift_range = 0.1,
                             height_shift_range = 0.1,
                             horizontal_flip = False,
                             vertical_flip = False
                            )
epochs = 32
batch_size = 256
history = [0]*num
for i in range(num):
    random_seed = i
    X_train_, X_val_, Y_train_, Y_val_ = train_test_split(X_train, Y_train, test_size = 0.2, random_state=random_seed)
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                                patience=3,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)
    early_stopping = EarlyStopping(monitor='val_accuracy',
                                   min_delta=0,
                                   patience=10, 
                                   verbose =1, 
                                   mode='auto')
    history[i] = model[i].fit_generator(datagen.flow(X_train_, Y_train_, batch_size = batch_size), epochs = epochs, validation_data = (X_val_,Y_val_), verbose = 1, steps_per_epoch = X_train.shape[0]//batch_size, callbacks=[learning_rate_reduction,early_stopping])
    print("CNN {0:d}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(i+1,epochs,max(history[i].history['accuracy']),max(history[i].history['val_accuracy'])))    
def predict(X_data):
    results = np.zeros((X_data.shape[0],10))
    for j in range(num):
        results = results + model[j].predict(X_data)
    return results
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
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

    
Y_pred = predict(X_val_)
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
Y_true = np.argmax(Y_val_,axis = 1) 
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
plot_confusion_matrix(confusion_mtx, classes = range(10)) 
test = process_data(test)
results = predict(test)
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("keras_cnn_mnist_aug_ensemble2.csv",index=False)