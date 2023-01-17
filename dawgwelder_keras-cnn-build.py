import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns 
sns.set()
from scipy import misc
import imageio as im
import os
import warnings
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import itertools
warnings.filterwarnings('ignore')
%config InlineBackend.figure_format = 'retina'
path = "/kaggle/input/shapes/shapes/"
os.chdir(path)
os.getcwd()
img = im.imread('circles/drawing(1).png')
plt.imshow(img, cmap='gray')
def make_labels(directory, data=[], y_hat=[], label=0):
    for root, dirs, files in os.walk(directory):
        for file in files:
            img = im.imread(directory+file)
            data.append(img)
        y_hat = [label] * len(data)
    return np.array(data), np.array(y_hat)
circles, y_circles = [], []
circles, y_circles = make_labels('circles/', data=circles, y_hat=y_circles)

squares, y_squares = [], []
squares, y_squares = make_labels('squares/', data=squares, y_hat=y_squares, label=1)

triangles, y_triangles = [], []
triangles, y_triangles = make_labels('triangles/', data=triangles, y_hat=y_triangles, label=2)
print(circles.shape, squares.shape, triangles.shape)
print(y_circles.shape, y_squares.shape, y_triangles.shape)
X, y = np.vstack((circles, squares, triangles)), np.hstack((y_circles, y_squares, y_triangles)).reshape(-1, 1)
X.shape, y.shape
import tensorflow as tf

config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
sess = tf.Session(config=config) 
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
 
# AlexNet-like
def createModel(input_shape, nclasses):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Flatten())
    model.add(Dense(512, activation='relu')) # we can drop 
    model.add(Dropout(0.5))                  # this layers
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nclasses, activation='softmax'))
         
    return model
model = createModel(img.shape, nclasses=3)
model_gen = createModel(img.shape, nclasses=3)
model.summary()
from sklearn.preprocessing import OneHotEncoder

oh = OneHotEncoder()
oh.fit(y)
y_hot = oh.transform(y)

from keras.utils.np_utils import to_categorical

y_cat = to_categorical(y)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y_hot, test_size=.2, random_state=42)
x_train, x_test, y_cat_train, y_cat_test = train_test_split(X, y_cat, test_size=.2, random_state=42)
%%time
batch_size = 40
epochs = 60

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model_gen.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
 
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, 
                   validation_data=(X_test, y_test))

scores = model.evaluate(X_test, y_test)
print(f'Logloss = {scores[0]: .5f} \nAccuracy = {scores[1]: .2f}')
from keras.preprocessing.image import ImageDataGenerator

# we can use image augmentation
# basically it needs to redifine for normal actual scores like 0.9 of accuracy and more
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=12,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=False)
datagen.fit(x_train)
%%time
epochs=120
history_generator = model_gen.fit_generator(datagen.flow(x_train, y_cat_train, batch_size=batch_size),
                    epochs=epochs, 
                    validation_data=(x_test, y_cat_test))
scores = model_gen.evaluate(x_test, y_cat_test)
print(f'Logloss = {scores[0]: .5f} \nAccuracy = {scores[1]: .2f}')
# https://www.kaggle.com/danbrice/keras-plot-history-full-report-and-grid-search
def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return 
    
    ## As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)
    
    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:    
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title='Normalized confusion matrix'
    else:
        title='Confusion matrix'

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
## multiclass or binary report
## If binary (sigmoid output), set binary parameter to True
def full_multiclass_report(model,
                           x,
                           y_true,
                           classes,
                           batch_size=32,
                           binary=False):

    # 1. Transform one-hot encoded y_true into their class number
    if not binary:
        y_true = np.argmax(y_true,axis=1)
    
    # 2. Predict classes and stores in y_pred
    y_pred = model.predict_classes(x, batch_size=batch_size)
    
    # 3. Print accuracy score
    print("Accuracy : "+ str(accuracy_score(y_true,y_pred)))
    
    print("")
    
    # 4. Print classification report
    print("Classification Report")
    print(classification_report(y_true,y_pred,digits=5))    
    
    # 5. Plot confusion matrix
    cnf_matrix = confusion_matrix(y_true,y_pred)
    print(cnf_matrix)
    plot_confusion_matrix(cnf_matrix,classes=classes)
plot_history(history)
full_multiclass_report(model,
                       X_test,
                       y_test,
                       ['circles', 'squares', 'triangles'])
plot_history(history_generator)
full_multiclass_report(model_gen,
                       X_test,
                       y_test,
                       ['circles', 'squares', 'triangles'])
