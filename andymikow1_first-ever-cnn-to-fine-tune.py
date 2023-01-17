# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from matplotlib import pyplot as plt #plotting and image showing
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
np.random.seed(95)
from keras import backend as K
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, GlobalMaxPooling2D, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import regularizers as r
from sklearn import metrics


train_datagen = image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

valid_datagen = image.ImageDataGenerator(rescale=1./255)

batch_size=32

train_generator = train_datagen.flow_from_directory(
        '../input/10-monkey-species/training/training',
        batch_size=batch_size,
        class_mode='categorical')

valid_generator = valid_datagen.flow_from_directory(
        '../input/10-monkey-species/validation/validation',
        batch_size=batch_size,
        shuffle=False,
        class_mode='categorical')

epoch = 100

model = Sequential()

model.add(Conv2D(32, (8,8), input_shape=(None,None,3)))
model.add(LeakyReLU())
model.add(Conv2D(32, (2,2),bias_regularizer=r.l2(0.)))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (4,4)))
model.add(LeakyReLU())
model.add(Conv2D(64, (2,2),bias_regularizer=r.l2(0.)))
model.add(LeakyReLU())
model.add(GlobalMaxPooling2D())
model.add(Dense(1024, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

filepath=str(os.getcwd()+"/model.h5f")
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# = EarlyStopping(monitor='val_acc', patience=15)
callbacks_list = [checkpoint]#, stopper]

trained = model.fit_generator(
        train_generator,
        steps_per_epoch=1097 // batch_size,
        epochs=epoch,
        validation_data=valid_generator,
        validation_steps=272 // batch_size, callbacks=callbacks_list, verbose = 1)
model.summary()
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

plot_history(trained)
validation_datagenerator = image.ImageDataGenerator(
    rescale=1. / 255)
testing_data = validation_datagenerator.flow_from_directory(
        '../input/10-monkey-species/validation/validation',
        target_size=(244, 244),
        shuffle=False,
        batch_size = 8,
        class_mode='categorical')
from keras.models import load_model
model_trained = load_model(filepath)

steps = 34
predictions = model_trained.predict_generator(testing_data, steps=steps, verbose=1)

val_preds = np.argmax(predictions, axis=1)
val_trues = testing_data.classes
cm = metrics.confusion_matrix(val_trues, val_preds)
labels = list(testing_data.class_indices.keys())
print(metrics.classification_report(val_trues, val_preds,target_names=labels))
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix - training',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

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
plot_confusion_matrix(cm,labels,normalize=True)
test_list = os.listdir("../input/test-monkeys/")
test_list.sort()
print(test_list)
model_test = load_model(filepath)
import cv2
import matplotlib.image as mpimg


d = {}
score_array = np.array([])
for i in range(10):
    d[str(test_list[i])]="../input/test-monkeys/"+str(test_list[i])
for i in test_list:
    imgorg = cv2.imread(str(d[i]))
    img = imgorg #cv2.resize(imgorg, dsize=(244, 244), interpolation=cv2.INTER_CUBIC)
    print(type(img))
    print(img.shape)
    test_img = np.expand_dims(img, axis=0)
    test_img = test_img/255
    print(test_img.shape)
    score = model_trained.predict(test_img, verbose=1)
    result = np.argmax(score, axis=1)
    score_values = list(range(10))
    score_dict = dict(zip(score_values,labels))
    print('Image was classified to class: ' + '"' + str(score_dict[int(result)]) + '"\nImage should be classified as "'+str(i)[0:2]+'"')
    #if str(score_dict[int(result)])[1:3] == str(i)[0:2]:
        #np.append(score_array,1,axis=1)
    #else:
        #np.append(score_array,1,axis=)
#accuracy_percent = np.sum(score_array) * 100
#print("Model has had " + str(accuracy_percent) +"% accuracy on own test dataset")

def plot_images(test_list, dictionary):
    for i in range(len(test_list)):
        plt.figure(i)
        plt.imshow(mpimg.imread(str(dictionary[test_list[i]])))
        plt.title(test_list[i])
    plt.show()
plot_images(test_list,d)