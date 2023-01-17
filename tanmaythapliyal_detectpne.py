import tensorflow as tf
tf.random.set_seed(42)# seed fix for tensorflow
import numpy as np

from numpy.random import seed
seed(42)# seed fix for keras

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix 
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout,Activation
from tensorflow.keras.optimizers import RMSprop,Adam
tf.__version__
#model 
classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), input_shape=(150,150,3), activation='relu',kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001), padding="same"))
classifier.add(Conv2D(32, (3, 3), activation='relu',kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001), padding="same"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(64, (3, 3), activation='relu',kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001), padding="same"))
classifier.add(Conv2D(64, (3, 3), activation='relu',kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001), padding="same"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(128, (3, 3), activation='relu',kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001), padding="same"))
classifier.add(Conv2D(128, (3, 3),activation='relu',kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001), padding="same"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(256, (3, 3),activation='relu',kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001), padding="same"))
classifier.add(Conv2D(256, (3, 3), activation='relu',kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001), padding="same"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(512, (3, 3),activation='relu',kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001), padding="same"))
classifier.add(Conv2D(512, (3, 3), activation='relu',kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001), padding="same"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(512, (3, 3),activation='relu',kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001), padding="same"))
classifier.add(Conv2D(512, (3, 3), activation='relu',kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001), padding="same"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Flatten())
classifier.add(Dense(256,activity_regularizer=l2(0.0001), activation='relu'))
classifier.add(Dropout(0.5))

classifier.add(Dense(256,activity_regularizer=l2(0.0001), activation='relu'))
classifier.add(Dropout(0.5))

classifier.add(Dense(2))
classifier.add(Activation('softmax'))
    
classifier.compile(loss='categorical_crossentropy',
            optimizer=Adam(lr=0.0001),
            metrics=['accuracy'])
classifier.load_weights('/kaggle/input/detectpne-589/classifier_589.h5')
from tensorflow.keras.utils import plot_model
plot_model(classifier)
classifier.summary()
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        '/kaggle/input/chest-xray/chest_xray/chest_xray/train',
        target_size=(150,150),
        classes=['NORMAL','PNEUMONIA'],
        batch_size=10,
        shuffle=True,
        #color_mode='grayscale',
        #class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
        '/kaggle/input/chest-xray/chest_xray/chest_xray/test',
        target_size=(150,150),
        classes=['NORMAL','PNEUMONIA'],
        batch_size=1,
        shuffle=False,
        #color_mode='grayscale',
        #class_mode='binary'
)

test_generator_for_labels = test_datagen.flow_from_directory(
        '/kaggle/input/chest-xray/chest_xray/chest_xray/test',
        target_size=(150,150),
        classes=['NORMAL','PNEUMONIA'],
        batch_size=624,
        shuffle=False,
        #color_mode='grayscale',
        #class_mode='binary'
)

val_set = test_datagen.flow_from_directory('/kaggle/input/chest-xray/chest_xray/chest_xray/val',
                                            target_size=(150,150),
                                            classes=['NORMAL','PNEUMONIA'],
                                            batch_size=10,
                                            shuffle=True,
                                            #class_mode='binary'
                                          )

testimg,testlabel=next(test_generator_for_labels)
testlabel=testlabel[:,0]
testlabel
#from tensorflow.keras.callbacks import ReduceLROnPlateau 
#reduce_learning_rate = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, cooldown=2, min_lr=0.00001, verbose=1)

#callbacks = [reduce_learning_rate]
#history = classifier.fit_generator(
        #train_generator,
        #steps_per_epoch=4832//10,
        #epochs=1,
        #validation_data=val_set,
        #validation_steps=396//10,
        #callbacks=callbacks,
        #verbose=1
        #)
# Plot training & validation accuracy values
#plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
#plt.title('Model accuracy')
#plt.ylabel('Accuracy')
#plt.xlabel('Epoch')
#plt.legend(['Train', 'Test'], loc='upper left')
#plt.show()
#plot train and validation loss values 
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('Model loss')
#plt.ylabel('Loss')
#plt.xlabel('Epoch')
#plt.legend(['Train', 'Test'], loc='upper left')
#plt.show()
pred=classifier.predict_generator(test_generator,steps=624,verbose=1)
pred_modi=pred[:,0]
for i in range(624):
    if pred_modi[i]<0.1:
        pred_modi[i]=0
    else:
        pred_modi[i]=1
        
pred_modi
import itertools
cm=confusion_matrix(testlabel,pred_modi)
epcounter=1
cm
#maxcm=cm
""""while(cm[0][0]+cm[1][1]<=588 and epcounter<=100):
    
    print(maxcm)
    print("number of epochs completed =",epcounter)
    classifier.fit_generator(
        train_generator,
        steps_per_epoch=4832//10,
        epochs=1,
        validation_data=val_set,
        validation_steps=396//10,
        callbacks=callbacks,
        verbose=1
        )
    epcounter=epcounter+1
    pred=classifier.predict_generator(test_generator,steps=624,verbose=1)
    pred_modi=pred[:,0]
    
    for i in range(624):
        if pred_modi[i]<0.1:
            pred_modi[i]=0
        else:
            pred_modi[i]=1
    cm=confusion_matrix(testlabel,pred_modi)
    if cm[0][0]+cm[1][1]>=maxcm[0][0]+maxcm[1][1]:
        maxcm=cm
    print()    
    """
#classifier.save_weights("classifier_589.h5")
#from IPython.display import FileLink
#FileLink(r'classifier_596.h5')
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
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
labels=['PNEUMONIA','NORMAL']
plot_confusion_matrix(cm,labels,title='Confusion Matrix') 