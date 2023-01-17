import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D,BatchNormalization,regularizers

from keras.models import load_model

from keras.preprocessing.image import ImageDataGenerator



import numpy as np

import matplotlib.pyplot as plt

import matplotlib

import os

from IPython.display import clear_output



import time

import datetime



import shutil

print('number of train samples')

print(len(os.listdir('../input/datatree/datatree/train/nv')))

print(len(os.listdir('../input/datatree/datatree/train/mel')))

print(len(os.listdir('../input/datatree/datatree/train/bkl')))

print(len(os.listdir('../input/datatree/datatree/train/bcc')))

print(len(os.listdir('../input/datatree/datatree/train/akiec')))

print(len(os.listdir('../input/datatree/datatree/train/vasc')))

print(len(os.listdir('../input/datatree/datatree/train/df')))


train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=1,zoom_range=0.1)



x_train = train_datagen.flow_from_directory(

    directory=r'../input/datatree/datatree/train/',

    batch_size=40,

    target_size=(75,100),

    class_mode="categorical",

    shuffle=True,

    seed=42

)



validation_datagen = ImageDataGenerator(rescale=1./255,rotation_range=1,zoom_range=0.1)



x_validation = validation_datagen.flow_from_directory(

    directory=r'../input/datatree/datatree/validation/',

    batch_size=31,

    target_size=(75,100),

    class_mode="categorical",

    shuffle=True,

    seed=42

)



test_datagen = ImageDataGenerator(rescale=1./255,rotation_range=1,zoom_range=0.1)



x_test = test_datagen.flow_from_directory(

    directory=r'../input/datatree/datatree/test/',

    batch_size=20,

    target_size=(75,100),

    class_mode="categorical",

    shuffle=False,

    seed=42

)



#ploting one image



p = x_train.next()

print((p[0][0]).shape)

(plt.imshow(p[0][0][:,:,:]) )







class PlotLearning(keras.callbacks.Callback):

    

    

    def on_train_begin(self, logs={}):

        self.i = 0

        self.x = []

        self.losses = []

        self.val_losses = []

        self.acc = []

        self.val_acc = []

        self.fig = plt.figure()

        self.logs = []



    def on_epoch_end(self, epoch, logs={}):

        

        self.logs.append(logs)

        self.x.append(self.i)

        self.losses.append(logs.get('loss'))

        self.val_losses.append(logs.get('val_loss'))

        self.acc.append(logs.get('acc'))

        self.val_acc.append(logs.get('val_acc'))

        self.i += 1

        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

        

        clear_output(wait=True)

        

        ax1.set_yscale('log')

        ax1.plot(self.x, self.losses, label="train loss")

        ax1.plot(self.x, self.val_losses, label="val loss")

        ax1.legend()

        ax2.plot(self.x, self.acc, label="train acc")

        ax2.plot(self.x, self.val_acc, label="validation acc")

        ax2.legend()

        

        plt.show();

        

plot = PlotLearning()


classes_count= 7



model = Sequential()

model.add(Conv2D(32,(3, 3),activation='relu',input_shape=(75,100,3)))



model.add(Conv2D(32,(3, 3),activation='relu'))



model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Dropout(0.2))



model.add(Conv2D(64,(3, 3),activation='relu'))



model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Dropout(0.3))



model.add(Flatten())





model.add(Dense(128, activation='relu'))

 

model.add(Dropout(0.1))



model.add(Dense(classes_count,activation='softmax'))



model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])



model.summary()


start=time.time()



cnn=model.fit_generator(x_train,steps_per_epoch=630,validation_data=x_validation,validation_steps=65,callbacks=[plot],epochs=30,verbose=2)



end=time.time()

print('training time: '+str(datetime.timedelta(seconds=(end-start))))


print('train accuracy     : '+str(cnn.history['acc'][-1]))

print('train loss         : '+str(cnn.history['loss'][-1]))

print('validation accuracy: '+str(cnn.history['val_acc'][-1]))

print('validation loss    : '+str(cnn.history['val_loss'][-1]))
name='model_'+str(cnn.history['acc'][-1])

model.save('model.h5')


predictions=model.predict_generator(x_test,steps=100,verbose=1)

print(predictions.shape)



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix')



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



    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.tight_layout()

    

    

test_labels = x_test.classes

print(test_labels.shape)


import numpy as np

import matplotlib.pyplot as plt

import itertools





from sklearn import svm, datasets

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.utils.multiclass import unique_labels

cm = confusion_matrix(test_labels, predictions.argmax(axis=1))

cm_plot_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel','nv', 'vasc']



plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')

# Get the index of the class with the highest probability score

y_pred = np.argmax(predictions, axis=1)



# Get the labels of the test images.

y_true = x_test.classes



from sklearn.metrics import classification_report



# Generate a classification report

report = classification_report(y_true, y_pred, target_names=cm_plot_labels)



print(report)



label_frac_correct=np.diag(cm)/np.sum(cm,axis=1)

plt.bar(cm_plot_labels,label_frac_correct)

plt.xlabel('True Label')

plt.ylabel('Fraction classified correctly')