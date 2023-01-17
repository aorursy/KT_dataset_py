# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
root_path='/kaggle/input/covid19-radiography-database/COVID-19 Radiography Database/'

size=320

n_epochs=50

current_subset=0

import os

import numpy as np

import pandas as pd

num_sample=0

train_path=root_path

for folder in os.listdir(train_path):

    if(os.path.isdir(train_path+folder)):

        print("Class name {} : {} samples".format(folder,len(os.listdir(train_path+folder))))

        num_sample=num_sample+len(os.listdir(train_path+folder))

print("total ",num_sample)
import os

import numpy as np

import pandas as pd

num_sample=0

train_path=root_path

label_list=[]

file_list=[]

train_list=[]

for folder in os.listdir(train_path):

    if(os.path.isdir(train_path+folder)):

        label_list.extend([folder]*len(os.listdir(train_path+folder)))

        file_list.extend([train_path+folder+'/'+image for image in os.listdir(train_path+folder)])

print(len(label_list),len(file_list))
total=pd.DataFrame({'image':file_list,'label':label_list})

total.sample(5)
from sklearn.model_selection import StratifiedKFold

kf = StratifiedKFold(n_splits=5,shuffle=True,random_state=32)
from sklearn.utils import shuffle

total = shuffle(total,random_state=23)

total_train=total.iloc[:int(0.8*total.shape[0]),:]

total_test=total.iloc[int(0.8*total.shape[0]):,:]

print(total_train.shape)

print(total_test.shape)
subset=0

subset_path='subset'

#os.mkdir('subset')

for train_index, test_index in kf.split(total_train.iloc[:,:-1],total_train.iloc[:,-1]):

    train_data=total.iloc[train_index]

    test_data=total.iloc[test_index]

    print("subset : ",subset,"\t train shape",train_data.shape,"\t test shape",test_data.shape)

    train_data.to_csv(subset_path+str(subset)+'_train.csv')

    test_data.to_csv(subset_path+str(subset)+'_test.csv')

    subset=subset+1
import pandas as pd

train_subset=pd.read_csv(subset_path+str(current_subset)+'_train.csv')

test_subset=pd.read_csv(subset_path+str(current_subset)+'_test.csv')
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dense, Conv2D, Flatten,Dropout,MaxPool2D,GlobalMaxPooling2D

from keras.applications import ResNet50, nasnet

from keras import optimizers

train_generator=ImageDataGenerator(

rescale = 1./255,

featurewise_center=False,  # set input mean to 0 over the dataset

samplewise_center=True,  # set each sample mean to 0

featurewise_std_normalization=False,  # divide inputs by std of the dataset

samplewise_std_normalization=True,  # divide each input by its std

zca_whitening=False,  # apply ZCA whitening

rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

zoom_range = 0.1, # Randomly zoom image 

width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

horizontal_flip=True,  # randomly flip images

vertical_flip=True)  # randomly flip images)



train_data= train_generator.flow_from_dataframe(

batch_size=4,

dataframe=train_subset,

shuffle=True,

seed=32,

class_mode="categorical",

x_col='image',

y_col='label',

target_size=(size,size))



val_data= train_generator.flow_from_dataframe(

batch_size=4,

dataframe=test_subset,

x_col='image',

y_col='label',

shuffle=False,

seed=32,

class_mode="categorical",

target_size=(size,size))
test_generator=ImageDataGenerator(rescale = 1./255,

samplewise_center=True,  # set each sample mean to 0

featurewise_std_normalization=False,  # divide inputs by std of the dataset

samplewise_std_normalization=True)

test_data= test_generator.flow_from_dataframe(

seed=45,



dataframe=total_test,

x_col='image',

y_col='label',shuffle=False,

batch_size=1,

class_mode="categorical",

target_size=(size,size))
!pip install git+https://github.com/titu1994/keras-efficientnets.git

from keras_efficientnets import EfficientNetB5



base_model = EfficientNetB5((size,size,3),  include_top=False,weights='imagenet')
model=Sequential()

model.add(base_model)

model.add(GlobalMaxPooling2D(name="gap"))



model.add(Dense(256,activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(512,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(3, activation='softmax'))
from keras import backend as K

import tensorflow as tf



def focal_loss(gamma=2., alpha=.25):

    def focal_loss_fixed(y_true, y_pred):

        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))

        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return focal_loss_fixed
from sklearn.utils import class_weight

import numpy as np



classes=np.unique(train_data.classes)

class_weight=class_weight.compute_class_weight('balanced', classes, train_data.classes)

class_weight_dict = dict(enumerate(list(class_weight)))

print(class_weight_dict)
train_data.class_indices
from keras.metrics import top_k_categorical_accuracy

from keras.callbacks import ModelCheckpoint

from keras.callbacks import ReduceLROnPlateau

from keras.callbacks import CSVLogger



csv_logger = CSVLogger('log_fold_'+str(current_subset)+'.csv', append=True, separator=',')

checkpoint = ModelCheckpoint('model-fold0-focal.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 

                                    patience=5, 

                                    verbose=1, 

                                    factor=0.1, 

                                    min_lr=0.0000001,mode='min')



model.compile(optimizer=optimizers.adam(lr=0.0001),loss=focal_loss(),metrics=["accuracy"])

history=model.fit_generator(generator=train_data,

                    steps_per_epoch=train_data.samples//train_data.batch_size,

                            validation_data=val_data,

                            verbose=1,class_weight=class_weight_dict,

                            validation_steps=val_data.samples//val_data.batch_size,

                    epochs=30,callbacks=[checkpoint,learning_rate_reduction,csv_logger])
val_pred=model.predict_generator(val_data,verbose=1)

val_pred=np.argmax(val_pred,axis=-1)

label=val_data.classes

from sklearn.metrics import classification_report

print(classification_report(label,val_pred,digits=3))
import numpy as np

import matplotlib.pyplot as plt

import numpy as np

import itertools



def plot_confusion_matrix(cm,

                          target_names,

                          title='Confusion matrix',

                          cmap=None,

                          normalize=True):





    accuracy = np.trace(cm) / np.sum(cm).astype('float')

    misclass = 1 - accuracy



    if cmap is None:

        cmap = plt.get_cmap('Blues')



    plt.figure(figsize=(8, 6))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()



    if target_names is not None:

        tick_marks = np.arange(len(target_names))

        plt.xticks(tick_marks, target_names, rotation=45)

        plt.yticks(tick_marks, target_names)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]





    thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        if normalize:

            plt.text(j, i, "{:0.4f}".format(cm[i, j]),

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")

        else:

            plt.text(j, i, "{:,}".format(cm[i, j]),

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")





    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))

    plt.show()
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(label, val_pred)

plot_confusion_matrix(cm,['COVID-19','Normal','ViralPneumonia'],normalize=False,cmap='Pastel2')
test_pred=model.predict_generator(test_data,verbose=1)

test_pred=np.argmax(test_pred,axis=-1)

label=test_data.classes

from sklearn.metrics import classification_report

print(classification_report(label,test_pred,digits=3))
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(label, test_pred)

plot_confusion_matrix(cm,['COVID-19','Normal','ViralPneumonia'],normalize=False,cmap='Pastel2')
from matplotlib import pyplot as plt

print(history.history.keys())

#  "Accuracy"

plt.style.use('ggplot')



plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['train accuracy', 'validation accuracy'], loc='lower right')

plt.show()

# "Loss"

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model Loss Values')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['train loss values', 'validation loss values'], loc='upper right')

plt.show()