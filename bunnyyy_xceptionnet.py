import warnings

#ignore Warnings

warnings.filterwarnings('always')
warnings.filterwarnings('ignore')


#Manipulasi data
import numpy as np
import pandas as pd

#Visualisasi Data
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline

#Model Selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

#preprocess
from keras.preprocessing.image import ImageDataGenerator

#dl libraries 
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD, Adagrad, Adadelta, RMSprop
from keras.utils import to_categorical

#specially for CNN
from keras.layers import Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPool2D, MaxPooling2D, BatchNormalization

import tensorflow as tf
import random as rn

#soepecially for manipulating Zipped image and getting numpy arrays
import cv2
import os
from PIL import Image

#save train data and model
import joblib
data = pd.read_csv('/kaggle/input/fer2013-dataset/fer2013.csv')
data.head()
import warnings
warnings.filterwarnings("ignore")

width, height = 48, 48

datapoints = data['pixels'].tolist()

#getting features for training
X = []
for xseq in datapoints:
    xx = [int(xp) for xp in xseq.split(' ')]
    xx = np.asarray(xx).reshape(width, height)
    X.append(xx.astype('float32'))

X = np.asarray(X)
X = np.expand_dims(X, -1)

#getting labels for training
y = data['emotion']

print("Preprocessing Done")
print("Number of Features: "+str(len(X[0])))
print("Number of Labels: "+ str(len(y)))
print("Number of examples in dataset:"+str(len(X)))
print("X,y stored in fdataX.npy and flabels.npy respectively")
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

import random as rn
w=10
h=10
fig=plt.figure(figsize=(10, 15))
columns = 4
rows = 5
for i in range(1, columns*rows +1):
#     img = np.random.randint(10, size=(h,w))
    fig.add_subplot(rows, columns, i)
    angka_random = rn.randint(1,len(X))
    plt.imshow(X[angka_random].reshape(48,48), cmap = 'gray')
    plt.title(labels[y[angka_random]])
plt.show()
# storing them using numpy
np.save('fdataX', X)
np.save('flabels', y)
Y = to_categorical(y,len(labels))
X = np.array(X)
X = X/255 #proses normalisasi
X_new = []
for i in range(len(X)):
    stacked_img = np.stack((X[i].reshape(48,48),)*3, axis=-1)
    X_new.append(stacked_img)

X_new = np.array(X_new)
x_train, x_test, y_train, y_test = train_test_split(X_new, Y, test_size = 0.2)
print("Data Train Shape")
print("X_TRAIN : ", x_train.shape)
print("Y_TRAIN : ", y_train.shape)

print("Data Test Shape")
print("X_TRAIN : ", x_test.shape)
print("Y_TRAIN : ", y_test.shape)
from keras.applications.xception import Xception
from keras.layers import Activation, Dense,GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.optimizers import SGD
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy

base_model = Xception(weights='imagenet', include_top=False )

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(512, activation='relu')(x)

x = Dropout(0.3)(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(7, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

model.summary()
batch_size= 64
epochs= 50

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
# red_lr= ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.1)


earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=epochs,
                          verbose=1,
                          restore_best_weights=True
                          )

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=5,
                              verbose=1,
                              min_delta=0.0001)

callbacks = [earlystop,reduce_lr]
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(x_train)
# Compile the model
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

History = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epochs, 
                              callbacks=callbacks,
                              validation_data = (x_test,y_test),
                              verbose = 1)
plt.figure(figsize=(15, 3))
plt.subplot(121)

plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])


plt.subplot(122)
plt.plot(History.history['accuracy'])
plt.plot(History.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])

plt.savefig('Xception.png')
plt.show()
#saving the  model to be used later
fer_json = model.to_json()
with open("mymodel.json", "w") as json_file:
    json_file.write(fer_json)
model.save_weights("model_weight.h5")
print("Saved model to disk")

joblib.dump(History, 'history.h5')
def confusion_matrix_img(y_pred, y_true, savename):
    cm = confusion_matrix(y_true, y_pred)
    labels = ['Angry','Sad', 'Happy','Fear','Surprise']
    title='Confusion matrix'
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(savename)
    plt.show()
truey=[]
predy=[]
x = x_test
y = y_test

yhat= model.predict(x)
yh = yhat.tolist()
yt = y.tolist()
count = 0

for i in range(len(y)):
    yy = max(yh[i])
    yyt = max(yt[i])
    predy.append(yh[i].index(yy))
    truey.append(yt[i].index(yyt))
    if(yh[i].index(yy)== yt[i].index(yyt)):
        count+=1

acc = (count/len(y))*100

#saving values for confusion matrix and analysis
np.save('truey', truey)
np.save('predy', predy)
print("Predicted and true label values saved")
print("Accuracy on test set :"+str(acc)+"%")
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


confusion_matrix_img(truey, predy, savename='Confusion Matrix')
