import numpy as np 

import pandas as pd 
data=pd.read_csv('../input/facial-expression/fer2013.csv')

data.head()
emotion=data['emotion'].values.tolist()

pixel=data['pixels'].values.tolist()

usage=data['Usage'].values.tolist()
print(len(emotion))

print(emotion[152])

print(type(usage[152]))

if(usage[152]=='Testing'):

    print(25)

else:

    print(55)
new_train_pixel=[]

new_train_emote=[]

new_test_pixel=[]

new_test_emote=[]
#custom dataset for or requirment

for i in range(0,len(emotion)):

    if(emotion[i]==3 or emotion[i]==5):

        if(usage[i]=='Training'):

            new_train_pixel.append(pixel[i])

            new_train_emote.append(0)

        else:

            new_test_pixel.append(pixel[i])

            new_test_emote.append(0)

            

    elif(emotion[i]==0 or emotion[i]==4):

        if(usage[i]=='Training'):

            new_train_pixel.append(pixel[i])

            new_train_emote.append(1)

        else:

            new_test_pixel.append(pixel[i])

            new_test_emote.append(1)

print(len(new_train_pixel))

print(len(new_train_emote))

print(len(new_test_pixel))

print(len(new_test_emote))
print(new_train_emote.count(0))

print(new_train_emote.count(1))

#print(new_train_emote.count(2))



print(new_test_emote.count(0))



print(new_test_emote.count(1))



#print(new_test_emote.count(2))
def convert(a):

    l=len(a)

    x=[]

    for i in range(len(a)):

        x.append(a[i].split(' '))

    x=np.array(x)

    x=x.astype('float32').reshape(l,48*48*1)

    

    return x
X_train=convert(new_train_pixel)

print("done converting train")

X_test=convert(new_test_pixel)

print("done converting test")
from sklearn.preprocessing import MinMaxScaler

X_train = MinMaxScaler().fit_transform(X_train)

X_test = MinMaxScaler().fit_transform(X_test)
X_test=np.reshape(X_test,(X_test.shape[0],48,48,1))

X_train=np.reshape(X_train,(X_train.shape[0],48,48,1))
y_train=np.array(new_train_emote)

y_test=np.array(new_test_emote)



from keras.utils.np_utils import to_categorical

y_train = to_categorical(y_train)

y_test = to_categorical(y_test)
from keras.models import Sequential

from keras.layers import Dense,InputLayer,Activation,Dropout

from keras.layers import Flatten,Dropout,BatchNormalization

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D



model = Sequential()

model.add(Conv2D(64, kernel_size = (3,3), activation='relu',padding="same", input_shape=(48, 48, 1)))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(BatchNormalization())

model.add(Conv2D(128, kernel_size = (3,3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(BatchNormalization())

model.add(Conv2D(256, kernel_size = (3,3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(BatchNormalization())

model.add(Conv2D(512, kernel_size = (3,3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(BatchNormalization())

model.add(Flatten())

model.add(Dense(100, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(10, activation='relu'))

model.add(Dense(2, activation='softmax'))



print("Model Developed")
model.summary()
import keras

from keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta



opt = Adam()

model.compile(loss='binary_crossentropy', optimizer=opt, metrics = ['accuracy'])





history = model.fit(X_train, y_train, batch_size=32, 

          epochs=100, verbose=1)



print("................TRAINING DONE....................")
target_names = ['Happy','No_Happy']

def reports(X_test,y_test):

    Y_pred = model.predict(X_test)

    y_pred = np.argmax(Y_pred, axis=1)

    classification = classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names)

    confusion = confusion_matrix(np.argmax(y_test, axis=1), y_pred)

    score = model.evaluate(X_test, y_test, batch_size=32)

    Test_Loss = score[0]*100

    Test_accuracy = score[1]*100

    kc=cohen_kappa_score(np.argmax(y_test, axis=1), y_pred)

    return classification, confusion, Test_Loss, Test_accuracy ,kc
from sklearn.metrics import classification_report, confusion_matrix,cohen_kappa_score

from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score

classification, confusion, Test_loss, Test_accuracy,kc = reports(X_test,y_test)

classification = str(classification)

confusion_str = str(confusion)
print("confusion matrix: ")

print('{}'.format(confusion_str))

print("KAppa Coeefecient=",kc)

print('Test loss {} (%)'.format(Test_loss))

print('Test accuracy {} (%)'.format(Test_accuracy))

print(classification)
import matplotlib.pyplot as plt

import itertools

%matplotlib inline

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.get_cmap("Blues")):

    Normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if normalize:

        cm = Normalized

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')

    plt.imshow(Normalized, interpolation='nearest', cmap=cmap)

    plt.colorbar()

    plt.title(title)

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=90)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        thresh = cm[i].max() / 2.

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')





plt.figure(figsize=(5,5))

plot_confusion_matrix(confusion, classes=target_names, normalize=False, 

                      title='Confusion matrix, without normalization')

plt.show()

plt.figure(figsize=(5,5))

plot_confusion_matrix(confusion, classes=target_names, normalize=True, 

                      title='Normalized confusion matrix')

plt.show()
from tensorflow.keras import Model

model.save('model_happy_detector.h5')
import cv2 

import dlib

from PIL import Image

from numpy import asarray

from skimage import io

import matplotlib.pyplot as plt

from glob import glob 



png = glob('../input/myfaces/*.png', recursive=True)

jpg = glob('../input/myfaces/*.jpg', recursive=True)

jpg2 = glob('../input/myfaces/*.JPG', recursive=True)



paths=png+jpg+jpg2



for wind in paths:

    image = io.imread(wind)

    face_detector = dlib.get_frontal_face_detector()

    detected_faces = face_detector(image, 1)

    face_frames = [(x.left(), x.top(),

                        x.right(), x.bottom()) for x in detected_faces]

    Image1 = Image.open(wind)

    to_test=[]

    for i in range(0,len(face_frames)):

        croppedIm = Image1.crop((face_frames[i]))

        croppedIm = croppedIm.resize((48,48))

        croppedIm = croppedIm.convert('L')

        data = asarray(croppedIm)

        #data = MinMaxScaler().fit_transform(data)

        to_test.append(data)

    kite = np.array(to_test)

    kite=np.reshape(kite,(kite.shape[0],kite.shape[1],kite.shape[2],1))

    final=model.predict(kite)

    final = np.argmax(np.round(final),axis=1)

    for i in range(0,len(final)):

        if(final[i]==0):

            x = kite[i].astype('float32').reshape(48, 48)

            plt.imshow(x)

            plt.show()

            print("Customer",i,"was HAPPY")

        else:

            x = kite[i].astype('float32').reshape(48, 48)

            plt.imshow(x)

            plt.show()

            print("Customer",i,"was NOT HAPPY")
import cv2 

import dlib

from PIL import Image

from numpy import asarray

from skimage import io

import matplotlib.pyplot as plt



img_path = '../input/myfaces/test.jpg'

image = io.imread(img_path)

face_detector = dlib.get_frontal_face_detector()

detected_faces = face_detector(image, 1)

face_frames = [(x.left(), x.top(),

                    x.right(), x.bottom()) for x in detected_faces]

Image1 = Image.open(img_path)

to_test=[]

for i in range(0,len(face_frames)):

    croppedIm = Image1.crop((face_frames[i]))

    croppedIm = croppedIm.resize((48,48))

    croppedIm = croppedIm.convert('L')

    data = asarray(croppedIm)

    #data = MinMaxScaler().fit_transform(data)

    to_test.append(data)

kite = np.array(to_test)

kite=np.reshape(kite,(kite.shape[0],kite.shape[1],kite.shape[2],1))

final=model.predict(kite)

final = np.argmax(np.round(final),axis=1)

for i in range(0,len(final)):

    if(final[i]==0):

        x = kite[i].astype('float32').reshape(48, 48)

        plt.imshow(x)

        plt.show()

        print("Customer",i,"was HAPPY")

    else:

        x = kite[i].astype('float32').reshape(48, 48)

        plt.imshow(x)

        plt.show()

        print("Customer",i,"was NOT HAPPY")