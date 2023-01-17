import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from keras.utils import to_categorical

from keras.callbacks import EarlyStopping

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from keras.losses import categorical_crossentropy

from sklearn.metrics import accuracy_score

from keras.optimizers import Adam

from keras.regularizers import l2

from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt

import cv2 

from skimage import io

from keras.preprocessing import image

from os import walk

import os

filename = '/kaggle/input/facial-expression/fer2013.csv'

data=pd.read_csv(filename)



data.head(20)
labels = {0: 'Angry', 1: 'Digust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

emotion_counts = data['emotion'].value_counts().reset_index()

emotion_counts.columns = ['emotion', 'number']

emotion_counts['emotion'] = emotion_counts['emotion'].map(labels)

emotion_counts

def row2image(row):

    pixels, emotion = row['pixels'], labels[row['emotion']]

    img = np.array(pixels.split())

    img = img.reshape(48,48)

    image = np.zeros((48,48,3))

    image[:,:,0] = img

    image[:,:,1] = img

    image[:,:,2] = img

    return np.array([image.astype(np.uint8), emotion])



plt.figure(0, figsize=(16,10))

for i in range(1,8):

    face = data[data['emotion'] == i-1].iloc[0]

    img = row2image(face)

    plt.subplot(2,4,i)

    plt.imshow(img[0])

    plt.title(img[1])



plt.show()  

data_train = data[data['Usage']=='Training'].copy()

data_val   = data[data['Usage']=='PublicTest'].copy()

data_test  = data[data['Usage']=='PrivateTest'].copy()
def pre_Processing(data):

    temp = data.pixels.apply(lambda row: [float(p) for p in row.split()])

    X = np.array(temp.tolist())

    X = X.reshape(-1,48,48,1)

    X = X/255.0

    Y = to_categorical(data['emotion'], len(labels)) 

    return X,Y



X_train,y_train = pre_Processing(data_train)

X_test,y_test = pre_Processing(data_test)

X_val,y_val = pre_Processing(data_val)





print(np.shape(X_train))
def get_model():

    model = Sequential()

    

    model.add(Conv2D(64,kernel_size=(3,3),input_shape=(48,48,1)))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Conv2D(64,kernel_size=(3,3),padding='same'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

    

    model.add(Conv2D(128,kernel_size=(3,3),padding='same'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Conv2D(128,kernel_size=(3,3),padding='same'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

    

    model.add(Conv2D(256,kernel_size=(3,3),padding='same'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

    

    model.add(Flatten())



    model.add(Dense(256))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    

    model.add(Dense(128))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    

    model.add(Dense(64))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    

    model.add(Dense(7, activation='softmax'))



    return model
cp_callBack = tf.keras.callbacks.ModelCheckpoint('model_filter.h5',verbose=0,save_freq=25,save_best_only=False)

model = get_model()

model.summary()

optimizer = tf.keras.optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-7)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
data_generator = ImageDataGenerator(

                        featurewise_center=False,

                        featurewise_std_normalization=False,

                        rotation_range=10,

                        width_shift_range=0.1,

                        height_shift_range=0.1,

                        zoom_range=.1,

                        horizontal_flip=True)



#history = model.fit_generator(data_generator.flow(X_train,y_train,64),steps_per_epoch=len(X_train) /64,epochs=50,verbose=1,callbacks = [cp_callBack],validation_data=(X_val, y_val),shuffle=True)

model.load_weights('/kaggle/input/facialexpressionweights/model_filter.h5')
model.evaluate(X_test,y_test)
def getFace(img,faces):

    for (x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        roi_color = img[y:y + h, x:x + w]

        return cv2.resize(roi_color,(48,48),interpolation=cv2.INTER_AREA)

   

def getPercentage(custom):

    sum =0

    for i in custom[0]:

        sum += i

    for i in range(len(custom[0])):

        custom[0][i] = int((custom[0][i] / sum) * 100)

    return custom[0]
def emotion_analysis(emotions):

    objects = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    y_pos = np.arange(len(objects))

    plt.bar(y_pos, emotions, align='center', alpha=0.9)

    plt.tick_params(axis='x', which='both', pad=10,width=4,length=10)

    plt.xticks(y_pos, objects)

    plt.ylabel('percentage')

    plt.title('emotion')

    

    plt.show()
def printOutput(image_path):

    show_img=image.load_img(image_path, grayscale=False, target_size=(400, 500))



    face_cascade = cv2.CascadeClassifier('/kaggle/input/facedetectionopencv/face_detection.xml') 

    img = cv2.imread(image_path,0)



    faces = face_cascade.detectMultiScale(

        img,

        scaleFactor=1.3,

        minNeighbors=3

    )





    croped_img = getFace(img,faces)

    if(np.shape(croped_img) == ()):

        croped_img = cv2.resize(img,(48,48),interpolation=cv2.INTER_AREA)



    x = image.img_to_array(croped_img)

    x = np.expand_dims(x, axis = 0)

    x /= 255

    plt.gray()

    plt.imshow(show_img)

    plt.show()

    custom = model.predict(x)

    

    emotion_analysis(custom[0])



    x = np.array(x, 'float32')

    x = x.reshape([48, 48]);



    



    

   



image_path = "/kaggle/input/facialexpresionimages/"





for (dirpath, dirnames, filenames) in walk(image_path):

    for i in range(len(filenames)):

        printOutput(image_path+''+filenames[i])

    