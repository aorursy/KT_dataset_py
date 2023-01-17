import os
import cv2
from random import shuffle
import  pandas as pd
import numpy as np
from random import shuffle
import  pandas as pd
import numpy as np
import shutil
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input, Dropout
from keras.models import Model, load_model
from keras.utils import np_utils
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import seaborn as sns
from  keras.applications import *
def resizer(from_path,to_path,HEIGHT=150,WIDTH=150,igore_files=None):

    to_path=to_path+"/Resized_"
    if not os.path.exists(to_path):
        os.mkdir(to_path)
    for file in os.listdir(from_path):
        if (igore_files is not None and file not in igore_files) or igore_files is None:
            image = cv2.imread(from_path + "/" + file)
            scaled_image = cv2.resize(image, (HEIGHT, WIDTH))
            cv2.imwrite(to_path + "/" + file, scaled_image)

path1 = "../input/all/All"
path2 = "./"
resizer(path1,path2,igore_files=["GTruth.csv"])
shutil.copy("../input/all/All/GTruth.csv", "Resized_/GTruth.csv")

def load_data(PATH,train,test,val,ground_truth):
    if round(train+test+val,5) != 1:
        print("Sum not One !!! "+str(train+test+val))
        return
    scaler = MinMaxScaler()
    image_id=[]
    trainX=[]
    testX=[]
    valX=[]
    trainY = []
    testY = []
    valY = []
    images=[]
    for file in os.listdir(PATH):
        if file!=ground_truth:
            image_id.append(file.split(".")[0])
            images.append(cv2.imread(PATH+"/"+file))
    #shuffle(image_id)
    images=np.array(images)/255
    gt=pd.read_csv(PATH+"/"+ground_truth)
    temp = gt["Id"].values.tolist()
    truth_value = gt["Ground_Truth"].values.tolist()

    from_length=0
    to_length=round(len(image_id)*train)
    id_list=image_id[from_length:to_length]
    trainX=images[from_length:to_length]
    trainY=[truth_value[temp.index(int(name))]for name in id_list if int(name) in temp]
    trainY=to_categorical(trainY,2)

    from_length = to_length
    to_length = to_length+round(len(image_id) * val)
    id_list = image_id[from_length:to_length]
    valX=images[from_length:to_length]
    valY = [truth_value[temp.index(int(name))] for name in id_list if int(name) in temp]
    valY = to_categorical(valY,2)

    from_length = to_length
    to_length = to_length + round(len(image_id) * val)
    id_list = image_id[from_length:to_length]
    testX=images[from_length:to_length]
    testY = [truth_value[temp.index(int(name))] for name in id_list if int(name) in temp]
    testY = to_categorical(testY,2)
    
    return trainX,valX,testX,trainY,valY,testY
def get_model(input_shape,output):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.125))
 
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
 
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
 
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.19))
    model.add(Dense(output, activation='softmax'))
    return model
def pre_trained_model(name,input_shape1,output):
  model = Sequential()
  model.add(name(weights='imagenet',include_top=False,input_shape=input_shape1))
  model.add(Flatten())
  model.add(Dense(512, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(output, activation='softmax'))
  return model
trainX,valX,testX,trainY,valY,testY=load_data("Resized_",0.7,0.2,0.1,"GTruth.csv")
model=get_model(trainX.shape[1:],2)
#model=pre_trained_model(InceptionV3,trainX.shape[1:],2)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
batch_size = 256
epochs = 20
history = model.fit(trainX, trainY, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(valX, valY))
model.evaluate(testX,testY)
plt.style.use('seaborn-ticks')
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
!rm -r Resized_