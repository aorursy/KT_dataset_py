import pandas as pd
import numpy as np
import h5py
file=h5py.File('/kaggle/input/classification-of-handwritten-letters/LetterColorImages_123.h5')
clms=list(file.keys())
clms
background=np.array(file[clms[0]])
img=np.array(file[clms[1]])
labels=np.array(file[clms[2]])
n=len(img)
images=img.reshape(n,32,32,3)
images=images/255
import pylab as pl
pl.figure(figsize=(3,3))
var=1
pl.title('Label:%s'%labels[var]+' Background:%s'%background[var])
pl.imshow(images[var])
pl.show
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
num_labels=33
y=OneHotEncoder(categories='auto')
y=y.fit_transform(labels.reshape(-1,1)).toarray().astype('int64')
x=images
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
model=Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(32,32,3)))
model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(Conv2D(128,kernel_size=(4,4),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_labels,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
history=model.fit(X_train,y_train,batch_size=64,epochs=50,verbose=1,validation_data=(X_test,y_test))
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Train', 'Validation'])
plt.show()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Train', 'Validation'])
plt.show()
model.load_weights('../input/best-weights/best_weights.hdf5')
model.evaluate(X_train,y_train)
import matplotlib.pyplot as plt
import glob
import imageio
import cv2
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

def import_data(path,csv_file):
    data = pd.read_csv(path + csv_file)
    data['source'] = csv_file[:-4] + '/'
    return data

#Creating a Dataframe:
path = '../input/classification-of-handwritten-letters/'
csv_files = ['letters.csv','letters2.csv','letters3.csv']
data1 = import_data(path,csv_files[0])
data2 = import_data(path,csv_files[1])
data3 = import_data(path,csv_files[2])
data = pd.concat([data1,data2,data3],ignore_index=True)


del(data1,data2,data3)
data.head()
data.tail()
all_letters=''
for i in data.letter.unique():
    all_letters+=i
print(all_letters)

import os
def to_img(filename):
    img = load_img(filename,target_size=(32,32))
    img = img_to_array(img)
    img = img.reshape(1,32,32,3)
    img = img.astype('float32')
    img = img/255.0
    return img

def actual_value(filename,df,column_name):
    file = os.path.basename(os.path.normpath(filename))
    index_row = df[df['file']==file].index[0]
    return df.loc[index_row,column_name]
test_img = to_img(path+'letters3/09_236.png')
predicted_letter = model.predict_classes(test_img)
plt.imshow(test_img[0])
print('predicted:',all_letters[predicted_letter[0]])
print('actual:',actual_value(path+'letters3/09_236.png',data,'letter'))
my_path = '../input/letter-k2/k2.jpg'
test_img = to_img(my_path)
predicted_letter = model.predict_classes(test_img)
plt.imshow(test_img[0])
print('predicted:',all_letters[predicted_letter[0]])
print('actual:k')