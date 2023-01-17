import numpy as np         
import os                  
import random 
import matplotlib.pyplot as plt
import cv2
mainDIR = os.listdir('../input/chest-xray-pneumonia/chest_xray')
print(mainDIR)
#TRAIN_DIR = 'C:/Users/Raj/Desktop/New folder/training_set/training_set'
#TEST_DIR = 'C:/Users/Raj/Desktop/New folder/test_set/test_set'
IMG_SIZE = 50
LR = 1e-3
train_folder= '../input/chest-xray-pneumonia/chest_xray/train/'
val_folder = '../input/chest-xray-pneumonia/chest_xray/val/'
test_folder = '../input/chest-xray-pneumonia/chest_xray/test/'
category=['PNEUMONIA','NORMAL']
datadir='../input/chest-xray-pneumonia/chest_xray/train/'
category=['PNEUMONIA','NORMAL']
for i in category:
    path=os.path.join(datadir,i)
    for img in os.listdir(path):
        img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array,cmap='gray')
        plt.show()
        break
    break
IMG_SIZE=60
new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE), interpolation=cv2.INTER_AREA)
plt.imshow(new_array,cmap='gray')
plt.show()
training_data=[]
def create_training_data():
    for i in category:
        path=os.path.join(train_folder,i)
        class_num=category.index(i)
        for img in os.listdir(path):
            try:
                img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE),interpolation=cv2.INTER_AREA)
                training_data.append([new_array,class_num])
            except Exception as e:
                pass
create_training_data()
training_data
len(training_data)
import random
random.shuffle(training_data)
for sample in training_data[:10]:
    print(sample[1])
testing_data=[]
def create_testing_data():
    for i in category:
        path=os.path.join(test_folder,i)
        class_num=category.index(i)
        for img in os.listdir(path):
            try:
                img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE),interpolation=cv2.INTER_AREA)
                testing_data.append([new_array,class_num])
            except Exception as e:
                pass
create_testing_data()
testing_data
len(testing_data)
import random
random.shuffle(testing_data)
for sample in testing_data[:10]:
    print(sample[1])
X=[]
y=[]
for features,label in training_data:
    X.append(features)
    y.append(label)
X=np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)
X.shape
X.shape[1:]
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D
model=Sequential()

model.add(Conv2D(64,(3,3),input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('sigmoid'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
X=X/255.0
X
y=np.array(y,dtype=np.uint8)
y
model.fit(X,y,validation_split=0.3,epochs=20)
!mkdir -p saved_model
model.save('saved_model/my_model')
import cv2
import tensorflow as tf
CATEGORIES = ["PNEUMONIA", "NORMAL"]

def prepare(filepath):
    IMG_SIZE = 50  # 50 in txt-based

    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    #new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    IMG_SIZE=60
    #new_array=cv2.resize(,(IMG_SIZE,IMG_SIZE), interpolation=cv2.INTER_AREA)
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = tf.keras.models.load_model('saved_model/my_model')

prediction = model.predict([prepare('../input/chest-xray-pneumonia/chest_xray/val/PNEUMONIA/person195_2bacteria_4883.jpeg')])
#print(prediction)  # will be a list in a list.
print(CATEGORIES[int(prediction[0][0])])
