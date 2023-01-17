import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import cv2
import scipy.stats as sp
import pydicom
import os
print(os.listdir("../input"))
class_info = pd.read_csv('../input/stage_2_detailed_class_info.csv')
tr_labels = pd.read_csv('../input/stage_2_train_labels.csv')
print(len(class_info))
class_info.head()
# Acccessing rows by index in pandas
class_info.loc[2][0]
class_info['class'].unique()
tr_labels.head()
images_list = os.listdir("../input/stage_2_train_images")
test_image = os.listdir("../input/stage_2_test_images")

print(images_list[0][:-4])
ty=[]
c=class_info[class_info['patientId']==images_list[1][:-4]]
ty.append(c['class'].unique()[0])
ty
print(len(images_list))
print(len(test_image))
def show(image):
    plt.figure(figsize=(10,10))
    plt.imshow(image,cmap='gray')
    plt.show()
path="../input/stage_2_train_images/"
dcm_data = pydicom.read_file(path+images_list[0])
print(dcm_data)
img = dcm_data.pixel_array
show(img)
l= list(dcm_data.elements())
dcm_data.keys()
dcm_data[0x10,0x10].value
# if we do nat exactly remember the keywords the following way can be conveniently used to first get
# the keys first and then using that key to access corresponding values.
dcm_data.dir()
dcm_data.PatientID
dcm_data.PatientSex
sample_sub = pd.read_csv("../input/stage_2_sample_submission.csv")
sample_sub.head(10)
l1 = list(class_info['patientId'])

not_labeled =[]

for i in range(0,1000):
    if test_image[i][:-4] not in l1:
        not_labeled.append(test_image[i])
# all image of test_images are not labeled 
len(not_labeled)
labeled =[]

for i in range(0,26684):
    if images_list[i][:-4] not in l1:
        labeled.append(images_list[i])
len(labeled)
# so all traing data is lebelled
len(set(l1))
s =test_image[10][:-4]
s in l1
s
pid = sample_sub['patientId']
string = sample_sub['PredictionString']


my_submission = pd.DataFrame({'patientId': pid, 'PredictionString': string})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)
my_submission.head()
import csv
import random
# empty dictionary
pneumonia_locations = {}
# load table
with open(os.path.join('../input/stage_1_train_labels.csv'), mode='r') as infile:
    # open reader
    reader = csv.reader(infile)
    # skip header
    next(reader, None)
    # loop through rows
    for rows in reader:
        # retrieve information
        filename = rows[0]
        location = rows[1:5]
        pneumonia = rows[5]
        # if row contains pneumonia add label to dictionary
        # which contains a list of pneumonia locations per filename
        if pneumonia == '1':
            # convert string to float to int
            location = [int(float(i)) for i in location]
            # save pneumonia location in dictionary
            if filename in pneumonia_locations:
                pneumonia_locations[filename].append(location)
            else:
                pneumonia_locations[filename] = [location]

img_with_pneumonia={}
for index,row in tr_labels.iterrows():
    filename=row['patientId']
    pneumonia= row['Target']
    if pneumonia==1:
        if filename in img_with_pneumonia:
            img_with_pneumonia[filename].append([int(row['x']),int(row['y']),int(row['height']),int(row['width'])])
        else:
            img_with_pneumonia[filename]=[[int(row['x']),int(row['y']),int(row['height']),int(row['width'])]]
    
img2= img
print(img2.shape)
img2 = np.expand_dims(img2,-1)
len(img2[0])
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import to_categorical
train_x=[]
train_y=[]
def makeDataset():
    for i in range(len(images_list)%1500):
        d=pydicom.read_file(path+images_list[i])
        c=class_info[class_info['patientId']==images_list[i][:-4]]
        train_y.append(c['class'].unique()[0])
        train_x.append(d.pixel_array)
makeDataset()
#function to convert string label to integer value
def label(s):
    if s=='Normal':
        return 0
    if s=='No Lung Opacity / Not Normal':
        return 1
    if s=='Lung Opacity':
        return 2
train_y = list(map(label,train_y))
train_y= np.stack(train_y)
train_x = np.stack(train_x)
train_y.shape
# converting train_y to one hot
train_y = to_categorical(train_y)
# reshaping train_x to standard form
train_x=train_x.reshape(-1,1024,1024,1)
train_x.shape
# Normalizing the pixel value
train_x = train_x.astype('float32')
train_x = train_x / 255.
print(train_x.shape)
print(train_y.shape)
#del train_x
#del train_y
train_y[0]
#The image have 1024X1024 and only one channel
img = pydicom.read_file(path+images_list[1])
img=img.pixel_array
img.shape

batch_size = 16
epochs = 15
num_classes = 3
model= Sequential()
# here input shape is (1024,1024) because the image is of size 1024X1024 and have only one channel
# here 32 is number of filters of kernel size (3,3)
# generally number of filters are increased and kernel size decreased but here is is constant

model.add(Conv2D(32,kernel_size=(7,7),activation='linear',input_shape=(1024,1024,1),padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(7,7),padding='same'))
model.add(Conv2D(64,kernel_size=(7,7),activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(7,7),padding='same'))
model.add(Conv2D(128,kernel_size=(7,7),activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(7,7),padding='same'))
model.add(Flatten())
model.add(Dense(128,activation='linear'))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(num_classes,activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
model.summary()
model_train = model.fit(train_x,train_y,batch_size=batch_size,epochs=epochs,verbose=1)
# Model Training Results
accuracy = model_train.history['acc']
loss = model_train.history['loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'r', label='Training accuracy')
plt.title('Training accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'g', label='Training loss')
plt.title('Training Loss')
plt.legend()
plt.show()
# check performance on train_data
test_eval = model.evaluate(train_x, train_y, verbose=0)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])
# check individual
print("Predicted label of the image",np.argmax(np.round(model.predict(np.expand_dims(train_x[3],axis=0)))))
print("Actual label of the image ",np.argmax(train_y[3]))

def check(i):
    dcm_img = pydicom.read_file(path+class_info.loc[i][0]+".dcm")
    img=dcm_img.pixel_array
    img2=img.reshape(1024,1024,1)
    plt.figure(figsize=(10,10))
    plt.imshow(img,cmap='gray')
    p=np.argmax(np.round(model.predict(np.expand_dims(img2,axis=0))))
    plt.title("Predicted {} ,Actual {}".format(p,label(class_info.loc[i][1])))
    plt.show()
check(25000)
batch_sz=20
for e in range(2):
    batch=0
    print("epochs : ",e)
    for image in range(int(100/batch_sz)):
        print("batch : ",batch)
        for i in range(batch_sz):
            print(batch+i)
        batch+=batch_sz
        
# Training on 