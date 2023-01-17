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
!cp -r /kaggle/input/covid19-detection-xray-dataset /kaggle/working/
!rm -rf /kaggle/working/covid19-detection-xray-dataset/TrainData/OversampledAugmentedCOVID-19
#!rm -rf /kaggle/working/covid19-detection-xray-dataset/TrainData/COVID-19/COVID-19
import os
import subprocess as sbp
S_dir = "/kaggle/input/covid19-detection-xray-dataset/TrainData/OversampledAugmentedCOVID-19/COVID-19"
D_dir = "/kaggle/working/covid19-detection-xray-dataset/TrainData/COVID-19"
for file in os.listdir(S_dir):
    path = os.path.join(S_dir, file)
    p3 = "cp -r " + path +" " + D_dir+"/"
    sbp.Popen(p3,shell=True)
    
!cd /kaggle/working/covid19-detection-xray-dataset/TrainData/COVID-19&& ls
import shutil
S_dir = "/kaggle/working/covid19-detection-xray-dataset/TrainData/COVID-19"
for file in os.listdir(S_dir):
    shutil.copy2(os.path.join(S_dir,file),os.path.join(S_dir,file.split('.')[0]+"_1"+file.split('.')[1]))
    shutil.copy2(os.path.join(S_dir,file),os.path.join(S_dir,file.split('.')[0]+"_2"+file.split('.')[1]))
    shutil.copy2(os.path.join(S_dir,file),os.path.join(S_dir,file.split('.')[0]+"_3"+file.split('.')[1]))
    shutil.copy2(os.path.join(S_dir,file),os.path.join(S_dir,file.split('.')[0]+"_4"+file.split('.')[1]))
        
import random
S_dir = "/kaggle/working/covid19-detection-xray-dataset/TrainData/Normal"
file_Normal = list(os.listdir(S_dir))
random.shuffle(file_Normal)
for i in file_Normal[0:300]:
    os.remove(S_dir + '/' + i)
import os
import seaborn as sns
counts = []
dir = '/kaggle/working/covid19-detection-xray-dataset/TrainData'
print("Total No of data present in each folder of TrainData\n")
for filename in os.listdir(dir):
        path = os.path.join(dir, filename)
        list1 = os.listdir(path) 
        number_files = len(list1)
        counts = counts + [filename]*number_files
        print (filename + ":"+ str(number_files))

sns.countplot(counts)
       
counts_val = []
dir = '/kaggle/working/covid19-detection-xray-dataset/ValData'
print("Total No of data present in each folder of ValData\n")
for filename in os.listdir(dir):
        path = os.path.join(dir, filename)
        list1 = os.listdir(path) 
        number_files = len(list1)
        counts_val = counts_val + [filename]*number_files
        print (filename + ":"+ str(number_files))
        
sns.countplot(counts_val)
!cd /kaggle/input/covid19-radiography-database&& ls
import tensorflow as tf
tf.test.is_gpu_available()
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        '/kaggle/working/covid19-detection-xray-dataset/TrainData',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')
test_generator = test_datagen.flow_from_directory(
        '/kaggle/working/covid19-detection-xray-dataset/ValData',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')
import matplotlib.pyplot as plt
from matplotlib.image import imread
import cv2
#covid = cv2.imread("/kaggle/input/covid19-detection-xray-dataset/NonAugmentedTrain/COVID-19/0.jpeg")
#viral = cv2.imread("/kaggle/input/covid19-detection-xray-dataset/NonAugmentedTrain/ViralPneumonia/0.jpeg")
#Normal = cv2.imread("/kaggle/input/covid19-detection-xray-dataset/NonAugmentedTrain/Normal/0.jpeg")
#bacterial = cv2.imread("/kaggle/input/covid19-detection-xray-dataset/NonAugmentedTrain/BacterialPneumonia/0.jpeg")

covid = cv2.imread("/kaggle/input/covid19-detection-xray-dataset/TrainData/COVID-19/_0_1465610.jpeg")
viral = cv2.imread("/kaggle/input/covid19-detection-xray-dataset/TrainData/ViralPneumonia/_0_9503218.jpeg")
Normal = cv2.imread("/kaggle/input/covid19-detection-xray-dataset/TrainData/Normal/_0_6256933.jpeg")
bacterial = cv2.imread("/kaggle/input/covid19-detection-xray-dataset/TrainData/BacterialPneumonia/_0_4754176.jpeg")
plt.imshow(covid)


plt.imshow(Normal)
plt.imshow(viral)
plt.imshow(bacterial)
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 224x224 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),


    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])
model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

history = model.fit_generator(train_generator, validation_data = test_generator, steps_per_epoch=64,epochs = 40,  verbose = 1)
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()


plt.show()
history.history
model.save("covid-19.h5")
train_generator.class_indices
img = cv2.imread("/kaggle/input/covid19-detection-xray-dataset/NonAugmentedTrain/Normal/111.jpeg")

img = cv2.resize(img,(224,224))

img = np.reshape(img,[1,224,224,3])

classes = model.predict_classes(img)

print (classes)
count1,count2,count3,count4 = 0,0,0,0
dir  = '/kaggle/input/normal'
for file in os.listdir(dir):
    path = os.path.join(dir,file)
    #list = os.listdir(path) 
    #number_files = len(list)
    img = cv2.imread(path)
    img = cv2.resize(img,(224,224))
    img = np.reshape(img,[1,224,224,3])
    classes = model.predict_classes(img)
    if(classes[0]==0):
        count1 = count1 + 1
    elif(classes[0]==1):
        count2 = count2 + 1
    elif(classes[0]==2):
        count3 = count3 + 1
    else:
        count4 = count4 + 1
        
    #print (classes)
sum1 = count1 + count2+ count3+ count4 
print ("BacterialPneumonia : " + str(count1) + "\nCOVID-19 : " + str(count2) + "\nNormal : " + str(count3) + "\nViralPneumonia : "+ str(count4) + "\nTotal : " + str(sum1))


dir = '/kaggle/input/covid19-detection-xray-dataset/ValData'
print("Total No of data present in each folder of NonAugmented\n")
for filename in os.listdir(dir):
        path = os.path.join(dir, filename)
        list = os.listdir(path) 
        number_files = len(list)
        #counts_val = counts_val + [filename]*number_files
        print (filename + ":"+ str(number_files))
import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
Y_preds=model.predict_generator(test_generator)
y_pred = np.argmax(Y_preds, axis=1)
Matrix = confusion_matrix(test_generator.classes, y_pred)
print('Confusion Matrix')
print(Matrix)
print('Classification Report')
target_names = ['BN', 'COVID-19', 'Normal', 'VN']
print(classification_report(test_generator.classes, y_pred, target_names=target_names))



df_cm = pd.DataFrame(Matrix, target_names, target_names)
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

plt.show()