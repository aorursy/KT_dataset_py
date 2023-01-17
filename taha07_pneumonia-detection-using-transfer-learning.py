# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import random
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
os.listdir("/kaggle/input/chest-xray-pneumonia/chest_xray/train")
train_dir = "/kaggle/input/chest-xray-pneumonia/chest_xray/train"
test_dir = "/kaggle/input/chest-xray-pneumonia/chest_xray/test"
valid_dir = "/kaggle/input/chest-xray-pneumonia/chest_xray/val"
alltype = ["NORMAL","PNEUMONIA"]
normal = random.sample(os.listdir(train_dir+"/NORMAL"),5)
pneumonia = random.sample(os.listdir(train_dir+"/PNEUMONIA"),5)
from tqdm import tqdm
import cv2
x_train =[]
y_train = []
#os.chdir = "/kaggle/input/chest-xray-pneumonia/chest_xray/train/NORMAL"
for i in tqdm(os.listdir("/kaggle/input/chest-xray-pneumonia/chest_xray/train/NORMAL")):
    img = cv2.imread("/kaggle/input/chest-xray-pneumonia/chest_xray/train/NORMAL" +"/"+i)
    img = cv2.resize(img,(256,256))
    x_train.append(img)
    y_train.append("Normal")

for i in tqdm(os.listdir("/kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA")):
    img = cv2.imread("/kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA" +"/"+i)
    img = cv2.resize(img,(256,256))
    x_train.append(img)
    y_train.append("PNEUMONIA")
fig = plt.figure(figsize=(12,6))
fig.set_size_inches(12,12)

for i,image in enumerate(normal):
    plt.subplot(1,5,i+1)
    img = load_img(train_dir+"/NORMAL"+"/"+image)
    plt.imshow(img)
    plt.xlabel("Normal")
    plt.xticks([])
    plt.yticks([])
plt.tight_layout();

plt.figure(figsize=(12,6))
fig.set_size_inches(12,12)

for i,image in enumerate(pneumonia):
    plt.subplot(1,5,i+1)
    img = load_img(train_dir+"/PNEUMONIA"+"/"+image)
    plt.imshow(img)
    plt.xlabel("PNEUMONIA")
    plt.xticks([])
    plt.yticks([])
plt.tight_layout();
    
print("Total images in Train directory: {}".format(len(os.listdir(train_dir+"/NORMAL")) + len(os.listdir(train_dir+"/PNEUMONIA"))))
print("Total images in Test directory: {}".format(len(os.listdir(test_dir+"/NORMAL")) + len(os.listdir(test_dir+"/PNEUMONIA"))))
print("Total images in Validation directory: {}".format(len(os.listdir(valid_dir+"/NORMAL")) + len(os.listdir(valid_dir+"/PNEUMONIA"))))
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=(1./255),shear_range = 0.2,zoom_range=0.3,
                                horizontal_flip=True)#fill_mode='nearest'
test_datagen = ImageDataGenerator(rescale = (1./255))

train_data = train_datagen.flow_from_directory(directory = train_dir,target_size=(224,224),
                                               class_mode = "categorical",batch_size=32)
test_data = test_datagen.flow_from_directory(directory = test_dir,target_size=(224,224),
                                                class_mode = "categorical",batch_size=32)

from tensorflow.keras.applications.vgg19 import VGG19
vgg = VGG19(weights = "imagenet",include_top = False,input_shape=(224,224,3))
for layer in vgg.layers:
    layer.trainable = False
from tensorflow.keras.layers import Flatten,Dense,Dropout,BatchNormalization
x = vgg.output
x = Flatten()(x)
#x = Dense(units=4096, activation='relu')(x)
#x = BatchNormalization()(x)
#x = Dropout(0.5)(x)
predictions = Dense(2,activation= "softmax")(x)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

model = Model(inputs = vgg.input,outputs = predictions)
model.compile(optimizer="adam",loss = "categorical_crossentropy",metrics =["accuracy"])#Adam(1e-4)
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping

checkpoint = ModelCheckpoint("vgg19.h5",monitor = "val_accuracy",save_best_only=True,
                             save_weights_only=False,mode='auto',verbose=1,period=1)

earlystop = EarlyStopping(monitor="val_acc",patience=5,verbose=1)
history = model.fit_generator(generator = train_data,validation_data = test_data,
                              epochs = 15,verbose = 1,callbacks=[checkpoint,earlystop])#steps_per_epoch = len(train_datagen),validation_steps = len(val_datagen)
model.evaluate_generator(test_data) #val_datagen
x_test =[]
y_test = []
for i in tqdm(os.listdir("/kaggle/input/chest-xray-pneumonia/chest_xray/test/NORMAL")):
    img = cv2.imread("/kaggle/input/chest-xray-pneumonia/chest_xray/test/NORMAL" +"/"+i)
    img = cv2.resize(img,(224,224))
    x_test.append(img)
    y_test.append("Normal")
    
for i in tqdm(os.listdir("/kaggle/input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA")):
    img = cv2.imread("/kaggle/input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA" +"/"+i)
    img = cv2.resize(img,(224,224))
    x_test.append(img)
    y_test.append("PNEUMONIA")
x_test = np.array(x_test)
y_test = np.array(y_test)

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred,axis=1)
y_pred[:15]
unique,counts = np.unique(y_pred,return_counts=True)
print(unique,counts)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_test = le.fit_transform(y_test)
unique,counts = np.unique(y_test,return_counts=True)
print(unique,counts)
from sklearn.metrics import accuracy_score,classification_report
print(classification_report(y_test,y_pred))
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
cm = confusion_matrix(y_test,y_pred)
plot_confusion_matrix(conf_mat = cm,figsize=(8,8),class_names = ["Normal","Pneumonia"],show_normed=True)
plt.style.use("ggplot")
fig = plt.figure(figsize=(12,7))
epochs = range(1,16)
plt.subplot(1,2,1)
plt.plot(epochs,history.history["accuracy"],"go-")
plt.plot(epochs,history.history["val_accuracy"],"ro-")
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["Train","val"],loc = "upper left")
#plt.show()

#fig = plt.figure(figsize=(12,8))    
plt.subplot(1,2,2)
plt.plot(epochs,history.history["loss"],"go-")
plt.plot(epochs,history.history["val_loss"],"ro-")
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Train","val"],loc = "upper left")
plt.show()
plt.figure(figsize=(12,8))
#plt.title("0 for Normal & 1 is for Pneumonia")
for i in range(8):
    sample_index = np.random.randint(1,len(y_test))
    plt.subplot(2,4,i+1)
    plt.imshow(x_test[sample_index])
    plt.xlabel("Actual: {}\n Predicted : {}".format(y_test[sample_index],y_pred[sample_index]))