# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
#Import necessary Libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import random
from tqdm import tqdm
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten,Dense
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
os.listdir("../input/intel-image-classification")
os.listdir("/kaggle/input/intel-image-classification/seg_train/seg_train/")
buildings = "/kaggle/input/intel-image-classification/seg_train/seg_train/buildings/"
street = "/kaggle/input/intel-image-classification/seg_train/seg_train/street/"
mountain = "/kaggle/input/intel-image-classification/seg_train/seg_train/mountain/"
glacier = "/kaggle/input/intel-image-classification/seg_train/seg_train/glacier/"
sea = "/kaggle/input/intel-image-classification/seg_train/seg_train/sea/"
forest = "/kaggle/input/intel-image-classification/seg_train/seg_train/forest/"
print("Number of images in Train Directory: ")

print("Buildings :",len(os.listdir("/kaggle/input/intel-image-classification/seg_train/seg_train/buildings/")))
print("Street: ",len(os.listdir("/kaggle/input/intel-image-classification/seg_train/seg_train/street/")))
print("Mountain:",len(os.listdir("/kaggle/input/intel-image-classification/seg_train/seg_train/mountain/")))
print("Glacier: ",len(os.listdir("/kaggle/input/intel-image-classification/seg_train/seg_train/glacier/")))
print("Sea: ",len(os.listdir("/kaggle/input/intel-image-classification/seg_train/seg_train/sea/")))
print("Forest: ",len(os.listdir("/kaggle/input/intel-image-classification/seg_train/seg_train/forest/")))

x = []
y = []
import cv2
def create_dataset(directory,label_name):
    for i in tqdm(os.listdir(directory)):
        full_path = os.path.join(directory,i)
        try:
            img = cv2.imread(full_path)
            img = cv2.resize(img,(120,120))
        except:
            continue

        x.append(img)
        y.append(label_name)
    return x,y
x,y = create_dataset(buildings,"buildings")
x,y = create_dataset(street,"street")
x,y = create_dataset(mountain,"mountain")
x,y = create_dataset(glacier,"glacier")
x,y = create_dataset(sea,"sea")
x,y = create_dataset(forest,"forest")
x = np.array(x)
y = np.array(y)
print(x.shape,y.shape)
fig =plt.figure(figsize=(12,7))
for i in range(10):
    sample = random.choice(range(len(x)))
    plt.subplot(2,5,i+1)
    plt.subplots_adjust(hspace=0.3)
    plt.imshow(x[sample])
    plt.xlabel(y[sample])
    
plt.tight_layout()
plt.show()
unique,counts = np.unique(y,return_counts=True)
print(unique,counts)
plt.style.use("ggplot")
plt.figure(figsize=(9,7))
sns.countplot(y)
plt.show()
from sklearn.preprocessing import LabelEncoder,LabelBinarizer
le = LabelEncoder()
y = le.fit_transform(y)

lb = LabelBinarizer()
y = lb.fit_transform(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
img_size=120
x_train = np.array(x_train)/255.0
x_test = np.array(x_test)/255.0

x_train = x_train.reshape(-1,img_size,img_size,3)
y_train = np.array(y_train)

x_test = x_test.reshape(-1,img_size,img_size,3)
y_test = np.array(y_test)
from tensorflow.keras.applications.vgg19 import VGG19
vgg= VGG19(weights="imagenet",include_top=False,input_shape = (img_size,img_size,3))
for layer in vgg.layers:
    layer.trainable=False
model = Sequential()
model.add(vgg)
model.add(Flatten())
model.add(Dense(6,activation = "softmax"))

model.summary()
checkpoint = ModelCheckpoint("vgg19.h5",monitor = "val_accuracy",save_best_only=True,
                                 save_weights_only = False,verbose=1)
earlystop = EarlyStopping(monitor='val_accuracy',patience=5,verbose=1)

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
history = model.fit(x_train,y_train,batch_size=32,validation_data = (x_test,y_test),
                    epochs=15,verbose=1,callbacks=[checkpoint,earlystop])
loss,accuracy = model.evaluate(x_test,y_test)

print(f"Loss: {loss}")
print(f"Accuracy: {accuracy*100}")
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
y_pred = model.predict_classes(x_test)
y_pred[:15]
y_test_le = np.argmax(y_test,axis=1)
y_test_le[:15]
print(classification_report(y_test_le,y_pred))
from mlxtend.plotting import plot_confusion_matrix
cm = confusion_matrix(y_test_le,y_pred)
plot_confusion_matrix(conf_mat = cm,figsize=(8,7),class_names=['glacier', 'sea', 'forest', 'street', 'mountain', 'buildings'],
                     show_normed = True);
plt.figure(figsize=(12,6))
plt.style.use("ggplot")
epochs = range(1,12)
plt.subplot(1,2,1)
plt.plot(epochs,history.history["accuracy"],"go-")
plt.plot(epochs,history.history["val_accuracy"],"ro-")
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["Train","val"],loc="upper left")

plt.subplot(1,2,2)
plt.plot(epochs,history.history["loss"],"go-")
plt.plot(epochs,history.history["val_loss"],"ro-")
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Train","val"],loc="upper left")
plt.show()
fig =plt.figure(figsize=(12,7))
for i in range(10):
    sample = random.choice(range(len(x_test)))
    plt.subplot(2,5,i+1)
    plt.subplots_adjust(hspace=0.3)
    plt.imshow(x_test[sample])
    plt.xlabel(f"Actual:{y_test_le[sample]}\n Predicted: {y_pred[sample]} ")
    
plt.tight_layout()
plt.show()
