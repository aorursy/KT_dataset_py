
!pip install imutils

import tensorflow as tf
import tensorflow_hub as hub

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import add
from tensorflow.keras.layers import Activation
from keras.utils import plot_model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential , Model , load_model
from tensorflow.keras.models import load_model 

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from tensorflow.keras.preprocessing.image import load_img , img_to_array
from tensorflow.keras.utils import to_categorical
from PIL import Image
import matplotlib.pyplot as plt

import cv2
from imutils import paths
import numpy as np
import os
import time
import warnings
warnings.filterwarnings("ignore")

#Dataset folder path
data = "/kaggle/input/face-mask-dataset/data"

Img_Paths = list(paths.list_images(data))

#Displaying sample image from dataset
sample1 = Image.open(Img_Paths[1])
plt.imshow(sample1)
sample1

print(type(Img_Paths[1]))
sample2 = Image.open(Img_Paths[-1])
plt.imshow(sample2)
sample2
#Initializing learning rate
INIT_LR = 0.0001
BATCH_SIZE = 32
EPOCHS = 20

#Getting all images and their labels in list
print("Loading images...")
Img_Paths = list(paths.list_images(data))
imgs = []
labels = []

#Looping over the image paths
for i in Img_Paths:
    #Extracting the class label
    label = i.split(os.path.sep)[-2]
    
    #Loading input image and processing it
    img = load_img(i,target_size=(224,224)) #Resizning all images with 224 Width and 224 height
    img = img_to_array(img) #Converting images to array
    img = preprocess_input(img)
    
    #updating imgs and labels respectively
    imgs.append(img)
    labels.append(label)
    
    
#Coverting imgs and labels to numpy array with float type
imgs = np.array(imgs,dtype="float32")
labels = np.array(labels)
print("...Done")

#Performing one-hot encoding on labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

#Splitting data into train and test

(X_train , X_test , y_train , y_test) = train_test_split(imgs,labels,test_size=0.20,stratify=labels,random_state=42)


#Constructing the generator for data augmentation

img_gen = ImageDataGenerator(rotation_range=40,
                            zoom_range=0.20,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            shear_range=0.15,
                            horizontal_flip=True,
                            fill_mode="nearest")

print("Train size: ",len(X_train),"Test size: ",len(X_test))


model = Sequential()
model.add(Conv2D(512,(2,2),padding="same",activation="relu",input_shape=(224,224,3)))
model.add(MaxPooling2D((2,2),strides=2))

model.add(Conv2D(256,(2,2),padding="same",activation="relu"))
model.add(MaxPooling2D((2,2),strides=2))

model.add(Conv2D(128,(2,2),padding="same",activation="relu"))
model.add(MaxPooling2D((2,2),strides=2))

model.add(Conv2D(64,(2,2),padding="same",activation="relu"))
model.add(MaxPooling2D((2,2),strides=2))

model.add(Conv2D(32,(2,2),padding="same",activation="relu"))
model.add(MaxPooling2D((2,2),strides=2))

#model.add(BatchNormalization())

model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(64,activation="relu"))
model.add(Dense(2,activation="softmax"))

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
model.summary()

start = time.time()

history = model.fit(img_gen.flow(X_train,y_train,batch_size=5),
                    steps_per_epoch=300,
                    validation_data=(X_test,y_test),
                    validation_steps=300,
                    epochs=45)
end = time.time()
print("Total train time: ",(end-start)/60," mins")


def plot_graph(history,string):
    plt.figure(figsize=(16,7))
    plt.plot(history.history[string],label=str(string))
    plt.plot(history.history["val_"+str(string)],label=str(string))
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string,"val_"+string])
    plt.show()


plot_graph(history,"accuracy")
plot_graph(history,"loss")
#Saving trained model
#model.save("Face_Mask_Net_Cunstom_CNN.h5")
input_image = Img_Paths[0]
image1 = load_img(input_image,target_size=(224,224))
image2 = img_to_array(image1)
image2 = preprocess_input(image2)
image2 = np.array([image2],dtype="float32")
detection = model.predict(image2)
print(detection)
labels_dict={0:'MASK',1:'NO MASK'}
print(labels_dict[np.argmax(detection)])
input_image = Image.open(input_image)
plt.imshow(input_image)
#Loading MobileNetV2 architecture
CLASSIFIER_URL ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"
IMAGE_RES = 224 #Image size in pixels as MobileNet is trained on same img size

model = Sequential()
model.add(hub.KerasLayer(CLASSIFIER_URL,input_shape=(IMAGE_RES,IMAGE_RES,3)))

model.add(Dense(2,activation="softmax"))

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
model.summary()


start = time.time()

history = model.fit(img_gen.flow(X_train,y_train,batch_size=5),
                    steps_per_epoch=300,
                    validation_data=(X_test,y_test),
                    validation_steps=300,
                    epochs=45)
end = time.time()
print("Total train time: ",(end-start)/60," mins")

plot_graph(history,"accuracy")
plot_graph(history,"loss")

#model.save("Transfer_Learning_Model.h5")

input_image = Img_Paths[0]
image1 = load_img(input_image,target_size=(224,224))
image2 = img_to_array(image1)
image2 = preprocess_input(image2)
image2 = np.array([image2],dtype="float32")
detection = model.predict(image2)
print(detection)
labels_dict={0:'MASK',1:'NO MASK'}
print(labels_dict[np.argmax(detection)])
input_image = Image.open(input_image)
plt.imshow(input_image)
# load the MobileNetV2 network, ensuring the head FC layer sets are
# left off
from tensorflow.keras.applications import MobileNetV2
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))
# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)
# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)
# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False
    
    
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
model.summary()

start = time.time()

history = model.fit(img_gen.flow(X_train,y_train,batch_size=5),
                    steps_per_epoch=300,
                    validation_data=(X_test,y_test),
                    validation_steps=300,
                    epochs=45)
end = time.time()
print("Total train time: ",(end-start)/60," mins")

#model.save("Transfer Learned.h5")
input_image = Img_Paths[0]
image1 = load_img(input_image,target_size=(224,224))
image2 = img_to_array(image1)
image2 = preprocess_input(image2)
image2 = np.array([image2],dtype="float32")
detection = model.predict(image2)
print(detection)
labels_dict={0:'MASK',1:'NO MASK'}
print(labels_dict[np.argmax(detection)])
input_image = Image.open(input_image)
plt.imshow(input_image)
plot_graph(history,"accuracy")
plot_graph(history,"loss")

maskNet = load_model("Face_Mask_Net_Cunstom_CNN.h5")
face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
labels_dict={0:'MASK',1:'NO MASK'}
color_dict={0:(0,255,0),1:(0,0,255)}


def Detector(image_file):
    
    input_image = image_file
    input_image1 = cv2.imread(input_image)
    
    faces=face_clsfr.detectMultiScale(input_image1,1.1,1)  
    
    for (x,y,w,h) in faces:
    
        face_img=input_image1[y:y+w,x:x+w]
        #resized=cv2.resize(face_img,(224,224))
        #image1 = load_img(face_img,target_size=(224,224))
        image1 = cv2.resize(face_img,(224,224))
        image2 = img_to_array(image1)
        image3 = preprocess_input(image2)
        
        image4 = np.array([image3],dtype="float32")
        result=maskNet.predict(image4)

        label=np.argmax(result,axis=1)[0]
      
        cv2.rectangle(input_image1,(x,y),(x+w,y+h),color_dict[label],1)
        cv2.rectangle(input_image1,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(input_image1, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
    cv2.imshow("Detection",input_image1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

Detector(str(image_file_path))
maskNet = load_model("Face_Mask_Net_Cunstom_CNN.h5")
face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


source=cv2.VideoCapture(0)
labels_dict={0:'MASK',1:'NO MASK'}
color_dict={0:(0,255,0),1:(0,0,255)}


while(True):

    ret,img=source.read()
    
    faces=face_clsfr.detectMultiScale(img,1.2,2)  
    
    

    for (x,y,w,h) in faces:
    
        face_img=img[y:y+w,x:x+w]
        #resized=cv2.resize(face_img,(224,224))
        #image1 = load_img(face_img,target_size=(224,224))
        image1 = cv2.resize(face_img,(224,224))
        image2 = img_to_array(image1)
        image3 = preprocess_input(image2)
        
        image4 = np.array([image3],dtype="float32")
        result=maskNet.predict(image4)

        label=np.argmax(result,axis=1)[0]
      
        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
        
    cv2.imshow('Face Mask Detector',img)
    key=cv2.waitKey(1)
    
    # if(key==27):
    #     break
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cv2.destroyAllWindows()
source.release()
