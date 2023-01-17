# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Creating all directories for storing training and validation images

!mkdir /kaggle/working/images

!mkdir /kaggle/working/images/train

!mkdir /kaggle/working/images/validation

!mkdir /kaggle/working/images/train/face_with_mask

!mkdir /kaggle/working/images/train/face_no_mask

!mkdir /kaggle/working/images/validation/face_with_mask

!mkdir /kaggle/working/images/validation/face_no_mask
# Reading all Annotated files belonging to face_with_mask and face_no_mask categories

import csv

import os

import shutil

import cv2



src_path="/kaggle/input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/images/"

res_path="/kaggle/working/images/train/face_with_mask/"

res2_path="/kaggle/working/images/train/face_no_mask/"



test_path="/kaggle/working/images/validation/face_with_mask/"

test2_path="/kaggle/working/images/validation/face_no_mask/"



count=0

with open('/kaggle/input/face-mask-detection-dataset/train.csv') as csvfile:

    readCSV = list(csv.reader(csvfile, delimiter=','))

    print(len([row for row in readCSV[1:] if(row[5]=="face_with_mask" or row[5]=="face_no_mask")]))

    #Train test split

    len_train_samples=int(len([row for row in readCSV[1:] if(row[5]=="face_with_mask" or row[5]=="face_no_mask")])*0.8)

    for row in readCSV[1:]:

        #print(row)

        if(row[5]=="face_with_mask" or row[5]=="face_no_mask"):

            count+=1

            x1=int(row[1])

            x2=int(row[2])

            y1=int(row[3])

            y2=int(row[4])

            

            image=cv2.imread(src_path+row[0])

            image=image[x2:y2,x1:y1]

            

            if(count<=len_train_samples and row[5]=="face_with_mask"):

                cv2.imwrite(res_path+str(count)+".jpg",image)

            

            elif(count<=len_train_samples and row[5]=="face_no_mask"):

                cv2.imwrite(res2_path+str(count)+".jpg",image)

            

            elif(count>len_train_samples and row[5]=="face_with_mask"):

                cv2.imwrite(test_path+str(count)+".jpg",image)

            

            elif(count>len_train_samples and row[5]=="face_no_mask"):

                cv2.imwrite(test2_path+str(count)+".jpg",image)

    
import matplotlib.pyplot as plt

%matplotlib inline

import os

for pic in os.listdir("/kaggle/working/images/train/face_with_mask/")[0:1]:

    print(pic)

    img=plt.imread("/kaggle/working/images/train/face_with_mask/"+pic)

    plt.imshow(img)

    
from tensorflow.keras.models import Sequential

from tensorflow.keras import Model

from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, BatchNormalization, Conv2D, Input, AveragePooling2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import MaxPooling2D, Activation, Dropout

from tensorflow.keras.applications import MobileNetV2, ResNet50, Xception

from tensorflow.keras.optimizers import Adam, Adadelta

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau



batch_size=32

momentum=0.9

epochs=7

epoch_size=400

lr=0.1

wd=0.0005



train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=20,zoom_range=0.15,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.15,horizontal_flip=True,fill_mode="nearest")

validation_datagen = ImageDataGenerator(rescale=1./255,rotation_range=20,zoom_range=0.15,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.15,horizontal_flip=True,fill_mode="nearest")
train_generator = train_datagen.flow_from_directory(

        "/kaggle/working/images/train/",  # This is the source directory for training images

        target_size=(224, 224),

        batch_size=batch_size,

        class_mode='categorical',

        color_mode="rgb",

        shuffle=False)



validation_generator = validation_datagen.flow_from_directory(

        "/kaggle/working/images/validation",  # This is the source directory for validation images

        target_size=(224, 224),

        batch_size=batch_size,

        class_mode='categorical',

        color_mode="rgb",

        shuffle=False)

nb_train_samples = 4599

nb_validation_samples = 1150
# The Base model is a pretrained Xception Model trained on imagenet



baseModel = Xception(weights="imagenet", include_top=False,

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



opt = Adam(lr=lr*0.01, decay=0.01)



model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])

model.summary()
# train the head of the network

H = model.fit(train_generator,

      steps_per_epoch=nb_train_samples // batch_size,

      validation_data=validation_generator, 

      validation_steps=nb_validation_samples // batch_size,

      epochs=epochs)



score = model.evaluate_generator(validation_generator,nb_validation_samples//batch_size)

print(" Total: ", len(validation_generator.filenames))

print("Loss: ", score[0], "Accuracy: ", score[1])
model.save("/kaggle/working/xception_model.h5")
# from IPython.display import FileLink, FileLinks

# FileLinks('.') #lists all downloadable files on server
import matplotlib.pyplot as plt

import numpy as np



N = epochs

plt.style.use("ggplot")

plt.figure()

plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")

plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")

plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")

plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")

plt.title("Training Loss and Accuracy")

plt.xlabel("Epoch #")

plt.ylabel("Loss/Accuracy")

plt.legend(loc="lower left")
def detect_and_predict_mask(frame, faceNet, maskNet):

    # grab the dimensions of the frame and then construct a blob

    # from it

    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 117.0, 123.0))

    # pass the blob through the network and obtain the face detections

    faceNet.setInput(blob)

    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,

    # and the list of predictions from our face mask network

    faces = []

    locs = []

    preds = []

    for i in range(0, detections.shape[2]):

        # extract the confidence (i.e., probability) associated with

        # the detection

        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is

        # greater than the minimum confidence

        if confidence > 0.5:

            # compute the (x, y)-coordinates of the bounding box for

            # the object

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of

            # the frame

            

            if(startX>w or startY>h or endX>w or endY>h):

                continue

            (startX, startY) = (max(0, startX), max(0, startY))

            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # crop detected face out of image

            face = frame[startY:endY, startX:endX]

            face = cv2.resize(face, (224, 224))

            face = img_to_array(face)

            face = preprocess_input(face)

            # add the face and bounding boxes to their respective

            # lists

            faces.append(face)

            locs.append((startX, startY, endX, endY))

            

    if len(faces) > 0:

        # for faster inference we'll make batch predictions on *all*

        # faces at the same time rather than one-by-one predictions

        # in the above `for` loop

        faces = np.array(faces, dtype="float32")

        preds = maskNet.predict(faces, batch_size=32)

        

    else:

        # in case the face gets obscured by the mask and doesn't get detected

        # run the model to detect for faces with masks on the entire image

        locs=[(20, 20, w-20, h-20)]

        face=frame

        face = cv2.resize(face, (224, 224))

        face = img_to_array(face)

        face = preprocess_input(face)

        faces = np.array([face], dtype="float32")

        preds = maskNet.predict(faces, batch_size=32)

        

        

    return (locs, preds)



    

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):

    # initialize the dimensions of the image to be resized and

    # grab the image size

    dim = None

    (h, w) = image.shape[:2]



    # if both the width and height are None, then return the

    # original image

    if width is None and height is None:

        return image



    # check to see if the width is None

    if width is None:

        # calculate the ratio of the height and construct the

        # dimensions

        r = height / float(h)

        dim = (int(w * r), height)



    # otherwise, the height is None

    else:

        # calculate the ratio of the width and construct the

        # dimensions

        r = width / float(w)

        dim = (width, int(h * r))



    # resize the image

    resized = cv2.resize(image, dim, interpolation=inter)



    # return the resized image

    return resized

import os

import numpy as np

import cv2

import matplotlib.pyplot as plt

%matplotlib inline

from IPython.display import Image, display

from tensorflow.keras.preprocessing.image import img_to_array

from tensorflow.keras.applications.xception import preprocess_input



src_path="/kaggle/input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/images/"

faceNet = cv2.dnn.readNet("/kaggle/input/face-detection-model/deploy.prototxt.txt", "/kaggle/input/face-detection-model/res10_300x300_ssd_iter_140000.caffemodel")

count=0



output=[]

for file in os.listdir(src_path):

    name=int(file.split(".")[0])

    if(name<1801):

        count+=1

        print(count)

        image=cv2.imread(src_path+file)        

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        

        print(file)

        (locs, preds) = detect_and_predict_mask(image, faceNet, model)



        flag=0

        for (box, pred) in zip(locs, preds):

            # unpack the bounding box and predictions

            flag=1

            (startX, startY, endX, endY) = box

            (withoutMask,mask) = pred

            # determine the class label and color we'll use to draw

            # the bounding box and text

            label = "face_with_mask" if mask > withoutMask else "face_no_mask"

            color = (0, 255, 0) if label == "face_with_mask" else (0, 0, 255)

            # include the probability in the label

            # display the label and bounding box rectangle on the output

            # frame

            cv2.putText(image, label, (startX, startY - 10),

                cv2.FONT_HERSHEY_SIMPLEX, 1.05, color, 2)

            cv2.rectangle(image, (startX, startY), (endX, endY), color, 3)

            output.append([file,startX,startY,endX,endY,label])

            

        if(count<15):

            plt.figure()

            plt.imshow(image)



        print(label)

print(output[:10])
import csv

with open("/kaggle/working/submission.csv","w+") as my_csv:

    csvWriter = csv.writer(my_csv,delimiter=',')

    csvWriter.writerow(["name","x1","x2","y1","y2","classname"])

    csvWriter.writerows(output)
import pandas as pd



data=pd.read_csv("/kaggle/working/submission.csv")

data.head()