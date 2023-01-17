# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
    #for filename in filenames:
        #print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
%reload_ext autoreload
%autoreload 2
%matplotlib inline
df = pd.read_csv('../input/face-mask-detection-dataset/train.csv')
df[:20]
#df5 = df[df.classname == 'face_no_mask','face_with_mask','face_with_mask_incorrect','mask_surgical','mask_colorful','face_other_covering','other']
#df5.head()
#indexNames = df[ (df['classname']!=  'face_no_mask') & (df['classname'] != 'face_with_mask')
#                &(df['classname']!=  'face_with_mask_incorrect') & (df['classname'] != 'mask_surgical')
#                &(df['classname']!=  'mask_colorful') & (df['classname'] != 'face_other_covering')& (df['classname'] != 'other') ].index
df1 =df[df.classname.str.contains('face_no_mask')| df.classname.str.contains('face_with_mask')]
#df5 = df.drop(indexNames , inplace=True)
df1.reset_index(drop=True, inplace=True)


df1 = df1.drop(df1.index[3111])
# df1.reset_index(drop=True, inplace=True)
#df1 = df1.drop(df1.index[1348])
df1.reset_index(drop=True, inplace=True)

#df1.reset_index(drop=True, inplace=True)
#df1[df1.name.str.contains('1861.jpg')]
df1[df1.name.str.contains('1861.jpg')]
df1[df1.classname.str.contains('face_with_mask_incorrect')]
df1 = df1[~df1.classname.str.contains('face_with_mask_incorrect')]
df1[df1.classname.str.contains('face_with_mask_incorrect')]
! pip install imutils
import cv2 
from imutils import paths as pth
import matplotlib.pyplot as plt
#parent_dir = ""
#path = os.path.join(parent_dir, 'face_no_mask')
#os.mkdir(path)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import resnet50
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
df2 = df1.iloc[np.random.permutation(len(df1))]
df2.reset_index(drop=True, inplace=True)
df2[:20], len(df2)
labels = df2['classname']
labels = np.array(labels)
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
print(labels[:5]),df2[:5]
(train_df,test_df,train_labels,test_labels) = train_test_split(df2,labels,test_size=0.20, stratify=labels, random_state=42)
len(train_df ), len(test_df ),len(train_labels),len(test_labels)
train_df.head(),test_df.head(),train_labels[:5],test_labels[:5]
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)
train_df.head(),  test_df.head(), len(train_df),len(test_df)
df3 = train_df.drop(['name','classname'], axis=1)
points = df3.to_numpy() 
int(points[3][0])
len(df1),len(df2)
train_data = []
for i,j in enumerate(train_df ['name']):
    direc = '../input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/images/%s'%j
    img1 = cv2.imread(direc,cv2.IMREAD_COLOR )
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
    x1= int(points[i][0])
    y1= int(points[i][1])
    x2= int(points[i][2])
    y2= int(points[i][3])
    #img2 = cv2.rectangle(img1.copy(), (x1,y1), (x2,y2), (0,0,255), 3)
    cropped = img1[y1:y2,x1:x2]
    
    image = cv2.resize(cropped,(224,224))
    image = img_to_array(image)
    image = preprocess_input(image)
    
    train_data.append(image)
    
#     fig = plt.figure()
#     a = fig.add_subplot(1, 2, 1)
#     plt.imshow(img2)
#     a = fig.add_subplot(1, 2, 2)
#     plt.imshow(cropped)
#     a.set_title(train_df['classname'][i])
#     print(i)

train_data = np.array(train_data, dtype="float32")

#print(data[5].shape)
train_labels = np.array(train_labels)
print(type(train_labels))
print(len(train_labels))

train_labels1 = to_categorical(train_labels)
print(train_labels1.shape)


len(train_data),i,j
train_labels1[:5]
df1[df1.name.str.contains('1861.jpg')]
df3 = test_df.drop(['name','classname'], axis=1)
points = df3.to_numpy() 
int(points[3][0])
test_data = []
for i,j in enumerate(test_df ['name']):
    direc = '../input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/images/%s'%j
    img1 = cv2.imread(direc,cv2.IMREAD_COLOR )
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
    x1= int(points[i][0])
    y1= int(points[i][1])
    x2= int(points[i][2])
    y2= int(points[i][3])
    #img2 = cv2.rectangle(img1.copy(), (x1,y1), (x2,y2), (0,0,255), 3)
    cropped = img1[y1:y2,x1:x2]
    
    image = cv2.resize(cropped,(224,224))
    image = img_to_array(image)
    image = preprocess_input(image)
   
    test_data.append(image)
    
#     fig = plt.figure()
#     a = fig.add_subplot(1, 2, 1)
#     plt.imshow(img2)
#     a = fig.add_subplot(1, 2, 2)
#     plt.imshow(cropped)
#     a.set_title(test_df['classname'][i])
#     print(i)

test_data = np.array(test_data, dtype="float32")

#print(data[5].shape)
test_labels = np.array(test_labels)
print(type(test_labels))
print(len(test_labels))

test_labels1 = to_categorical(test_labels)
print(test_labels1.shape)
train_labels.shape, train_data.shape
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")
baseModel = resnet50.ResNet50(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)
model = Model(inputs=baseModel.input, outputs=headModel)
for layer in baseModel.layers:
	layer.trainable = False
INIT_LR = 1e-3
EPOCHS = 5

print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])
BS = 32

print("[INFO] training head...")
H = model.fit(
	aug.flow(train_data, train_labels1, batch_size=BS),
	steps_per_epoch=len(train_data) // BS,
	validation_data=(test_data, test_labels1),
	validation_steps=len(test_data) // BS,
	epochs=EPOCHS)
# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(test_data, batch_size=BS)
# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)
# show a nicely formatted classification report
print(classification_report(test_labels.argmax(axis=1), predIdxs,
	target_names=lb.classes_))
# serialize the model to disk
# print("[INFO] saving mask detector model...")
# model.save(args["model"], save_format="h5")
# plot the training loss and accuracy
N = EPOCHS
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
#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
df5 =  pd.read_csv('../input/face-mask-detection-dataset/submission.csv')
df5.head(), len(df5)
df5.drop_duplicates(subset ="name", 
                     keep = 'first', inplace = True)
df5.reset_index(drop=True, inplace=True)
df5.head()
df5[df5.name.str.contains('0619.jpg')]
df5['name'][0]
net = cv2.dnn.readNetFromCaffe('../input/facedetector-model/deploy.prototxt.txt', '../input/facedetector-model/res10_300x300_ssd_iter_140000.caffemodel')
plt.rcParams["axes.grid"] = False
classname = []
imgname   = []
x1 = []
y1 = []
x2 = []
y2 = []
for n,i in enumerate(df5['name']):
    direc = '../input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/images/%s'%i
    image = cv2.imread(direc)
    image2 =cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image3 = image2.copy()
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
	(300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    for j in range(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with
	# the detection
        confidence = detections[0, 0, j, 2]
	# filter out weak detections by ensuring the confidence is
	# greater than the minimum confidence
        if confidence > 0.5:
		# compute the (x, y)-coordinates of the bounding box for
		# the object
            box = detections[0, 0, j, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
		# ensure the bounding boxes fall within the dimensions of
		# the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            # draw the bounding box of the face along with the associated
            if startX > w or startY > h:
                break
		# probability
            face = image2[startY:endY, startX:endX]
            #face = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)
            
            # pass the face through the model to determine if the face
		# has a mask or not
            (withoutMask, mask) = model.predict(face)[0]
            
            labl = "face_with_mask" if mask > withoutMask else "face_no_mask"
        
            classname.append(labl)
            imgname.append(i)
            x1.append(startX)
            y1.append(startY)
            x2.append(endX)
            y2.append(endY)
#             text = "{:.2f}%".format(confidence * 100)
#             y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image3, (startX, startY), (endX, endY),
 			(0, 0, 255),10)
#             cv2.putText(image, text, (startX, y),
# 			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
    if n < 10:
        plt.axis('off')
        plt.grid(b=None)
        fig = plt.figure()
        a = fig.add_subplot(1, 2, 1)
        plt.imshow(image3)
        a.set_title(labl)
        
classname[:20], imgname[:20], len(classname),len(imgname),
len(x1),len(y1),len(x2),len(y2)
name = pd.Series(imgname)
x1 = pd.Series(x1)
y1 = pd.Series(y1)
x2 = pd.Series(x2)
y2 = pd.Series(y2)
classname =  pd.Series(classname)
submit = pd.DataFrame({ 'name': name, 'x1': x1,'y1': y1, 'x2': x2,'y2': y2,'classname': classname })

len(submit), submit.tail(), df5.tail()
submit.to_csv('submission.csv')    