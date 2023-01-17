#installing imutils
!pip install imutils
#import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

#the libraries that we will use in the deploment step
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import imutils
import time
import cv2
#initialize learning rate , number of epochs and batch_size
INIT_LR = 1e-4
EPOCHS = 20
BATCH_SIZE = 32
directory = r"../input/face-datasets"
categories = ["with_mask" , 'without_mask']
#grab the list of images in our dataset directory , then initialize the list of data( images) and class names
print("[INFO] loading images...")
data = []
labels = []

for category in categories:
    path = os.path.join(directory , category)
    for img in os.listdir(path):
        img_path = os.path.join(path , img)
        image = load_img(img_path , target_size=(224 , 224))
        image = img_to_array(image)
        image = preprocess_input(image)
        
        data.append(image)
        labels.append(category)
#perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
data = np.array(data , "float32")
labels = np.array(labels)
#splitting the data
(Xtrain , Xtest , ytrain , ytest) = train_test_split(data , labels , test_size = 0.20 , random_state = 42 , stratify = labels)
#construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20,
                        zoom_range=0.15,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        shear_range=0.15,
                        horizontal_flip=True,
                        fill_mode = "nearest")
#load the MobileNetV2 network , ensuring the head FC layer sets are left off
base_model = MobileNetV2(weights="imagenet" , include_top=False , input_tensor=Input(shape = (224 , 224 , 3)))
#construct the head of the model that will be placed on top of the base_model
head_model = base_model.output
head_model = AveragePooling2D(pool_size = (7,7))(head_model)
head_model = Flatten(name = "flatten")(head_model)
head_model = Dense(128 , activation = "relu")(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(2 , activation = "softmax")(head_model)
#place the head_model on top of the base_model (this will be the actual model that we will train)
model = Model(inputs = base_model.input , outputs = head_model)
#loop over all layers of base model and freeze them so they will not be updated during the first training process
for layer in base_model.layers:
    layer.trainable = True
#compiling the model
print("[INFO] compiling the model...")
opt = Adam(lr = INIT_LR , decay = INIT_LR / EPOCHS)
model.compile(loss = "binary_crossentropy" , optimizer = opt , metrics=["accuracy"])
#train the head of the network
print("[info] training the head...")
history = model.fit(aug.flow(Xtrain , ytrain , batch_size=BATCH_SIZE),
             steps_per_epoch = len(Xtrain)//BATCH_SIZE , 
             validation_data = (Xtest , ytest),
             validation_steps = len(Xtest)//BATCH_SIZE,
             epochs = EPOCHS)
#make predictions on the testing set
print("[INFO] evaluating the model...")
pred_idxs = model.predict(Xtest , batch_size=BATCH_SIZE)
#for each image in the testing set we need to find the index of the label with corresponding to the largest predicted prbability
pred_idxs = np.argmax(pred_idxs , axis=1)
#show a nicely formatted classification report
print(classification_report(ytest.argmax(axis = 1) , pred_idxs , target_names=lb.classes_))
#saving the model
print("[INFO] saving mask detector model...")
model.save("face_mask_detector.model" , save_format="h5")
#plot the loss
N = EPOCHS
plt.plot(np.arange(0,N) , history.history["loss"] , "r--")
plt.plot(np.arange(0,N) , history.history["val_loss"] ,"b-")
plt.legend(["Training Loss" , "Validation Loss"])
plt.xlabel("EPOCHS")
plt.ylabel("Loss")
plt.savefig("loss_plot.png")
plt.show()
#plot the Accuracy
N = EPOCHS
plt.plot(np.arange(0,N) , history.history["accuracy"] , "r--")
plt.plot(np.arange(0,N) , history.history["val_accuracy"] ,"b-")
plt.legend(["Training Accuracy" , "Validation Accuracy"])
plt.xlabel("EPOCHS")
plt.ylabel("Accuracy")
plt.show()
plt.savefig("accuracy_plot.png")
def detect_and_predict_mask(frame , face_net , mask_net):
    #grap the dimensions of the frame and then construct a blob from it
    (h,w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame , 1.0 , (224,224) , (104.0 , 177.0 , 123.0))
    
    #pass the blob through the network and obtain the face detections
    face_net.setInput(blob)
    detections = face_net.forward()
    print(detections.shape)
    
    #initialize our list of faces , their corresponding locations and the list of predictions from our face mask network
    faces = []
    locs  = []
    preds = []
    
    #loop over the detections
    for i in range(0 , detections.shape[2]):
        #extract the confidence (propability) associated with the detection
        confidence = detections[0,0,i,2]
        
        #filter out weak detections by ensuring the confidence is greater than the minimum confidence
        if confidence > 0.5:
            #compute the (x,y) coordinates of the bounding box for the object
            box = detections[0,0,i,3:7]*np.array([w,h,w,h])
            (startX , starty , endX , endy) = box.astype("int")
            
            #ensure bounding boxes fall within the dimension of the frame
            (startX , starty) = (max(0 , startX) , max(0 , starty))
            (endX , endy) = (min(w-1 , endX) , min(h-1 , endy))
            
            #extract the face ROI , convert it from BGR to RGB channel ordering , resize it to 224 * 224 and preprocess it
            face = frame[starty:endy , startX:endX]
            face = cv2.cvtColor(face , cv2.COLOR_BGR2RGB)
            face = cv2.resize(face , (224,224))
            face = img_to_array(face)
            face = preprocess_input(face)
            
            #add the face and bounding boxes to their respective lists
            faces.append(face)
            locs.append((startX , starty , endX , endy))
            
    #only make a predictions if at least one face was detected 
    if  len(faces) > 0:
        #for faster inference we will make batch predictions on "all" faces at the same time rather than one-by-one
        faces = np.array(faces , dtype = "float32")
        preds = mask_net.predict(faces , batch_size = 32)
        
    
    #return a 2-tuple of the face locations and their corresponding locations
    return(locs , preds)
    
#load our serialized face detector model from disk
prototxt_path = r"../input/face-detector/deploy.prototxt"
weights_path  = r"../input/face-detector/res10_300x300_ssd_iter_140000.caffemodel"
face_net = cv2.dnn.readNet(prototxt_path , weights_path)
#load the Face Mask Detection model from disk
mask_net = load_model("./face_mask_detector.model")
#initialize the videostream
print("[INFO] starting the video stream...")
vs = VideoStream(src = 0).start()
#loop over the frames from video stream
while True:
    #grap the frame from the threaded videostream and resize it to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame , width=400)
    
    #detect faces in the frame and determine if they are wearing face mask or not
    (locs , preds) = detect_and_predict_mask(frame , face_net , mask_net)
    
    #loop over the detected face locations and their corresponding locations
    for (box , pred) in zip(locs , preds):
        (startX , starty , endX , endy) = box
        (mask , without_mask) = pred
        
        #determine the class label and color we will use to draw the bounding box and text
        label = "Mask" if mask > without_mask else "No Mask"
        color = (0,255,0) if label == "Mask" else (0,0,255)
        
        #include the probability in the label
        label = "{}:{:.2f}".format(label , max(mask , without_mask)*100)
        
        #display the label and bounding box rectangle on the output frame
        cv2.putText(frame , label , (startX , starty-10) , cv2.FONT_HERSHEY_SIMPLEX , 0.45 , color , 2)
        cv2.rectangle(frame , (startX , starty) , (endX , endy) , color , 2)
        
    #show the output frame
    cv2.imshow("Frame" , frame)
    key = cv2.waitKey(1)&0xff
    
    #if the "q" key is pressed , break from the loop
    if key == ord("q"):
        break
cv2.destroyAllWindows()
vs.stop()
from IPython.display import display, Image

without_mask = "../input/model-deploying-pictures/without_mask.png"
display(Image(filename = with_mask))

with_mask = "../input/model-deploying-pictures/with_mask.png"
display(Image(filename = with_mask))
