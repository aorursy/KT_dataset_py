import numpy as np 
import matplotlib.pyplot as plt 
from keras.models import Sequential 
from keras.layers import Dense, BatchNormalization, Dropout, Flatten 
from keras.optimizers import Adam 
from keras.utils.np_utils import to_categorical 
from keras.layers.convolutional import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import pickle 
import cv2 
import os 
import pandas as pd 
import random 
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array 
from scipy import misc , ndimage
from tensorflow import keras
import tensorflow as tf 
############################ Parameters ###############################

imageDimensions = (32,32)
testRatio = 0.2   #if  1000 images split will 200 for testing 
validationRation = 0.2 #if 1000 images 20% of remaining 800 will be 160 for validation
################## the way ####################################
path ="../input/traffic-sign-images-from-turkey/Trafik/Trafik"
labelFile = "../input/traffic-sign-images-from-turkey/labels.csv"

count = 0
images = []
classNo = []
myList = os.listdir(path)
print("Total Classes Detected:",len(myList))
noOfClasses=len(myList)
print("Importing Classes.....")
for x in range (0,len(myList)):
    myPicList = os.listdir(path+"/"+str(count))
    for y in myPicList:
        curImg = cv2.imread(path+"/"+str(count)+"/"+y,cv2.IMREAD_GRAYSCALE ) # Tek kanallı yapıyoruz.. 
        curImg=cv2.resize(curImg, (32,32)) # Boyutlar eşitleniyor.
        images.append(curImg)
        classNo.append(count)
    print(count, end =" ")
    count +=1
print(" ")
images = np.array(images)
classNo = np.array(classNo)
print(images.shape)
print(images[0])
print(images.dtype)### Veri type #### 
########################################## Split Data
X_train, X_test , y_train , y_test = train_test_split(images, classNo, test_size=testRatio)
X_train, X_validation , y_train , y_validation = train_test_split(X_train, y_train , test_size = validationRation)
########################## to check if number of images matches to number of labels for each data set 

print("Data Shapes")
print("Train", end="");print(X_train.shape,y_train.shape)
print("Validation",end="");print(X_validation.shape,y_validation.shape)
print("Test",end=""); print(X_test.shape , y_test.shape)

assert(X_train.shape[0] == y_train.shape[0]),"The number of images in not equal to the number of labels(hedef değişken) in training set"
assert(X_validation.shape[0] == y_validation.shape[0]),"The number of images in not equal to the number of labels validation set"
assert(X_test.shape[0] == y_test.shape[0]),"The number of images in not equal to the number of labels test set"
assert(X_train.shape[1:]== (imageDimensions)), "The dimension of the Training images are wrong"
assert(X_validation.shape[1:]==(imageDimensions)),"The dimension of the Validation images are wrong"
assert(X_test.shape[1:]== imageDimensions),"The dimension of test images aste wrong"
################################# READ CSV FILE ###################
data = pd.read_csv(labelFile, encoding="ISO-8859-1")
print("Data Shape: ",data.shape, type(data))
data
###################33 DISPLAY SOME SAMPLES OF ALL CLASSES ###################

plt.figure(figsize=(35,6))
for i in range(20):
    plt.subplot(2,10,i+1)
    plt.imshow(X_train[i])
    plt.title("{}".format(data.Name[y_train[i]]))    
    plt.axis("off")
######################## DISPLAY A BAR CHART SHOWING NO OF SAMPLES FOR EACH CATEGORY 

plt.figure(figsize=(12,4))
plt.hist(classNo, bins=len(data.Classid))
plt.title("Distribution of the training dataset")
plt.xlabel("Class Number")
plt.ylabel("Number of images")
plt.show()
X_train = X_train.astype(np.uint8)
X_validation = X_validation.astype(np.uint8)
X_test = X_test.astype(np.uint8)
############################### PREPROCESSING THE IMAGES 

def equalize(img):
    img=cv2.equalizeHist(img)
    return img 

rand_num = random.randint(0,len(X_train)-1)
plt.imshow(X_train[rand_num])
plt.title("GrayScale Images {}".format(data.Name[y_train[rand_num]]))
plt.show()

 
X_train = np.array(list(map(equalize, X_train)))
X_validation = np.array(list(map(equalize, X_validation)))
X_test = np.array(list(map(equalize, X_test)))
plt.imshow(X_train[rand_num])
plt.title("GrayScale Images {}".format(data.Name[y_train[rand_num]]))
plt.show()
######################### ADD A DEPTH OF 1 ######## Bır katmanlı yaptık X_train.shape = (13443,32,32,1)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1],X_train.shape[2],1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1],X_validation.shape[2],1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1],X_test.shape[2],1)
########################## AUGMENTATION OF IMAGES TO MAKE IT MORE GENERIC 

dataGen = ImageDataGenerator(width_shift_range=0.1, 
                            height_shift_range=0.1,
                            zoom_range=0.2,
                            shear_range=0.1,
                            rotation_range=10)
dataGen.fit(X_train)
batches = dataGen.flow(X_train, y_train, batch_size=20)
X_batch, y_batch = next(batches)
# TO SHOW AGMENTED IMAGE SAMPLES 

fig, axs = plt.subplots(1,15, figsize= (20,5))
fig.tight_layout()

for i in range(15):
    axs[i].imshow(X_batch[i].reshape(imageDimensions[0], imageDimensions[1]))
    axs[i].axis('off')
plt.show()
############################ CONVOLUTION NEURAL NETWORK MODEL 

def myModel():
    no_Of_Filters = 64 
    size_of_Filter = (5,5)
    size_of_Filter2 = (3,3)
    size_of_pool= (3,3)
    no_Of_Nodes = 500
    
    model = Sequential()
    model.add((Conv2D(no_Of_Filters, size_of_Filter, input_shape=(imageDimensions[0], imageDimensions[1],1), activation="relu")))
    model.add((Conv2D(no_Of_Filters // 2, size_of_Filter2, activation="relu")))
    model.add(MaxPooling2D(pool_size = size_of_pool))
    model.add(BatchNormalization())
    model.add(Flatten())
    
    model.add(Dense(no_Of_Nodes, activation="relu"))
    model.add(Dense(noOfClasses, activation="softmax"))
    
    #COMPILE MODEL
    model.compile('rmsprop',loss='sparse_categorical_crossentropy',metrics=['accuracy']) 
    return model
batch_size_val=40 
 
model = myModel()
print(model.summary())
history=model.fit_generator(dataGen.flow(X_train,y_train,batch_size=batch_size_val),
                            epochs=25,validation_data=(X_validation,y_validation))
############################### PLOT veriler düzeltilmeden önce
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Acurracy')
plt.xlabel('epoch')
plt.show()
score =model.evaluate(X_test,y_test,verbose=0)
print('Test Loss Score:',score[0])
print('Test Accuracy:',score[1])
model.save("MY_h5_model_5.h5") ### Tam dosya yolu gerekli.
import numpy as np 
import cv2 
import pandas as pd 
##################################33 
label = pd.read_csv('../input/traffic-sign-images-from-turkey/labels.csv',encoding="ISO-8859-1")
frameWidth = 640 
frameHeight = 480 
brightness = 180 
threshold = 0.75
font = cv2.FONT_HERSHEY_SIMPLEX
# Setup the video camera 

cap = cv2.VideoCapture(0)
cap.set(3,frameWidth)
cap.set(4,frameHeight)
cap.set(10, brightness)

# import the trannined model 
model = keras.models.load_model("MY_h5_model_5.h5")
def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img =cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    return img

def getCalssName(classNo):
    labels=pd.read_csv("labels.csv",encoding='ISO-8859-1')
    a=labels[labels["Classid"]==classNo]["Name"]
    return a


while True:
 
    # READ IMAGE
    success, imgOrignal = cap.read()
 
    # PROCESS IMAGE
    img = np.asarray(imgOrignal) 
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    
    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 32, 32, 1)
    
    cv2.putText(imgOrignal, "CLASS: " , (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA)
    
    # PREDICT IMAGE
    predictions = model.predict(img)
    probabilityValue =np.amax(predictions)
    classIndex = np.where(predictions == probabilityValue)[1][0]
    
    if probabilityValue > threshold:
        #print(getCalssName(classIndex))
        cv2.putText(imgOrignal,str(classIndex)+" "+str(getCalssName(classIndex)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOrignal, str(round(probabilityValue*100,2) )+"%", (180, 75), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow("Result", imgOrignal)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
