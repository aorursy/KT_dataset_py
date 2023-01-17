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
!pip install imutils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten,AveragePooling2D
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.optimizers import Adam
from keras.layers import Input 
#from keras.appications.inception_v3 import preprocess_input, decode_predictions
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import shutil
import cv2
import os
from imutils import paths
dataset_path = '/kaggle/input/who-is-she-for-soobi/Who_is_she/training_set/training_set'
def ceildiv(a, b):
    return -(-a // b)

def plots_from_files(imspaths, figsize=(100, 100), rows=1, titles=None, maintitle=None):
    """Plot the images in a grid"""
    f = plt.figure(figsize=figsize)
    if maintitle is not None: plt.suptitle(maintitle, fontsize=48)
    for i in range(len(imspaths)):
        if i <3 :
            sp = f.add_subplot(rows, ceildiv(len(imspaths), rows), i+1)
            sp.axis('Off')
            if titles is not None: sp.set_title(titles[i] , fontsize= 48)
            img = plt.imread(imspaths[i])
            plt.imshow(img)
akane_images = list(paths.list_images(f"{dataset_path}/akane"))
neru_images = list(paths.list_images(f"{dataset_path}/neru"))
rika_images = list(paths.list_images(f"{dataset_path}/rika"))
risa_images = list(paths.list_images(f"{dataset_path}/risa"))
yui_images = list(paths.list_images(f"{dataset_path}/yui"))
%matplotlib inline
plots_from_files(akane_images, rows=10, maintitle="akane")
plots_from_files(neru_images, rows=10, maintitle="neru")
plots_from_files(rika_images, rows=10, maintitle="rika")
plots_from_files(risa_images, rows=10, maintitle="risa")
plots_from_files(yui_images, rows=10, maintitle="yui")
# remove directories : '.ipynb_checkpoints' and '.ipynb_checkpoints'
imagePaths = list(paths.list_images(dataset_path))
for imagePath in imagePaths:
    # extract the class label from the filename
    if imagePath.split(os.path.sep)[-2] == '.ipynb_checkpoints' or imagePath.split(os.path.sep)[-2] == '.ipynb_checkpoints':
        os.remove(imagePath)
# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images

print("[INFO] loading images...")
imagePaths = list(paths.list_images(dataset_path))

data = []
labels = []

# loop over the image paths
for imagePath in imagePaths:
    # extract the class label from the filename    
    label = imagePath.split(os.path.sep)[-2]
    # load the image, swap color channels, and resize it to be a fixed       
    image = cv2.imread(imagePath)
    #轉為RGB格式
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #resize # 224x224 pixels while ignoring aspect ratio
    image = cv2.resize(image, (128, 128))
    # update the data and labels lists, respectively
    data.append(image)
    labels.append(label)
# convert the data and labels to NumPy arrays while scaling the pixel
# intensities to the range [0, 1]
print(labels)
#normalize到0-1之前的值，可試試不要除
data = np.array(data) / 255.0
labels = np.array(labels)
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
print(labels.shape)
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
#labels = to_categorical(labels)
#lb.classes_
# if only two categories
# perform one-hot encoding on the labels
#lb = LabelBinarizer()
#labels = lb.fit_transform(labels)
#labels = to_categorical(labels)
print(labels)
# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=5)
# initialize the training data augmentation object
trainAug = ImageDataGenerator(rotation_range=15,width_shift_range=1.0,height_shift_range=1, fill_mode="nearest")
lb.classes_
# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-3
EPOCHS = 30
BS = 3
from keras.layers import Input 

# load the VGG16 network, ensuring the head FC layer sets are left
# off
model= ""
baseModel = ""
baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(128, 128, 3)))
# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(2, 2))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel) # the same with resize
headModel = Dropout(0.5)(headModel)
headModel = Dense(5, activation="softmax")(headModel)
# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)
# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
    layer.trainable = False
    
model.summary()
# compile our model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the head of the network
print("[INFO] training head...")
H = model.fit_generator(
    trainAug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS)
# the meaning of loss
# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
#plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
#plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Who is she")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")
# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)
# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)
# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))
# compute the confusion matrix and and use it to derive the raw
# accuracy, sensitivity, and specificity
cm = confusion_matrix(testY.argmax(axis=1), predIdxs)

total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]+ cm[2, 2]+ cm[3, 3]+ cm[4, 4]) / total
#sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
#specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
# show the confusion matrix, accuracy, sensitivity, and specificity
print(cm)
print("acc: {:.4f}".format(acc))
#print("sensitivity: {:.4f}".format(sensitivity))
#print("specificity: {:.4f}".format(specificity))
def loadimg(imagfile):
    # 224x224 pixels while ignoring aspect ratio
    image = cv2.imread(imagfile)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128)) 
    return (np.array(image)/255).reshape(1,128,128,3)


test_d= loadimg("/kaggle/input/who-is-she-for-soobi/Who_is_she/data/testing_set/FE7571BBC1.png")
#test_d= loadimg("/kaggle/input/covid19/covid19/test_image/cov.jpg")
predIdxs = model.predict(test_d, batch_size=BS)
# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
#predIdxs = np.argmax(predIdxs, axis=1)
print(predIdxs)
predIdxs = np.argmax(predIdxs)
if predIdxs == 0 :
    print('akane(守屋茜)')
elif predIdxs == 1 :
    print('neru(長濱ねる)')
elif predIdxs == 2 :
    print('rika(渡边梨加)')
elif predIdxs == 3 :
    print('risa(渡邊理佐)')
elif predIdxs == 4 :
    print('yui(小林由依)')

             
import matplotlib.pyplot as plt

img = plt.imread("/kaggle/input/who-is-she-for-soobi/Who_is_she/data/testing_set/FE7571BBC1.png")
plt.imshow(img)

