%matplotlib inline

import os

import matplotlib.pyplot as plt



data={}

for dirname, _, filenames in os.walk('/kaggle/input/cash-recognition-system/Cash'):

    print(dirname)

    item = dirname.split('/')[-1]

    if len(filenames)>0:

        data[item]=len(filenames)
my_colors = ['brown','pink', 'red', 'green', 'blue', 'cyan','orange','purple'] 

plt.bar([i for i in data.keys()],[j for j in data.values()],color=my_colors)

plt.xlabel('Notes')

plt.ylabel('No of image')

plt.show()


import matplotlib.image as mpimg

BasePath = '/kaggle/input/cash-recognition-system/Cash/Cash/100/'

fileNames= os.listdir('/kaggle/input/cash-recognition-system/Cash/Cash/100/')

Img_path=[]

for file in fileNames[:10]:

    IMG_PATH = os.path.join(BasePath,file)

    Img_path.append(IMG_PATH)

    

#print(Img_path)

plt.figure(figsize=(50,50))

for i in range(10):

    plt.subplot(2,5,i+1)

    plt.axis('off')

    plt.xticks([]),plt.yticks([])

    image = mpimg.imread(Img_path[i])

    plt.imshow(image)

plt.show()

%%writefile hdf5datasetwriter.py

import h5py

import os



class HDF5DatasetWriter:

    def __init__(self, dims, outputPath, dataKey="images",

    bufSize=500):

        # check to see if the output path exists, and if so, raise

        # an exception

        if os.path.exists(outputPath):

            raise ValueError("The supplied ‘outputPath‘ already "

            "exists and cannot be overwritten.Manually delete "

            "the file before continuing.", outputPath)

        self.db = h5py.File(outputPath, "w")

        self.data = self.db.create_dataset(dataKey, dims,

                                           dtype="float",compression='gzip',compression_opts=9)

        self.labels = self.db.create_dataset("labels", (dims[0],),

                                             dtype="int",compression='gzip',compression_opts=9)

        self.bufSize = bufSize

        self.buffer = {"data": [], "labels": []}

        self.idx = 0



    def add(self, rows, labels):



        # add the rows and labels to the buffer

        self.buffer["data"].extend(rows)

        self.buffer["labels"].extend(labels)

        if len(self.buffer["data"]) >= self.bufSize:

            self.flush()



    def flush(self):



        # write the buffers to disk then reset the buffer

        i = self.idx + len(self.buffer["data"])

        self.data[self.idx:i] = self.buffer["data"]

        self.labels[self.idx:i] = self.buffer["labels"]

        self.idx = i

        self.buffer = {"data": [], "labels": []}



    def storeClassLabels(self, classLabels):



        # create a dataset to store the actual class label names,

        # then store the class labels

        dt = h5py.special_dtype(vlen=str)

        labelSet = self.db.create_dataset("label_names",

                                          (len(classLabels),), dtype=dt)

        labelSet[:] = classLabels



    def close(self):



        # check to see if there are any other entries in the buffer

        # that need to be flushed to disk

        if len(self.buffer["data"]) > 0:

            self.flush()



        # close the dataset

        self.db.close()
import os

from sklearn.preprocessing import LabelEncoder

from keras.applications import VGG16

from keras.applications import imagenet_utils

from keras.preprocessing.image import img_to_array

from keras.preprocessing.image import load_img

from sklearn.preprocessing import LabelEncoder

from hdf5datasetwriter import HDF5DatasetWriter

#from imutils import paths

import numpy as np

from tqdm import tqdm

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_PATH = []

labels =[]



for dirname, _, filenames in os.walk('/kaggle/input/cash-recognition-system/Cash/Cash/'):

    for filename in filenames:

        imagepath = os.path.join(dirname,filename)

        IMG_PATH.append(imagepath)

        label = imagepath.split('/')[-2]

        labels.append(label)

       # print(imagepath)

        #print(label)

print(len(IMG_PATH))

print(len(labels))

le = LabelEncoder()

labels=le.fit_transform(labels)

le.classes_

classTotals = labels.sum(axis=0)

classWeight = classTotals.max()/classTotals
Base_model = VGG16(weights="imagenet", include_top=False,input_shape=(224,224,3))

Base_model.summary()
writer = HDF5DatasetWriter((len(IMG_PATH),512*7*7),'Nepali_Cash_Features.hdf5',bufSize=100)

writer.storeClassLabels(le.classes_)
Batch_size=32

for i in tqdm(np.arange(0,len(IMG_PATH),Batch_size)):

    batchPaths = IMG_PATH[i:i+Batch_size]

    batchLabels = labels[i:i+Batch_size]

    batchImages = []

    

    for (j,imagePath) in enumerate(batchPaths):

        image = load_img(imagePath,target_size = (224,224))

        image = img_to_array(image)

        image=np.expand_dims(image,axis=0)

        image = imagenet_utils.preprocess_input(image)

        batchImages.append(image)

    

    batchImages = np.vstack(batchImages)

    features = Base_model.predict(batchImages,batch_size=Batch_size)

    features = features.reshape((features.shape[0],512*7*7))

    writer.add(features,batchLabels)

writer.close()



        
import h5py

db=h5py.File('/kaggle/working/Nepali_Cash_Features.hdf5')

list(db.keys())
print(db['images'].shape)

print(db['labels'].shape)

print(db['label_names'].shape)
db = h5py.File('/kaggle/working/Nepali_Cash_Features.hdf5', "r")

i = int(db["labels"].shape[0] * 0.75)
trainX=db["images"][:i]

testX=db["images"][i:]

trainY=db["labels"][:i]

testY = db["labels"][i:]

print(trainX[0])
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=0)

trainX_res,trainY_res = ros.fit_resample(trainX, trainY)
from collections import Counter

print(sorted(Counter(trainY_res).items()))
from keras.models import Sequential

from keras.layers import Flatten,Dense,GlobalAveragePooling2D



model=Sequential()

model.add(GlobalAveragePooling2D(input_shape=(7,7,512)))

model.add(Dense(6,activation='softmax'))

model.summary()
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=["acc"])

model.fit(trainX_res.reshape(-1,7,7,512), trainY_res,epochs=100,validation_data=(testX.reshape(-1,7,7,512),testY))
db.close()
model.save('NepaliCash.hdf5')
!pip install imutils

from imutils import paths

batchImages = []

imagePaths = list(paths.list_images('../input/neplesenotes/Neplese notes/'))

for image in imagePaths:

    img=load_img(image ,target_size=(224, 224,3))

    img=img_to_array(img)

    img=np.expand_dims(img,axis=0)

    img=imagenet_utils.preprocess_input(img)

    batchImages.append(img)

batchImages=np.vstack(batchImages)

    

features=Base_model.predict(batchImages)

features=features.reshape(features.shape[0],7*7*512)
prediction=model.predict(features.reshape(-1,7,7,512)).argmax(axis=1)

import cv2

classLabels=list(le.classes_)

fig=plt.figure(figsize=(50, 50))

columns = 2

rows = 3

for (i, imagePath) in enumerate(imagePaths):

    # load the example image, draw the prediction, and display it

    # to our screen

    image = cv2.imread(imagePath)

    image= cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    cv2.putText(image, "Label: {}".format(classLabels[prediction[i]]),

    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    fig.add_subplot(rows, columns,i+1)

    plt.imshow( image)

plt.show()