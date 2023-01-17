# https://www.pyimagesearch.com/2017/NUM_CLASSES/11/image-classification-with-keras-and-deep-learning/

# https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/

# https://machinelearningmastery.com/save-load-keras-deep-learning-models/

#https://www.learnopencv.com/image-classification-using-convolutional-neural-networks-in-keras/



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import random

import sys

import cv2

import matplotlib

from subprocess import check_output



from keras.models import Sequential

from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

#list the files in the input directory

#print(os.listdir("../input"))

#print(check_output(["ls", "../input"]).decode("utf8")) #trainLabels.csv

#print(check_output(["pwd", ""]).decode("utf8")) # /kaggle/working/

#classes : 0 - No DR, 1 - Mild, 2 - Moderate, 3 - Severe, 4 - Proliferative DR

def classes_to_int(label):

    # label = classes.index(dir)

    label = label.strip()

    if label == "No DR":  return 0

    if label == "Mild":  return 1

    if label == "Moderate":  return 2

    if label == "Severe":  return 3

    if label == "Proliferative DR":  return 4

    print("Invalid Label", label)

    return 5



def int_to_classes(i):

    if i == 0: return "No DR"

    elif i == 1: return "Mild"

    elif i == 2: return "Moderate"

    elif i == 3: return "Severe"

    elif i == 4: return "Proliferative DR"

    print("Invalid class ", i)

    return "Invalid Class"
NUM_CLASSES = 5

# we need images of same size so we convert them into the size

WIDTH = 128

HEIGHT = 128

DEPTH = 3

inputShape = (HEIGHT, WIDTH, DEPTH)

# initialize number of epochs to train for, initial learning rate and batch size

EPOCHS = 15

INIT_LR = 1e-3

BS = 32

#global variables

ImageNameDataHash = {}

uniquePatientIDList = []
def readTrainData(trainDir):

    global ImageNameDataHash

    # loop over the input images

    images = os.listdir(trainDir)

    print("Number of files in " + trainDir + " is " + str(len(images)))

    for imageFileName in images:

        if (imageFileName == "trainLabels.csv"):

            continue

        # load the image, pre-process it, and store it in the data list

        imageFullPath = os.path.join(os.path.sep, trainDir, imageFileName)

        #print(imageFullPath)

        img = load_img(imageFullPath)

        arr = img_to_array(img)  # Numpy array with shape (233,233,3)

        dim1 = arr.shape[0]

        dim2 = arr.shape[1]

        dim3 = arr.shape[2]

        if (dim1 < HEIGHT or dim2 < WIDTH or dim3 < DEPTH):

            print("Error image dimensions are less than expected "+str(arr.shape))

        arr = cv2.resize(arr, (HEIGHT,WIDTH)) #Numpy array with shape (HEIGHT, WIDTH,3)

        #print(arr.shape) # 128,128,3

        dim1 = arr.shape[0]

        dim2 = arr.shape[1]

        dim3 = arr.shape[2]

        if (dim1 != HEIGHT or dim2 != WIDTH or dim3 != DEPTH):

            print("Error after resize, image dimensions are not equal to expected "+str(arr.shape))

        #print(type(arr))

        # scale the raw pixel intensities to the range [0, 1] - TBD TEST

        arr = np.array(arr, dtype="float") / 255.0

        imageFileName = imageFileName.replace('.jpeg','')

        ImageNameDataHash[str(imageFileName)] = np.array(arr) 

    return
from datetime import datetime

print("Loading images at..."+ str(datetime.now()))

sys.stdout.flush()

readTrainData("/kaggle/working/../input/")

print("Loaded " + str(len(ImageNameDataHash)) + " images at..."+ str(datetime.now())) # 1000
#csv contains image	level

#10_left 0

#10_right 0

import csv

def readTrainCsv():

    raw_df = pd.read_csv('/kaggle/working/../input/trainLabels.csv', sep=',')

    print(type(raw_df)) #<class 'pandas.core.frame.DataFrame'>

    row_count=raw_df.shape[0] #gives number of row count row_count=35126 

    col_count=raw_df.shape[1] #gives number of col count col count=2

    print("row_count="+str(row_count)+" col count="+str(col_count))

    raw_df["PatientID"] = ''

    header_list = list(raw_df.columns)

    print(header_list) # ['image', 'level', 'PatientID']

    # double check if level of left and right are same or not

    ImageLevelHash = {}

    patientIDList = []

    for index, row in raw_df.iterrows():

        # 0 is image, 1 is level, 2 is PatientID, 3 is data

        key = row[0] + ''

        patientID = row[0] + ''

        patientID = patientID.replace('_right','')

        patientID = patientID.replace('_left','')

        #print("Adding patient ID"+ patientID)

        raw_df.at[index, 'PatientID'] = patientID

        patientIDList.append(patientID)

        ImageLevelHash[key] = str(row[1]) # level

                

    global uniquePatientIDList

    uniquePatientIDList = sorted(set(patientIDList))

    count=0;

    for patientID in uniquePatientIDList:

        left_level = ImageLevelHash[str(patientID+'_left')]

        right_level = ImageLevelHash[str(patientID+'_right')]

        #right_exists = str(patientID+'_right') in raw_df.values

        if (left_level != right_level):

            count = count+1

            #print("Warning for patient="+ str(patientID) + " left_level=" + left_level+ " right_level=" +right_level)

    print("count of images with both left and right eye level not matching="+str(count)) # 2240

    print("number of unique patients="+str(len(uniquePatientIDList))) # 17563

    return raw_df
random.seed(10)

print("Reading trainLabels.csv...")

df = readTrainCsv()
for i in range(0,10):

    s = df.loc[df.index[i], 'PatientID'] # get patient id of patients

    print(str(i) + " patient's patientID="+str(s))
# df has 3 columns ['image', 'level', 'PatientID']

keepImages =  list(ImageNameDataHash.keys())

df = df[df['image'].isin(keepImages)]

print(len(df)) # 1000
#convert hash to dataframe

imageNameArr = []

dataArr = []

for index, row in df.iterrows():

    key = str(row[0])

    if key in ImageNameDataHash:

        imageNameArr.append(key)

        dataArr.append(np.array(ImageNameDataHash[key])) # np.array



df2 = pd.DataFrame({'image': imageNameArr, 'data': dataArr})

df2_header_list = list(df2.columns) 

print(df2_header_list) # ['image', 'data']

print(len(df2)) # 1000

#print(df2.describe(include='all'))

#print(df2.sample(3)) # 3 rows x 2 columns
if len(df) != len(df2):

    print("Error length of df != df2")

    

for idx in range(0,len(df)):

    if (df.loc[df.index[idx], 'image'] != df2.loc[df2.index[idx], 'image']):

        print("Error " + df.loc[df.index[idx], 'image'] +"==" + df2.loc[df2.index[idx], 'image'])

        

print(df2.dtypes)

print(df.dtypes)
df = pd.merge(df2, df, left_on='image', right_on='image', how='outer')

df_header_list = list(df.columns) 

print(df_header_list) # 'image', 'data', level', 'PatientID'

print(len(df)) # 1000

print(df.sample())
sample0 = df.loc[df.index[0], 'data']

print(sample0)

print(type(sample0)) # <class 'numpy.ndarray'>

print(sample0.shape) # 128,128,3

from matplotlib import pyplot as plt

plt.imshow(sample0, interpolation='nearest')

plt.show()

print("Sample Image")
X = df['data']

Y = df['level']

# scale the raw pixel intensities to the range [0, 1]

#print(type(X)) # 'pandas.core.series.Series'

#X = np.array(X, dtype="float") / 255.0 -- TBD moved to top

Y = np.array(Y)

# convert the labels from integers to vectors

Y =  to_categorical(Y, num_classes=NUM_CLASSES)
# partition the data into training and testing splits using 75% training and 25% for validation

print("Parttition data into 75:25...")

sys.stdout.flush()

print("Unique patients in dataframe df=" + str(df.PatientID.nunique())) # 500

unique_ids = df.PatientID.unique()

print('unique_ids shape='+ str(len(unique_ids))) #500



# Refer https://www.kaggle.com/kmader/tf-data-tutorial-with-retina-and-keras

train_ids, valid_ids = train_test_split(unique_ids, test_size = 0.25, random_state = 10) #stratify = rr_df['level'])

trainid_list = train_ids.tolist()

print('trainid_list shape=', str(len(trainid_list))) # 375



traindf = df[df.PatientID.isin(trainid_list)]

valSet = df[~df.PatientID.isin(trainid_list)]
print(traindf.head())

print(valSet.head())



traindf = traindf.reset_index(drop=True)

valSet = valSet.reset_index(drop=True)



print(traindf.head())

print(valSet.head())
trainX = traindf['data']

trainY = traindf['level']



valX = valSet['data']

valY = valSet['level']



#(trainX, valX, trainY, valY) = train_test_split(X,Y,test_size=0.25, random_state=10)

print('trainX shape=', trainX.shape[0], 'valX shape=', valX.shape[0]) # 750, 250
trainY =  to_categorical(trainY, num_classes=NUM_CLASSES)

valY =  to_categorical(valY, num_classes=NUM_CLASSES)
#construct the image generator for data augmentation

print("Generating images...")

sys.stdout.flush()

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, \

    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,\

    horizontal_flip=True, fill_mode="nearest")
def createModel():

    model = Sequential()

    # first set of CONV => RELU => MAX POOL layers

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=inputShape))

    model.add(Conv2D(32, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))

    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))

    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(output_dim=NUM_CLASSES, activation='softmax'))

    # returns our fully constructed deep learning + Keras image classifier 

    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

    # use binary_crossentropy if there are two classes

    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    return model
print("Reshaping trainX at..."+ str(datetime.now()))

#print(trainX.sample()) 

print(type(trainX)) # <class 'pandas.core.series.Series'>

print(trainX.shape) # (750,)

from numpy import zeros

Xtrain = np.zeros([trainX.shape[0],HEIGHT, WIDTH, DEPTH])

for i in range(trainX.shape[0]): # 0 to traindf Size -1

    Xtrain[i] = trainX[i]

print(Xtrain.shape) # (750,128,128,3)

print("Reshaped trainX at..."+ str(datetime.now()))
print("Reshaping valX at..."+ str(datetime.now()))

print(type(valX)) # <class 'pandas.core.series.Series'>

print(valX.shape) # (250,)

from numpy import zeros

Xval = np.zeros([valX.shape[0],HEIGHT, WIDTH, DEPTH])

for i in range(valX.shape[0]): # 0 to traindf Size -1

    Xval[i] = valX[i]

print(Xval.shape) # (250,128,128,3)

print("Reshaped valX at..."+ str(datetime.now()))
# initialize the model

print("compiling model...")

sys.stdout.flush()

model = createModel()



# print the summary of model

from keras.utils import print_summary

print_summary(model, line_length=None, positions=None, print_fn=None)



# add some visualization

from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))
# train the network

print("training network...")

sys.stdout.flush()

#class_mode ='categorical', # 2D one-hot encoded labels

H = model.fit_generator(aug.flow(Xtrain, trainY, batch_size=BS), \

    validation_data=(Xval, valY), \

    steps_per_epoch=len(trainX) // BS, \

    epochs=EPOCHS, verbose=1)



# save the model to disk

print("Saving model to disk")

sys.stdout.flush()

model.save("/tmp/mymodel")
# set the matplotlib backend so figures can be saved in the background

# plot the training loss and accuracy

print("Generating plots...")

sys.stdout.flush()

matplotlib.use("Agg")

matplotlib.pyplot.style.use("ggplot")

matplotlib.pyplot.figure()

N = EPOCHS

matplotlib.pyplot.plot(np.arange(0, N), H.history["loss"], label="train_loss")

matplotlib.pyplot.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")

matplotlib.pyplot.plot(np.arange(0, N), H.history["acc"], label="train_acc")

matplotlib.pyplot.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")

matplotlib.pyplot.title("Training Loss and Accuracy on diabetic retinopathy detection")

matplotlib.pyplot.xlabel("Epoch #")

matplotlib.pyplot.ylabel("Loss/Accuracy")

matplotlib.pyplot.legend(loc="lower left")

matplotlib.pyplot.savefig("plot.png")