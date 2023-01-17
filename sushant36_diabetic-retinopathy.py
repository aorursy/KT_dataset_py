import numpy as np 
import pandas as pd 
import os, random, sys, cv2, matplotlib, csv, keras
from subprocess import check_output
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import SGD, Adam
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing import image
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
#Storing images in image-data pair

print("Loading images at..."+ str(datetime.now()))
sys.stdout.flush()

ImageNameDataHash = {}            #For storing image information in image-data pair
images = os.listdir("/kaggle/working/../input/")

for imageFileName in images:
    if(imageFileName == "trainLabels.csv"):
        continue
    img = load_img(os.path.join(os.path.sep, "/kaggle/working/../input/", imageFileName))
    arr = img_to_array(img)
    dim1 = arr.shape[0]
    dim2 = arr.shape[1]
    dim3 = arr.shape[2]
    if (dim1 < HEIGHT or dim2 < WIDTH or dim3 < DEPTH):
        print("Error image dimensions are less than expected "+str(arr.shape))
    
    arr = cv2.resize(arr, (HEIGHT,WIDTH))       
    
    dim1 = arr.shape[0]
    dim2 = arr.shape[1]
    dim3 = arr.shape[2]
    
    if (dim1 != HEIGHT or dim2 != WIDTH or dim3 != DEPTH):
        print("Error after resize, image dimensions are not equal to expected "+str(arr.shape))
    
    arr = np.array(arr, dtype="float") / 255.0
    imageFileName = imageFileName.replace('.jpeg','')
    
    ImageNameDataHash[str(imageFileName)] = np.array(arr)       #Storing Image-data pair
        
print("Loaded " + str(len(ImageNameDataHash)) + " images at..."+ str(datetime.now())) 
random.seed(10)
print("Reading trainLabels.csv...")
df = pd.read_csv('/kaggle/working/../input/trainLabels.csv', sep=',')
print(type(df))

row_count = df.shape[0]
col_count = df.shape[1]
print("row_count="+str(row_count)+" col count="+str(col_count))

df["PatientID"] = ''
header_list = list(df.columns)
print(header_list)

ImageLevelHash = {}     # Storing Image-level pair
patientIDList = []
uniquePatientIDList = []

for index, row in df.iterrows():
    key = row[0] + ''
    patientID = row[0] + ''
    patientID = patientID.replace('_right', '')
    patientID = patientID.replace('_left', '')
    df.at[index, 'PatientID'] = patientID
    patientIDList.append(patientID)
    ImageLevelHash[key] = str(row[1])
    
uniquePatientIDList = sorted(set(patientIDList))
count = 0

for patientID in uniquePatientIDList:
    left_level = ImageLevelHash[str(patientID + '_left')]
    right_level = ImageLevelHash[str(patientID + '_right')]
    
    if(left_level != right_level):
        count = count + 1

print("count of images with both left and right eye level not matching="+str(count))
print("number of unique patients="+str(len(uniquePatientIDList)))
#Merging two dataframes to give columns/features as follows: ['image', 'data', 'level', 'PatientID']

imageNameArr = []
dataArr = []

keepImages =  list(ImageNameDataHash.keys())
df = df[df['image'].isin(keepImages)]
for index, row in df.iterrows():
    key = str(row[0])
    if key in ImageNameDataHash:
        imageNameArr.append(key)
        dataArr.append(np.array(ImageNameDataHash[key]))
        
df2 = pd.DataFrame({'image': imageNameArr, 'data': dataArr})
df2_header_list = list(df2.columns)
        
df = pd.merge(df2, df, left_on='image', right_on='image', how='outer')
df_header_list = list(df.columns) 
print(df_header_list) # 'image', 'data', level', 'PatientID'
print(len(df)) # 1000
print(df.sample())
#Plotting a sample data

sample = df.loc[df.index[0], 'data']
print("Sample Image")
print(type(sample)) # <class 'numpy.ndarray'>
print(sample.shape) # 128,128,3
from matplotlib import pyplot as plt
plt.imshow(sample, interpolation='nearest')
plt.show()
#Partitioning of data in 60:20:20 ratio for training, evaluation, testing respectively

X = df['data']
Y = to_categorical(np.array(df['level']), num_classes=NUM_CLASSES)
print("Partition of image into 60:20:20")
sys.stdout.flush()
unique_ids = df.PatientID.unique()
print('unique_ids shape='+ str(len(unique_ids))) #500

train_ids, not_train_ids = train_test_split(unique_ids, test_size = 0.40, random_state = 10)
valid_ids, test_ids = train_test_split(not_train_ids, test_size = 0.50, random_state = 10)

trainid_list = train_ids.tolist()
validid_list = valid_ids.tolist()
testid_list = test_ids.tolist()

traindf = df[df.PatientID.isin(trainid_list)]
valSet = df[df.PatientID.isin(validid_list)]
testSet = df[df.PatientID.isin(testid_list)]

traindf = traindf.reset_index(drop=True)
valSet = valSet.reset_index(drop=True)
testSet = testSet.reset_index(drop=True)
print(traindf.head())
print(valSet.head())
print(testSet.head())
#Saving data and level of each image in separate lists, each for training, evaluating and testing

trainX = traindf['data']
trainY = traindf['level']

valX = valSet['data']
valY = valSet['level']

testX = testSet['data']
testY = testSet['level']

print('trainX shape=', trainX.shape[0], 'valX shape=', valX.shape[0], 'testX shape=', testX.shape[0]) 
trainY =  to_categorical(trainY, num_classes=NUM_CLASSES)
valY =  to_categorical(valY, num_classes=NUM_CLASSES)
testY =  to_categorical(testY, num_classes=NUM_CLASSES)
#Reshaping Image Data

from numpy import zeros

Xtrain = np.zeros([trainX.shape[0],HEIGHT, WIDTH, DEPTH])
for i in range(trainX.shape[0]): # 0 to (traindf Size - 1)
    Xtrain[i] = trainX[i]
Xval = np.zeros([valX.shape[0],HEIGHT, WIDTH, DEPTH])
for i in range(valX.shape[0]): # 0 to (traindf Size - 1)
    Xval[i] = valX[i]
Xtest = np.zeros([testX.shape[0],HEIGHT, WIDTH, DEPTH])
for i in range(testX.shape[0]): # 0 to (traindf Size - 1)
    Xtest[i] = testX[i]

print(Xtrain.shape) # (600,128,128,3)
print(Xval.shape) # (200,128,128,3)
print(Xtest.shape) # (200,128,128,3)
model = Sequential()
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
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
print("compiling model..")
sys.stdout.flush()
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
#summary
from keras.utils import print_summary
print_summary(model, line_length=None, positions=None, print_fn=None)
#construct the image generator for data augmentation
print("Generating images...")
sys.stdout.flush()
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

print("training network...")
sys.stdout.flush()

H = model.fit_generator(aug.flow(Xtrain, trainY, batch_size=BS), validation_data=(Xval, valY), steps_per_epoch=len(trainX) // BS, epochs=EPOCHS, verbose=1)
#Prediction

predict = model.predict(Xtest, batch_size=BS, verbose = 1, steps = None)
print(predict)
evaluate = model.evaluate(Xtest, testY, verbose = 1, steps = None)
print(evaluate)




