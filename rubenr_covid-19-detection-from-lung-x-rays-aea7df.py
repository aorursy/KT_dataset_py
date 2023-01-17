from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D
from keras.layers import Activation, Dropout, BatchNormalization, Flatten, Dense, AvgPool2D,MaxPool2D
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.optimizers import Adam, SGD, RMSprop

import tensorflow as tf

import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
%matplotlib inline
DATASET_DIR = "../input/covid-19-x-ray-10000-images/dataset"



os.listdir(DATASET_DIR)

from numpy import inf
import numpy as npy
from timeit import default_timer as timer
import numpy as np

import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline

def InsideCircle(xc, yc, R): # returns list of [x,y] points inside a circle with center (xc,yc) and radius R
    L = []
    for i in range(2*R+1):  
        for j in range(2*R+1): 
            xx = xc-R+j
            yy = yc-R+i
            ll = (R-j)*(R-j)+(R-i)*(R-i)
            if (ll <= R*R):
                q = [xx,yy]
                L.append(q) 

    return L

def InsideCircleX(xc, yc, R): # returns list of [x] points inside a circle with center (xc,yc) and radius R
    Lx = []
    for i in range(2*R+1):  
        for j in range(2*R+1): 
            xx = xc-R+j
            ll = (R-j)*(R-j)+(R-i)*(R-i)
            if (ll <= R*R):
                Lx.append(xx)

    return Lx

def InsideCircleY(xc, yc, R): # returns list of [y] points inside a circle with center (xc,yc) and radius R
    Ly = []
    for i in range(2*R+1):  
        for j in range(2*R+1): 
            yy = yc-R+i
            ll = (R-j)*(R-j)+(R-i)*(R-i)
            if (ll <= R*R):
                Ly.append(yy)
    return Ly


def DistanceMatrix_cpu(boundary_x, boundary_y, internal_points_x, internal_points_y):
    dist = []
    dist_x = (boundary_x[:,npy.newaxis] - internal_points_x[:])**2

    dist_y = (boundary_y[:,npy.newaxis] - internal_points_y[:])**2
    #print(boundary_x)
    #print(internal_points_x)
    #print(internal_points_x)
    #print("Dist_X")
    #print(dist_x.shape)
    #print("Dist_Y")
    #print(dist_y.shape)
    #print(dist_x)
   
    return np.sqrt(dist_x+dist_y)

def normalize(array):
    return (array - array.mean()) / array.std()

#initialize object that keeps track of our boundary values
dataset = {
    'n' :[],
    'c' : []
}
normal_images = []

for img_path in glob.glob(DATASET_DIR + '/normal/*'):
    img = mpimg.imread(img_path)
    if len(img.shape) > 2:
        img = rgb2gray(mpimg.imread(img_path))
    normal_images.append(img)


'''
1. Iterate through every normal image 
2. Find center of image
   2a. then select a circle near the center 
   2b. Compute boundary function for circle
   2c. Repeat N times (experiment to generate more boundary values)
'''
dataset['n'] = []

for img in normal_images:
#     img = normal_images[0]
#     print("XRay grayscale intensity for normal_case[0]")
#     print(img.shape)

    # Apply a circular probe (circle1) to the Normal Image
    # with (Center_Normal_X, Center_Normal_Y) and radius Rad_Normal
    # A concentric circumference (circle1_1) with Rad_Normal + Delta_Normal 
    # is applied as a measurement boundary (Delta_Normal: a gap between probe and boundary)

    #Center_Normal_X, Center_Normal_Y = 1300, 800 # 100, 100
#     Center_Normal_X, Center_Normal_Y = int(img.shape[0] *.75) , 600 # 100, 100????????
    Rad_Normal = 40 #13 #4 #10 #100
    Delta_Normal = 5 #40
    for _ in range(10):
        
        
        Center_Normal_X, Center_Normal_Y = int(img.shape[0])//2 , int(img.shape[0] *.75) //2 # 100, 100????????

        ## Randomly select an x/y point near the center of the circle
        Center_Normal_X += random.randint(50, int((img.shape[0]-Center_Normal_X) *.6))
        Center_Normal_Y += random.randint(50, int((img.shape[1]-Center_Normal_Y) *.6))
        
        
        # Find points on the boundary
        Boundary_Points = 1000 #100 #10 #1000
        R = Rad_Normal + Delta_Normal
        Center_x = Center_Normal_X
        Center_y = Center_Normal_Y
        Circ_Bound = np.linspace(0, 2*np.pi, Boundary_Points); 
        #print(Circ_Bound)
        Circ_Bound_x = R * np.cos(Circ_Bound) + Center_x
        Circ_Bound_y = R * np.sin(Circ_Bound) + Center_y    
        #print(Circ_Bound_x)
        #print(Circ_Bound_y)

        #Find all points within circular probe
        #In_Circle = InsideCircle(Center_x, Center_y, Rad_Normal) # Returns (x,y) vector
        In_CircleX = InsideCircleX(Center_x, Center_y, Rad_Normal) # Returns X-components
        In_CircleY = InsideCircleY(Center_x, Center_y, Rad_Normal) # Returns Y components
        #print("In_CircleX")
        #print(In_CircleX)
        #print("In_CircleY")
        #print(In_CircleY)
        #print("Circ_Bound_x (in this example 10 points on the boundary)")
        #print(Circ_Bound_x)

        # Distance matrix between all points in the circular probe and the external circular boundary
        DM_Distance = DistanceMatrix_cpu((Circ_Bound_x), (Circ_Bound_y),(In_CircleX), (In_CircleY));
        #print("Distance Matrix (in this example 10 by 49)")
        #print(DM_Distance)
        Kernel1 = 1./DM_Distance
        #print("Kernel1 (Inverse Radial Function)")
        #print(Kernel1)

        #Create ImageSample
        ImageSample = []
        PointsInside = len(In_CircleX)
        #print("PointsInside")
        #print(PointsInside)
        for i in range(PointsInside):
            x = In_CircleX[i]
            y = In_CircleY[i]
            ImageSample.append(img[x,y])
        # print(ImageSample)

        #BoundaryFunction_unnorm = np.dot(Kernel1, ImageSample)
        #print(BoundaryFunction_unnorm)
        #plt.suptitle("Boundary Function - before normalization")
        #plt.plot(BoundaryFunction_unnorm)
        #plt.ylabel('Boundary Value')
        #plt.show()


        BoundaryFunction = normalize(np.dot(Kernel1, ImageSample))
        #print(BoundaryFunction)

        dataset['n'].append(BoundaryFunction)


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])


'''
Covid images return an rgb shape sometimes so I have to convert those to grayscale
'''
covid_images = []
for img_path in glob.glob(DATASET_DIR + '/covid/*'):
    img = mpimg.imread(img_path)
    if len(img.shape) > 2:
        img = rgb2gray(mpimg.imread(img_path))
    covid_images.append(img)
    
import random

'''
1. Iterate through every covid image 
2. Find center of image
   2a. then select a circle near the center 
   2b. Compute boundary function for circle
   2c. Repeat N times (experiment to generate more boundary values)
'''
for img in covid_images:
#     img = normal_images[0]
    print("XRay grayscale intensity for normal_case[0]")
    print(img.shape)

    # Apply a circular probe (circle1) to the Normal Image
    # with (Center_Normal_X, Center_Normal_Y) and radius Rad_Normal
    # A concentric circumference (circle1_1) with Rad_Normal + Delta_Normal 
    # is applied as a measurement boundary (Delta_Normal: a gap between probe and boundary)

    #Center_Normal_X, Center_Normal_Y = 1300, 800 # 100, 100
    Center_Normal_X, Center_Normal_Y = int(img.shape[0])//2 , int(img.shape[0] *.75) //2 # 100, 100????????
    Rad_Normal = 40 #13 #4 #10 #100
    Delta_Normal = 5 #40
    
    
    
    for _ in range(10):
        Center_Normal_X, Center_Normal_Y = int(img.shape[0])//2 , int(img.shape[0] *.75) //2 # 100, 100????????

        ## Randomly select an x/y point near the center of the circle
        Center_Normal_X += random.randint(50, int((img.shape[0]-Center_Normal_X) *.6))
        Center_Normal_Y += random.randint(50, int((img.shape[1]-Center_Normal_Y) *.6))
        print('center: ({}, {})'.format(Center_Normal_X, Center_Normal_Y))
        # Find points on the boundary
        Boundary_Points = 1000 #100 #10 #1000
        R = Rad_Normal + Delta_Normal
        Center_x = Center_Normal_X
        Center_y = Center_Normal_Y
        Circ_Bound = np.linspace(0, 2*np.pi, Boundary_Points); 
        #print(Circ_Bound)
        Circ_Bound_x = R * np.cos(Circ_Bound) + Center_x
        Circ_Bound_y = R * np.sin(Circ_Bound) + Center_y    
        #print(Circ_Bound_x)
        #print(Circ_Bound_y)

        #Find all points within circular probe
        #In_Circle = InsideCircle(Center_x, Center_y, Rad_Normal) # Returns (x,y) vector
        In_CircleX = InsideCircleX(Center_x, Center_y, Rad_Normal) # Returns X-components
        In_CircleY = InsideCircleY(Center_x, Center_y, Rad_Normal) # Returns Y components
        #print("In_CircleX")
        #print(In_CircleX)
        #print("In_CircleY")
        #print(In_CircleY)
        #print("Circ_Bound_x (in this example 10 points on the boundary)")
        #print(Circ_Bound_x)

        # Distance matrix between all points in the circular probe and the external circular boundary
        DM_Distance = DistanceMatrix_cpu((Circ_Bound_x), (Circ_Bound_y),(In_CircleX), (In_CircleY));
        #print("Distance Matrix (in this example 10 by 49)")
        #print(DM_Distance)
        Kernel1 = 1./DM_Distance
        #print("Kernel1 (Inverse Radial Function)")
        #print(Kernel1)

        #Create ImageSample
        ImageSample = []
        PointsInside = len(In_CircleX)
        #print("PointsInside")
        #print(PointsInside)
        for i in range(PointsInside):
            x = In_CircleX[i]
            y = In_CircleY[i]
            ImageSample.append(img[x,y])
    #     print(ImageSample)

        #BoundaryFunction_unnorm = np.dot(Kernel1, ImageSample)
        #print(BoundaryFunction_unnorm)
        #plt.suptitle("Boundary Function - before normalization")
        #plt.plot(BoundaryFunction_unnorm)
        #plt.ylabel('Boundary Value')
        #plt.show()


        BoundaryFunction = normalize(np.dot(Kernel1, ImageSample))


        dataset['c'].append(BoundaryFunction)


'''
At this stage, I like to keep the original dataset intact and experiment 
with different transformations by copying the values over to a new object
'''
data_modified = {
    'n' : [],
    'c' : [],
}


import copy
for shape in dataset:
    
    ##Normalize each class by L2 norm
    t = copy.deepcopy(dataset[shape])
    t/=np.linalg.norm(t)
#     t -=np.mean(t)
#     t/=np.std(t)
    for item in t:
        data_modified[shape].append(item)
        


for _ in range(5):
    idx = random.randint(0, 139)
    bf = data_modified['n'][idx]
    bfc = data_modified['c'][idx]

    plt.plot(bf, label='normal')
    plt.plot(bfc, label='covid')
    plt.legend()
    plt.show()
train_x = []
test_x = []
train_y=[]
test_y=[]



i = 0
max_len = []
for s in data_modified:
    max_len.append(len(data_modified[s]))
    
max_len = np.min(max_len)

for shape in data_modified:
    #select only the first 140 elements to make both labels have same number of objects
    data = data_modified[shape][:max_len]

    if len(data) == 0:
        continue
        
    print('shape', shape, len(data))

    data = np.asarray(data)
    print(data.shape)
#     data = data.reshape(data.shape[0], data.shape[1], 1)
#     data /= np.linalg.norm(data)
    # data = data[:,0]
    #data = np.abs(np.apply_along_axis(np.fft.fft, 1, data))
    random_range = np.arange(data.shape[0])
    np.random.shuffle(random_range)
    train_range = int(random_range.shape[0] *.99)
    
    if i== 0:
        train_y = [i] * data[random_range[:train_range]].shape[0]
        test_y = [i] * data[random_range[train_range:]].shape[0]
        train_x = data[random_range[:train_range]]
        test_x = data[random_range[train_range:]]
    else:
        train_y = np.concatenate((train_y, [i] * data[random_range[:train_range]].shape[0]), axis=0)
        test_y = np.concatenate((test_y, ([i] * data[random_range[train_range:]].shape[0])), axis=0)

        train_x = np.concatenate((train_x, data[random_range[:train_range]]), axis=0)

        test_x = np.concatenate((test_x, data[random_range[train_range:]]), axis=0)
    i+=1
train_x.shape
# from keras.models import Sequential
# from keras.layers.normalization import BatchNormalization
# from keras.layers.convolutional import Conv1D, MaxPooling1D
# from keras.layers.core import Activation
# from keras.layers.core import Flatten
# from keras.layers.core import Dropout
# from keras.layers import Input, Dense, Concatenate
# from keras.layers import LSTM

# from keras.layers.core import Dense
# from keras import backend as K
# from keras.optimizers import Adam
# from keras.models import Model
# from sklearn.model_selection import StratifiedKFold
# from keras.utils import to_categorical
# import os
# from keras.callbacks import ModelCheckpoint

# input_shape = Input(shape=(train_x.shape[1], train_x.shape[2]))

# def getLSTMModel():
#     dropout_rate_=.2
#     input2 = LSTM(128, recurrent_dropout=dropout_rate_ )(input_shape)
#     # input2 = LSTM(128, recurrent_dropout=dropout_rate_)(input2)
#     input2 = Dense(128, activation='relu')(input2)
#     FC1 = Dropout(0.4)(input2)
#     predictions = Dense(2, activation='softmax')(FC1) 
    
#     return Model(inputs=[input_shape], outputs=[predictions])

# lstmModel = getLSTMModel()
# lstmModel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# lstmModel.summary()
train_y.shape
# train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], 1)
# input_shape = Input(shape=(train_x.shape[1], train_x.shape[2]))

# def getCNNModel():
#     kernel_s = (2)
#     input1 = Conv1D(32, (kernel_s), activation = "relu")(input_shape)
#     input1 = BatchNormalization()(input1)

#     input1 = Conv1D(filters = 64, kernel_size = (kernel_s), activation ='relu')(input1)
#     input1 = BatchNormalization()(input1)

#     input1 = Conv1D(filters = 128, kernel_size = (kernel_s), activation ='relu')(input1)
#     input1 = BatchNormalization()(input1)

#     input1 = Flatten()(input1)
#     input1 = Dense(256, activation='relu')(input1)
#     FC1 = Dropout(0.7)(input1)
#     predictions = Dense(2, activation='softmax')(FC1) 

#     return Model(inputs=[input_shape], outputs=[predictions])
# lstmModel = getCNNModel()
# lstmModel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# combined_history = lstmModel.fit(train_x, 
#                             to_categorical(train_y), 
#                             batch_size=100, 

#                             epochs=10, verbose=1, validation_split=.2
#                             # callbacks=[csv_logger]
#                             ) 
# # print(lstmModel.evaluate(x_test, to_categorical(y_test)))

# combined_history = lstmModel.fit(train_x, 
#                             to_categorical(train_y), 
#                             batch_size=5, 

#                             epochs=10, verbose=1, validation_split=.5
#                             # callbacks=[csv_logger]
#                             ) 
# # print(lstmModel.evaluate(x_test, to_categorical(y_test)))
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=2)
neigh.fit(np.nan_to_num(train_x), train_y)
pred_i = neigh.predict(np.nan_to_num(test_x))
neigh.score(np.nan_to_num(test_x), test_y)
from sklearn.metrics import accuracy_score
from sklearn import svm

clf = svm.SVC(decision_function_shape='ovo', probability=True)
clf.fit(np.nan_to_num(train_x), train_y)

predicted = clf.predict(np.nan_to_num(test_x))

# get the accuracy
accuracy_score(test_y, predicted)
IMG_W = 150
IMG_H = 150
CHANNELS = 3

INPUT_SHAPE = (IMG_W, IMG_H, CHANNELS)
NB_CLASSES = 2
EPOCHS = 48
BATCH_SIZE = 6
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=INPUT_SHAPE))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(Conv2D(250,(3,3)))
model.add(Activation("relu"))
  
model.add(Conv2D(128,(3,3)))
model.add(Activation("relu"))
model.add(AvgPool2D(2,2))
model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(AvgPool2D(2,2))

model.add(Conv2D(256,(2,2)))
model.add(Activation("relu"))
model.add(MaxPool2D(2,2))
    
model.add(Flatten())
model.add(Dense(32))
model.add(Dropout(0.25))
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


model.summary()
train_datagen = ImageDataGenerator(rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.3)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_H, IMG_W),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training')

validation_generator = train_datagen.flow_from_directory(
    DATASET_DIR, 
    target_size=(IMG_H, IMG_W),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle= False,
    subset='validation')

history = model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // BATCH_SIZE,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // BATCH_SIZE,
    epochs = EPOCHS)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
print("training_accuracy", history.history['accuracy'][-1])
print("validation_accuracy", history.history['val_accuracy'][-1])
label = validation_generator.classes
pred= model.predict(validation_generator)
predicted_class_indices=np.argmax(pred,axis=1)
labels = (validation_generator.class_indices)
labels2 = dict((v,k) for k,v in labels.items())
predictions = [labels2[k] for k in predicted_class_indices]
print(predicted_class_indices)
print (labels)
print (predictions)
from sklearn.metrics import confusion_matrix

cf = confusion_matrix(predicted_class_indices,label)
cf
exp_series = pd.Series(label)
pred_series = pd.Series(predicted_class_indices)
pd.crosstab(exp_series, pred_series, rownames=['Actual'], colnames=['Predicted'],margins=True)
plt.matshow(cf)
plt.title('Confusion Matrix Plot')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show();