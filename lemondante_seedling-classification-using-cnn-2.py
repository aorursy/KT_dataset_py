import numpy as np # MATRIX OPERATIONS

import pandas as pd # EFFICIENT DATA STRUCTURES

import matplotlib.pyplot as plt # GRAPHING AND VISUALIZATIONS

import math # MATHEMATICAL OPERATIONS

import cv2 # IMAGE PROCESSING - OPENCV

from glob import glob # FILE OPERATIONS

import itertools



# KERAS AND SKLEARN MODULES

from keras.utils import np_utils

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import Flatten

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.layers import BatchNormalization

from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,CSVLogger

from keras import initializers

from keras import optimizers





from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix





# GLOBAL VARIABLES

scale = 70

seed = 7

subscale = 299
path_to_images = '../input/plant-seedlings-classification/train/*/*.png'

images = glob(path_to_images)

trainingset = []

traininglabels = []

subtrainingset = []

subtraininglabels = []

num = len(images)

count = 1

#READING IMAGES AND RESIZING THEM

for i in images:

    print(str(count)+'/'+str(num),end='\r')

    trainingset.append(cv2.resize(cv2.imread(i),(scale,scale)))

    tempLabel = i.split('/')[-2]

    traininglabels.append(tempLabel)

    if tempLabel == "Black-grass" or tempLabel == "Loose Silky-bent":

        subtrainingset.append(cv2.resize(cv2.imread(i),(subscale,subscale)))

        subtraininglabels.append(tempLabel)

    count=count+1

trainingset = np.asarray(trainingset)

traininglabels = pd.DataFrame(traininglabels)
subtrainingset = np.asarray(subtrainingset)

subtraininglabels = pd.DataFrame(subtraininglabels)
new_train = []

sets = []; getEx = True

for i in trainingset:

    blurr = cv2.GaussianBlur(i,(5,5),0)

    hsv = cv2.cvtColor(blurr,cv2.COLOR_BGR2HSV)

    #GREEN PARAMETERS

    lower = (25,40,50)

    upper = (75,255,255)

    mask = cv2.inRange(hsv,lower,upper)

    struc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))

    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,struc)

    boolean = mask>0

    new = np.zeros_like(i,np.uint8)

    new[boolean] = i[boolean]

    new_train.append(new)

    

    if getEx:

        plt.subplot(2,3,1);plt.imshow(i) # ORIGINAL

        plt.subplot(2,3,2);plt.imshow(blurr) # BLURRED

        plt.subplot(2,3,3);plt.imshow(hsv) # HSV CONVERTED

        plt.subplot(2,3,4);plt.imshow(mask) # MASKED

        plt.subplot(2,3,5);plt.imshow(boolean) # BOOLEAN MASKED

        plt.subplot(2,3,6);plt.imshow(new) # NEW PROCESSED IMAGE

        plt.show()

        getEx = False

new_train = np.asarray(new_train)



# CLEANED IMAGES

for i in range(8):

    plt.subplot(2,4,i+1)

    plt.imshow(new_train[i])
labels = preprocessing.LabelEncoder()

labels.fit(traininglabels[0])

print('Classes'+str(labels.classes_))

encodedlabels = labels.transform(traininglabels[0])

clearalllabels = np_utils.to_categorical(encodedlabels)

classes = clearalllabels.shape[1]

print(str(classes))

traininglabels[0].value_counts().plot(kind='pie')
new_train = new_train/255

x_train,x_test,y_train,y_test = train_test_split(new_train,clearalllabels,test_size=0.1,random_state=seed,stratify=clearalllabels)
generator = ImageDataGenerator(rotation_range = 180,zoom_range = 0.1,width_shift_range = 0.1,height_shift_range = 0.1,horizontal_flip = True,vertical_flip = True)

generator.fit(x_train)
np.random.seed(seed)



model = Sequential()



model.add(Conv2D(filters=64, kernel_size=(5, 5), input_shape=(scale, scale, 3), activation='relu'))

model.add(BatchNormalization(axis=3))

model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))

model.add(MaxPooling2D((2, 2)))

model.add(BatchNormalization(axis=3))

model.add(Dropout(0.1))



model.add(Conv2D(filters=128, kernel_size=(5, 5), activation='relu'))

model.add(BatchNormalization(axis=3))

model.add(Conv2D(filters=128, kernel_size=(5, 5), activation='relu'))

model.add(MaxPooling2D((2, 2)))

model.add(BatchNormalization(axis=3))

model.add(Dropout(0.1))



model.add(Conv2D(filters=256, kernel_size=(5, 5), activation='relu'))

model.add(BatchNormalization(axis=3))

model.add(Conv2D(filters=256, kernel_size=(5, 5), activation='relu'))

model.add(MaxPooling2D((2, 2)))

model.add(BatchNormalization(axis=3))

model.add(Dropout(0.1))



model.add(Flatten())



model.add(Dense(256, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))



model.add(Dense(256, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))



model.add(Dense(classes, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



model.summary()



# SETTING UP CHECKPOINTS, CALLBACKS AND REDUCING LEARNING RATE

lrr = ReduceLROnPlateau(monitor='val_acc', 

                        patience=3, 

                        verbose=1, 

                        factor=0.4, 

                        min_lr=0.00001)



filepath="drive/DataScience/PlantReco/weights.best_{epoch:02d}-{val_acc:.2f}.hdf5"

checkpoints = ModelCheckpoint(filepath, monitor='val_acc', 

                              verbose=1, save_best_only=True, mode='max')

filepath="drive/DataScience/PlantReco/weights.last_auto4.hdf5"

checkpoints_full = ModelCheckpoint(filepath, monitor='val_acc', 

                                 verbose=1, save_best_only=False, mode='max')



callbacks_list = [checkpoints, lrr, checkpoints_full]



#MODEL

# hist = model.fit_generator(datagen.flow(trainX, trainY, batch_size=75), 

#                            epochs=35, validation_data=(testX, testY), 

#                            steps_per_epoch=trainX.shape[0], callbacks=callbacks_list)



# LOADING MODEL

model.load_weights("../input/plantrecomodels/weights.best_17-0.96.hdf5")

dataset = np.load("../input/plantrecomodels/Data.npz")

data = dict(zip(("x_train","x_test","y_train", "y_test"), (dataset[k] for k in dataset)))

x_train = data['x_train']

x_test = data['x_test']

y_train = data['y_train']

y_test = data['y_test']



print(model.evaluate(x_train, y_train))  # Evaluate on train set

print(model.evaluate(x_test, y_test))  # Evaluate on test set
# PREDICTIONS

y_pred = model.predict(x_test)

y_class = np.argmax(y_pred, axis = 1) 

y_check = np.argmax(y_test, axis = 1) 



cmatrix = confusion_matrix(y_check, y_class)

print(cmatrix)
print(len(subtrainingset))
new_sub_train = []

sets = []; getEx = True

for i in subtrainingset:

    blurr = cv2.GaussianBlur(i,(5,5),0)

    hsv = cv2.cvtColor(blurr,cv2.COLOR_BGR2HSV)

    #GREEN PARAMETERS

    lower = (25,40,50)

    upper = (75,255,255)

    mask = cv2.inRange(hsv,lower,upper)

    struc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21))

    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,struc)

    boolean = mask>0

    new = np.zeros_like(i,np.uint8)

    new[boolean] = i[boolean]

    new_sub_train.append(new)

    

    if getEx:

        plt.subplot(2,3,1);plt.imshow(i) # ORIGINAL

        plt.subplot(2,3,2);plt.imshow(blurr) # BLURRED

        plt.subplot(2,3,3);plt.imshow(hsv) # HSV CONVERTED

        plt.subplot(2,3,4);plt.imshow(mask) # MASKED

        plt.subplot(2,3,5);plt.imshow(boolean) # BOOLEAN MASKED

        plt.subplot(2,3,6);plt.imshow(new) # NEW PROCESSED IMAGE

        plt.show()

        getEx = False

new_sub_train = np.asarray(new_sub_train)

print(len(new_sub_train))
# CLEANED IMAGES

for i in range(8):

    plt.subplot(2,4,i+1)

    plt.imshow(new_sub_train[i])
sublabels = preprocessing.LabelEncoder()

sublabels.fit(subtraininglabels[0])

print('Classes'+str(sublabels.classes_))

subencodedlabels = sublabels.transform(subtraininglabels[0])

clearalllabels = np_utils.to_categorical(subencodedlabels)

subclasses = clearalllabels.shape[1]

print(str(subclasses))

subtraininglabels[0].value_counts().plot(kind='pie')
new_sub_train = new_sub_train/255

x_train,x_test,y_train,y_test = train_test_split(new_sub_train,clearalllabels,test_size=0.2,random_state=seed,stratify=clearalllabels)
generator = ImageDataGenerator(rotation_range = 360,zoom_range = 0.3,width_shift_range = 0.1,height_shift_range = 0.5,horizontal_flip = True,vertical_flip = True)

generator.fit(x_train)
# np.random.seed(seed)



# model = Sequential()



# model.add(Conv2D(filters=128, kernel_size=(5, 5), input_shape=(scale, scale, 3), activation='relu'))

# model.add(BatchNormalization(axis=3))

# model.add(Conv2D(filters=128, kernel_size=(5, 5), activation='relu'))

# model.add(BatchNormalization(axis=3))

# model.add(Conv2D(filters=256, kernel_size=(5, 5), activation='relu'))

# model.add(MaxPooling2D((2, 2)))

# model.add(BatchNormalization(axis=3))

# model.add(Dropout(0.1))



# model.add(Conv2D(filters=256, kernel_size=(5, 5), activation='relu'))

# model.add(BatchNormalization(axis=3))

# model.add(Conv2D(filters=256, kernel_size=(5, 5), activation='relu'))

# model.add(BatchNormalization(axis=3))

# model.add(Conv2D(filters=512, kernel_size=(5, 5), activation='relu'))

# model.add(MaxPooling2D((2, 2)))

# model.add(BatchNormalization(axis=3))

# model.add(Dropout(0.1))



# # model.add(Conv2D(filters=512, kernel_size=(5, 5), activation='relu'))

# # model.add(BatchNormalization(axis=3))

# # model.add(Conv2D(filters=512, kernel_size=(5, 5), activation='relu'))

# # model.add(BatchNormalization(axis=3))

# # model.add(Dropout(0.1))



# model.add(Flatten())



# model.add(Dense(256, activation='relu', kernel_initializer='he_normal'))

# model.add(BatchNormalization())

# model.add(Dropout(0.5))



# model.add(Dense(256, activation='relu'))

# model.add(BatchNormalization())

# model.add(Dropout(0.5))



# model.add(Dense(2, activation='sigmoid'))



# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



# model.summary()



from keras.applications import xception

from keras.preprocessing import image

from keras.models import Model

from keras.layers import GlobalAveragePooling2D

from keras import backend as K



# 构建不带分类器的预训练模型

base_model = xception.Xception(weights='imagenet', include_top=False)



# 添加全局平均池化层

x = base_model.output

x = GlobalAveragePooling2D()(x)



x = Dense(1024, activation='relu')(x)

x = BatchNormalization()(x)

x = Dropout(0.5)(x)



predictions = Dense(2, activation='sigmoid')(x)



# 构建我们需要训练的完整模型

submodel = Model(inputs=base_model.input, outputs=predictions)



# 首先，我们只训练顶部的几层（随机初始化的层）

# 锁住所有 InceptionV3 的卷积层

# for layer in base_model.layers[:126]:

#     layer.trainable = False

# for layer in model.layers[126:]:

#     layer.trainable = True

for layer in base_model.layers:

    layer.trainable = False



# 编译模型（一定要在锁层以后操作）

submodel.compile(optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), loss='binary_crossentropy', metrics = ['accuracy'])



submodel.summary()
# SETTING UP CHECKPOINTS, CALLBACKS AND REDUCING LEARNING RATE

lrr = ReduceLROnPlateau(monitor='val_acc', 

                        patience=4, 

                        verbose=1, 

                        factor=0.3, 

                        min_lr=0.000001)



filepath="subweights.best.hdf5"

checkpoints = ModelCheckpoint(filepath, monitor='val_acc', 

                              verbose=1, save_best_only=True, mode='max')

filepath="subweights.last_auto4.hdf5"

checkpoints_full = ModelCheckpoint(filepath, monitor='val_acc', 

                                 verbose=1, save_best_only=False, mode='max')



callbacks_list = [checkpoints, lrr, checkpoints_full]



# submodel.load_weights("../input/subweights2/subweights.best.hdf5")



hist = submodel.fit_generator(generator.flow(x_train, y_train, batch_size=25), 

                           epochs=45, validation_data=(x_test, y_test), 

                           steps_per_epoch=x_train.shape[0], callbacks=callbacks_list,shuffle =True)
# submodel.load_weights("../input/subweights2/subweights.best.hdf5")

# ./input/Seeding-Classification-using-CNN-2
print(submodel.evaluate(x_train, y_train))  # Evaluate on train set

print(submodel.evaluate(x_test, y_test))  # Evaluate on test set
path_to_test = '../input/plant-seedlings-classification/test/*.png'

pics = glob(path_to_test)



testimages = []

tests = []

count=1

num = len(pics)



for i in pics:

    print(str(count)+'/'+str(num),end='\r')

    tests.append(i.split('/')[-1])

    testimages.append(cv2.resize(cv2.imread(i),(scale,scale)))

    count = count + 1



testimages = np.asarray(testimages)
newtestimages = []

sets = []

getEx = True

for i in testimages:

    blurr = cv2.GaussianBlur(i,(5,5),0)

    hsv = cv2.cvtColor(blurr,cv2.COLOR_BGR2HSV)

    

    lower = (25,40,50)

    upper = (75,255,255)

    mask = cv2.inRange(hsv,lower,upper)

    struc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))

    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,struc)

    boolean = mask>0

    masking = np.zeros_like(i,np.uint8)

    masking[boolean] = i[boolean]

    newtestimages.append(masking)

    

    if getEx:

        plt.subplot(2,3,1);plt.imshow(i)

        plt.subplot(2,3,2);plt.imshow(blurr)

        plt.subplot(2,3,3);plt.imshow(hsv)

        plt.subplot(2,3,4);plt.imshow(mask)

        plt.subplot(2,3,5);plt.imshow(boolean)

        plt.subplot(2,3,6);plt.imshow(masking)

        plt.show()

        getEx=False



newtestimages = np.asarray(newtestimages)

# OTHER MASKED IMAGES

for i in range(6):

    plt.subplot(2,3,i+1)

    plt.imshow(newtestimages[i])
newtestimages=newtestimages/255

prediction = model.predict(newtestimages)

# PREDICTION TO A CSV FILE

pred = np.argmax(prediction,axis=1)

predStr = labels.classes_[pred]

count_num = 0

change_num = 0



for i in range(0,794):

    if predStr[i] == "Black-grass" or predStr[i] == "Loose Silky-bent":

        subtestimage = cv2.resize(cv2.imread(pics[i]),(subscale,subscale))

        

        blurr = cv2.GaussianBlur(subtestimage,(5,5),0)

        hsv = cv2.cvtColor(blurr,cv2.COLOR_BGR2HSV)



        lower = (25,40,50)

        upper = (75,255,255)

        mask = cv2.inRange(hsv,lower,upper)

        struc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))

        mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,struc)

        boolean = mask>0

        masking = np.zeros_like(subtestimage,np.uint8)

        masking[boolean] = subtestimage[boolean]

        subtestimage = masking/255

        subtestimage = np.expand_dims(subtestimage, axis=0)

        

        subprediction = submodel.predict(subtestimage)

        subpred = np.argmax(subprediction, axis = 1)

        

        templabel = sublabels.classes_[subpred][0]

        if predStr[i] != templabel:

            change_num+=1

        predStr[i] = templabel

        count_num+=1



print(count_num)

print(change_num)

result = {'file':tests,'species':predStr}

result = pd.DataFrame(result)

result.to_csv("Prediction.csv",index=False)