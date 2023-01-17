import numpy 

import matplotlib.pyplot as plt

import tensorflow as tf

import keras

import glob

import cv2

import os

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
#reading the training data

octImagesTrain = []

octImagesTrain_Labels = []

octImagesTrain_Labels_ID = []

octImagesTest = []

octImagesTest_Labels = []

octImagesTest_Labels_ID = []

octImagesVal = []

octImagesVal_Labels = []

octImagesVal_Labels_ID = []



for image_path in glob.glob("../input/kermany2018/oct2017/OCT2017 /train/DME/*.jpeg"):

    image = cv2.imread(image_path)

    image = cv2.resize(image, (45,45))

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    octImagesTrain.append(image)

    octImagesTrain_Labels.append("DME")

    octImagesTrain_Labels_ID.append(0) # label 0 -> DME

    

for image_path in glob.glob("../input/kermany2018/oct2017/OCT2017 /train/NORMAL/*.jpeg"):

    image = cv2.imread(image_path)

    image = cv2.resize(image, (45,45))

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    octImagesTrain.append(image)

    octImagesTrain_Labels.append("NORMAL")

    octImagesTrain_Labels_ID.append(1) #label 1 -> Normal

    

for image_path in glob.glob("../input/kermany2018/oct2017/OCT2017 /train/CNV/*.jpeg"):

    image = cv2.imread(image_path)

    image = cv2.resize(image, (45,45))

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    octImagesTrain.append(image)

    octImagesTrain_Labels.append("CNV")

    octImagesTrain_Labels_ID.append(2) #label 2 -> CNV 

    

for image_path in glob.glob("../input/kermany2018/oct2017/OCT2017 /train/DRUSEN/*.jpeg"):

    image = cv2.imread(image_path)

    image = cv2.resize(image, (45,45))

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    octImagesTrain.append(image)

    octImagesTrain_Labels.append("DRUSEN")

    octImagesTrain_Labels_ID.append(3) #label 3 -> Drusen



#transforming the train data to numpy arrays

octImagesTrain = numpy.array(octImagesTrain)

octImagesTrain_Labels = numpy.array(octImagesTrain_Labels)

octImagesTrain_Labels_ID = numpy.array(octImagesTrain_Labels_ID)



#reading the test data

for image_path in glob.glob("../input/kermany2018/oct2017/OCT2017 /test/DME/*.jpeg"):

    image = cv2.imread(image_path)

    image = cv2.resize(image, (45,45))

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    octImagesTest.append(image)

    octImagesTest_Labels.append("DME")

    octImagesTest_Labels_ID.append(0) # label 0 -> DME

    

for image_path in glob.glob("../input/kermany2018/oct2017/OCT2017 /test/NORMAL/*.jpeg"):

    image = cv2.imread(image_path)

    image = cv2.resize(image, (45,45))

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    octImagesTest.append(image)

    octImagesTest_Labels.append("NORMAL")

    octImagesTest_Labels_ID.append(1) #label 1 -> Normal

    

for image_path in glob.glob("../input/kermany2018/oct2017/OCT2017 /test/CNV/*.jpeg"):

    image = cv2.imread(image_path)

    image = cv2.resize(image, (45,45))

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    octImagesTest.append(image)

    octImagesTest_Labels.append("CNV")

    octImagesTest_Labels_ID.append(2) #label 2 -> CNV



for image_path in glob.glob("../input/kermany2018/oct2017/OCT2017 /test/DRUSEN/*.jpeg"):

    image = cv2.imread(image_path)

    image = cv2.resize(image, (45,45))

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    octImagesTest.append(image)

    octImagesTest_Labels.append("DRUSEN")

    octImagesTest_Labels_ID.append(3) #label 3 -> DRUSEN

    



#transforming the test data to numpy arrays



octImagesTest = numpy.array(octImagesTest)

octImagesTest_Labels = numpy.array(octImagesTest_Labels)

octImagesTest_Labels_ID = numpy.array(octImagesTest_Labels_ID)



#Validation Data

for image_path in glob.glob("../input/kermany2018/oct2017/OCT2017 /val/DME/*.jpeg"):

    image = cv2.imread(image_path)

    image = cv2.resize(image, (45,45))

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    octImagesVal.append(image)

    octImagesVal_Labels.append("DME")

    octImagesVal_Labels_ID.append(0) # label 0 -> DME

    

for image_path in glob.glob("../input/kermany2018/oct2017/OCT2017 /val/NORMAL/*.jpeg"):

    image = cv2.imread(image_path)

    image = cv2.resize(image, (45,45))

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    octImagesVal.append(image)

    octImagesVal_Labels.append("NORMAL")

    octImagesVal_Labels_ID.append(1) #label 1 -> Normal



for image_path in glob.glob("../input/kermany2018/oct2017/OCT2017 /val/CNV/*.jpeg"):

    image = cv2.imread(image_path)

    image = cv2.resize(image, (45,45))

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    octImagesVal.append(image)

    octImagesVal_Labels.append("CNV")

    octImagesVal_Labels_ID.append(2) #label 2 -> CNV  

    

for image_path in glob.glob("../input/kermany2018/oct2017/OCT2017 /val/DRUSEN/*.jpeg"):

    image = cv2.imread(image_path)

    image = cv2.resize(image, (45,45))

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    octImagesVal.append(image)

    octImagesVal_Labels.append("DRUSEN")

    octImagesVal_Labels_ID.append(3) #label 3 -> Drusen

    

    

#transforming the train data to numpy arrays

octImagesVal = numpy.array(octImagesVal)

octImagesVal_Labels = numpy.array(octImagesVal_Labels)

octImagesVal_Labels_ID = numpy.array(octImagesVal_Labels_ID)
catagories = ["Normal" , "DME" , "CNV" , "DRUSEN"]

trainData = [octImagesTrain_Labels[octImagesTrain_Labels == "NORMAL"].shape[0] ,

             octImagesTrain_Labels[octImagesTrain_Labels == "DME"].shape[0] , 

             octImagesTrain_Labels[octImagesTrain_Labels == "CNV"].shape[0],

             octImagesTrain_Labels[octImagesTrain_Labels == "DRUSEN"].shape[0]]

y_pos = numpy.arange(len(catagories))

plt.bar(y_pos, trainData, align='center', alpha=0.5, color = "Blue")

plt.xticks(y_pos, catagories)

plt.legend(["Train"])

plt.show()



testData = [octImagesTest_Labels[octImagesTest_Labels == "NORMAL"].shape[0] ,

            octImagesTest_Labels[octImagesTest_Labels == "DME"].shape[0] , 

            octImagesTest_Labels[octImagesTest_Labels == "CNV"].shape[0],

            octImagesTest_Labels[octImagesTest_Labels == "DRUSEN"].shape[0]]

plt.bar(y_pos, testData, align='center', alpha=0.5, color = "Green")

plt.legend(["Test"])

plt.xticks(y_pos, catagories)

plt.show()



valData =  [octImagesVal_Labels[octImagesVal_Labels == "NORMAL"].shape[0] ,

            octImagesVal_Labels[octImagesVal_Labels == "DME"].shape[0] , 

            octImagesVal_Labels[octImagesVal_Labels == "CNV"].shape[0],

            octImagesVal_Labels[octImagesVal_Labels == "DRUSEN"].shape[0]]

plt.bar(y_pos, valData, align='center', alpha=0.5, color = "Red")

plt.legend(["Validation"])

plt.xticks(y_pos, catagories)

plt.show()
#normalization of the images from 0-255 to 0-1

octImagesTest = octImagesTest / 255

octImagesTrain = octImagesTrain / 255

octImagesVal = octImagesVal / 255



#one-hot encode the catagorical data.

octImagesTest_Labels_ID = keras.utils.to_categorical(octImagesTest_Labels_ID)

octImagesTrain_Labels_ID = keras.utils.to_categorical(octImagesTrain_Labels_ID)

octImagesVal_Labels_ID = keras.utils.to_categorical(octImagesVal_Labels_ID)



model = Sequential()

#first layer

model.add(Conv2D(64,(3,3) , input_shape = octImagesTest.shape[1:] , activation="relu"))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

#second layer

model.add(Conv2D(64,(3,3) , activation="relu"))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

#third layer

model.add(Conv2D(64,(3,3) , activation="relu"))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

#regular neural network layer

model.add(Flatten())

model.add(Dense(512))

model.add(Activation("relu"))

#output layer

model.add(Dense(4))

model.add(Activation("softmax"))



#compile the model.

model.compile(loss="categorical_crossentropy" , optimizer = "Adam" , metrics = ['accuracy'])



#train the model

history = model.fit(octImagesTrain,

         octImagesTrain_Labels_ID,

         batch_size = 32,

         epochs = 10, 

         validation_data = (octImagesVal, octImagesVal_Labels_ID),

         shuffle=True)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

ax1.plot(history.history['loss'], color='b', label="Training loss")

ax1.plot(history.history['val_loss'], color='r', label="validation loss")

ax1.set_xticks(numpy.arange(1, 10, 1))

ax1.set_yticks(numpy.arange(0, 1, 0.1))



ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")

ax2.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")

ax2.set_xticks(numpy.arange(1, 10, 1))



legend = plt.legend(loc='best', shadow=True)

plt.tight_layout()

plt.show()



testDataActual = [octImagesTest_Labels[octImagesTest_Labels == "DME"].shape[0] ,

             octImagesTest_Labels[octImagesTest_Labels == "NORMAL"].shape[0] , 

             octImagesTest_Labels[octImagesTest_Labels == "CNV"].shape[0],

             octImagesTest_Labels[octImagesTest_Labels == "DRUSEN"].shape[0]

]

plt.bar(y_pos, testDataActual, align='center', alpha=0.5 , color = "Green")

plt.xticks(y_pos, catagories)

plt.legend(["Actual Data"])

plt.show()



prediction_res = numpy.argmax(model.predict(octImagesTest) , axis = -1)



testDataPredicted = [

    prediction_res[prediction_res == 0].shape[0], 

    prediction_res[prediction_res == 1].shape[0], 

    prediction_res[prediction_res == 2].shape[0], 

    prediction_res[prediction_res == 3].shape[0]

]

plt.bar(y_pos, testDataPredicted, align = "center" , alpha = 0.5, color = "Blue")

plt.xticks(y_pos, catagories)

plt.legend(["Predicted Data"])

plt.show()