import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers import Conv2D,MaxPool2D,Dense,Dropout,Softmax,Input,Flatten

from keras.optimizers import Adam,RMSprop,SGD

from keras.layers.merge import add

from keras.layers import Dense, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from keras.layers import BatchNormalization

from math import ceil









# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



#attempt to fix reproducibility issue (hard to get reproducible results even after setting seed)

from tensorflow import set_random_seed

os.environ['PYTHONHASHSEED'] = "0"

np.random.seed(1)

set_random_seed(2)



print(os.listdir("../input/chest_xray/chest_xray"))



# Any results you write to the current directory are saved as output.
auggen = ImageDataGenerator(

#         rotation_range=40,

#         width_shift_range=0.1,

#         height_shift_range=0.1,

# #         shear_range=0.2,

# #         zoom_range=0.2,

#         horizontal_flip=True,

#         vertical_flip=True,

        rescale=1./255

        )

auggen = auggen.flow_from_directory(directory="../input/chest_xray/chest_xray/train",

                                    target_size=(256, 256), color_mode='rgb',  class_mode='binary', 

         batch_size=32, shuffle=True, seed=1)
auggen.class_indices
i=0

k=0

fig,axis1 = plt.subplots(1,10,figsize=(60,60))



#augggen.flow_from_directory returns a DirectoryIterator yielding tuples of (x, y) for each iteration where x is a numpy array containing a batch of images with shape (batch_size, *target_size, channels) and y is a numpy array of corresponding labels.

#here we unpack the tuple into the first element, images, which is of dimension (batch_size,length, width, number of color channels)

#think of the number of iterations that the DirectoryIterator object undergoes as the steps_per_epoch parameter in the fit method



#if we set class_mode= categorical then the labels will be hot encoded, if we set class_mode=binary then it will be a 1d array

#outer loop is used to iterate ove auggen (generator object)

for images,labels in  auggen:

    print(images.shape)

    print(labels[0])

    

    #iterating over images in the first batch in order to plot

    for image in images:

        print(image.shape)



        axis1[k].imshow(image)

        axis1[k].set_title(labels[k],fontdict={'fontsize':50})

        k=k+1

        #I only want to plot the first 10 images but we have all 32 available in images variable after 1st iteration

        if k==10:

            break

    

    i=i+1

    if i==1:

        break

    
print('total number of positive instances (pneumonia) in first batch of 32: {}'.format(sum(labels)))
traingen = ImageDataGenerator(

        rotation_range=40,

        width_shift_range=0.2,

        height_shift_range=0.2,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True,

        rescale=1./255

        )

testgen = ImageDataGenerator(

        rescale=1./255

        )



valgen = ImageDataGenerator(

        rescale=1./255

        )



traingen = traingen.flow_from_directory(directory="../input/chest_xray/chest_xray/train", 

    target_size=(256, 256), color_mode='grayscale',  class_mode='binary', 

         batch_size=32, shuffle=True, seed=1)

        

                                   



                                   

                                                                      

                                   
testgen = testgen.flow_from_directory(directory="../input/chest_xray/chest_xray/test", 

                                      target_size=(256, 256), color_mode='grayscale',  class_mode='binary', 

         batch_size=25, shuffle=False)

                                   
valgen = valgen.flow_from_directory(directory="../input/chest_xray/chest_xray/val", 

                                      target_size=(256, 256), color_mode='grayscale',  class_mode='binary', 

         batch_size=16, shuffle=False)
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu", padding="same",

                 input_shape=(256,256,1)))

model.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu", padding="same"))

# model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

# model.add(Dropout(rate=0.25))

model.add(Conv2D(filters=64, kernel_size=(3,3), activation="relu", padding="same"))

model.add(Conv2D(filters=64, kernel_size=(3,3), activation="relu", padding="same"))

# model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

# model.add(Dropout(rate=0.25))

model.add(Flatten())

model.add(Dense(1024,activation="relu"))

# model.add(BatchNormalization())

# model.add(Dropout(rate=0.4))

model.add(Dense(1, activation="sigmoid"))



model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['binary_accuracy'])
model.summary()
history=model.fit_generator(

        traingen,

    

        steps_per_epoch=5216/32,

        validation_data=valgen,

        validation_steps=4,

        epochs=3,

        class_weight = {0:2.94,

                        1:1}

)
len(testgen)
type(testgen)
model.evaluate_generator(testgen, steps = len(testgen))

testgen = ImageDataGenerator(

        rescale=1./255

        )

testgen = testgen.flow_from_directory(directory="../input/chest_xray/chest_xray/test", 

                                      target_size=(256, 256), color_mode='grayscale',  class_mode='binary', 

         batch_size=25, shuffle=False)
y_pred=model.predict_generator(testgen, steps = len(testgen), verbose=1)
y_pred[0:10]
def map_probs(y_predict, T):   

    k=0

    for i in y_predict:

        if y_predict[k]>=T:

            y_predict[k]=1

        else:

            y_predict[k]=0

        k=k+1

    return y_predict

y_pred_binary=map_probs(y_pred,.5)    

y_pred_binary[0:10]
y_true=testgen.classes
y_true
from sklearn.metrics import confusion_matrix



confusion_m=confusion_matrix(y_true, y_pred_binary)
confusion_m
tn=confusion_m[0][0]

fp=confusion_m[0][1]

fn=confusion_m[1][0]

tp=confusion_m[1][1]
accuracy=(tp+tn)/(tp+fp+fn+tn)

print("the accuracy of the model is {} %".format(accuracy*100))
recall = (tp)/(tp+fn)

print("the recall of the model is {} %".format(recall*100))
#

labels = ['normal', 'pneumonia']

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(confusion_m)

plt.title('Confusion matrix of the classifier', pad =20)

fig.colorbar(cax)

ax.set_xticklabels([''] + labels)

ax.set_yticklabels([''] + labels)

plt.xlabel('Predicted')

plt.ylabel('True')

plt.show()



traingen2 = ImageDataGenerator(

        rotation_range=40,

        width_shift_range=0.2,

        height_shift_range=0.2,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True,

        rescale=1./255

        )

testgen2 = ImageDataGenerator(

        rescale=1./255

        )



valgen2 = ImageDataGenerator(

        rescale=1./255

        )



traingen2 = traingen2.flow_from_directory(directory="../input/chest_xray/chest_xray/train", 

    target_size=(256, 256), color_mode='grayscale',  class_mode='binary', 

         batch_size=32, shuffle=True, seed=1)

        

                                   



                                   

                                                                      

                                   
testgen2 = testgen2.flow_from_directory(directory="../input/chest_xray/chest_xray/test", 

                                      target_size=(256, 256), color_mode='grayscale',  class_mode='binary', 

         batch_size=25, shuffle=False)

                                   
valgen2 = valgen2.flow_from_directory(directory="../input/chest_xray/chest_xray/val", 

                                      target_size=(256, 256), color_mode='grayscale',  class_mode='binary', 

         batch_size=16, shuffle=False)
from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense

from keras import backend as K

from keras.layers.convolutional import SeparableConv2D

from keras.layers.normalization import BatchNormalization

from math import ceil

import matplotlib.pyplot as plt







chanDim = -1



model2 = Sequential()



model2.add(SeparableConv2D(32, (3, 3), padding="same",

input_shape=(256,256,1)))

model2.add(Activation("relu"))

model2.add(BatchNormalization(axis=chanDim))

model2.add(MaxPooling2D(pool_size=(2, 2)))

model2.add(Dropout(0.25))

 

# (CONV => RELU => POOL) * 2

model2.add(SeparableConv2D(64, (3, 3), padding="same"))

model2.add(Activation("relu"))

model2.add(BatchNormalization(axis=chanDim))

model2.add(SeparableConv2D(64, (3, 3), padding="same"))

model2.add(Activation("relu"))

model2.add(BatchNormalization(axis=chanDim))

model2.add(MaxPooling2D(pool_size=(2, 2)))

model2.add(Dropout(0.25))

 

# (CONV => RELU => POOL) * 3

model2.add(SeparableConv2D(128, (3, 3), padding="same"))

model2.add(Activation("relu"))

model2.add(BatchNormalization(axis=chanDim))

model2.add(SeparableConv2D(128, (3, 3), padding="same"))

model2.add(Activation("relu"))

model2.add(BatchNormalization(axis=chanDim))

model2.add(SeparableConv2D(128, (3, 3), padding="same"))

model2.add(Activation("relu"))

model2.add(BatchNormalization(axis=chanDim))

model2.add(MaxPooling2D(pool_size=(2, 2)))

model2.add(Dropout(0.25))



model2.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model2.add(Dense(256, activation ='relu'))

model2.add(BatchNormalization())



model2.add(Dropout(0.5))

model2.add(Dense(1, activation ='sigmoid'))

# model.add(Activation('softmax'))





model2.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['binary_accuracy'])



model2.summary()

history2=model2.fit_generator(

        traingen2,

    

        steps_per_epoch=5216/32,

        validation_data=valgen,

        validation_steps=4,

        epochs=3,

        class_weight = {0:2.94,

                        1:1}

)
model2.evaluate_generator(testgen2, steps = len(testgen2))

testgen = ImageDataGenerator(

        rescale=1./255

        )



testgen = testgen.flow_from_directory(directory="../input/chest_xray/chest_xray/test", 

                                      target_size=(256, 256), color_mode='grayscale',  class_mode='binary', 

         batch_size=100, shuffle=False)

#iteration 1

images,labels = testgen.next()
labels
#iteration 2

images2,labels2 = testgen.next()
labels2
#iteration 3 

images3,labels3 = testgen.next()
labels3
images4,labels4 = testgen.next()
labels4
images5,labels5 = testgen.next()
labels5
images6,labels6 = testgen.next()
labels6
images7,labels7 = testgen.next()
#last batch, 6*100=600 images and labels generated so far, we have a total of 624 total files, therefor our last batch has 24 images and this is what we see

labels7


images8,labels8 = testgen.next()
labels8
images9,labels9 = testgen.next()
labels9
images10,labels10 = testgen.next()
labels10
testgen.classes
images11,labels11 = testgen.next()
labels11
testgen.classes
images12,labels12 = testgen.next()
labels12
images13,labels13 = testgen.next()
labels13
images14,labels14 = testgen.next()
labels14
testgen = ImageDataGenerator(

        rescale=1./255

        )



testgen = testgen.flow_from_directory(directory="../input/chest_xray/chest_xray/test", 

                                      target_size=(256, 256), color_mode='grayscale',  class_mode='binary', 

         batch_size=100, shuffle=False)

model.evaluate_generator(testgen, steps = len(testgen))

y_pred=model.predict_generator(testgen, steps = len(testgen))
#run the next iteration yielding 100 images

images,labels=testgen.next()
labels
images2,labels2=testgen.next()
labels2