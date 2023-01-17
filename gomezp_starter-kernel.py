!pip install keras-metrics #used for F1 score

import keras_metrics as km

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #plotting 

import matplotlib.patches as patches

from tqdm import tqdm #progress bar

from glob import glob #finds files matching regex 

import os,imageio,time #for IO, resizing and runtime profiling

from skimage import img_as_ubyte

from skimage.transform import resize



import keras #keras related

from keras.applications.imagenet_utils import preprocess_input

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation

from keras.layers import Conv2D, MaxPool2D

from keras.applications.resnet50 import ResNet50
#Now, let's load the data and split it into training and validation set.

files = pd.read_csv("../input/data/training.csv")["Id"].values

train_labels = np.asarray(pd.read_csv("../input/data/training.csv")["Expected"].values)

train_imgs = []

for file in tqdm(files):

    img = img_as_ubyte(resize(imageio.imread("../input/data/training/"+file),(112,112))) #read image and resize all images to  112,112 for convenience

    train_imgs.append(preprocess_input(img)) #normalize images

train_imgs = np.asarray(train_imgs,dtype="uint8")
#Plot some examples, the title 1 will indicate the presence of the parasite

fig = plt.figure(figsize=(8, 5), dpi=100)

for idx,file in enumerate(files[:32]):

    img = img_as_ubyte(resize(imageio.imread("../input/data/training/"+file),(112,112)))

    label = train_labels[idx]

    ax = fig.add_subplot(4, 8, idx+1, xticks=[], yticks=[])

    plt.imshow(img)

    plt.title(str(label))
#just some network parameters, see above link regarding the layers for details

kernel_size = (3,3)

pool_size= (2,2)

first_filters = 32

second_filters = 64

third_filters = 128



#dropout is used for regularization here with a probability of 0.3 for conv layers, 0.5 for the dense layer at the end

dropout_conv = 0.3

dropout_dense = 0.5



#initialize the model

small_model = Sequential()



#now add layers to it



#conv block 1

small_model.add(Conv2D(first_filters, kernel_size, input_shape = (112, 112, 3)))

small_model.add(BatchNormalization())

small_model.add(Activation("relu"))

small_model.add(Conv2D(first_filters, kernel_size, use_bias=False))

small_model.add(BatchNormalization())

small_model.add(Activation("relu"))

small_model.add(MaxPool2D(pool_size = pool_size)) 

small_model.add(Dropout(dropout_conv))



#conv block 2

small_model.add(Conv2D(second_filters, kernel_size, use_bias=False))

small_model.add(BatchNormalization())

small_model.add(Activation("relu"))

small_model.add(Conv2D(second_filters, kernel_size, use_bias=False))

small_model.add(BatchNormalization())

small_model.add(Activation("relu"))

small_model.add(MaxPool2D(pool_size = pool_size))

small_model.add(Dropout(dropout_conv))



#conv block 3

small_model.add(Conv2D(third_filters, kernel_size, use_bias=False))

small_model.add(BatchNormalization())

small_model.add(Activation("relu"))

small_model.add(Conv2D(third_filters, kernel_size, use_bias=False))

small_model.add(BatchNormalization())

small_model.add(Activation("relu"))

small_model.add(MaxPool2D(pool_size = pool_size))

small_model.add(Dropout(dropout_conv))



#a fully connected (also called dense) layer at the end

small_model.add(Flatten())

small_model.add(Dense(256, use_bias=False))

small_model.add(BatchNormalization())

small_model.add(Activation("relu"))

small_model.add(Dropout(dropout_dense))



#finally convert to values of 0 to 1 using the sigmoid activation function

small_model.add(Dense(1, activation = "sigmoid"))
#Now, we will compile the model and train it for 5 epochs

small_model.compile(loss=keras.losses.binary_crossentropy,

              optimizer=keras.optimizers.Adam(0.00075), 

              metrics=['accuracy',km.binary_f1_score()])

small_model.fit(x=train_imgs,y=train_labels,batch_size=50,epochs=5,validation_split=0.05)
inputs = keras.layers.Input((112, 112, 3)) #declare input shape

base_model = ResNet50(include_top=False, input_tensor=inputs, weights='imagenet') #load pretrained model

x = base_model(inputs) #get resnet output

out = Flatten()(x) #flatten output

out = Dropout(0.5)(out) #perform dropout

out = Dense(1, activation="sigmoid", name="out_")(out) # convert to values of 0 to 1 using the sigmoid activation function

model = keras.models.Model(inputs, out) #define model
#Now, we will compile the ResNet-50 model and also train it for 5 epochs

model.compile(loss=keras.losses.binary_crossentropy,

              optimizer=keras.optimizers.Adam(0.00075), 

              metrics=['accuracy',km.binary_f1_score()])

model.fit(x=train_imgs,y=train_labels,batch_size=50,epochs=5,validation_split=0.05)
#Load test data

N = 7558

test_files = ["../input/data/test/" + str(idx) + ".png" for idx in range(N)]

test_imgs = []

for file in tqdm(test_files):

    img = img_as_ubyte(resize(imageio.imread(file),(112,112))) #read image and resize all images to  112,112 for convenience

    test_imgs.append(preprocess_input(img)) #normalize images

test_imgs = np.asarray(test_imgs,dtype="uint8")

test_files = [file.split("test/")[1] for file in test_files] #only remember the final name for the submission csv
# Get number of model parameters

resnet_params = model.count_params() / 1e6

small_model_params = small_model.count_params() / 1e6



print("ResNet-50: Number of parameters in millions: ",resnet_params)

print("Small Model: Number of parameters in millions: ",small_model_params)
#Infer test data predictions using both models

for i in range(3): # we run it thrice to avoid potential overhead during the first run for initializations

    start = time.time()

    resnet_preds = model.predict(test_imgs,batch_size = 50,verbose=0)

    resnet_time = (1000*(time.time() - start) / N)

    print("ResNet-50: Inference runtime per image [ms]: ",str(resnet_time))
for i in range(3): # we run it thrice to avoid potential overhead during the first run for initializations

    start = time.time()

    small_model_preds = small_model.predict(test_imgs,batch_size = 50,verbose=0)

    small_model_time = (1000*(time.time() - start) / N)

    print("Small Model: Inference runtime per image [ms]: ",str(small_model_time))
resnet_submission = pd.DataFrame(data = {"Id" : test_files,

                                  "Predicted" : np.round(resnet_preds.squeeze())})

resnet_submission.Predicted = resnet_submission.Predicted.apply(int) #convert to 0 or 1

resnet_submission.to_csv("resnet_submission.csv", index = False, header = True)

resnet_submission.head(5)