#Import some useful libraries

import numpy as np #array handling and linear algebra

import matplotlib.pyplot as plt #plotting lib



#Data loading

from PIL import Image

import cv2

import os



#Machine learning framework

import keras

from sklearn.model_selection import train_test_split

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, AveragePooling2D, Dropout, Input

from keras.layers.normalization import BatchNormalization

from keras.layers.merge import concatenate

from keras import Model, Sequential



print("Importing finished!")
#Load the data

data=[]

labels=[]

parasitized_path =os.listdir("../input/cell_images/cell_images/Parasitized/")

for pars in parasitized_path:

    try:

        image=cv2.imread("../input/cell_images/cell_images/Parasitized/"+ pars)

        image_from_array = Image.fromarray(image, 'RGB')

        size_image = image_from_array.resize((64, 64))

        data.append(np.array(size_image))

        labels.append(0)

    except:

        continue
uninfected_path=os.listdir("../input/cell_images/cell_images/Uninfected/")

for unef in uninfected_path:

    try:

        image=cv2.imread("../input/cell_images/cell_images/Uninfected/" + unef)

        image_from_array = Image.fromarray(image, 'RGB')

        size_image = image_from_array.resize((64, 64))

        data.append(np.array(size_image))

        labels.append(1)

    except:

        continue
#Shape of the data

data = np.array(data)

labels = np.array(labels)



print("Shape of the data array: ", np.shape(data))

print("Shape of the label array: ", np.shape(labels))
#Split the data in training and test set

#Shuffle the data randomly

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=5)
#In earlier versions of this notebook I used a different label approach with just 0's and 1's. But I recognize a much better result,

#if I hot encoded the data. Therefore I have to change the label representation



train_y = np.zeros((len(y_train), 2))

test_y = np.zeros((len(y_test), 2))



for i in range(len(y_train)):

    if y_train[i] == 1:

        train_y[i][1] = 1

    else:

        train_y[i][0] = 1

        

for i in range(len(y_test)):

    if y_test[i] == 1:

        test_y[i][1] = 1

    else:

        test_y[i][0] = 1

y_train = train_y

y_test = test_y
print("Shape of the train data array: ", np.shape(X_train))

print("Shape of the train label array: ", np.shape(y_train))

print("Shape of the test data array: ", np.shape(X_test))

print("Shape of the test label array: ", np.shape(y_test))
#Define a function for a convolutional layer with batch normalization

#The commands in this function will be used very much, so it's simpler to define this function once. 

def conv(input_, filters_, kernel_, strides_, bias_, padding_):

    conv_ = Conv2D(filters=filters_, kernel_size=kernel_, strides=strides_, use_bias=bias_, padding=padding_)(input_)

    #The batch normalization helps to prevent overfitting and better learning results by removing the covariance shift.

    conv_ = BatchNormalization(axis = -1, momentum = 0.9997, scale = False)(conv_)

    conv_ = Activation("relu")(conv_)

    return conv_
#All kinds of inception networks starting with a stem. The stem preprocesses the input.

def stem(input_):

    #First convolutional block

    stem_ = conv(input_, 32, (3,3), (2,2), False, "valid")

    stem_ = conv(stem_, 32, (1,3), (1,1), False, "same")

    stem_ = conv(stem_, 32, (3,1), (1,1), False, "same")

    stem_ = conv(stem_, 64, (1,3), (1,1), False, "same")

    stem_ = conv(stem_, 64, (3,1), (1,1), False, "same")

    

    #Instead of going deeper the network will becoming wider!

    stem_1 = conv(stem_, 96, (3,3), (2,2), False, "valid")

    stem_2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="valid")(stem_)

    #Concatenate stem_1 and stem_2

    stem_ = concatenate([stem_1, stem_2], axis = -1)

    

    #In the next block we will also parallize two convolutional blocks

    #Here I reuse the two variable names from above

    stem_1 = conv(stem_, 64, (1,1), (1,1), False, "same")

    stem_1 = conv(stem_1, 64, (7,1), (1,1), False, "same")

    stem_1 = conv(stem_1, 64, (1,7), (1,1), False, "same")

    stem_1 = conv(stem_1, 96, (1,3), (1,1), False, "valid")

    stem_1 = conv(stem_1, 96, (3,1), (1,1), False, "valid")

    stem_2 = conv(stem_, 64, (1,1), (1,1), False, "same")

    stem_2 = conv(stem_2, 96, (3,3), (1,1), False, "valid")

    #Concatenate stem_1 and stem_2

    stem_ = concatenate([stem_1, stem_2], axis = -1)

    

    #Third concatenation block

    #Reuse stem_1 and stem_2

    stem_1 = MaxPooling2D(pool_size=(1,1), strides=(2,2), padding="valid")(stem_)

    stem_2 = stem_1 = conv(stem_, 192, (3,3), (1,1), False, "valid")

    #Concatenate stem_1 and stem_2

    stem_ = concatenate([stem_1, stem_2], axis = -1)

    

    return stem_  
def inception_A(input_):

    #In this block we parallize four convolutional blocks

    #First

    A_1 = conv(input_, 64, (1,1), (1,1), False, "same")

    A_1 = conv(A_1, 96, (1,3), (1,1), False, "same")

    A_1 = conv(A_1, 96, (3,1), (1,1), False, "same")

    A_1 = conv(A_1, 96, (1,3), (1,1), False, "same")

    A_1 = conv(A_1, 96, (3,1), (1,1), False, "same")

    

    #Second

    A_2 = conv(input_, 64, (1,1), (1,1), False, "same")

    A_2 = conv(A_2, 96, (1,3), (1,1), False, "same")

    A_2 = conv(A_2, 96, (3,1), (1,1), False, "same")

    

    #Third

    A_3 = conv(input_, 96, (1,1), (1,1), False, "same")

    

    #Fourth

    A_4 = AveragePooling2D((3, 3), strides = (1, 1), padding = "same")(input_)

    A_4 = conv(A_4, 96, (1,1), (1,1), False, "same")

    

    A = concatenate([A_1, A_2, A_3, A_4], axis=-1)

    

    return A    

    
def inception_B(input_):

    #Similiar to A

    #In this block we also parallize four convolutional blocks

    

    #First

    B_1 = conv(input_, 192, (1,1), (1,1), False, "same")

    B_1 = conv(B_1, 192, (1,7), (1,1), False, "same")

    B_1 = conv(B_1, 224, (7,1), (1,1), False, "same")

    B_1 = conv(B_1, 224, (1,7), (1,1), False, "same")

    B_1 = conv(B_1, 256, (7,1), (1,1), False, "same")

    

    #Second

    B_2 = conv(input_, 192, (1,1), (1,1), False, "same")

    B_2 = conv(B_2, 224, (7,1), (1,1), False, "same")

    B_2 = conv(B_2, 256, (1,7), (1,1), False, "same")

    

    #Third

    B_3 = conv(input_, 384, (1,1), (1,1), False, "same")

    

    #Fourth

    B_4 = AveragePooling2D((3, 3), strides = (1, 1), padding = "same")(input_)

    B_4 = conv(B_4, 128, (1,1), (1,1), False, "same")

    

    B = concatenate([B_1, B_2, B_3, B_4], axis=-1)

    

    return B    

    
def inception_C(input_):

    #This block is different to the structures of the other two blocks

    

    #First

    C_1 = conv(input_, 384, (1,1), (1,1), False, "same")

    C_1 = conv(C_1, 448, (1,3), (1,1), False, "same")

    C_1 = conv(C_1, 512, (3,1), (1,1), False, "same")

    #Split it up again

    C_11 = conv(C_1, 256, (1,3), (1,1), False, "same")

    C_12 = conv(C_1, 256, (3,1), (1,1), False, "same")

    #Concatenate it again

    C_1 = concatenate([C_11, C_12], axis=-1)

    

    #Second

    C_2 = conv(input_, 384, (1,1), (1,1), False, "same")

    #Split it up again

    C_21 = conv(C_2, 256, (1,3), (1,1), False, "same")

    C_22 = conv(C_2, 256, (3,1), (1,1), False, "same")

    #Concatenate it again

    C_2 = concatenate([C_21, C_22], axis=-1)

    

    #Third

    C_3 = conv(input_, 256, (1,1), (1,1), False, "same")

    

    #Fourth

    C_4 = AveragePooling2D((3, 3), strides = (1, 1), padding = "same")(input_)

    C_4 = conv(C_4, 128, (1,1), (1,1), False, "same")

    

    C = concatenate([C_1, C_2, C_3, C_4], axis=-1)

    

    return C

    
def reduction_1(input_):

    #Three parallized branches

    #We must choose four parameters (k,l,m,n) depending on the used network

    #The parameters are listed in a look up table in Paper 2

    k = 192

    l = 224

    m = 256

    n = 384

    

    #First

    R_1 = conv(input_, k, (1,1), (1,1), False, "same")

    R_1 = conv(R_1, l, (1,3), (1,1), False, "same")

    R_1 = conv(R_1, l, (3,1), (1,1), False, "same")

    R_1 = conv(R_1, m, (3,3), (2,2), False, "same")

    

    #Second

    R_2 = conv(input_, n, (1,3), (2,2), False, "same")

    R_2 = conv(input_, n, (3,1), (2,2), False, "same")

    

    #Third

    R_3 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="same")(input_)

    

    R = concatenate([R_1, R_2, R_3], axis=-1)

    

    return R
def reduction_2(input_):

    #Second reduction module

    

    #First

    R_1 = conv(input_, 256, (1,1), (1,1), False, "same")

    R_1 = conv(R_1, 256, (1,7), (1,1), False, "same")

    R_1 = conv(R_1, 320, (7,1), (1,1), False, "same")

    R_1 = conv(R_1, 320, (3,3), (2,2), False, "same")

    

    #Second

    R_2 = conv(input_, 192, (1,1), (1,1), False, "same")

    R_2 = conv(R_2, 192, (1,1), (2,2), False, "same")

    

    #Third

    R_3 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="same")(input_)

    

    R = concatenate([R_1, R_2, R_3], axis=-1)

    

    return R
def pure_inception_v4(load_weights=True):

    

    starter = Input((64, 64, 3))

    

    #Start with the stem

    inc = stem(starter)

    

    #inception block A

    inc = inception_A(inc)

    inc = Dropout(0.2)(inc)

    

    #First Reduction

    inc = reduction_1(inc)

    

    #innception block B

    inc = inception_B(inc)    

    inc = Dropout(0.2)(inc)

    

    #Second Reduction

    inc = reduction_2(inc)

    

    #inception block C

    inc = inception_C(inc)

    

    #Average pooling

    inc = AveragePooling2D((3, 3))(inc)



    # Dropout

    inc = Dropout(0.2)(inc) # Keep dropout 0.2 as mentioned in the paper

    inc = Flatten()(inc)



    # Output layer

    output = Dense(units = 2, activation = "softmax")(inc)

    

    model = Model(starter, output, name = "Inception-v4")   

        

    return model    
Model_ = pure_inception_v4()
print(Model_.summary())
Model_.compile(optimizer="adam", metrics=["accuracy"], loss="categorical_crossentropy")
history = Model_.fit(X_train, y_train, verbose=1, batch_size=500, epochs=5, shuffle=True, validation_data=(X_test, y_test))
plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend(["Train", "Test"])

plt.title("Model accuracy")

plt.show()
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend(["Train", "Test"])

plt.title("Model loss")

plt.show()