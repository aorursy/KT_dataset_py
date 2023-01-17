
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from keras import backend as K
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from keras.utils import to_categorical
from keras.models import Model

# specifically for cnn
from keras.layers import Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

TypeFlowers = os.listdir("../input/flowers/flowers/") #list of all the possible flowers

# Any results you write to the current directory are saved as output.
X = [] #all the inputs
Y= [] #all the outputs (remember to use softmax)
SizeImage = 150 #Lets get all the images the same size
DictType  ={}
j = 0
# creating a dictionary to identifiy the outputs
for i in TypeFlowers:
    DictType[i] = j
    j = j+ 1
print(DictType["dandelion"])

def GettingTrainingData(flower): #here we're gonna get all the data (then we'll separated in test and training mode)
    Directory = "../input/flowers/flowers/" + flower + "/"
   
    for img in tqdm(os.listdir(Directory)):
        path = Directory + img
        _, ftype = os.path.splitext(path)
        if ftype == ".jpg":  #there is a diferent extension that doesnt aceppt resize function in dandelion folder
            Image = cv2.imread(path)
            Image = cv2.resize(Image, (SizeImage,SizeImage))
            X.append(np.array(Image))
            Y.append(DictType[flower])
        
    
    
for i in TypeFlowers:
    GettingTrainingData(i) #to get all the data to X and Y variables




# in order to get a better result, it's suggested to normalize the input
X = np.array(X)
Y = np.array(Y)
X = X/255
print(X.shape)
print(Y.shape)
plt.imshow(X[2000])
plt.title(TypeFlowers[Y[2000]])
#now it's time to separate the training set, the dev set, and the test set
# it is quite important, to get them randomly
def SeparateData(X, Y):
    ##before I forget, it's really importante to hot one encode Y
    Y = to_categorical(Y,5)
    X_train, X_test,Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=13)
    return X_train, Y_train, X_test, Y_test

print(X[1])
## now it's time to train our modeeel,

def BuildModel(input_shape):
    #at this example, we're using the API model in Keras, learned in Andrew NG`s specialization
    # Define the input placeholder as a tensor with shape input_shape.
    
    #We're gonna use 3 convolution networks, increasing the number of filters, while reducing the window sizes
    X_input = Input(input_shape)

    X = Conv2D(filters = 32, kernel_size = (5, 5), strides = (1, 1), name = 'conv0', padding = 'same', activation = 'relu')(X_input)
    X = MaxPooling2D((2, 2), name='max_pool0')(X)
    X = Conv2D(filters = 64, kernel_size = (5,5), name = 'conv1', padding = 'same', activation = 'relu')(X)
    X = MaxPooling2D((2, 2), name='max_pool1')(X)
    X = Conv2D(filters = 96, kernel_size = (3,3), strides = (1, 1), name = 'conv2', padding = 'same', activation = 'relu')(X)
    X = MaxPooling2D((2, 2), name='max_pool2')(X)
    
    X = Conv2D(filters = 128, kernel_size = (3,3), strides = (1, 1), name = 'conv3', padding = 'same', activation = 'relu')(X)
    
    X = MaxPooling2D((2, 2), name='max_pool3')(X)
    
    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(512, activation='sigmoid', name='fc')(X)
    # lastly, we need to get our output to the right format, to this we use a sofmax - Remember that we have 5 flowers options
    # later on, we could use other images, to represent "not a flower". Then, the softmax would increase to 6
    X = Dense(5, activation = "softmax", name="finalout") (X)
    
    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='Flower Recognition')

    return model
    
    
    
#now it's time to call the function and actively build the model. 

X_train, Y_train, X_test, Y_test = SeparateData(X,Y)



FlowerRecon = BuildModel(X_train.shape[1:])
#ow it's time to compile the network to get ready for training

FlowerRecon.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
FlowerRecon.summary()
print(X_train.shape)
print(Y_train.shape)
#now it's time to train and gett all the paramameters
#it's important to choose wisely the epochs and batch_sizes hyperparameters
FlowerRecon.fit(x = X_train, y = Y_train, epochs = 20, batch_size = 128)
#at this step we're going to test the results. 
EvalResult = FlowerRecon.evaluate(X_test, Y_test)

print ("Loss = " + str(EvalResult[0]))
print ("Test Accuracy = " + str(EvalResult[1]))

#Gettin results to evalueate
EvalResult = FlowerRecon.predict(X_test)
#Then, we're gonna separate between corrected label, and uncorrectedlabel

Correctlbl = []
Uncorlbl =[]
print(EvalResult.shape)

#separating between corrected and uncorrected values
for i in range(len(Y_test)):
    if np.argmax(Y_test[i])  == np.argmax(EvalResult[i]):
        Correctlbl.append(i)
    else:
        Uncorlbl.append(i)
    


# function to see Results
def ResultsVisualization (X_test, Y_test, EvalResult,labels):
                        
    fig,ax=plt.subplots(4,2)
    fig.set_size_inches(15,15)
    for i in range (4):
        for j in range (2):
            cont = j+ 2*i
            actual = TypeFlowers[np.argmax(Y_test[labels[cont]])]
            predict = TypeFlowers[np.argmax(EvalResult[labels[cont]])]
            ax[i,j].imshow(X_test[labels[cont]])
            ax[i,j].set_title("Predicted Flower : "+ predict +"\n"+"Actual Flower : "
                              +actual)
            plt.tight_layout()
        
#Seeing correct examples

ResultsVisualization (X_test, Y_test, EvalResult,Correctlbl)

#Seeing uncorrected results
ResultsVisualization (X_test, Y_test, EvalResult,Uncorlbl)
