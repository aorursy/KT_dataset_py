import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
import pandas as pd
import os
import random
from PIL import Image
import skimage
import glob
from skimage import transform
import matplotlib.pyplot as plt
from keras import layers ,regularizers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from sklearn.preprocessing import LabelBinarizer
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
# this fun load data set and shuffle it too 
def loadData():
    '''read data, shuffle it  , split it into training and test and reshape it to (100*100*3)
    
    - parameters 
    
        - no inputs
        
        - outputs :
            # assume (tr) num of images of training & (te) num of images of test
            - trainingX : np array and shape of (tr,100,100,3)
            - trainigY : np array shape of (tr,1 )
            - testX : np array and shape of (te,100,100,3)
            - testY : np array shape of (te,1 )
            
    '''
    # read images names and labels 
    imageNames = np.array(glob.glob("../input/train/*.jpg"))
    imageLabels = np.array([f[15:18] for f in imageNames])
    #imageNames = np.array([f for f in os.listdir("../input/train/")])
    #imageLabels = np.array([f[0:3] for f in imageNames])
    
    # shuffl data 
    m = len(imageLabels)
    randomIndex = list(np.random.permutation(m))
    Y_shuffled = imageLabels[randomIndex]
    imageName_shuffled = imageNames[randomIndex]
    
    # splid data into 80% traning and 20% test 
    trainingLen = int(m * .8)
    testLen = m - trainingLen
    
    trainingY = Y_shuffled[0:trainingLen]
    trainingX_names = imageName_shuffled[0:trainingLen]
    
    testY = Y_shuffled[trainingLen:m]
    testX_names = imageName_shuffled[trainingLen:m]
    
    # load images as rgb
    trainingX = []
    testX = []
    
    for imgName in trainingX_names: #load training images
        img = np.array(Image.open(imgName))
        img = skimage.transform.resize(img, output_shape =(100,100))
        trainingX.append(img)
    
    for imgName in testX_names: #load test images
        img = np.array(Image.open(imgName))
        img = skimage.transform.resize(img, output_shape =(100,100))
        testX.append(img)
    
    # convert list of trainingX and testX to matrix
    trainingX = np.array(trainingX,dtype="float32")
    testX = np.array(testX,dtype="float32")
    


    return trainingX,trainingY,testX,testY
def oneHot(y):
    ''' convert labels into one hot coding 
        parameters:
            - input : list of labels 
            - output : np array of one hot encoding 
    '''
    encoder = LabelBinarizer()
    y = encoder.fit_transform(y)
    return y

# this fun plot example of our images data 
def plot_figures(x,y,nrows = 3, ncols=4):
    """Plot random figures of data.

    Parameters
    ----------
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """
    # generate random indexs 
    indexs = random.sample(range(0, len(y)), nrows*ncols)
    
    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows,figsize=(10,10))
    for ind,i in enumerate(indexs):
        axeslist.ravel()[ind].imshow(x[i], cmap=plt.gray())
        axeslist.ravel()[ind].set_title(y[i])
        axeslist.ravel()[ind].set_axis_off()
    #plt.tight_layout() # optional

def identityBlock(inputs,stage,block,f,kernal_size=3,s=1):
    ''' Implementation of indentity block for resNet when input's shape of block == output's shape
        # using f = 3*3 and num of skipping layers is 3 and s = 1
        args:
            - inpute : input tensor of indentity block-- shape (m,h_prev,w_prev,c_prev)
            - f : int num -- num of filters for every layer in this block 
            - kernal_size : int num -- kernal_size of conv layers of main path 
            - stage : int num -- the number of current stage of resNet, we use it to set names of conv leayers
            - block :int num -- the number of current block of current stage, we use it to set names of conv leayers
            - s :  int num -- stride

        output :
         - x tensor with shape (m,h,w,c) -- c=f
    '''
    
    x = inputs
    stage = str(stage)
    block =str(block)
    nameBase = 'stage'+stage+'-block'+block
    # first 
    x = Conv2D(f, (kernal_size,kernal_size),padding='same',strides = (s,s),kernel_initializer='he_normal',name = 'a-'+nameBase)(x)
    x = BatchNormalization(name='bn-a-'+nameBase,axis = 3)(x)
    x = Activation('relu')(x)
    
    # seconde 
    x = Conv2D(f, (kernal_size,kernal_size),padding='same',strides = (s,s),kernel_initializer='he_normal',name = 'b-'+nameBase)(x)
    x = BatchNormalization(name='bn-b-'+nameBase,axis = 3)(x)
    x = Activation('relu')(x)
    
    #third
    x = Conv2D(f, (kernal_size,kernal_size),padding='same',strides = (s,s),kernel_initializer='he_normal',name = 'c-'+nameBase)(x)
    x = BatchNormalization(name='bn-c-'+nameBase,axis = 3)(x)
    
    #short 
    x = Add()([x,inputs])
    x = Activation('relu')(x)
    
    return x
def convBlock(inputs,stage,block,f,kernal_size=3,s=2):
    ''' Implementation of indentity block for resNet when input's shape of block != output's shape
        # using f = 3*3 and num of skipping layers is 3 and s = 1
        args:
            - inpute : input tensor of indentity block-- shape (m,h_prev,w_prev,c_prev)
            - f : int num -- num of filters for every layer in this block 
            - kernal_size : int num -- kernal_size of conv layers of main path 
            - stage : int num -- the number of current stage of resNet, we use it to set names of conv leayers
            - block :int num -- the number of current block of current stage, we use it to set names of conv leayers
            - s :  int num -- stride

        output :
         - x tensor with shape (m,h,w,c) -- c=f
    '''
    x = inputs
    stage = str(stage)
    block = str(block)
    nameBase = 'stage'+stage+'-block'+block
    #first
    x = Conv2D(f, (kernal_size,kernal_size),padding='valid',strides = (s,s),kernel_initializer='he_normal',name = 'a-'+nameBase)(x)
    x = BatchNormalization(name='bn-a-'+nameBase,axis = 3)(x)
    x = Activation('relu')(x)
    
    #seconde
    x = Conv2D(f,(kernal_size,kernal_size),padding='same',strides = (1,1),kernel_initializer='he_normal',name = 'b-'+nameBase)(x)
    x = BatchNormalization(name='bn-b-'+nameBase,axis = 3)(x)
    x = Activation('relu')(x)
    
    #third
    x = Conv2D(f, (kernal_size,kernal_size),padding='same',strides = (1,1),kernel_initializer='he_normal',name = 'c-'+nameBase)(x)
    x = BatchNormalization(name='bn-c-'+nameBase,axis = 3)(x)

    #conv for inputs before add it to x 
    inputs = Conv2D(f,(kernal_size,kernal_size),padding='valid',strides = (2,2),kernel_initializer='he_normal',name = 'short-'+nameBase)(inputs)
    inputs = BatchNormalization(name='bn-short-'+nameBase,axis = 3)(inputs)
    
    #add shortcut to x
    x = Add()([x,inputs])
    x = Activation('relu')(x)
    
    return x
def resNet35(inputShape):
    '''Implementation of ResNet with 35 layers 
    Architecture :
        ->conv2D -> batchNorm -> maxPool = 1 layer 
        -> stage1 =  2*identityBlock = 6 layers
        -> stage2 = 1*convBlock + 2*identityBlock = 9 layers
        -> stage3 = 1*convBlock + 3*identityBlock = 12 layers
        -> stage4 = 1*convBlock + 1*identityBlock = 6 layers
        -> avgPool -> FC softmax with 1 node (num of class = 2 so output nodes = 1 ) = 1 layer
    
    '''

    
    ###############################
    
    BlockNum = [2,3,4,2] # list for number of blocks in every stage 
    kernal_size = [64,128,256,512]
    modelInput =Input(inputShape[1:]) # define input tensor to feed it later to keras Model
    x=modelInput
    print(x.shape)
    # first layer conv2D ->  batchNorm -> maxPool
    x = Conv2D(64,(7,7),padding='same',strides=(2,2),name='conv-first',kernel_initializer='he_normal')(x)
    x = BatchNormalization(name='bn-first',axis = 3)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    ## stages and blocks loop
    for stage in range(1,5):
        f = kernal_size[stage-1] #num of filters for current stage 
        block =(BlockNum[stage-1])
        for b in range(block):
            if stage==1: #first stage so we don't need convBlock
                x = identityBlock(x,stage,b,f,kernal_size=3,s=1)

            else : 
                if b ==0 : # not first stage but first block
                    x = convBlock(x,stage,b,f,kernal_size=3,s=2)
                else:
                    x = identityBlock(x,stage,b,f,kernal_size=3,s=1)
    
    # avgPool layer
    x= AveragePooling2D(pool_size=(2, 2),name='avg_pool',padding='same')(x)
    # FC layer
    x = Flatten()(x)
    #x = Dense(100,name="fc1", kernel_initializer = 'glorot_uniform',activation='relu',kernel_regularizer=regularizers.l2(0.01))(x)
    x = Dense(1,name="fc", kernel_initializer = 'glorot_uniform',activation='sigmoid')(x)
    model = Model(inputs = modelInput, outputs = x, name='ResNet35')
    return model

## let's start work 

#load data 
trainingX,trainingY,testX,testY = loadData()

# encoding trainingY and testY
trainingY=oneHot(trainingY)
testY=oneHot(testY)

# check our dim
print("training X shape is ", trainingX.shape)
print("training y shape is ", trainingY.shape)
print("test X shape is ", testX.shape)
print("test y shape is ", testY.shape)


# plot Figures 
plot_figures(trainingX,trainingY,nrows = 3, ncols=4)
# call model
model = resNet35(trainingX.shape)
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#start training
history = model.fit(trainingX,
                    trainingY,
                    epochs = 50,
                    batch_size = 64,
                    validation_data=(testX,testY), 
                    shuffle = True)
def plotModelHistory(modeHistory):
    # summarize history for accuracy
    history = modeHistory
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
plotModelHistory(history)
#save model
model.save_weights('resNet35_wieghts.h5')
model.save('model_keras.h5')
# del some data 
del trainingX,trainingY , testX,testY
#load test data
def loadTestData():
    path = '../input/test/*jpg'
    imageNames=glob.glob(path)
    m = len(imageNames)
    test_x = np.zeros((m,100,100,3),dtype=np.float32)
    test_y = np.zeros((m,1))
    for i in range(m):
        img = np.array(Image.open(imageNames[i]))
        img = skimage.transform.resize(img, output_shape =(100,100))
        test_x [i] = img
    return test_x , test_y
test_x , test_y = loadTestData()
test_y = model.predict(test_x)

# plot some test image with there predicted label
plot_figures(test_x,test_y,nrows = 3, ncols=4)
# save submission file 
frame = pd.DataFrame({'label': test_y.T.squeeze()})
frame = frame.reset_index(drop=True)
frame.index += 1 
frame.to_csv("Dogs Vs. Cats.csv", index_label='id')
print(frame)
