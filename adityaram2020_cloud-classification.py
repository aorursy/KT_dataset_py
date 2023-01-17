# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import cv2                  # importing open cv
import numpy as np           # numpy for conversion into numpy array
import pandas as pd             # pandas for csv
import matplotlib.pyplot as plt        # matplot for graphs
%matplotlib inline                
import os                               #for listing directories
import random                             # for shuffling images
import gc                                     #garbage collection after deletion
train_Ac = '../input/cloud-classification/data/train/Ac'  # Loading  directories
train_As = '../input/cloud-classification/data/train/As'
train_Cb = '../input/cloud-classification/data/train/Cb'
train_Cc = '../input/cloud-classification/data/train/Cc'
train_Ci = '../input/cloud-classification/data/train/Ci'
train_Cs = '../input/cloud-classification/data/train/Cs'
train_Ct = '../input/cloud-classification/data/train/Ct'
train_Cu = '../input/cloud-classification/data/train/Cu'
train_Ns = '../input/cloud-classification/data/train/Ns'
train_Sc = '../input/cloud-classification/data/train/Sc'
train_St = '../input/cloud-classification/data/train/St'
test_dir = '../input/cloud-classification/data/test'

train_Ac = ['../input/cloud-classification/data/train/Ac/{}'.format(i) for i in os.listdir(train_Ac)] # list of training images for each category
train_As = ['../input/cloud-classification/data/train/As/{}'.format(i) for i in os.listdir(train_As)]
train_Cb = ['../input/cloud-classification/data/train/Cb/{}'.format(i) for i in os.listdir(train_Cb)]
train_Cc = ['../input/cloud-classification/data/train/Cc/{}'.format(i) for i in os.listdir(train_Cc)]
train_Ci = ['../input/cloud-classification/data/train/Ci/{}'.format(i) for i in os.listdir(train_Ci)]
train_Cs = ['../input/cloud-classification/data/train/Cs/{}'.format(i) for i in os.listdir(train_Cs)]
train_Ct = ['../input/cloud-classification/data/train/Ct/{}'.format(i) for i in os.listdir(train_Ct)]
train_Cu = ['../input/cloud-classification/data/train/Cu/{}'.format(i) for i in os.listdir(train_Cu)]
train_Ns = ['../input/cloud-classification/data/train/Ns/{}'.format(i) for i in os.listdir(train_Ns)]
train_Sc = ['../input/cloud-classification/data/train/Sc/{}'.format(i) for i in os.listdir(train_Sc)]
train_St = ['../input/cloud-classification/data/train/St/{}'.format(i) for i in os.listdir(train_St)]
test_imgs = ['../input/cloud-classification/data/test/{}'.format(i) for i in os.listdir(test_dir)]
train_imgs = train_Ac +train_As+train_Cb+train_Cc+train_Ci+train_Cs+train_Ct+train_Cu+train_Ns+train_Sc+train_St # final list of images
random.shuffle(train_imgs) # shuffle images
 




import matplotlib.image as mpimg # matplotlib.image uses pillow convert images to numpy arrays
for ima in train_imgs[0:3]:
        img = mpimg.imread(ima)       # convert the image file stored to arrays
        imgplot = plt.imshow(img)                          # checking if the images are loaded properly
        plt.show()    # shows the images 
nrows = 150
ncolumns = 150         # nrows = height of the images , width of the images
channels = 3            # 3 channels for 3 array for red , blue and green pixel values

def process_image(image_list):                # function for process images using cv2 
    X = []  # Empty list for label list and image list
    Y = []
    for image in image_list:
        X.append(cv2.resize(cv2.imread(image,cv2.IMREAD_COLOR),(nrows,ncolumns),interpolation=cv2.INTER_CUBIC))  # cv2.imread is for reading the image file with color according to the specified height and width 
                                                                                                                    #while using cubic interpolation method
        if 'Ac' in image:
            Y.append(0)                           # labeling each image according from 1 to 10 for different classes of clouds
        elif 'As' in image:
            Y.append(1)
        elif 'Cb' in image:
            Y.append(2)
        elif 'Cc' in image:
            Y.append(3)
        elif 'Ci' in image:
            Y.append(4)
        elif 'Cs' in image:
            Y.append(5)
        elif 'Ct' in image:
            Y.append(6)
        elif 'Cu' in image:
            Y.append(7)
        elif 'Ns' in image:
            Y.append(8)
        elif 'Sc' in image:
            Y.append(9)
        elif 'St' in image:
            Y.append(10)
     
    return X,Y
            
            
X,Y = process_image(train_imgs)# function process the images and stores the return values in X and Y

plt.figure(figsize=(20,10))
columns =5 
for i in range(columns):
    plt.subplot(5/columns+1,columns,i+1)   # subplot to plot multiple images
    plt.imshow(X[i])        #  to check images are loaded properly
import seaborn as sns                # import seaborn library
del train_imgs                       # del train imgs directory list as its not needed
gc.collect()                           # garbage collection

X = np.array(X)                       # converting to numpy arrays
Y = np.array(Y)
sns.countplot(Y)                      # plot showing distribution of different labels
plt.title("Label for different clouds")
from sklearn.model_selection import train_test_split   # train test split for splitting into training data and validation data 
from keras.utils import to_categorical  # to_categorical for converting list of integers to binary matrix 0's and 1's
X_train, X_val , y_train , y_test = train_test_split(X,Y,test_size=0.10,random_state=2) #random_state for randomising the data being the split

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)  # conversion of both y train and y test into a binary matrix
print(y_train.shape) # checking if the conversion is successful by checking shape.
del X
del Y                           # X,Y are deleted as its no longer needed and garbage is collected
gc.collect()
ntrain = len(X_train)
nval = len(X_val)               #length of X_train and X_val is collected
batch_size = 64                  # defining batch size
from keras import layers                              #layers in keras performs calculation and send it to next layers
from keras import models                            # models class defines the way in which model is going to be built.
from keras import optimizers                        # contains algorithms that determines the step in which the neural network after each layer takes in weight to move towards solution
from keras.preprocessing.image import ImageDataGenerator # ImageDataGenerator augments / performs the incoming data and makes it generalisable.
from keras.preprocessing.image import img_to_array , load_img  # converts image to array and for loading image
model = models.Sequential()                                                         #Sequential API for linear stacking model
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))            #VGG net architechture (Increasing filter size for each layer)
model.add(layers.MaxPooling2D((2,2)))                                                       # Input layer takes 3,3 size window with output size of 32 and input shape of the image
model.add(layers.Conv2D(64,(3,3),activation='relu'))# activation function rectified linear unit
model.add(layers.MaxPooling2D((2,2)))                # Max pooling reduces the parameters , computational load (2,2) reduces chances of over fitting
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())          # flattens the neural network model          
model.add(layers.Dropout(0.5))                    # Dropout is for regularisation drops 50% of neurons to reduce overfitting
#model.add(layers.Dense(256,activation='relu',input_shape=(150,150,3)))
model.add(layers.Dense(512,activation='relu')) # Dense calculates the final computation based on the model returned by flatten using relu
model.add(layers.Dense(11,activation='softmax')) # softmax calculates the probablities for multiple classes
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy']) # defining optimisers , loss function = categorical for multiclass optimiser =adam sgd algorithm
train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,)
val_datagen = ImageDataGenerator(rescale=1./255) # data augmentation converts pixels to range of 0,1 normalisation of the data 
#image will be rescaled based on the features , pixels having mean of 0 and SD of 1
# for val image is not augmented just normalised
train_generator = train_datagen.flow(X_train,y_train,batch_size=batch_size) #.flow takes the data and creates batches of augmented data
val_generator = val_datagen.flow(X_val,y_test,batch_size=batch_size)
model.summary()
history = model.fit(train_generator,steps_per_epoch= ntrain//batch_size,epochs=40,validation_data=val_generator,validation_steps=nval//batch_size)# training the data
model.save_weights('model_weights.h5') # saves the model weights
model.save('model_keras.h5') # saves the models
acc = history.history['accuracy']
loss = history.history['loss']                  # recording training parameters and checking for overfitting 
val_loss = history.history['val_loss']
val_accuracy = history.history['val_accuracy']
epochs = range(1,len(acc)+1)
plt.plot(epochs,acc,'b',label='Training Accuracy')
plt.plot(epochs,val_accuracy,'r',label='Validation accuracy')
plt.title("Training and validation accuracy")
plt.legend()
plt.plot(epochs,loss,'b',label='Training Loss')
plt.plot(epochs,val_loss,'r',label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

X_test , y_test = process_image(test_imgs)
x= np.array(X_test)
print(x.shape)                          # Testing data ( unseen)
test_datagen = ImageDataGenerator(rescale=1./255)# normalising the data 
i =0 
text_labels = []
plt.figure(figsize=(30,20))
for batch in test_datagen.flow(x,batch_size=1,shuffle=False):                   #predicting classes and appending labels
    pred = model.predict_classes(batch)
    
    if pred==0:
        text_labels.append('Ac')
    elif pred==1:
        text_labels.append("As")
    elif pred==2:
        text_labels.append("Cb")
    elif pred==3:
        text_labels.append("Cc")
    elif pred==4:
        text_labels.append("Ci")
    elif pred==5:
        text_labels.append("Cs")
    elif pred==6:
        text_labels.append("Ct")
    elif pred==7:
        text_labels.append("Cu")
    elif pred==8:
        text_labels.append("Ns")
    elif pred==9:
        text_labels.append("Sc")
    elif pred==10:
        text_labels.append("St")
    plt.subplot(5/columns+1  , columns , i+1)
    plt.title('This is a '+text_labels[i]+test_imgs[i])
    imgplot = plt.imshow(batch[0])
    i +=1                                            
    if i % x.shape[0]==0:
        break
plt.show()
