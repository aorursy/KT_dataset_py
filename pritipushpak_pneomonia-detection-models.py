# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



# import numpy as np # linear algebra

# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# # Input data files are available in the read-only "../input/" directory

# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#import required libraries

import os

import io



import pandas as pd

import numpy as np

import random



import matplotlib.pyplot as plt

import pydicom

import cv2

import matplotlib.pyplot as plt

import matplotlib.patches as patches

import seaborn as sns



import skimage.io as io

from sklearn.utils import shuffle

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from skimage.transform import resize

from skimage import measure



import tensorflow as tf

from tensorflow.keras.losses import binary_crossentropy

from tensorflow.keras.backend import log, epsilon

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import MaxPooling2D

from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import Dense, Dropout, InputLayer, BatchNormalization, Input

from tensorflow.keras.layers import Concatenate, UpSampling2D, Conv2D, Reshape

from tensorflow.keras.optimizers import Adam



%matplotlib inline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  





def printConfusionMatrix(title, y_true, y_pred):

    print (title)

    print ('-------------------')

    print ('Validation Accuracy: ', accuracy_score(y_true,y_pred))

    print ('Validation Precision: ', precision_score(y_true,y_pred))

    print ('Validation Recall: ', recall_score(y_true,y_pred))

    print ('Validation F1-Score: ', f1_score(y_true,y_pred))

    

    cm=confusion_matrix(y_true, y_pred, labels=[1, 0])

    df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],

                      columns = [i for i in ["Predict Yes","Predict No"]])

    #Plot the heat map for the confusion matrix

    plt.figure(figsize = (5,5))   

    sns.heatmap(df_cm, annot=True, fmt="g", yticklabels=["Actual Yes", "Actual No"], cmap="YlGnBu_r")

    ax = plt.gca()

    ax.set_title(title)    

    plt.show()

    return accuracy_score(y_true,y_pred), precision_score(y_true,y_pred),recall_score(y_true,y_pred),f1_score(y_true,y_pred)
def saveOrLoadWeights(model,fileName):

    h5Path = os.path.join(root_dir, 'input', 'h5files',fileName)

    if not loadweightsDirectlyToModel:

        model.save_weights(fileName)

    else:

        model.load_weights(h5Path)
def plotGraphs(model):

    if not loadweightsDirectlyToModel:

        plt.figure(figsize=(12,4))

        plt.subplot(131)

        #history = model.history



        plt.plot(history.epoch, history.history["loss"], label="Train loss")

        plt.plot(history.epoch, history.history["val_loss"], label="Valid loss")

        plt.xlabel('Epochs')

        plt.ylabel('Loss')

        plt.legend()



        plt.subplot(132)

        plt.plot(history.epoch, history.history["mean_iou"], label="Mean IOU")

        plt.plot(history.epoch, history.history["val_mean_iou"], label="Mean IOU")

        plt.legend()



        plt.subplot(133)

        plt.plot(history.epoch, history.history["accuracy"], label="Accuracy")

        plt.plot(history.epoch, history.history["val_accuracy"], label="Valid Accuracy")

        plt.legend()



        plt.show()
def displayImagesWithBoundingBox(model):

    for imgs, msks in val_gen:

        # predict batch of images

        preds = model.predict(imgs)

        # create figure

        f, axarr = plt.subplots(4, 8, figsize=(20,15))

        axarr = axarr.ravel()

        axidx = 0

        # loop through batch

        for img,msk, predm in zip(imgs,msks, preds):

            # plot image

            axarr[axidx].imshow(img[:, :, 0])

            detectandMask(msk,False, axarr[axidx])

            detectandMask(predm,True, axarr[axidx])

            axidx += 1

        plt.show()



        # only plot one batch

        break
def displayMetrics(modelName, model):

    y_predval = []

    y_trueval = []





    for imgs, msks in val_gen:

        # predict batch of images

        preds = model.predict(imgs)



        # loop through batch

        for img, msk, pred in zip(imgs, msks, preds):

            #  detect Pneumonia in mask actual & predicted

            y_trueval.append(detectandMask(msk, False))

            y_predval.append(detectandMask(pred))





    return printConfusionMatrix(modelName, y_trueval, y_predval)
#Initialize the variables

print(f'Current working directory: {os.getcwd()}')

print('Folder and Files in current directory: {}'.format(os.listdir()))



doProcessingFromDCMImages = False

loadweightsDirectlyToModel = False

root_dir = '/kaggle/'

if doProcessingFromDCMImages:

    input_dir = os.path.join(root_dir, 'input/rsna-pneumonia-detection-challenge/')



    #input_dir = 'input'

    output_dir = os.path.join(root_dir, 'output', )

    if not os.path.exists(output_dir):

        os.makedirs(output_dir)

else:

    input_dir = os.path.join(root_dir, 'input/rsna-pneumonia-detection-challenge/')

    

    processed_input_dir = os.path.join(root_dir, 'input', 'processed-rsna-image-files','output')

    



input_train = 'stage_2_train_images'

input_test = 'stage_2_test_images'

class_file = 'stage_2_detailed_class_info.csv'

label_file = 'stage_2_train_labels.csv'

output_train = 'train'

output_test = 'test'

output_val = 'val'







IMAGE_HEIGHT = 224

IMAGE_WIDTH = 224
#     #read the classes to create folders as per the category. 



#     #Reading the class file to get the list of patients and shuffle them to create training and validation set

df_class = pd.read_csv(os.path.join( root_dir, input_dir, class_file))



df_class["class"] = df_class["class"].astype('category')

print("Mapped values for SMOKER column ====> %s" %( dict(enumerate(df_class["class"].cat.categories))))

df_class["class"] = df_class["class"].cat.codes
df_class.head()
#shuffle the data and split the data as train and valid



df_class = shuffle(df_class)

# stratify=y creates a balanced validation set.

y = df_class['class']



df_train, df_val = train_test_split(df_class, test_size=0.10, random_state=101, stratify=y)



print('Train set size: ', df_train.shape)

print('Validation set size: ', df_val.shape)
# read dicom files and resize them to 256 x 256 and save them as .png file in train, validation and test folder.



def preprocessimages(root_dir,input_images , outputpath, filenames , image_width= 256 ,image_height= 256):

    """

    Args:

    :param root_dir (string): Root Directory with all the images

    :param input_images : Input image directory

    :param outputpath: Output root directory

    :param image_width (optional): Output Image width size, Default is 256

    :param image_height (optional): Output Image width size, Default is 256

    """

    dest_path = os.path.join(root_dir, output_dir)

    if not os.path.exists(dest_path):

        os.mkdir(dest_path)

    dest_path = os.path.join(dest_path, outputpath)

    if not os.path.exists(dest_path):

        os.mkdir(dest_path)

    

    # Loop through all the DICOM files under the folder and collect the information from DICOM file

    for file in filenames:

        path = os.path.join(root_dir, input_dir, input_images, file)

        dicom_data = pydicom.read_file(path)

        

        # Transform the image to smaller size and store it under data\images\train\

        dicom_img = dicom_data.pixel_array



        img = cv2.resize(dicom_img, dsize=(image_height, image_width), interpolation=cv2.INTER_CUBIC)

        cv2.imwrite(os.path.join(dest_path, dicom_data.PatientID + '.png'),img)

        

trainfiles= list(map(lambda x: x + '.dcm' , df_train['patientId']))

valfiles= list(map(lambda x: x + '.dcm' , df_val['patientId']))



if doProcessingFromDCMImages:

    

    #Convert the training images

    preprocessimages(root_dir,input_train, output_train ,trainfiles ,image_height=IMAGE_HEIGHT, image_width=256)

    #Convert the validation images

    preprocessimages(root_dir,input_train, output_val,valfiles ,image_height=IMAGE_HEIGHT, image_width=256)

    #Convert the test images

    preprocessimages(root_dir,input_test, output_test,os.listdir(os.path.join(root_dir, input_dir,input_test)), image_height=IMAGE_HEIGHT, image_width=256)

else:

    print(f'using preprocessed png images from dcm images')
if doProcessingFromDCMImages:

    #initialize folders train, val , test, 

    output_dir = os.path.join(root_dir, output_dir)



    train_dir = os.path.join(output_dir,output_train)



    val_dir = os.path.join(output_dir,output_val)



    test_dir = os.path.join(output_dir, output_test)

else:

    output_dir = processed_input_dir

    output_dir = os.path.join(root_dir, output_dir)



    train_dir = os.path.join(output_dir,output_train)



    val_dir = os.path.join(output_dir,output_val)



    test_dir = os.path.join(output_dir, output_test)

print(f"output_dir: {output_dir}\ntrain_dir:{train_dir}\nval_dir:{val_dir}\ntest_dir:{test_dir}")
# Get a list of train and val images to create a datagenerator

train_list = (os.listdir(train_dir))

val_list = (os.listdir(val_dir))



num_train_samples = len(os.listdir(train_dir))

num_val_samples = len(os.listdir(val_dir))

train_batch_size = 10

val_batch_size = 10



#initialize the training and validation steps

train_steps = np.ceil(num_train_samples / train_batch_size)

val_steps = np.ceil(num_val_samples / val_batch_size)



print ('Training sample size: ', num_train_samples)

print ('Training steps: ', train_steps)

print ('Validation sample size: ', num_val_samples)

print ('Validation steps : ', val_steps)
# Store the weights

fileName = 'unet_model.h5'

saveOrLoadWeights(unet_model,fileName)
# Generator class which handles large dataset and creates batches of images.

class generator(tf.keras.utils.Sequence):

    

    def __init__(self, img_dir,label_file,  batch_size=32, image_size=256, shuffle=True, augment=False, predict=False):

        """

        Args:

        :param img_dir (string): Directory with all the images(train, val or test)

        :param batch_size (optional): Defines the batch size of the generator. when the value is increased memory usage will be more. default is 32.

        :param image_size: Imagesize which will be input to the model. Images will be loaded to memory and resized them to this resize before inputting to model. default is 256

        :param shuffle (optional): defult is True, when True the dataset is shuffled before each epoch.

        :param augment (optional): defult is False, when True the dataset is flipped horizontally

        :param predict (optional): defult is False, when True only the images are returned , When False, Images and its masks are returned

        """

        self.folder = img_dir

        self.filenames = os.listdir(img_dir)

        self.batch_size = batch_size

        self.image_size = image_size

        self.shuffle = shuffle

        self.augment = augment

        self.predict = predict

        self.on_epoch_end()

        # read the label Label file which contains the bounding boxes

        self.df_data = pd.read_csv(label_file)

        # set Nan values to 0

        self.df_data= self.df_data.fillna(0)

        self.ratio = image_size / 1024 # Original size

        

    def __load__(self, filename):

        # load input png image file as numpy array

        img = cv2.imread(os.path.join(self.folder ,filename), 0) 

        # create empty mask

        msk = np.zeros((self.image_size, self.image_size))

        patient = self.df_data[self.df_data.patientId==filename[:-4]]



        for i in range(len(patient)):

            #Calculate the bounding box based on the resized image

            x1 = int(self.ratio * patient['x'].iloc[i]  )

            y1 = int(self.ratio * patient['y'].iloc[i] )

            x2 = x1 + int(self.ratio * patient['width'].iloc[i] )

            y2 = y1 + int(self.ratio * patient['height'].iloc[i])

            msk[y1:y2, x1:x2] = 1

        # resize both image and mask

        img = resize(img, (self.image_size, self.image_size), mode='reflect')

        msk = resize(msk, (self.image_size, self.image_size), mode='reflect') > 0.5

        # if augment then horizontal flip half the time

        if self.augment and random.random() > 0.5:

            img = np.fliplr(img)

            msk = np.fliplr(msk)

        # add trailing channel dimension

        img = np.expand_dims(img, -1)

        msk = np.expand_dims(msk, -1)

        return img, msk

    

    def __loadpredict__(self, filename):

        # load png image file as numpy array

        img = cv2.imread(os.path.join(self.folder ,filename), 0) 

        # resize image

        img = resize(img, (self.image_size, self.image_size), mode='reflect')

        # add trailing channel dimension

        img = np.expand_dims(img, -1)

        return img

        

    def __getitem__(self, index):

        # select batches of files

        filenames = self.filenames[index*self.batch_size:(index+1)*self.batch_size]

        

        # predict mode: return images and filenames

        if self.predict:

            # load files

            imgs = [self.__loadpredict__(filename) for filename in filenames]

            # create numpy batch

            imgs = np.array(imgs)

            return imgs#, filenames

        # train mode: return images and masks

        else:

            # load files

            items = [self.__load__( filename) for filename in filenames]

            # unzip images and masks

            imgs, msks = zip(*items)

            # create numpy batch

            imgs = np.array(imgs)

            msks = np.array(msks)

            return imgs, msks

        

    def on_epoch_end(self):

        if self.shuffle:

            shuffle(self.filenames)

        

    def __len__(self):

        if self.predict:

            # return everything

            return int(np.ceil(len(self.filenames) / self.batch_size))

        else:

            # return full batches only

            return int(len(self.filenames) / self.batch_size)
class generatorWithBoxType(tf.keras.utils.Sequence):

    

    def __init__(self, img_dir,label_file,  batch_size=32, image_size=256, shuffle=True, augment=False, predict=False, ytype=['box','lbl', 'both']):

        """

        Args:

        :param img_dir (string): Directory with all the images(train, val or test)

        :param batch_size (optional): Defines the batch size of the generator. when the value is increased memory usage will be more. default is 32.

        :param image_size: Imagesize which will be input to the model. Images will be loaded to memory and resized them to this resize before inputting to model. default is 256

        :param shuffle (optional): defult is True, when True the dataset is shuffled before each epoch.

        :param augment (optional): defult is False, when True the dataset is flipped horizontally

        :param predict (optional): defult is False, when True only the images are returned , When False, Images and its masks are returned

        :param ytype : List of possible values=['box','lbl', 'both']. box - returns only the boundarybox as ylable, lbl - returns only the classification lable, both - returns both.

        """

        self.folder = img_dir

        self.filenames = os.listdir(img_dir)

        self.batch_size = batch_size

        self.image_size = image_size

        self.shuffle = shuffle

        self.augment = augment

        self.predict = predict

        self.ytype = ytype

        self.on_epoch_end()

        # read the label Label file which contains the bounding boxes

        self.df_data = pd.read_csv(label_file)

        # set Nan values to 0

        self.df_data= self.df_data.fillna(0)

        self.ratio = image_size / 1024 # Original size

        

    def __load__(self, filename):

        # load input png image file as numpy array

        img = cv2.imread(os.path.join(self.folder ,filename), 0) 

        # create empty mask

        msk = np.zeros((self.image_size, self.image_size))

        patient = self.df_data[self.df_data.patientId==filename[:-4]]

        #initialize the lable

        lungopacity = [1,0] # Normal

        for i in range(len(patient)):

            #Calculate the bounding box based on the resized image

            x1 = int(self.ratio * patient['x'].iloc[i]  )

            y1 = int(self.ratio * patient['y'].iloc[i] )

            x2 = x1 + int(self.ratio * patient['width'].iloc[i] )

            y2 = y1 + int(self.ratio * patient['height'].iloc[i])

            msk[y1:y2, x1:x2] = 1

            if ((x2 - x1) > 0) & ((y2 - y1) >0):

                lungopacity = [0,1] # update the labels to Pneumonia

        # resize both image and mask

        img = resize(img, (self.image_size, self.image_size, 1), mode='reflect')

        msk = resize(msk, (self.image_size, self.image_size), mode='reflect') > 0.5

        # if augment then horizontal flip half the time

        if self.augment and random.random() > 0.5:

            img = np.fliplr(img)

            msk = np.fliplr(msk)

        # add trailing channel dimension

        #img = np.expand_dims(img, -1)

        #msk = np.expand_dims(msk, -1)

        return img, lungopacity,msk

    

    def __loadpredict__(self, filename):

        # load png image file as numpy array

        img = cv2.imread(os.path.join(self.folder ,filename), 0) 

        # resize image

        img = resize(img, (self.image_size, self.image_size, 1), mode='reflect')

        # add trailing channel dimension

        #img = np.expand_dims(img, -1)

        return img

        

    def __getitem__(self, index):

        # select batches of files

        filenames = self.filenames[index*self.batch_size:(index+1)*self.batch_size]

        

        # predict mode: return images and filenames

        if self.predict:

            # load files

            imgs = [self.__loadpredict__(filename) for filename in filenames]

            # create numpy batch

            imgs = np.array(imgs)

            return imgs#, filenames

        # train mode: return images and masks

        else:

            # load files

            items = [self.__load__( filename) for filename in filenames]

            # unzip images and masks

            imgs, lbls, msks = zip(*items)

            # create numpy batch

            imgs = np.array(imgs)

            msks = np.array(msks)

            lbls = np.array(lbls)

            if self.ytype == 'box':

                return imgs, msks

            elif self.ytype == 'lbl':

                return imgs, lbls

            else:    

                return imgs, [lbls, msks]

            

    def on_epoch_end(self):

        if self.shuffle:

            shuffle(self.filenames)

        

    def __len__(self):

        if self.predict:

            # return everything

            return int(np.ceil(len(self.filenames) / self.batch_size))

        else:

            # return full batches only

            return int(len(self.filenames) / self.batch_size)
# # In the mask, find the bounding box, if there is one then the image consist of Pneumonia infection

# def detectandMask(mask_img, pred = True, ax=None):

#     comp = mask_img[:, :] > 0.5

#     # apply connected components

#     comp = measure.label(comp)

#     regionFound_true = 0

#     # apply bounding boxes

#     for region in measure.regionprops(comp):

#         color = 'b'

#         if (pred == True):

#             color = 'r'

#             y, x, y2, x2 = region.bbox

#         else:

#             y, x,_, y2, x2 ,_= region.bbox

            

#         height = y2 - y

#         width = x2 - x

#         if ax != None:

#             ax.add_patch(patches.Rectangle((x,y),width,height,linewidth=2,edgecolor=color,facecolor='none'))

#         #Ignore small patches

#         if (width > 10) & (height > 10):

#             regionFound_true = 1



#     return regionFound_true
def detectandMask(mask_img, pred = True, ax=None):

    comp = mask_img[:, :] > 0.5

    # apply connected components

    comp = measure.label(comp)

    regionFound_true = 0

    # apply bounding boxes

    for region in measure.regionprops(comp):

        color = 'b'

        if (pred == True):

            color = 'r'

        try:

            y, x, y2, x2 = region.bbox

            

        except:

            y, x,_, y2, x2 ,_= region.bbox

            

        height = y2 - y

        width = x2 - x

        if ax != None:

            ax.add_patch(patches.Rectangle((x,y),width,height,linewidth=2,edgecolor=color,facecolor='none'))

        #Ignore small patches

        if (width > 10) & (height > 10):

            regionFound_true = 1



    return regionFound_true
# Define Dice coefficient

def dice_coefficient(y_true, y_pred):

    numerator = 2.0 *  tf.reduce_sum(y_true * y_pred) 

    denominator = tf.reduce_sum(y_true + y_pred)

    return numerator / (denominator + tf.keras.backend.epsilon() ) 



# Combine Bce loss and dice coefficient

def loss(y_true, y_pred):

    y_true = tf.dtypes.cast(y_true, tf.float32)

    y_pred = tf.dtypes.cast(y_pred, tf.float32)

    return binary_crossentropy(y_true, y_pred) - log(dice_coefficient(y_true, y_pred) + epsilon())



# Iou loss

def iou_loss(y_true, y_pred):

    y_true = tf.dtypes.cast(tf.reshape(y_true, [-1]), tf.float32)

    y_pred = tf.dtypes.cast(tf.reshape(y_pred, [-1]), tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred)

    score = (intersection + 1.) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection + 1.)

    return 1 - score



# combine bce loss and iou loss

def iou_bce_loss(y_true, y_pred):

    return 0.5 * tf.keras.losses.binary_crossentropy(y_true, y_pred) + 0.5 * iou_loss(y_true, y_pred)



# define mean iou 

def mean_iou(y_true, y_pred):

    y_pred = tf.round(y_pred)

    intersect = tf.reduce_sum(y_true * y_pred, axis=[1, 2])

    union = tf.reduce_sum(y_true, axis=[1, 2]) + tf.reduce_sum(y_pred, axis=[1, 2])

    smooth = tf.ones(tf.shape(intersect))

    return tf.reduce_mean((intersect + smooth) / (union - intersect + smooth))



# Pixel accuracy

accu = tf.keras.metrics.BinaryAccuracy(

    name="binary_accuracy", dtype=None, threshold=0.5

)



filepath = "epoch_{epoch:02d}-val_loss_{val_loss:.2f}.h5" #"model.h5"

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

stop = EarlyStopping(monitor="val_accuracy", patience=5, mode="max")

reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2, verbose=1, mode='max', min_lr=0.00001)

callbacks_list = [checkpoint , reduce_lr, stop]
# create Image generator for train, validation and test images

label_path = os.path.join(root_dir, input_dir, label_file)

label_path
No_OfFiles = 100

train_gen = generator(train_dir, label_path[0:500], batch_size=32, image_size=IMAGE_HEIGHT, shuffle=True, augment=True, predict=False)

val_gen = generator(val_dir,label_path[0:100],  batch_size=32, image_size=IMAGE_HEIGHT, shuffle=False, predict=False)

test_gen = generator(test_dir, label_path[0:100], batch_size=32, image_size=IMAGE_HEIGHT, shuffle=False, predict=True)
def createUnetModel():

    input_l = Input((IMAGE_HEIGHT, IMAGE_WIDTH, 1))

    layer0 =Conv2D(16, (3, 3), activation='relu', padding="same")(input_l)

    b0 = BatchNormalization()(layer0)

    mp0 = MaxPooling2D(pool_size=(2, 2))(b0)



    layer1 =Conv2D(32, (3, 3), activation='relu', padding="same")(mp0)

    b1 = BatchNormalization()(layer1)

    mp1 = MaxPooling2D(pool_size=(2, 2))(b1)



    layer2 = Conv2D(64, (3, 3), activation='relu', padding="same")(mp1)

    b2 = BatchNormalization()(layer2)

    mp2 = MaxPooling2D(pool_size=(2, 2))(b2)



    layer3 = Conv2D(96, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same")(mp2)

    b3 = BatchNormalization()(layer3)

    mp3 = MaxPooling2D(pool_size=(2, 2))(b3)



    layer4 = Conv2D(128, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same")(mp3)

    b4 = BatchNormalization()(layer4)

    mp4 = MaxPooling2D(pool_size=(2, 2))(b4)    

    

    layer5 = Conv2D(256, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same")(mp4)

    

    up4 = Conv2D(128, (3, 3), dilation_rate=(2, 2), activation = 'relu', padding = 'same')(UpSampling2D()(layer5))

    merge4 = Concatenate()([layer4,up4])

    

    up3 = Conv2D(96, (3, 3), dilation_rate=(2, 2), activation = 'relu', padding = 'same')(UpSampling2D()(merge4))

    merge3 = Concatenate()([layer3,up3])

    

    up2 = Conv2D(64, (3, 3),  activation = 'relu', padding = 'same')(UpSampling2D()(merge3))

    merge2 = Concatenate()([layer2,up2])

    

    up1= Conv2D(32, (3, 3), activation = 'relu', padding = 'same')(UpSampling2D()(merge2))

    merge1 = Concatenate()([layer1,up1])

    

    up0 = Conv2D(16, (3, 3), activation = 'relu', padding = 'same')(UpSampling2D()(merge1))

    merge0 = Concatenate()([layer0,up0])

   

    conv = Conv2D(1, kernel_size=1, activation='sigmoid') (merge0)

    

    outputs = Reshape((IMAGE_HEIGHT, IMAGE_WIDTH))(conv)

    

    return  Model(inputs=[input_l], outputs=[outputs]) 

unet_model = createUnetModel()

unet_model.summary()



unet_model.compile(optimizer='adam',

              loss=iou_bce_loss,

              metrics=[accu, mean_iou])
if not loadweightsDirectlyToModel:

    numberofEpochs=1,

    unet_History = unet_model.fit(train_gen,

                           validation_data=val_gen,

                           epochs=5,

                           verbose=1,

              callbacks=callbacks_list

    )
if not loadweightsDirectlyToModel:

    print('List of metrics available: ', unet_model.metrics_names)
#plotGraphs(unet_History) #unet_model.history)

if not loadweightsDirectlyToModel:

    plt.figure(figsize=(12,4))

    plt.subplot(131)

    history = unet_model.history



    plt.plot(history.epoch, history.history["loss"], label="Train loss")

    plt.plot(history.epoch, history.history["val_loss"], label="Valid loss")

    plt.xlabel('Epochs')

    plt.ylabel('Loss')

    plt.legend()



    plt.subplot(132)

    plt.plot(history.epoch, history.history["mean_iou"], label="Mean IOU")

    plt.plot(history.epoch, history.history["val_mean_iou"], label="Mean IOU")

    plt.legend()



    plt.subplot(133)

    plt.plot(history.epoch, history.history["binary_accuracy"], label="Accuracy")

    plt.plot(history.epoch, history.history["val_binary_accuracy"], label="Valid Accuracy")

    plt.legend()



    plt.show()
displayImagesWithBoundingBox(unet_model)
unet_accuracy_score, unet_precision_score,unet_recall_score,unet_f1_score=  displayMetrics('Unet Model',unet_model)
# No_OfFiles = 100

# train_gen = generatorWithBoxType(train_dir, label_path, batch_size=32, image_size=IMAGE_HEIGHT, shuffle=True, augment=True, predict=False, ytype='box')

# val_gen = generatorWithBoxType(val_dir,label_path,  batch_size=32, image_size=IMAGE_HEIGHT, shuffle=False, predict=False,ytype='box')

# test_gen = generatorWithBoxType(test_dir, label_path, batch_size=32, image_size=IMAGE_HEIGHT, shuffle=False, predict=True,ytype='box')
# Create the following block of layers

# Convolution layer with given kernal size and filter. 

# If requested adds Batch norm layer 

# ReLu activation layer

# These convolution locks ared added 'ndeeplayer' times to create a deepConvolution layer.

def convolutionBlock(input_tensor,ndeeplayer, nfilters, kernelsize=3, batchnorm=True):

    layer = input_tensor

    

    for n in np.arange(ndeeplayer):

        # first layer

        layer = Conv2D(filters=nfilters, kernel_size=(kernelsize, kernelsize), kernel_initializer="he_normal",

                   padding="same")(layer)

        if batchnorm:

            layer = BatchNormalization()(layer)

        layer = Activation("relu")(layer)

    

    return layer


from tensorflow.keras.layers import  Activation



#This function creates Unet model with deep convolution layer.

def createDeepUNETModel():

    input_img = Input((IMAGE_HEIGHT, IMAGE_WIDTH, 1), name = 'input_1')

    ndeeplayer=8

    nfilters=16

    dropout=0.5

    batchnorm=True

    # contracting path

    layer0 = convolutionBlock(input_img, ndeeplayer=ndeeplayer, nfilters=16, kernelsize=3, batchnorm=batchnorm)

    tmplayer = MaxPooling2D((2, 2)) (layer0)

    tmplayer = Dropout(dropout*0.5)(tmplayer)

    

    layer1 = convolutionBlock(tmplayer, ndeeplayer=ndeeplayer, nfilters=32, kernelsize=3, batchnorm=batchnorm)

    tmplayer = MaxPooling2D((2, 2)) (layer1)

    tmplayer = Dropout(dropout*0.5)(tmplayer)



    layer2 = convolutionBlock(tmplayer, ndeeplayer=ndeeplayer, nfilters=64, kernelsize=3, batchnorm=batchnorm)

    tmplayer = MaxPooling2D((2, 2)) (layer2)

    tmplayer = Dropout(dropout)(tmplayer)



    layer3 = convolutionBlock(tmplayer,ndeeplayer=ndeeplayer, nfilters=96, kernelsize=3, batchnorm=batchnorm)

    tmplayer = MaxPooling2D((2, 2)) (layer3)

    tmplayer = Dropout(dropout)(tmplayer)



    layer4 = convolutionBlock(tmplayer,ndeeplayer=ndeeplayer, nfilters=128, kernelsize=3, batchnorm=batchnorm)

    tmplayer = MaxPooling2D(pool_size=(2, 2)) (layer4)

    tmplayer = Dropout(dropout)(tmplayer)

    

    layer5 = convolutionBlock(tmplayer, ndeeplayer= ndeeplayer, nfilters=256, kernelsize=3, batchnorm=batchnorm)

    

    # expansive path

    tmplayer = Conv2D(128, (3, 3), dilation_rate=(2, 2), activation = 'relu', padding = 'same')(UpSampling2D()(layer5))

    tmplayer = Concatenate()([layer4,tmplayer])

    tmplayer = Dropout(dropout)(tmplayer)

    tmplayer = convolutionBlock(tmplayer, ndeeplayer=ndeeplayer, nfilters=128, kernelsize=3, batchnorm=batchnorm)



    tmplayer = Conv2D(96, (3, 3), dilation_rate=(2, 2), activation = 'relu', padding = 'same')(UpSampling2D()(tmplayer))

    tmplayer = Concatenate()([layer3,tmplayer])

    tmplayer = Dropout(dropout)(tmplayer)

    tmplayer = convolutionBlock(tmplayer, ndeeplayer=ndeeplayer, nfilters=96, kernelsize=3, batchnorm=batchnorm)



    tmplayer = Conv2D(64, (3, 3),  activation = 'relu', padding = 'same')(UpSampling2D()(tmplayer))

    tmplayer = Concatenate()([layer2,tmplayer])

    tmplayer = Dropout(dropout)(tmplayer)

    tmplayer = convolutionBlock(tmplayer, ndeeplayer=ndeeplayer, nfilters=64, kernelsize=3, batchnorm=batchnorm)



    

    tmplayer= Conv2D(32, (3, 3), activation = 'relu', padding = 'same')(UpSampling2D()(tmplayer))

    tmplayer = Concatenate()([layer1,tmplayer])

    tmplayer = Dropout(dropout)(tmplayer)

    tmplayer = convolutionBlock(tmplayer, ndeeplayer=ndeeplayer, nfilters=32, kernelsize=3, batchnorm=batchnorm)



    

    tmplayer = Conv2D(16, (3, 3), activation = 'relu', padding = 'same')(UpSampling2D()(tmplayer))

    tmplayer = Concatenate(name ='concat_last')([layer0,tmplayer])

    tmplayer = Dropout(dropout)(tmplayer)

    tmplayer = convolutionBlock(tmplayer, ndeeplayer=ndeeplayer, nfilters=16, kernelsize=3, batchnorm=batchnorm)



    outputs = Conv2D(1, (1, 1), activation='sigmoid') (tmplayer)

    model = Model(inputs=[input_img], outputs=[outputs])

    return model
deepunet_model = createDeepUNETModel()

deepunet_model.summary()

deepunet_model.compile(optimizer='adam',

           loss=iou_loss,

           metrics=['accuracy', mean_iou])
for layer in deepunet_model.layers:

    layer.trainable = False
if not loadweightsDirectlyToModel:

    numberofEpochs=1,

    history = deepunet_model.fit(train_gen,

                           validation_data=val_gen,

                           epochs=5,

                           verbose=1,

              callbacks=callbacks_list

    )
fileName = 'deepunet_model.h5'

saveOrLoadWeights(deepunet_model,fileName)
if not loadweightsDirectlyToModel:

    print('List of metrics available: ', deepunet_model.metrics_names)
plotGraphs(deepunet_model)
#mask_pred = deepunetmodel.predict(test_gen, steps=300)
displayImagesWithBoundingBox(deepunet_model)
deepunet_accuracy_score, deepunet_precision_score,deepunet_recall_score,deepunet_f1_score=  displayMetrics('Deep UNet Model',deepunet_model)
from tensorflow import keras

def create_downsample(channels, inputs):

    x = keras.layers.BatchNormalization(momentum=0.9)(inputs)

    x = keras.layers.LeakyReLU(0)(x)

    x = keras.layers.Conv2D(channels, 1, padding='same', use_bias=False)(x)

    x = keras.layers.MaxPool2D(2)(x)

    return x





def create_resblock(channels, inputs):

    x = keras.layers.BatchNormalization(momentum=0.9)(inputs)

    x = keras.layers.LeakyReLU(0)(x)

    x = keras.layers.Conv2D(channels, 3, padding='same', use_bias=False)(x)

    x = keras.layers.BatchNormalization(momentum=0.9)(x)

    x = keras.layers.LeakyReLU(0)(x)

    x = keras.layers.Conv2D(channels, 3, padding='same', use_bias=False)(x)

    return keras.layers.add([x, inputs])



def create_network(input_size, channels, n_blocks=2, depth=4):

    # input

    inputs = keras.Input(shape=(input_size, input_size, 1))

    x = keras.layers.Conv2D(channels, 3, padding='same', use_bias=False)(inputs)

    # residual blocks

    for d in range(depth):

        channels = channels * 2

        x = create_downsample(channels, x)

        for b in range(n_blocks):

            x = create_resblock(channels, x)

    # output

    x = keras.layers.BatchNormalization(momentum=0.9)(x)

    x = keras.layers.LeakyReLU(0)(x)

    x = keras.layers.Conv2D(1, 1, activation='sigmoid')(x)

    outputs = keras.layers.UpSampling2D(2**depth)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

cnn_model = create_network(input_size=128, channels=32, n_blocks=2, depth=4)

cnn_model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy', mean_iou])



print("model summary:", cnn_model.summary())
if not loadweightsDirectlyToModel:

#fit the model for 5 epoch.

    cnn_history = cnn_model.fit(train_gen,

                           validation_data=val_gen,

                           epochs=5,

                           verbose=1,

              callbacks=callbacks_list

    )
fileName = "cnn_model.h5"

saveOrLoadWeights(cnn_model,fileName)
plotGraphs(cnn_model)
displayImagesWithBoundingBox(cnn_model)
cnn_accuracy_score, cnn_precision_score,cnn_recall_score,cnn_f1_score= displayMetrics('CNN Model',cnn_model)
from tensorflow.keras.layers import add

from tensorflow.keras.layers import LeakyReLU, SpatialDropout2D, MaxPool2D

from tensorflow.keras import regularizers

from tensorflow.keras.layers import Conv2DTranspose, Activation



def create_downsample(channels, inputs):

    layer = BatchNormalization(momentum=0.9)(inputs)

    layer = LeakyReLU(0)(layer)

    layer = Conv2D(channels, 1, padding='same', use_bias=False)(layer)

    layer = MaxPool2D(2)(layer)

    return layer



def create_resblock(channels, inputs):

    layer = BatchNormalization(momentum=0.9)(inputs)

    layer = LeakyReLU(0)(layer)

    layer = Conv2D(channels, 3, padding='same', use_bias=False,activity_regularizer=regularizers.l2(0.001))(layer)

    layer = BatchNormalization(momentum=0.9)(layer)

    layer = LeakyReLU(0)(layer)

    layer = Conv2D(channels, 3, padding='same', use_bias=False,activity_regularizer=regularizers.l2(0.001))(layer)

    return add([layer, inputs])



def createResnetModel():

    # input

    channels= 32

    depth = 4

    n_blocks = 2

    # Input layer

    inputs = Input(shape=(224, 224, 1))

    layer = Conv2D(channels, 3, padding='same', use_bias=False)(inputs)

    # residual blocks

    for d in range(depth):

        channels = channels * 2

        layer = create_downsample(channels, layer)

        for b in range(n_blocks):

            layer = create_resblock(channels, layer)

    # output

    layer = BatchNormalization(momentum=0.9)(layer)

    layer = LeakyReLU(0)(layer)

    layer = Conv2D(256, 1, activation=None)(layer)

    layer = SpatialDropout2D(0.25)(layer)

    layer = BatchNormalization(momentum=0.9)(layer)

    layer = LeakyReLU(0)(layer)

    layer = Conv2DTranspose(128, (8,8), (4,4), padding="same", activation=None)(layer)

    layer = SpatialDropout2D(0.25)(layer)

    layer = BatchNormalization(momentum=0.9)(layer)

    layer = LeakyReLU(0)(layer)

    layer = Conv2D(1, 1, activation='sigmoid')(layer)

    outputs = UpSampling2D(2**(depth-2))(layer)

    model = Model(inputs=inputs, outputs=outputs)

    model.summary()

    return model



resnet_model = createResnetModel()

resnet_model.summary()
resnet_model.compile(optimizer='adam',

              loss=loss,

              metrics=['accuracy', mean_iou])

if not loadweightsDirectlyToModel:

#fit the model for 5 epoch.

    resnet_history = resnet_model.fit(train_gen,

                           validation_data=val_gen,

                           epochs=5,

                           verbose=1,

              callbacks=callbacks_list

    )
fileName = "resnet_model.h5"

saveOrLoadWeights(resnet_model,fileName)
plotGraphs(resnet_history)
displayImagesWithBoundingBox(resnet_model)


resnet_accuracy_score, resnet_precision_score,resnet_recall_score,resnet_f1_score=  displayMetrics('Resnet Model',resnet_model)
print(f"Resnet: accuracy: {resnet_accuracy_score}, precision: {resnet_precision_score}, recall: {resnet_recall_score},f1: {resnet_f1_score}")

print(f"CNN: accuracy: {cnn_accuracy_score}, precision: {cnn_precision_score}, recall: {cnn_recall_score},f1: {cnn_f1_score}")



print(f"Deep UNet: accuracy: {deepunet_accuracy_score}, precision: {deepunet_precision_score}, recall: {deepunet_recall_score},f1: {deepunet_f1_score}")



print(f"Unet: accuracy: {unet_accuracy_score}, precision: {unet_precision_score}, recall: {unet_recall_score},f1: {unet_f1_score}")
