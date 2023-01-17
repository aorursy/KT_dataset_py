# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import os

import glob

import h5py

import shutil

import imgaug as aug

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.image as mimg

import imgaug.augmenters as iaa

from os import listdir, makedirs, getcwd, remove

from os.path import isfile, join, abspath, exists, isdir, expanduser

from PIL import Image

from pathlib import Path

from skimage.io import imread

from skimage.transform import resize

from keras.models import Sequential, Model

from keras.applications.vgg16 import VGG16, preprocess_input

from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, SeparableConv2D

from keras.layers import GlobalMaxPooling2D

from keras.layers.normalization import BatchNormalization

from keras.layers.merge import Concatenate

from keras.models import Model

from keras.optimizers import Adam, SGD, RMSprop

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from mlxtend.plotting import plot_confusion_matrix

from sklearn.metrics import confusion_matrix

import cv2

from keras import backend as K

from matplotlib import pylab

color = sns.color_palette()

%matplotlib inline

%pylab inline

# 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

print(os.listdir("../input"))
from keras.regularizers import l2
import tensorflow as tf

os.environ['PYTHONHASHSEED']='0'

np.random.seed(111)

session_conf =  tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)

tf.random.set_seed(111)

sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(),config=session_conf)

# K.set_session(sess)

tf.compat.v1.keras.backend.get_session(sess)

aug.seed(111)
data_dir = Path("../input/chest-xray-pneumonia/chest_xray")

train_dir = data_dir / 'train'

val_dir = data_dir / 'val'

test_dir = data_dir / 'test'

train_dir,val_dir,test_dir

#get path to sub directories

normal_cases_dir = train_dir / 'NORMAL'

pneumonia_cases_dir = train_dir / 'PNEUMONIA'



#get list of images

normal_cases = normal_cases_dir.glob('*.jpeg')

pneumonia_cases = pneumonia_cases_dir.glob('*.jpeg')



#now an empty list is taken to get all data



train_data =[]



for img in normal_cases:

    train_data.append((img,0))



for img in pneumonia_cases:

    train_data.append((img,1))

    

#make a data frame

train_data = pd.DataFrame(train_data,columns=['image','label'],index=None)

#shuffle image

train_data = train_data.sample(frac=1).reset_index(drop=True)

train_data.head()
%pylab inline

#use for figsize

cases_count = train_data['label'].value_counts()

print("Total cases are -")

print(cases_count)



#ploting the result

plt.figure(figsize=(10,8))

sns.barplot(x=cases_count.index,y=cases_count.values)

plt.title("Number of cases",fontsize=14)

plt.xlabel("Case Type",fontsize=12)

plt.ylabel("Count",fontsize=12)

plt.xticks(range(len(cases_count.index)),['Normal(0)','Pneumonia(1)'])

plt.show()
#trying to different

#get path to sub directories

normal_cases_dir = train_dir / 'NORMAL'

pneumonia_cases_dir = train_dir / 'PNEUMONIA'



#get list of images

normal_cases = normal_cases_dir.glob('*.jpeg')

pneumonia_cases = pneumonia_cases_dir.glob('*.jpeg')



#now an empty list is taken to get all data



train_data =[]



for img in normal_cases:

    train_data.append((img,0))

i=0

for img in pneumonia_cases:

    if i == 2300:

        break

    train_data.append((img,1))

    i=i+1

    



#make a data frame

train_data = pd.DataFrame(train_data,columns=['image','label'],index=None)

#shuffle image

train_data = train_data.sample(frac=1).reset_index(drop=True)

print(train_data.head())

%pylab inline

#use for figsize

cases_count = train_data['label'].value_counts()

print("Total cases are -")

print(cases_count)



#ploting the result

plt.figure(figsize=(10,8))

sns.barplot(x=cases_count.index,y=cases_count.values)

plt.title("Number of cases",fontsize=14)

plt.xlabel("Case Type",fontsize=12)

plt.ylabel("Count",fontsize=12)

plt.xticks(range(len(cases_count.index)),['Normal(0)','Pneumonia(1)'])

plt.show()

#get some samples

pneumonia_samples = (train_data[train_data['label']==1]['image'].iloc[:5]).tolist()

normal_samples = (train_data[train_data['label']==0]['image'].iloc[:5]).tolist()



#concat the data into a single data list del the above two list



samples = pneumonia_samples + normal_samples

del pneumonia_samples,normal_samples
#plot the data

f , ax = plt.subplots(2,5,figsize=(30,10))

for i in range(10):

    img = imread(samples[i])

    ax[i//5, i%5].imshow(img,cmap='gray')

    

    if i<5:

        ax[i//5,i%5].set_title("Pneumonia")

    else:

        ax[i//5,i%5].set_title("Normal")

        

    ax[i//5,i%5].axis("off")

    ax[i//5,i%5].set_aspect("auto")

    

plt.show()

    
normal_cases_dir = val_dir/'NORMAL'

pneumonia_cases_dir = val_dir/'PNEUMONIA'



normal_cases = normal_cases_dir.glob("*.jpeg")

pneumonia_cases = pneumonia_cases_dir.glob("*.jpeg")



valid_data =[]

valid_labels =[]





#most of the image in RGB but some are in greyscale. So grayscale image will be converted in RGB and get Normalize

#normal cases

for img in normal_cases:

    img = cv2.imread(str(img))

    img = cv2.resize(img,(224,224))

    

    if img.shape[2]==1:

        img = np.dstack([img,img,img])

        

    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32)/255.0

    label = to_categorical(0,num_classes=2)

    valid_data.append(img)

    valid_labels.append(label)



#pneumonia cases

for img in pneumonia_cases:

    img = cv2.imread(str(img))

    img = cv2.resize(img,(224,224))

    

    if img.shape[2]==1:

        img = np.dstack([img,img,img])

    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32)/255.0

    label = to_categorical(1,num_classes=2)

    valid_data.append(img)

    valid_labels.append(label)

    



#convert the list into numpy array

valid_data = np.array(valid_data)

valid_labels = np.array(valid_labels)



print("Total number of validation data : ",valid_data.shape)

print("Total number of validation labels : ",valid_labels.shape)

#here imgaug is used

seq = iaa.OneOf([

    iaa.Fliplr(), #horizontal

    iaa.Affine(rotate=20), #rotation

    iaa.Multiply((1.2,1.5)) #random brightness

])
def data_gen(data,batch_size):

    #find total number of samples

    n = len(data)

    steps = n//batch_size

    

    #make batch data and label

    

    batch_data = np.zeros((batch_size,224,224,3),dtype=np.float32)

    batch_labels= np.zeros((batch_size,2),dtype=np.float32)

    

    #get all indices of the input data

    indices = np.arange(n)

    

    i=0

    while True:

        np.random.shuffle(indices)

        count =0

        next_batch = indices[(i*batch_size):(i+1)*batch_size]

        

        for j ,idx in enumerate(next_batch):

            img_name = data.iloc[idx]['image']

            label = data.iloc[idx]['label']

            

            #one hot encoding

            encoded_label = to_categorical(label,num_classes=2)

            

            #read the image and resize

            img = cv2.imread(str(img_name))

            img = cv2.resize(img,(224,224))

            

            #check for grayscale

            

            if img.shape[2]==1:

                img = np.dstack([img,img,img])

            #read for RGB default

            

            orig_img = img.astype(np.float32)/255.0

            batch_data[count]=orig_img

            batch_labels[count]= encoded_label

            

            #generating more samples of the undersampled class

            

            if label==0 and count < batch_size-2:

                aug_img1 = seq.augment_image(img)

                aug_img2 = seq.augment_image(img)

                aug_img1 = cv2.cvtColor(aug_img1,cv2.COLOR_BGR2RGB)

                aug_img2 = cv2.cvtColor(aug_img2,cv2.COLOR_BGR2RGB)

                

                aug_img1 = aug_img1.astype(np.float32)/255.0

                aug_img2 = aug_img2.astype(np.float32)/255.0

                

                batch_data[count+1] = aug_img1

                batch_labels[count+1]= encoded_label

                batch_data[count+2]= aug_img2

                batch_labels[count+2]=encoded_label

                count+=2

            else:

                count+=1

            if count==batch_size-1:

                break

            

            i+=1

            yield batch_data,batch_labels

            

            if i>=steps:

                i=0
image_input = Input(shape=(224,224,3))

vgg_mod = VGG16(input_tensor=image_input,include_top=True,weights='imagenet')

vgg_mod.summary()
#editing the last layer for our classes

num_classes=2

last_layer = vgg_mod.get_layer('fc2').output

out = Dense(num_classes,activation='softmax',name='output')(last_layer)

cust_vgg_model = Model(image_input,out)

cust_vgg_model.summary()
opt = Adam(lr=0.0001,decay=1e-5)

es = EarlyStopping(patience=5)

chkpt = ModelCheckpoint(filepath='best_model_todate',save_best_only=True,save_weights_only=True)

cust_vgg_model.compile(loss='binary_crossentropy',metrics=['accuracy'],optimizer=opt)
batch_size = 16

np_epochs =20

#get a train data generator

train_data_gen = data_gen(data=train_data,batch_size=batch_size)

#define num of training steps

nb_train_steps = train_data.shape[0]//batch_size

# print("size of train data",train_data_gen.shape[0])

print("Number of training and validation steps : {} and {}".format(nb_train_steps,len(valid_data)))
history= cust_vgg_model.fit_generator(train_data_gen,

                                      epochs=np_epochs,

                                     steps_per_epoch=nb_train_steps,

                                      validation_data=(valid_data,valid_labels),

                                      callbacks=[es,chkpt],

                                      class_weight={0:1.0,1:0.4}

                                     )
def build_model():

    input_img = Input(shape=(224,224,3), name='ImageInput')

    x = Conv2D(64, (3,3), activation='relu', padding='same', name='Conv1_1')(input_img)

    x = Conv2D(64, (3,3), activation='relu', padding='same', name='Conv1_2')(x)

    x = MaxPooling2D((2,2), name='pool1')(x)

    

    x = SeparableConv2D(128, (3,3), activation='relu', padding='same', name='Conv2_1')(x)

    x = SeparableConv2D(128, (3,3), activation='relu', padding='same', name='Conv2_2')(x)

    x = Dropout(0.5, name='dropout1')(x)

    x = MaxPooling2D((2,2), name='pool2')(x)

    

    x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_1')(x)

    x = BatchNormalization(name='bn1')(x)

    x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_2')(x)

    x = BatchNormalization(name='bn2')(x)

    x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_3')(x)

    x = Dropout(0.7, name='dropout2')(x)

    x = MaxPooling2D((2,2), name='pool3')(x)

    

    x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_1')(x)

    x = BatchNormalization(name='bn3')(x)

    x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_2')(x)

    x = BatchNormalization(name='bn4')(x)

    x = Dropout(0.5, name='dropout4')(x)

    x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_3')(x)

    x = MaxPooling2D((2,2), name='pool4')(x)

    

    x = Flatten(name='flatten')(x)

    x = Dense(1024, activation='relu', name='fc1')(x)

    x = Dropout(0.7, name='dropout5')(x)

#     x = BatchNormalization(name='bn5')(x)    

    x = Dense(512, activation='relu', name='fc2')(x)

    x = Dropout(0.5, name='dropout3')(x)

    x = Dense(2, activation='softmax', name='fc3')(x)

    

    model = Model(inputs=input_img, outputs=x)

    return model
model =  build_model()

model.summary()
regularizer = tf.keras.regularizers.l2(0.2)



for layer in model.layers:

    for attr in ['kernel_regularizer']:

        if hasattr(layer, attr):

          setattr(layer, attr, regularizer)
model.summary()
#initialize vgg 16 weights

# print(os.listdir("../input/vgg-16/"))

f = h5py.File("../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5",'r')

#selecting the layers

w,b = f['block1_conv1']['block1_conv1_W_1:0'], f['block1_conv1']['block1_conv1_b_1:0']

model.layers[1].set_weights = [w,b]



w,b = f['block1_conv2']['block1_conv2_W_1:0'], f['block1_conv2']['block1_conv2_b_1:0']

model.layers[2].set_weights = [w,b]



w,b = f['block2_conv1']['block2_conv1_W_1:0'], f['block2_conv1']['block2_conv1_b_1:0']

model.layers[4].set_weights = [w,b]



w,b = f['block2_conv2']['block2_conv2_W_1:0'], f['block2_conv2']['block2_conv2_b_1:0']

model.layers[5].set_weights = [w,b]

f.close()

opt = Adam(lr=0.001, decay=1e-5)

# es = EarlyStopping(patience=4)

chkpt = ModelCheckpoint(filepath='best_model_todate_two', save_best_only=True, save_weights_only=True)

model.compile(loss='binary_crossentropy', metrics=['accuracy'],optimizer=opt)
from keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,

                              patience=5, min_lr=0.001)
def calculate_class_weights(train_label):

    list = train_label.tolist()



    num_neg = list.count(0)

    num_pos = list.count(1)



    duplicate = num_pos / num_neg



    class_weights={0 : (num_neg * (duplicate)) , 1: num_pos }

    return class_weights

class_weight=calculate_class_weights(train_data['label'])

print(class_weight)

# val_class_weight = calculate_class_weights(valid_labels)
import numpy as np

import math



# labels_dict : {ind_label: count_label}

# mu : parameter to tune 



def create_class_weight(labels_dict,mu=0.15):

    total = np.sum(list(labels_dict.values()))

    keys = labels_dict.keys()

    class_weight = dict()



    for key in keys:

        score = math.log(mu*total/float(labels_dict[key]))

        class_weight[key] = score if score > 1.0 else 1.0



    return class_weight

class_weight2=create_class_weight(class_weight)

print(class_weight2)
batch_size = 16

nb_epochs = 20



# Get a train data generator

train_data_gen = data_gen(data=train_data, batch_size=batch_size)



# Define the number of training steps

nb_train_steps = train_data.shape[0]//batch_size



print("Number of training and validation steps: {} and {}".format(nb_train_steps, len(valid_data)))
history = model.fit_generator(train_data_gen, epochs=nb_epochs, steps_per_epoch=nb_train_steps,

                              validation_data=(valid_data, valid_labels),callbacks=[reduce_lr],

                             class_weight={0:1.0, 1:0.4} )
model.save_weights('new_weights_three.h5')
model.save("try_model")
model.load_weights("../working/new_weights.h5")
normal_cases_dir = test_dir / 'NORMAL'

pneumonia_cases_dir = test_dir / 'PNEUMONIA'



normal_cases = normal_cases_dir.glob('*.jpeg')

pneumonia_cases = pneumonia_cases_dir.glob('*.jpeg')



test_data = []

test_labels = []



for img in normal_cases:

    img = cv2.imread(str(img))

    img = cv2.resize(img, (224,224))

    if img.shape[2] ==1:

        img = np.dstack([img, img, img])

    else:

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32)/255.

    label = to_categorical(0, num_classes=2)

    test_data.append(img)

    test_labels.append(label)

                      

for img in pneumonia_cases:

    img = cv2.imread(str(img))

    img = cv2.resize(img, (224,224))

    if img.shape[2] ==1:

        img = np.dstack([img, img, img])

    else:

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32)/255.

    label = to_categorical(1, num_classes=2)

    test_data.append(img)

    test_labels.append(label)

    



test_data = np.array(test_data)

test_labels = np.array(test_labels)



print("Total number of test examples: ", test_data.shape)

print("Total number of labels:", test_labels.shape)
test_loss, test_score = model.evaluate(test_data, test_labels, batch_size=11)

print("Loss on test set: ", test_loss)

print("Accuracy on test set: ", test_score)
# Get predictions

preds = model.predict(test_data, batch_size=16)

preds = np.argmax(preds, axis=-1)



# Original labels

orig_test_labels = np.argmax(test_labels, axis=-1)



print(orig_test_labels.shape)

print(preds.shape)
preds.shape
cm  = confusion_matrix(orig_test_labels, preds)

plt.figure()

plot_confusion_matrix(cm,figsize=(12,8), hide_ticks=True,cmap=plt.cm.Blues)

plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)

plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)

plt.show()
# Calculate Precision and Recall

tn, fp, fn, tp = cm.ravel()



precision = tp/(tp+fp)

recall = tp/(tp+fn)



print("Recall of the model is {:.2f}".format(recall))

print("Precision of the model is {:.2f}".format(precision))
# normal_image = test_dir + '/NORMAL'

import cv2

test_dir
img_normal = imread("../input/chest-xray-pneumonia/chest_xray/test/NORMAL/IM-0006-0001.jpeg")

imshow(img_normal,cmap='gray')

plt.show()
img_normal.shape
img_normal=np.resize(img_normal,(1,224,224,3))
img_normal.shape
y_pred = model.predict(img_normal)
y_pred = np.argmax(y_pred)

y_pred
#all in one

img_normal = imread("../input/chest-xray-pneumonia/chest_xray/test/NORMAL/IM-0007-0001.jpeg")

imshow(img_normal,cmap='gray')

img_normal=np.resize(img_normal,(1,224,224,3))

y_pred = model.predict(img_normal)

y_pred = np.argmax(y_pred)

plt.title(str(y_pred))



plt.show()
# summarize history for accuracy

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

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