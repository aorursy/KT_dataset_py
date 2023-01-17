import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from random import randint
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

from subprocess import check_output
print(check_output(["ls","../input"]).decode("utf8"))
df_train=pd.read_csv("../input/labels.csv")
df_train.head(10)
ax= pd.value_counts(df_train['breed'],ascending=True).plot(kind='barh',
                                                         fontsize ='20',
                                                         title ="Class Distribution",
                                                         figsize = (30,60))
ax.set(xlabel="Images per class",ylabel ='Classes')
ax.xaxis.label.set_size(20)
ax.yaxis.label.set_size(20)
ax.title.set_size(80)
def plot_images(images, classes):
    assert len(images) == len(classes) ==9
    fig,axes = plt.subplots(3,3,figsize =(60,60),sharex =True)
    fig.subplots_adjust(hspace =0.3, wspace = 0.3)
    
    for i,ax in enumerate(axes.flat):
        ax.imshow(cv2.cvtColor(images[i],cv2.COLOR_BGR2RGB).reshape(img_width,img_height,3),cmap ='hsv')
        xlabel = "BreedL{0}".format(classes[i])
        ax.set_xlabel(xlabel)
        ax.xaxis.label.set_size(60)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
img_width =250
img_height = 250
images=[]
classes =[]

for f,breed in tqdm(df_train.values):
    img = cv2.imread('../input/train/{}.jpg'.format(f))
    classes.append(breed)
    images.append(cv2.resize(img,(img_width,img_height)))
random_numbers = [randint(0,len(images)) for p in range(0,9)]
images_to_show =[images[i] for i in random_numbers]
classes_to_show =[classes[i] for i in random_numbers]
print("Images to show:{0}".format(len(images_to_show)))
print("Classes to show :{0}".format(len(classes_to_show)))

plot_images(images_to_show,classes_to_show)
import numpy as np
import pandas as pd
from random import randint
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

from subprocess import check_output
print(check_output(["ls","../input"]).decode("utf8"))

df_train = pd.read_csv("../input/labels.csv")
df_train.head(10)
ax = pd.value_counts(df_train['breed'],ascending =True).plot(kind ='barh',
                                                            fontsize ='40',
                                                            title="Class Distribution",
                                                            figsize=(50,100))
ax.set(xlabel ="Images per class",ylabel ="Classes")
ax.xaxis.label.set_size(40)
ax.yaxis.label.set_size(40)
ax.title.set_size(60)
def plot_images(images,classes):
    assert len(images) == len(classes) ==9
    fig,axes = plt.subplots(3,3,figsize = (60,60),sharex=True)
    fig.subplots_adjust(hspace=0.3,wspace=0.3)
    
    for i,ax in enumerate(axes.flat):
        ax.imshow(cv2.cvtColor(images[i],cv2.COLOR_BGR2RGB).reshape(img_width,img_height,3),cmap='hsv')
        xlabel ="Breed:{0}".format(classes[i])
        
        #show the classes as the label on the x_axis
        ax.set_xlabel(xlabel)
        ax.xaxis.label.set_size(60)
        #remove ticks from the plot
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
img_width=250
img_height =250
classes =[]
images=[]
for f,breed in tqdm(df_train.values):
    img = cv2.imread('../input/train/{}.jpg'.format(f))
    classes.append(breed)
    images.append(cv2.resize(img,(img_width,img_height)))
random_numbers =[randint(0,len(images)) for p in range(0,9)]
images_to_show =[images[i] for i in random_numbers]
classes_to_show =[classes[i] for i in random_numbers]
print('Images to show:{0}'.format(len(images_to_show)))
print("classes to show:{0}".format(len(classes_to_show)))

plot_images(images_to_show,classes_to_show)
import numpy as np
import pandas as pd
import keras
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.layers import Dense,Dropout,Flatten

import os
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import cv2

from subprocess import check_output
print(check_output(["ls","../input"]).decode("utf8"))
df_train =pd.read_csv('../input/labels.csv')
df_test = pd.read_csv('../input/sample_submission.csv')
df_train.head(10)
# we can see that breed nees to be one-hot encoded for the final submission
targets_series = pd.Series(df_train['breed'])
one_hot = pd.get_dummies(targets_series,sparse= True)

one_hot_labels = np.asarray(one_hot)
# read images for test and tain via for loop in csv files. resise image to 90 X 90px

im_size =90
x_train =[]
y_train =[]
x_test =[]

i =0
for f,breed in tqdm(df_train.values):
    img = cv2.imread('../input/train/{}.jpg'.format(f))
    label = one_hot_labels[i]
    x_train.append(cv2.resize(img,(im_size,im_size)))
    y_train.append(label)
    i +=1
for f in tqdm(df_test['id'].values):
    img = cv2.imread('../input/test/{}.jpg'.format(f))
    x_test.append(cv2.resize(img,(im_size,im_size)))
y_train_raw = np.array(y_train,np.uint8)
x_train_raw = np.array(x_train,np.float32)/255.
x_test = np.array(x_test,np.float32)/255.
#check the shape of the output to make sure everything went as expected
print(x_train_raw.shape)
print(y_train_raw.shape)
print(x_test.shape)
#put 120 breeds in a num_class variables
num_class = y_train_raw.shape[1]
#put validation sets 30% from training set. make sure it can contains images 
#from every class.
X_train,X_valid,Y_train,Y_valid =train_test_split(x_train_raw,
                                                 y_train_raw,
                                                 test_size =0.3,
                                                 random_state =1)
#CNN via VGG19
base_model = VGG19(weights = None, include_top =False, input_shape=(im_size,im_size,3))

#add a new top layer
x= base_model.output
x =Flatten()(x)
predictions =Dense(num_class,activation='softmax')(x)

#This is the model we will train
model =Model(inputs=base_model.input,outputs =predictions)

# First: train only the top layers(which were randomly initialized)
for layer in base_model.layers:
    layer.trainable = False

model.compile(loss='categorical_crossentropy',
             optimizer ='adam',
             metrics=['accuracy'])

callbacks_list =[keras.callbacks.EarlyStopping(monitor ='val_acc',
                                              patience =3,
                                              verbose =1)]
model.summary()
model.fit(X_train,Y_train,epochs =1, validation_data =(X_valid,Y_valid),
         verbose =1)
# accuracy is low here because we are not taking advantage of the pre-trained weights as they cannot be downloaded in the kernel.
#This means we are training the weights from scratch and we have only 1 epoch due to the hardware constrains in the kernel.

#next we will make out predictions.

preds = model.predict(X_test,verbose =1)
sub = pd.DataFrame(preds)
#set column names to those generated by the one_hot encoding earlier
col_names =one_hot.columns.values
sub,columns = col.names
# Insert the column id from the sample_submission at the start of the data frame
sub.insert(0,'id',df_test['id'])
sub.head(5)
from tqdm import tqdm

for i in tqdm(range(1000)):
    pass

for char in tqdm(["a","b","c","d"]):
    pass
from tqdm import trange
for i in trange(100):
    pass

pbar = tqdm(["a","b","c","d"])
for char in pbar:
    print(char)
    pbar.set_description("Processing %s" %char)


