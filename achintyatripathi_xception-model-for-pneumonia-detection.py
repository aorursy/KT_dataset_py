# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import cv2

from glob import glob

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import matplotlib.pyplot as plt 

import seaborn as sns

from sklearn.model_selection import train_test_split

import tensorflow as tf 

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications.xception import Xception

from tensorflow.keras.optimizers import RMSprop,Adam



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_folder = '../input/chest-xray-pneumonia/chest_xray/train'

test_folder = '../input/chest-xray-pneumonia/chest_xray/test'

val_folder = '../input/chest-xray-pneumonia/chest_xray/val'
print('Number of images in training set = ',str(len(glob(train_folder+'/*/*'))))

print('Number of images in validation set = ',str(len(glob(val_folder+'/*/*'))))

print('Number of images in testing set = ',str(len(glob(test_folder+'/*/*'))))
train_f = tf.io.gfile.glob(train_folder+'/*/*')

validation_f = tf.io.gfile.glob(val_folder+'/*/*')



total_files = train_f 

total_files.extend(validation_f)



print("Total no. of images we have [train_f + validation_f] = {}".format(len(total_files)))
## So now spliting this data into 80:20 



train_images,val_images = train_test_split(total_files,test_size =0.2)

print("Train_images = {}".format(len(train_images)))

print("Validation_images = {}".format(len(val_images)))
tf.io.gfile.makedirs('/kaggle/working/val_dataset/NORMAL/')

tf.io.gfile.makedirs('/kaggle/working/val_dataset/PNEUMONIA/')

tf.io.gfile.makedirs('/kaggle/working/train_dataset/NORMAL/')

tf.io.gfile.makedirs('/kaggle/working/train_dataset/PNEUMONIA/')
for element in train_images:

    parts_of_path = element.split('/')



    if 'PNEUMONIA' == parts_of_path[-2]:

        tf.io.gfile.copy(src = element, dst = '/kaggle/working/train_dataset/PNEUMONIA/' +  parts_of_path[-1])

    else:

        tf.io.gfile.copy(src = element, dst = '/kaggle/working/train_dataset/NORMAL/' +  parts_of_path[-1])
for element in val_images:

    parts_of_path = element.split('/')



    if 'PNEUMONIA' == parts_of_path[-2]:

        tf.io.gfile.copy(src = element, dst = '/kaggle/working/val_dataset/PNEUMONIA/' +  parts_of_path[-1])

    else:

        tf.io.gfile.copy(src = element, dst = '/kaggle/working/val_dataset/NORMAL/' +  parts_of_path[-1])
print('Pneumonia x-ray images in training set after split = ',len(os.listdir('/kaggle/working/train_dataset/PNEUMONIA/')))

print('Normal x-ray images in training set after split = ',len(os.listdir('/kaggle/working/train_dataset/NORMAL/')))

print('Pneumonia x-ray images in validation set after split = ',len(os.listdir('/kaggle/working/val_dataset/PNEUMONIA/')))

print('Normal x-ray images in validation set after split = ',len(os.listdir('/kaggle/working/val_dataset/NORMAL/')))

print('Pneumonia x-ray images in test set = ',len(os.listdir('../input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA/')))

print('Normal x-ray images in test set = ',len(os.listdir('../input/chest-xray-pneumonia/chest_xray/test/NORMAL')))
dataset_info = {'Dataset': ['train_dataset','val_dataset','test_dataset'],

        'Pneumonia': [3107,776,390],

        'Normal' : [1078,271,234]

        }



df = pd.DataFrame(dataset_info, columns = ['Dataset', 'Pneumonia','Normal'])



print (df)
train_dir='/kaggle/working/train_dataset/'

val_dir='/kaggle/working/val_dataset/'

test_dir='../input/chest-xray-pneumonia/chest_xray/test/'







train_normal_dir='/kaggle/working/train_dataset/NORMAL/'

train_pneumonia_dir='/kaggle/working/train_dataset/PNEUMONIA/'

val_normal_dir='/kaggle/working/val_dataset/NORMAL'

val_pneumonia_dir='/kaggle/working/val_dataset/PNEUMONIA'





train_normal_fnames=os.listdir(train_normal_dir)

train_pneumonia_fnames=os.listdir(train_pneumonia_dir)
%matplotlib inline 



import matplotlib.pyplot as plt 

import matplotlib.image as mpimg



ncols = 4

nrows = 2



pic_index = 0
fig = plt.gcf()

fig.set_size_inches(4*ncols,4*nrows)



pic_index+=4



normal = [os.path.join(train_normal_dir,fname) for fname in train_normal_fnames[pic_index-4:pic_index]]

pneumonia = [os.path.join(train_pneumonia_dir,fname) for fname in train_pneumonia_fnames[pic_index-4:pic_index]]



for i,img in enumerate(pneumonia+normal):

    ax = plt.subplot(nrows,ncols,i+1)

    ax.axis()

    

    imgs = mpimg.imread(img)

    plt.imshow(imgs,cmap='gray')

    if(i<4):

        plt.title('Pneumonia')

    else:

        plt.title('Normal')

plt.show()
train_datagen2=ImageDataGenerator(rescale=1.0/255,

                                 rotation_range=30,

                                 width_shift_range=0.2,

                                 height_shift_range=0.2,

                                 zoom_range=0.2,

                                 )



val_datagen2=ImageDataGenerator(rescale=1.0/255)



test_datagen2=ImageDataGenerator(rescale=1.0/255)



train_generator2=train_datagen2.flow_from_directory(train_dir,target_size=(180,180),batch_size=128,class_mode='binary')



val_generator2=val_datagen2.flow_from_directory(val_dir,target_size=(180,180),batch_size=128,class_mode='binary')



test_generator2=test_datagen2.flow_from_directory(test_dir,target_size=(180,180),batch_size=128,class_mode='binary')
#load pre trained Xception model



model = Xception(weights= None, include_top=False, input_shape= (180,180,3))



#freazing the trained layers

for layers in model.layers:

    layers.trainable = False



model.summary()
last_layer=model.get_layer('block14_sepconv2_act')

last_output = last_layer.output



x=tf.keras.layers.Flatten()(last_output)

x=tf.keras.layers.Dense(1024,activation='relu')(x)

x=tf.keras.layers.Dropout(0.2)(x)

x=tf.keras.layers.Dense(256,activation='relu')(x)

x=tf.keras.layers.Dropout(0.2)(x)

x=tf.keras.layers.Dense(1,activation='sigmoid')(x)



model=tf.keras.Model(model.input,x)



model.compile(optimizer=RMSprop(lr=0.001),

              loss='binary_crossentropy',

               metrics=['accuracy','Precision','Recall'])



model.summary()
history = model.fit(train_generator2,validation_data=val_generator2,epochs=10,steps_per_epoch=5,verbose=2)
acc2 = history.history['accuracy']

val_acc2 = history.history['val_accuracy']



train_precision2=history.history['precision']

val_precision2=history.history['val_precision']



train_recall2=history.history['recall']

val_recall2=history.history['val_recall']



loss2 = history.history['loss']

val_loss2 = history.history['val_loss']

epochs = range(len(acc2))



plt.plot(epochs, acc2, 'r', label='Training accuracy')

plt.plot(epochs, val_acc2, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend()

plt.show()



plt.plot(epochs, train_precision2, 'r', label='Training precision')

plt.plot(epochs, val_precision2, 'b', label='Validation precision')

plt.title('Training and validation precision')

plt.legend()

plt.show()



plt.plot(epochs, train_recall2, 'r', label='Training recall')

plt.plot(epochs, val_recall2, 'b', label='Validation recall')

plt.title('Training and validation recall')

plt.legend()

plt.show()



plt.plot(epochs, loss2, 'r', label='Training Loss')

plt.plot(epochs, val_loss2, 'b', label='Validation Loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()
eval_result3 = model.evaluate_generator(test_generator2, 624)

print('loss  :', eval_result3[0])

print('accuracy  :', eval_result3[1])

print('Precision :', eval_result3[2])

print('Recall :', eval_result3[3])