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


import os

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras import Model

from tensorflow.keras import layers

from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3





weights_path = '/kaggle/input/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pretrained_model = InceptionV3(input_shape = (150, 150, 3), 

                                include_top = False, 

                                weights = None)



pretrained_model.load_weights(weights_path)



for layer in pre_trained_model.layers:

     layer.trainable = False


last_layer=pretrained_model.get_layer('mixed7')

print('last layer output shape: ', last_layer.output_shape)

last_output=last_layer.output
x=layers.Flatten()(last_output)

x=layers.Dense(1024,activation='relu')(x)

x=layers.Dropout(0.2)(x)  #20% dropout

x=layers.Dense(6,activation='softmax')(x)

model=Model(pretrained_model.input,x)

model.compile(optimizer = RMSprop(lr=0.0001), 

              loss = 'categorical_crossentropy', 

              metrics =['acc'])
###########  WITHOUT TRANSFER LEARNING



# model=tf.keras.models.Sequential([

#     layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)),

#     layers.MaxPool2D(2,2),

#     layers.Conv2D(32,(3,3),activation='relu'),

#     layers.MaxPooling2D(2,2),

#     layers.Conv2D(64,(3,3),activation='relu'),

#     layers.MaxPooling2D(2,2),

#     layers.Conv2D(128, (3,3), activation='relu'),

#     layers.MaxPooling2D(2,2),

#     layers.Flatten(),

#     layers.Dense(512,activation='relu'),

#     layers.Dense(6,activation='softmax')  

# ])



# model.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['acc'])
TRAINING_DIR = '/kaggle/input/intel-image-classification/seg_train/seg_train/'

train_datagen = ImageDataGenerator(

                  rescale=1/255.0,

                  rotation_range=40,

                  horizontal_flip=True,

                  width_shift_range=0.2,

                  height_shift_range=0.2,

                  shear_range=0.2,

                  zoom_range=0.2,

                  fill_mode='nearest'

)



train_generator = train_datagen.flow_from_directory(

                      TRAINING_DIR,target_size=(150,150),batch_size=10,class_mode='categorical')



VALIDATION_DIR = '/kaggle/input/intel-image-classification/seg_test/seg_test/'

validation_datagen = ImageDataGenerator(rescale=1/255.0)



validation_generator = validation_datagen.flow_from_directory(

                      VALIDATION_DIR,target_size=(150,150),batch_size=10,class_mode='categorical')
history = model.fit_generator(train_generator,

                              epochs=30,

                              steps_per_epoch=10,

                              verbose=1,

                              validation_data=validation_generator,

                             validation_steps=50)
%matplotlib inline



import matplotlib.pyplot as plt

train_acc=history.history['acc']

val_acc=history.history['val_acc']

loss=history.history['loss']

val_loss=history.history['val_loss']



epochs=range(len(history.history['acc']))



plt.plot(epochs,train_acc,'r','Training acc')

plt.plot(epochs,val_acc,'b','Validation acc')

plt.figure()

plt.plot(epochs,loss,'r','Train loss')

plt.plot(epochs,val_loss,'b','Val loss')
%matplotlib inline

import numpy as np

from keras.preprocessing import image

import matplotlib.image as mpimg



files=os.listdir('/kaggle/input/sample/sample/')



path='/kaggle/input/sample/sample/'

for filename in files:

    file=path+filename

    img=image.load_img(file, target_size=(150, 150))

    x=image.img_to_array(img)

    x=x/255

    x=np.expand_dims(x, axis=0)

    images = np.vstack([x])

    classes = model.predict(images, batch_size=10)

    plt.imshow(mpimg.imread(file))

    plt.figure()

    i,j=np.unravel_index(classes.argmax(), classes.shape)

    if j==0:

        print('Building')

    elif j==1:

        print('forest')

    elif j==2:

        print('glacier')    



    elif j==3:

        print('mountain')

    elif j==4:

        print('sea')

    else:

        print('street')    

        

  

#total_images=len(os.listdir('/kaggle/input/intel-image-classification/seg_pred/seg_pred/'))





files=os.listdir('/kaggle/input/intel-image-classification/seg_pred/seg_pred/')[:100]

path='/kaggle/input/intel-image-classification/seg_pred/seg_pred/'

results=[]

for filename in files:

    file=path+filename

    img=image.load_img(file, target_size=(150, 150))

    x=image.img_to_array(img)

    x=x/255

    x=np.expand_dims(x, axis=0)

    images = np.vstack([x])

    classes = model.predict(images, batch_size=10)

    i,j=np.unravel_index(classes.argmax(), classes.shape)

    if j==0:

        classname='Building'

    elif j==1:

        classname='forest'

    elif j==2:

        classname='glacier'    



    elif j==3:

        classname='mountain'

    elif j==4:

        classname='sea'

    else:

        classname='street'

    

    results.append(classname)

    

results=pd.Series(results,name="Label")   

submission = pd.concat([pd.Series(range(1,101),name = "ImageId"),results],axis = 1)  

submission.head()

submission.to_csv("intel_image_classification.csv",index=False)