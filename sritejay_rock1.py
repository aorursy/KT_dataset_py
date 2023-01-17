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

#present working directory
!pwd
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import os
import shutil
from tqdm import tqdm_notebook
input_data_dir = '/kaggle/input/rockpaperscissors'
WORKING_dir = "/kaggle/working"
TRAIN_dir = 'Train'
VALIDATION_dir = 'Validation'
PAPER_dir = 'paper'
ROCK_dir = 'rock'
SCISSORS_dir = 'scissors'
os.mkdir(TRAIN_dir)
os.mkdir(VALIDATION_dir)
os.mkdir(os.path.join(TRAIN_dir,ROCK_dir))
os.mkdir(os.path.join(VALIDATION_dir,ROCK_dir))
os.mkdir(os.path.join(TRAIN_dir,PAPER_dir))
os.mkdir(os.path.join(VALIDATION_dir,PAPER_dir))
os.mkdir(os.path.join(TRAIN_dir,SCISSORS_dir))
os.mkdir(os.path.join(VALIDATION_dir,SCISSORS_dir))

for dirpath,dirnames,filesnames in tqdm_notebook(os.walk('/kaggle/input')):
    #print(f"dirpath : {dirpath}, dirnames: {dirnames}, filenames: {filesnames}")
    count = 1
    for file in filesnames:        
        if dirpath.endswith("/"+ROCK_dir):
            if count> 700:
                shutil.copy(os.path.join(input_data_dir,ROCK_dir,file),os.path.join(WORKING_dir,VALIDATION_dir,ROCK_dir,file))  
            shutil.copy(os.path.join(input_data_dir,ROCK_dir,file),os.path.join(WORKING_dir,TRAIN_dir,ROCK_dir,file))
            count +=1
        if dirpath.endswith("/"+PAPER_dir):
            if count> 700:
                shutil.copy(os.path.join(input_data_dir,PAPER_dir,file),os.path.join(WORKING_dir,VALIDATION_dir,PAPER_dir,file))  
            shutil.copy(os.path.join(input_data_dir,PAPER_dir,file),os.path.join(WORKING_dir,TRAIN_dir,PAPER_dir,file)) 
            count +=1
        if dirpath.endswith("/"+SCISSORS_dir):
            if count> 700:
                shutil.copy(os.path.join(input_data_dir,SCISSORS_dir,file),os.path.join(WORKING_dir,VALIDATION_dir,SCISSORS_dir,file))  
            shutil.copy(os.path.join(input_data_dir,SCISSORS_dir,file),os.path.join(WORKING_dir,TRAIN_dir,SCISSORS_dir,file)) 
            count +=1
    
    
from tensorflow.keras.preprocessing.image import ImageDataGenerator
img_datagen = ImageDataGenerator(rescale=1/255,horizontal_flip=True,zoom_range=0.3)
train_generator = img_datagen.flow_from_directory(os.path.join(WORKING_dir,TRAIN_dir),target_size=(200,300))
validation_generator = img_datagen.flow_from_directory(os.path.join(WORKING_dir,VALIDATION_dir),target_size=(200,300))
train_generator.class_indices
validation_generator.class_indices
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from tensorflow.keras.activations import relu,softmax
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import SGD,RMSprop
model = Sequential()

model.add(Conv2D(32,kernel_size=(3,3),activation=relu,input_shape=(200,300,3)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,kernel_size=(3,3),activation=relu))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
          
model.add(Dense(512,activation=relu))
model.add(Dropout(0.3))
model.add(Dense(3,activation=softmax))
model.summary()
model.compile(loss=categorical_crossentropy,optimizer=RMSprop(),metrics=['acc'])
batch_size=100
model.fit_generator(train_generator,epochs=20,steps_per_epoch=(train_generator.n/batch_size),validation_data=validation_generator,validation_steps=(validation_generator.n/batch_size))