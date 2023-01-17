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


import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.preprocessing import image
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications.vgg16 import preprocess_input
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

tf.__version__


# loading Test Data
def test_data():
    # Loading Important Files
    ListOfTestSet = pd.read_csv("../input/kagglefsd/ListOfTestSet.csv")
    test_df = pd.read_csv("../input/kagglefsd/test_post_competition_scoring_clips.csv")
    Test_Path = "../input/kagglefsd/Spectrogram/Test/"
    # list to hold Test data
    Labels = []
    Images = []
    # for loading spectrogram Images
    length = len(test_df)
    for i in range(length):
        if ListOfTestSet["Spectrogram"][i] != 0:
            fname = Test_Path + test_df["fname"][i] + ".png"
            img = image.load_img(fname)
            image_array = image.img_to_array(img)
            Images.append(image_array)
            if test_df["label"][i] == "Cough" or test_df["label"][i] == "cough":
                Labels.append(1)
            else:
                Labels.append(0)
        print("***** "+ str(i) +" Done "+"*****")

    x_test = np.array(Images)
    y_test = np.array(Labels)
    x_test = preprocess_input(x_test)
    #x_test, val_x, y_test, val_y = train_test_split(x_test, y_test, test_size=0.51)
    return x_test, y_test

val_x, val_y = test_data()
# loading Train Data
def train_data():
    # Loading Important Files
    train_df = pd.read_csv("../input/kagglefsd/train.csv")
    ListOfTrainSet = pd.read_csv("../input/kagglefsd/ListOfTrainSet.csv")
    Train_Path = "../input/kagglefsd/Spectrogram/Train1/"
    # list to hold Train data
    Labels = []
    Images = []
    # for loading spectrogram Images
    length = len(train_df)
    for i in range(length):
        if ListOfTrainSet["Spectrogram"][i] != 0:
            fname = Train_Path + train_df["fname"][i] + ".png"
            img = image.load_img(fname)
            image_array = image.img_to_array(img)
            Images.append(image_array)
            if train_df["label"][i] == "Cough" or train_df["label"][i] == "cough":
                Labels.append(1)
            else:
                Labels.append(0)
        print("***** "+ str(i) +" Done "+"*****") 
    x_train = np.array(Images)
    y_train = np.array(Labels)
    x_train = preprocess_input(x_train)
    return x_train, y_train

x_train, y_train = train_data()

y_train
es = EarlyStopping(monitor='loss', patience=3)
filepath="/kaggle/working/bestmodel.h5"
md = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min')
# defininig ImageDataGeneratore to increase data

'''
datagen = ImageDataGenerator(zoom_range = 0.1,
                            height_shift_range = 0,
                            width_shift_range = 0,
                            rotation_range = 10)
                            '''
##### Important Variables
epochs = 18
num_classes = 2
batch_size = 1024
input_shape = (120, 124, 3)
adam = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model = Sequential()

# Filter 1
model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape, activation= 'relu')) 
model.add(Conv2D(32, (3, 3), padding='same', activation= 'relu'))    
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
#model.add(BatchNormalization())

# Filter 2
model.add(Conv2D(16, (3, 3), padding='same', activation= 'relu'))                          
model.add(Conv2D(16, (3, 3), padding='same', activation= 'relu'))    
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
#model.add(BatchNormalization())


# Filter 3 
#model.add(Conv2D(16, (3, 3), padding='same', activation= 'relu'))                         
#model.add(Conv2D(16, (3, 3), padding='same', activation= 'relu'))                        
#model.add(MaxPool2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
#model.add(BatchNormalization())


# 1st Dense Layer
model.add(Flatten())
#model.add(Dense(1024, activation='relu'))                                                
#model.add(Dropout(0.25))
#model.add(BatchNormalization())


# 2nd Dense Layer
model.add(Dense(512, activation='relu'))                                                
model.add(Dropout(0.25))

# 3rd Dense Layer
model.add(Dense(256, activation='relu'))                                                
model.add(Dropout(0.3))

# 4th Dense Layer
model.add(Dense(128, activation='relu'))                                                
model.add(Dropout(0.3))

# 5th Dense Layer

model.add(Dense(64, activation='relu'))                                                
model.add(Dropout(0.5))

# 6th Dense Layer
#model.add(Dense(32, activation='relu'))                                                
#model.add(Dropout(0.5))

# Output Layer
model.add(Dense(1, activation= 'sigmoid'))   

# Model Compile
model.compile(optimizer= adam, loss= tf.keras.losses.binary_crossentropy, metrics=["accuracy"])

# Model Summery
model.summary()

History = model.fit(x_train,
                    y_train, 
                    batch_size=batch_size,
                    #steps_per_epoch=2048,
                    epochs = epochs,
                    verbose=2,
                    validation_data = (val_x, val_y),
                    callbacks = [es,md],
                    shuffle= True
                    )
cnn = load_model("/kaggle/working/bestmodel.h5")
cnn.summary()
'''
pred = cnn.predict(x_test)
pred[0][0]
'''


'''from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test, pred_class))
df = pd.DataFrame(columns=['prediction','Actual_Result'])
df['prediction'] = pred_class
df['Actual_Result'] = y_test
df.to_csv('/kaggle/working/prediction.csv', index=False)'''


# Saving Structre Of Neural Network
from pathlib import Path
model_structure = cnn.to_json()
saving_m = Path("/kaggle/working/model_structure_17_256_0.1293_0.9722_0.1457_0.9812.jason")
saving_m.write_text(model_structure)
# Saving Model
model.save("/kaggle/working/17_256_0.1293_0.9722_0.1457_0.9812.h5")
# Saving weights only 
model.save_weights("/kaggle/working/W_17_256_0.1293_0.9722_0.1457_0.9812.h5")

