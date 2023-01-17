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
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
%matplotlib inline
train = pd.read_csv('../input/coronahack-chest-xraydataset/Chest_xray_Corona_Metadata.csv') #reading the csv file
train.head()      # printing first five rows of the file
train.columns
train_image = []
path = '../input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/'
for i in tqdm(range(train.shape[0])):
    put = 'train' if (train['Dataset_type'][i] == "TRAIN") else 'test'
    img = image.load_img('../input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/'+put+'/'+train['X_ray_image_name'][i],target_size=(256,256,3))
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)
X = np.array(train_image)
X.shape
for i, item in train.iteritems():
    print(item.unique())
train["Normal"] = 0
train["Pnemonia"] = 0
train["Virus"] = 0
train["bacteria"] = 0
train["Stress-Smoking"] = 0
train["Streptococcus"] = 0
train["COVID-19"] = 0
train["ARDS"] = 0
train["SARS"] = 0
train.head()
train.loc[train.Label == "Normal", 'Normal'] = 1
train.loc[train.Label == "Pnemonia", 'Pnemonia'] = 1
train.loc[train.Label_2_Virus_category == "Streptococcus", 'Streptococcus'] = 1
train.loc[train.Label_2_Virus_category == "COVID-19", 'COVID-19'] = 1
train.loc[train.Label_2_Virus_category == "ARDS", 'ARDS'] = 1
train.loc[train.Label_2_Virus_category == "SARS", 'SARS'] = 1
train.loc[train.Label_1_Virus_category == "Virus", 'Virus'] = 1
train.loc[train.Label_1_Virus_category == "bacteria", 'bacteria'] = 1
train.loc[train.Label_1_Virus_category == "Stress-Smoking", 'Stress-Smoking'] = 1
train.head()
y = np.array(train.drop(['Unnamed: 0', 'X_ray_image_name', 'Dataset_type','Label_2_Virus_category','Label_1_Virus_category','Label'],axis=1))
y.shape
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=42, test_size=0.3)
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(256,256,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(9, activation='sigmoid'))
'''
model = Sequential()

model.add(Conv2D(filters=64, kernel_size=(5, 5), input_shape=(256, 256, 3), activation='relu'))
model.add(BatchNormalization(axis=3))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.1))

model.add(Conv2D(filters=128, kernel_size=(5, 5), activation='relu'))
model.add(BatchNormalization(axis=3))
model.add(Conv2D(filters=128, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.1))

model.add(Conv2D(filters=256, kernel_size=(5, 5), activation='relu'))
model.add(BatchNormalization(axis=3))
model.add(Conv2D(filters=256, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.1))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(9, activation='sigmoid'))
'''
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_test, y_test))