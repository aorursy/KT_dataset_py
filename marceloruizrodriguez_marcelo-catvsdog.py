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
import numpy as np
import pandas as pd 
import os
import cv2

import matplotlib.pyplot as plt
!unzip ../input/dogs-vs-cats/train.zip -d train
!unzip ../input/dogs-vs-cats/test1.zip -d test
TRAIN_DIR = 'train/train/'
TEST_DIR = 'test/test1/'

train_images_filepaths = [TRAIN_DIR + last_file_name for last_file_name in os.listdir(TRAIN_DIR)]
test_images_filepaths = [TEST_DIR + last_file_name for last_file_name in os.listdir(TEST_DIR)]

print("Done")
train_dogs_filepaths = [TRAIN_DIR+ dog_file_name for dog_file_name in os.listdir(TRAIN_DIR) if 'dog' in dog_file_name]
train_cats_filepaths = [TRAIN_DIR+ cat_file_name for cat_file_name in os.listdir(TRAIN_DIR) if 'cat' in cat_file_name]

print("Done")
#Seeing a "color" image
test_img_file_path = train_dogs_filepaths[10]
img_array = cv2.imread(test_img_file_path,cv2.IMREAD_COLOR) #The last parameter can be switched with cv2.IMREAD_GRAYSCALE too
plt.imshow(img_array)
plt.show()
img_array_gray = cv2.imread(test_img_file_path,cv2.IMREAD_GRAYSCALE)

plt.imshow(img_array_gray, cmap = "gray")
plt.show()

print(img_array_gray.shape)
ROW_DIMENSION = 150
COLUMN_DIMENSION = 150
CHANNELS = 3 #For greyscale images put it to 1; put it to 3 if you want color image data

new_array = cv2.resize(img_array_gray,(ROW_DIMENSION,COLUMN_DIMENSION)) #A squarish compression on it's width will take place
plt.imshow(new_array,cmap = 'gray')
plt.show()
def read_converted_img(to_read_img_array):
    plt.imshow(to_read_img_array,cmap = 'gray')
    plt.show()
    
def prep_img(single_image_path):
    img_array_to_resize = cv2.imread(single_image_path,cv2.IMREAD_COLOR)
    resized = cv2.resize(img_array_to_resize,(ROW_DIMENSION,COLUMN_DIMENSION),interpolation = cv2.INTER_CUBIC)
    return resized

def prep_data(list_of_image_paths):
    
    size = len(list_of_image_paths)
    
    #preped_data = np.ndarray((size, ROW_DIMENSION, COLUMN_DIMENSION,CHANNELS), dtype=np.uint8)
    preped_data = []
    
    '''
    for i in range(size):
        list_of_image_paths[i] = prep_img(list_of_image_paths)
    '''
    
    for i, image_file_path in enumerate(list_of_image_paths):
        '''
        image = prep_img(image_file_path)
        #preped_data[i] = image.T
        preped_data.append(image)
        '''
        preped_data.append(cv2.resize(cv2.imread(image_file_path), (ROW_DIMENSION,COLUMN_DIMENSION), interpolation=cv2.INTER_CUBIC))
        
        if(i%1000==0):
            print("Processed",i,"of",size)
        
        #print(image.shape)
        #print(preped_data.shape)
        
    return preped_data
print("PREPING TRAINING SET")
train_data = prep_data(train_images_filepaths)
print("\nPREPING TEST SET")
test_data = prep_data(test_images_filepaths)
print("\nDone")
X_train = np.array(train_data)

print(X_train.shape)
#print(train_data.shape)
#print(test_data.shape)
read_converted_img(X_train[0])

y_train = []

for path_name in train_images_filepaths:
    print(path_name)
    if('dog' in path_name):
        y_train.append(1)
    else:
        y_train.append(0)

print("Percentage of dogs is",sum(y_train)/len(y_train))
y_train = np.array(y_train)
print(y_train)
from keras.utils.np_utils import to_categorical   

categorical_labels = to_categorical(y_train)
print(categorical_labels)

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Dropout

print("Import Successful")
dvc_classifier = Sequential()

dvc_classifier.add(Conv2D(32,kernel_size = (3,3),
                         activation = 'relu',
                         input_shape = (ROW_DIMENSION,COLUMN_DIMENSION,3)))

dvc_classifier.add(Conv2D(32,kernel_size = (3,3),
                        activation = 'relu'))

dvc_classifier.add(Conv2D(64,kernel_size = (3,3),
                        activation = 'relu'))

dvc_classifier.add(Flatten())

dvc_classifier.add(Dense(128,activation = 'relu'))

dvc_classifier.add(Dense(2,activation = 'softmax'))

dvc_classifier.summary()
dvc_classifier.compile(loss = "categorical_crossentropy",
                      optimizer = 'adam',
                      metrics = ['accuracy'])
dvc_classifier.fit(X_train,categorical_labels,
               batch_size = 128,
               epochs = 5,
               validation_split = 0.2)
#Trying to save a model
model_json = dvc_classifier.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
dvc_classifier.save_weights("model.h5")
from keras.models import model_from_json

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
loaded_model.summary()
arr_test = np.array(test_data)
print(np.shape(arr_test))
prediction_probabilities = dvc_classifier.predict(arr_test, verbose=0)
print(prediction_probabilities)

for i in range(0,100):
    if prediction_probabilities[i, 0] >= prediction_probabilities[i, 1]: 
        print(f'I am {prediction_probabilities[i][0]} sure this is a Cat')
    else: 
        print(f'I am {prediction_probabilities[i][1]} sure this is a Dog')
    plt.imshow(arr_test[i])
    plt.show()