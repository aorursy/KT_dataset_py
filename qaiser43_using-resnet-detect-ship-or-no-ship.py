# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import gc
print(os.listdir("../input"))
import numpy as np 
import pandas as pd
import time

# Any results you write to the current directory are saved as output.


train = pd.read_csv('../input/airbus-ship-detection/train_ship_segmentations_v2.csv')
train.head()
list(train.columns.values)
train['exist_ship'] = train['EncodedPixels'].fillna(0)
train.head()
train['exist_ship'] != 0
train.loc[train['exist_ship'] != 0 , 'exist_ship'] = 1
train.head()
del train['EncodedPixels']
train.head()
print(len(train['ImageId']))
print(train['ImageId'].value_counts().shape[0])
train_gp = train.groupby(['ImageId']).sum().reset_index()
train_gp.loc[train_gp['exist_ship']>0,'exist_ship']=1

train_sample = train_gp.sample(5000)
test_sample = train_gp.sample(1000)
print(train_gp['exist_ship'].value_counts())
print(train_sample['exist_ship'].value_counts())
print(test_sample['exist_ship'].value_counts())
print (train_sample.shape)
print (test_sample)
from keras.utils import np_utils
import numpy as np
from glob import glob

Train_path = '../input/airbus-ship-detection/train_v2/'
Test_path = '../input/airbus-ship-detection/test_v2/'
# define function to load train, test, and validation datasets
def load_dataset(path):
    files_array = []
    if str(path) == str(Train_path):
        data = np.array(train_sample['ImageId'])
        data_targets = np_utils.to_categorical(np.array(train_sample['exist_ship']), 133)

        for idx, element in  enumerate(data): 
            files_array.append(Train_path + element)

        data = np.array(files_array)
    else:
        data = np.array(test_sample['ImageId'])
        data_targets = np_utils.to_categorical(np.array(test_sample['exist_ship']), 133)

        for idx, element in  enumerate(data): 
            files_array.append(Train_path + element)

        data = np.array(files_array)
    
    return data, data_targets


# load train, test, and validation datasets
train_files, train_targets = load_dataset(Train_path)
test_files, test_targets = load_dataset(Test_path)

from keras.preprocessing import image 
from tqdm import tqdm

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)
from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 


test_tensors = paths_to_tensor(test_files).astype('float32')/255
train_tensors = paths_to_tensor(train_files).astype('float32')/255
from keras.applications.resnet50 import ResNet50

img_width, img_height = 224, 224
#model = ResModel(weights = 'imagenet')
model = ResNet50(weights='../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels.h5', include_top=True, input_shape = (224, 224, 3))
from keras.applications.resnet50 import preprocess_input, decode_predictions

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    pred = model.predict(img)
    print('Predicted:', decode_predictions(pred, top=3))
    return np.argmax(pred)
### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 
human_files = np.array(glob(Test_path+ "*"))
human_files_short = human_files[:10]
import cv2                
import matplotlib.pyplot as plt                        
%matplotlib inline 

for human_file in human_files_short:
    fd = dog_detector(human_file)
    img = cv2.imread(human_file)
    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(cv_rgb)
    plt.show()
        