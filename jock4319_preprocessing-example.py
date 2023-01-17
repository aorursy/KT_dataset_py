import os, cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
#from keras import backend as K
from keras.utils import np_utils
path = '/data/examples/may_the_4_be_with_u/where_am_i/train'
data_path_list = [path]

img_rows=200
img_cols=200
num_channel=1

# Define number of classes
num_classes = 15

img_data_list=[]
for data_path in data_path_list:
    for dataset in sorted(os.listdir(data_path), key=str.lower): 
        #dataset = ['bedroom', 'CALsuburb', 'coast', 'forest', 'highway', 'industrial', 'insidecity', 'kitchen',
        # 'livingroom', 'mountain', 'opencountry', 'PARoffice', 'store', 'street', 'tallbuilding']
        img_list = os.listdir(data_path+'/'+ dataset)
        print('Loaded the images of dataset-' +  '%s/%s\n' % (data_path, dataset))
        for img in img_list:
            if img.endswith('.jpg'):
                input_img = cv2.imread(data_path + '/'+ dataset + '/'+ img, 0)
                input_img_resize=cv2.resize(input_img,(128,128)) # 你要resize成為多少，我預設128
                img_data_list.append(input_img_resize)

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255
print(img_data.shape)
for data_path in data_path_list:
    for dataset in sorted(os.listdir(data_path), key=str.lower): 
        img_list = os.listdir(data_path+'/'+ dataset)
        print(dataset,'裡面有',len(img_list),'張圖')
        

num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64') # 一熱

index = 0
label = 0
for data_path in data_path_list:
    for dataset in sorted(os.listdir(data_path), key=str.lower): 
        img_list = os.listdir(data_path+'/'+ dataset)
        labels[index: index+len(img_list)] = label
        label+=1
        index+=len(img_list)
        #print(dataset,'裡面有',len(img_list),'張圖')



img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255
print(img_data.shape)

img_data = img_data.reshape(-1, 128, 128, 1) # 128是resize時候自己設定的
print(img_data.shape)

# One-hot encoding
Y = np_utils.to_categorical(labels, num_classes)

# Shuffle the data
from sklearn.utils import shuffle
x,y = shuffle(img_data,Y, random_state=2)
# Split the data into training set and validation set
print(x.shape)
print(y.shape)
from sklearn.model_selection import train_test_split
X_train, X_valid, Y_train, Y_valid = train_test_split(x, y, test_size=0.25, random_state=2)
