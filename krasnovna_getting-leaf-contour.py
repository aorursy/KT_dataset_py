import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing import image
%matplotlib inline 
train = pd.read_csv('../input/train.csv')
# Make a dictionary of leaf labels
df = pd.DataFrame( { 'index': np.arange(99), 'class': np.sort(train['species'].unique()) } )
class_dict = dict(zip(df['class'],df.index ))
#list(class_dict.keys())[list(class_dict.values()).index(6)]
# get fi,r for image contur 
def edge_features(img):

    fi_ = np.linspace(0,2*np.pi, 360)
    r_ = np.zeros(360)
    i = 0 
    for fi in np.linspace(0,2*np.pi, 360):
        # look for nearest radius value// compare neighbour values 
        r_previous = 0 
        for r in np.linspace(0, np.sqrt(2)*112, 360):
            #print(r)
            x_previous =  112 + np.int(r_previous*np.cos(fi))
            x_previous = max( min(x_previous, 223 ), 0 )
            y_previous = 112 + np.int(r_previous*np.sin(fi))
            y_previous = max( min(y_previous, 223 ), 0 )
            
            x_ = 112 +  np.int(r*np.cos(fi))
            y_ = 112 +  np.int(r*np.sin(fi))
            x_ = max( min(x_, 223 ), 0 )
            y_ = max( min(y_, 223 ), 0 )
            pixel = img.getpixel((x_, y_))
            pixel_previous = img.getpixel((x_previous, y_previous))
            #print(pixel_previous[0],pixel[0])
            if( pixel_previous[0] > 0 and pixel[0] > 0 ):
                continue 
            else:
                break 
            r_previous = r 
        r_[i] = r
        i = i + 1
    return fi_, r_     
img_path = '../input/images/' + str(1) + '.jpg' 
img = image.load_img(img_path, target_size=(224, 224)) 
fi, r = edge_features(img)
fig, ax = plt.subplots(1,1)
ax.axis('equal')
plt.scatter(r*np.cos(fi),r*np.sin(fi),s = 2)
plt.plot(r*np.cos(fi),r*np.sin(fi))
def get_edge_features(img_id_list, labels_list = [] , flag_test = 1 , augment_list = ['same'] ):
    
    img_count = len(img_id_list)
    augment_count = 1 + len(augment_list)
    features = np.zeros(shape=(augment_count*img_count, 1, 1, 2048))
    labels = np.zeros(shape=(augment_count*img_count))   
    img_path = ['../input/images/' + str(id_) + '.jpg' for id_ in img_id_list ]
    img = [ image.load_img(path, target_size=(224, 224)) for path in img_path]
    # Augmentation 
    features_ = []
    labels_ = []
    j = 0 
    for val in augment_list:
        if (val == 'same'):
            img_ = img
        if (val == 'rotate_45'):
            img_ = [ x.rotate(45) for x in img ]
        if (val == 'rotate_90'):
            img_ = [ x.rotate(90) for x in img ]
        #print(img_arr.shape)
        features_[j*img_count:(j+1)*img_count] = [edge_features(x)[1] for x in img_]
        j = j + 1
        if( flag_test == 0 ):
            labels_ = labels_ + labels_list 
            
    return np.asarray(features_), np.asarray( labels_ )
import time
m = 10 # size of dataset 
# Extract features ( example for 10 observation) m = 10
train_id_list =  train['id'].tolist()[0:m]
labels_ = [ class_dict[train[train['id'] == x ]['species'].values[0]] for x in train_id_list ] 
start_time = time.time()
train_features, train_labels = get_edge_features( train_id_list, labels_, flag_test = 0 , 
                                                     augment_list = ['same'] )
elapsed_time = time.time() - start_time
print( elapsed_time)
# Several cycle transfer dont change our plot
# Rotation by 30% 
train_features_1 = np.zeros((m*12,360))
train_labels_1 =  np.zeros((m*12),dtype = np.int64)
for i in range(m-1):
    for j in range(12):
        #print(np.asarray( list(train_features[i,j:359]) + list(train_features[i,0:j]) ).shape)
        train_features_1[12*i+j,:] = np.asarray( list(train_features[i,30*j:360]) + list(train_features[i,0:30*j]) )
        train_labels_1[12*i+j,] = train_labels[i]
fig, ax = plt.subplots(2,2)
fig.set_size_inches(10, 10)
plt.suptitle('Example of rotation 30 degree : ' + list(class_dict.keys())[list(class_dict.values()).index(train_labels_1[0])], fontsize=16 )
ax[0,0].axis('equal')
for i in range(1,5):
    plt.subplot(2,2,i)
    r = train_features_1[i,:]
    plt.scatter(r*np.cos(fi),r*np.sin(fi),s = 2)
    plt.plot(r*np.cos(fi),r*np.sin(fi))
