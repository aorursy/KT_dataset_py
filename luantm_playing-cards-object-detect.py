import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
#import and explore data
path = '../input/card_dataset'
path_train = os.path.join(path,'train')
path_test = os.path.join(path,'test')
print(path_train)

train = pd.read_csv(os.path.join(path,'train_labels.csv'))
test = pd.read_csv(os.path.join(path,'test_labels.csv'))
print(train.shape)
print(test.shape)
print('_' * 49)
print(train.head())
print('_' * 100)
print(test.head())
print(train['class'].unique())

#display crop image
for i in range(10):
    row = train.iloc[i]
#     print(path_train)
    img = cv2.imread(os.path.join(path_train, row['filename']))
    
    print(img.shape)
    plt.imshow(img)
    plt.show()
train = train.set_index('filename')
test = test.set_index('filename')

print(train.head())
# print(enumerate(set(train.index)))
# print(train.loc['cam_image1.jpg'])
#Create data
indexes = list(set(train.index))

m = len(indexes)
height = 480
width = 480

y_width = 11
y_height = 11
anchor_box = 1
#y example [pc, bx, by, bh, bw]

def create_data():
    x = np.zeros((m, height, width, 3))
    y = np.zeros((m, y_height, y_width, 5))

    for idx in range(m):
        img = cv2.imread(os.path.join(path_train, indexes[idx]))
        resize = cv2.resize(img, (width, height))
        x[idx] = resize
        
        
        rows = train.loc[[indexes[idx]]]
        
        for i, row in rows.iterrows():
            xmin = row['xmin']
            xmax = row['xmax']
            ymin = row['ymin']
            ymax = row['ymax']
            
            grid_width = (row['width']  / y_width)
            grid_height = (row['height']  / y_height)
            
            bx = (xmax + xmin) / 2 
            by = (ymax + ymin) / 2 
            bh = (ymax - ymin) / grid_height #ok
            bw = (xmax - xmin) / grid_width #ok
            pw = int(bx // grid_width)
            ph = int(by // grid_height)
            bx = bx % grid_width / grid_width
            by = by % grid_height/ grid_height
#             print(f'ph {ph}, pw {pw}')
            y[idx, ph, pw] = np.array([1, bx, by, bw, bh]) 
        
        
    return x, y
x, y = create_data()    
    
print(train.loc[indexes[1]])
img = cv2.imread(os.path.join(path_train, indexes[1]))
img = cv2.resize(img, (width, height))
print(img.shape)
img[:,::int(width//y_width)] = [0, 0, 0]
img[::int(height//y_height),:] = [0,0,0]
plt.imshow(img)
print(y[1])
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
%matplotlib inline
import keras

import keras.backend as K
def create_model(): # 480 x 480
    X_input = Input((480, 480, 3))

    X = Conv2D(6, (5, 5), kernel_initializer = glorot_uniform(seed=0))(X_input) #480 - 4 = 476
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X) # 476 / 2 = 238
    
    X = Conv2D(16, (5, 5), kernel_initializer = glorot_uniform(seed=0))(X) #238 - 4 = 234
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X) # 234 / 2 = 117
    
    X = Conv2D(32, (5, 5), kernel_initializer = glorot_uniform(seed=0))(X) #117 - 4 = 113
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X) # 113 / 2 = 56
    
    X = Conv2D(16, (5, 5), kernel_initializer = glorot_uniform(seed=0))(X) #56 - 4 = 52
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X) # 52 / 2 = 26
    
    X = Conv2D(5, (5, 5), kernel_initializer = glorot_uniform(seed=0))(X) #26 - 4 = 22
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X) # 22 / 2 = 11
    
    model = Model(inputs = X_input, outputs = X, name='ResNet50')
    return model

model = create_model()
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
print(model.summary())
model.fit(x, y, epochs=1000)

#Create data
indexes = list(set(test.index))

m = len(indexes)
height = 480
width = 480

y_width = 11
y_height = 11
anchor_box = 1
#y example [pc, bx, by, bh, bw]


def create_test_data():
    x = np.zeros((m, height, width, 3))
    y = np.zeros((m, y_height, y_width, 5))

    for idx in range(m):
        img = cv2.imread(os.path.join(path_test, indexes[idx]))
        resize = cv2.resize(img, (width, height))
        x[idx] = resize
        
        
        rows = test.loc[[indexes[idx]]]
        
        for i, row in rows.iterrows():
            xmin = row['xmin']
            xmax = row['xmax']
            ymin = row['ymin']
            ymax = row['ymax']
            
            grid_width = (row['width']  / y_width)
            grid_height = (row['height']  / y_height)
            
            bx = (xmax + xmin) / 2 
            by = (ymax + ymin) / 2 
            bh = (ymax - ymin) / grid_height #ok
            bw = (xmax - xmin) / grid_width #ok
#             print(row)
#             print(f'bx = {bx}, by = {by}, bw={bw}, bh={bh}')
            pw = int(bx // grid_width)
            ph = int(by // grid_height)
            bx = bx % grid_width / grid_width
            by = by % grid_height/ grid_height
            
            
            
#             print(f'ph {ph}, pw {pw}')
            y[idx, ph, pw] = np.array([1, bx, by, bw, bh]) 
        
        
    return x, y
x_test, y_test = create_test_data()    
    
model.evaluate(x=x, y=y)
model.evaluate(x=x_test, y=y_test)
import matplotlib.patches as patches
def print_bounding_box(img, bounding_box):
    width = img.shape[0]
    height = img.shape[1]
    
    no_grid_x = bounding_box.shape[0]
    no_grid_y = bounding_box.shape[1]
    
    grid_width = width / no_grid_x
    grid_height = height / no_grid_y

    for x in range(no_grid_x):
        for y in range(no_grid_y):
            pc, bx, by, bw, bh = bounding_box[x][y]
            if pc >= 0.2:
                
                x_centroid = (x + by) * grid_width
                y_centroid = (y + bx) * grid_height
#                 print('x_centroid ', x_centroid)
#                 print('y_centroid ', y_centroid)
                actual_width = bw * grid_width
                actual_height = bh * grid_height
                
#                 print('actual widht ', actual_width/2)
#                 print('actuan height ',actual_height/2)
                
                left = 50
                top = 70
                left = x_centroid - actual_height/2 
                top = y_centroid - actual_width/2
                
                
#                 print('left ', left)
#                 print('top ', top)
                
                fig,ax = plt.subplots(1)
                ax.imshow(img.astype(int))
                ax.add_patch(patches.Rectangle((top, left), actual_width, actual_height,linewidth=1,edgecolor='r',facecolor='none'))

y_pred = model.predict(x_test)
print(y_pred.shape)
for i in range(10):
    print_bounding_box(x_test[i], y_pred[i])
print(train.head())
