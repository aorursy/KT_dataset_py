# Import library

import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from matplotlib.pyplot import imshow

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf
from keras import layers
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from keras.models import Model

import keras.backend as K
from keras.models import Sequential

import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)
'''
import pickle
X = pickle.load( open( "../input/photo-thaimnist-transform/save_X.p", "rb" ) )
y_cat = pickle.load( open( "../input/photo-thaimnist-transform/save_y.p", "rb" ) )
X_test = pickle.load( open( "../input/thaimnistclassification/X_test.p", "rb" ) )
'''
train_map = '../input/thai-mnist-classification/mnist.train.map.csv'
train_path ='../input/thai-mnist-classification/train'
def prepareImages(data, m, dataset):
    '''
    Function for preprocessing input
    '''
    print("Preparing images")
    X_train = np.zeros((m, 32, 32, 3))
    count = 0
    
    for fig in data['id']:
        #load images into images of size 100x100x3
        img = image.load_img("../input/thai-mnist-classification/"+str(dataset)+"/"+str(fig), grayscale=True)
        x = image.img_to_array(img)
        x = preprocess_input(x)

        X_train[count] = x
        if (count%500 == 0):
            print("Processing image: ", count+1, ", ", fig)
        count += 1
    
    return X_train
def prepare_labels(y):
    '''
    Function for encoding label
    '''
    values = np.array(y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # print(integer_encoded)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # print(onehot_encoded)

    y = onehot_encoded
    # print(y.shape)
    return y, label_encoder
# Thanks to https://www.kaggle.com/layyer/thai-mnist-lenet5
class getdata():
    def __init__(self,data_path,label_path):
        self.dataPath = data_path
        self.labelPath = label_path
        self.label_df = pd.read_csv(label_path)
        self.dataFile = self.label_df['id'].values
        self.label = self.label_df['category'].values
        self.n_index = len(self.dataFile)
        
    
    def get1img(self,img_index,mode='rgb',label = False):
        img = cv2.imread( os.path.join(self.dataPath,self.label_df.iloc[img_index]['id']) )
        if mode == 'rgb':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif mode == 'gray':
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        if label:
            return img,self.label_df.iloc[img_index]['category']
        return img
from skimage.morphology import convex_hull_image
from skimage.util import invert
import cv2
from skimage import feature
from skimage import measure
def convex_crop(img,pad=20):
    convex = convex_hull_image(img)
    r,c = np.where(convex)
    while (min(r)-pad < 0) or (max(r)+pad > img.shape[0]) or (min(c)-pad < 0) or (max(c)+pad > img.shape[1]):
        pad = pad - 1
    return img[min(r)-pad:max(r)+pad,min(c)-pad:max(c)+pad]
def convex_resize(img):
    img = invert(img)
    img = convex_crop(img,pad=20)
    img = cv2.resize(img,(32,32))
    return img
def thes_resize(img,thes=40):
    img = invert(img)
    img = convex_crop(img,pad=20)
    img = ((img > thes)*255).astype(np.uint8)
    if(min(img.shape) > 300):
        img = cv2.resize(img,(300,300))
        img = ((img > thes)*255).astype(np.uint8)
    if(min(img.shape) > 150):
        img = cv2.resize(img,(150,150))
        img = ((img > thes)*255).astype(np.uint8)
    img = cv2.resize(img,(80,80))
    img = ((img > thes)*255).astype(np.uint8)
    img = cv2.resize(img,(50,50))
    img = ((img > thes)*255).astype(np.uint8)
    img = cv2.resize(img,(32,32))
    img = ((img > thes)*255).astype(np.uint8)
    return img
# Prepare data
gdt = getdata(train_path,train_map)

X = []
for i in range(gdt.n_index):
    X.append(thes_resize(gdt.get1img(i,'gray')))
    if (i+1) % 100 == 0:
        print(i)
X = np.array(X)

y = gdt.label
X = X.reshape((-1,32,32,1))
X.shape,y.shape
'''
X_train = prepareImages(test_df,test_df.shape[0], "test")
X_test /= 255.
'''
y_cat = tf.keras.utils.to_categorical(y)
y_cat.shape
# Create Train and Validation set
from sklearn.model_selection import train_test_split
X = X / 255.
X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, random_state=1)
'''
# Model 1

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense

vgg = VGG16(include_top=False, weights='imagenet') # FCM = Fully Convolution Network = No input size
# fit input
x_in = layers.Input(shape=(32, 32, 1))
x = layers.Conv2D(3, 1)(x_in) # Add filter 1x1 to change from 1 to 3 dimension to match VGG
x = vgg(x)
# fit output
x = layers.Flatten()(x)
x = layers.Dense(10, activation='softmax')(x)
model = Model(x_in, x)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics="accuracy")

learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.0000001)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=10,verbose=1)

# Fit Model
#history = model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=1, validation_data=(X_val, y_val), callbacks=[learning_rate_reduction,early_stop])
history = model.fit(X, y_cat, epochs=80, batch_size=64, verbose=1)
'''
'''
# Model 2
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(6, (5,5), input_shape=(32, 32, 1), activation='relu'))
model.add(tf.keras.layers.MaxPool2D()) 
model.add(tf.keras.layers.Conv2D(16, (5,5), activation='relu')) 
model.add(tf.keras.layers.MaxPool2D()) 
model.add(tf.keras.layers.Flatten()) 
model.add(tf.keras.layers.Dense(120, activation='relu'))
model.add(tf.keras.layers.Dense(84, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['acc'])
history = model.fit(X, y_cat, batch_size=64, epochs=100)
'''
# Model 3

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense

vgg = VGG16(include_top=False, weights='imagenet') # FCM = Fully Convolution Network = No input size
# fit input
x_in = layers.Input(shape=(32, 32, 1))
x = layers.Conv2D(3, 1)(x_in) # Add filter 1x1 to change from 1 to 3 dimension to match VGG
x = vgg(x)
# fit output
x = layers.Flatten()(x)
x = layers.Dense(10, activation='softmax')(x)
model = Model(x_in, x)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics="accuracy")

learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.0000001)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=10,verbose=1)

# Fit Model
# history = model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=1, validation_data=(X_val, y_val), callbacks=[learning_rate_reduction,early_stop])
history = model.fit(X, y_cat, epochs=25, batch_size=50, verbose=1, callbacks=[learning_rate_reduction,early_stop])
# Import test set
test = os.listdir("../input/thai-mnist-classification/test")
col = ['id']
test_df = pd.DataFrame(test, columns=col)
test_df['category'] = ''
# test_df.to_csv (r'test_blank.csv', index = False, header=True)
# Import test set
test = os.listdir("../input/thai-mnist-classification/test")
col = ['id']
test_df = pd.DataFrame(test, columns=col)
test_df['category'] = ''

# Prepare test data
X_test = prepareImages(test_df,test_df.shape[0], "test")
X_test /= 255.
# Predict test set
predictions = model.predict(np.array(X_test), verbose=1)
test_df['category'] = predictions.argsort()[:,8]
test_df.head()
# Create a submission file
test_df.to_csv('test_prediction.csv', index=False)
train_rule = pd.read_csv('../input/thai-mnist-classification/train.rules.csv')
test_rule = pd.read_csv('../input/thai-mnist-classification/test.rules.csv')
train_map_file = pd.read_csv('../input/thai-mnist-classification/mnist.train.map.csv')
# train_rule.head()
# train_rule.info()
# there are 284 null rows in feature 1
'''
train_rule.fillna(-1, inplace=True) # replace na with -1
# Replace train_rule with train map value
m = train_map_file.set_index('id')['category'].to_dict()
v = train_rule.filter(like='feature')
train_rule[v.columns] = v.replace(m)
'''
# train_rule.head()
# Count number of each number for each feature
# train_rule.drop(['id','predict'],axis=1).apply(pd.value_counts)
'''
X = train_rule.drop(['id','predict'],axis=1)
y = train_rule['predict']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
'''
'''
from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(max_depth=20, random_state=0)
regr.fit(X_train, y_train)
'''
'''
# Measure MAE of validation set
from sklearn import metrics

y_pred_val = regr.predict(X_val)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_val, y_pred_val)) '''
# Predict test set
# Replace test_rule with train map value
test_map = pd.read_csv('./test_prediction.csv')
m_test = test_map.set_index('id')['category'].to_dict()
v_test = test_rule.filter(like='feature')
test_rule[v_test.columns] = v_test.replace(m_test)

# fill na
test_rule.fillna(-1, inplace=True)

# Predict from model
'''
X_test = test_rule.drop(['id','predict'],axis=1)
y_pred = regr.predict(X_test)
test_rule['predict'] = y_pred
'''
# train_rule.head()
'''
# Test with train
train_rule.loc[train_rule['feature1']==-1,'predict2'] = train_rule['feature2']+train_rule['feature3']
train_rule.loc[train_rule['feature1']==0,'predict2'] = train_rule['feature2']*train_rule['feature3']
train_rule.loc[train_rule['feature1']==1,'predict2'] = abs(train_rule['feature2']-train_rule['feature3'])
train_rule.loc[train_rule['feature1']==2,'predict2'] = (train_rule['feature2']+train_rule['feature3'])*abs(train_rule['feature2']-train_rule['feature3'])
train_rule.loc[train_rule['feature1']==3,'predict2'] = abs((train_rule['feature3']*(train_rule['feature3']+1)-train_rule['feature2']*(train_rule['feature2']-1))/2)
train_rule.loc[train_rule['feature1']==4,'predict2'] = 50+train_rule['feature2']-train_rule['feature3']
train_rule.loc[train_rule['feature1']==5,'predict2'] = np.minimum(train_rule['feature2'],train_rule['feature3'])
train_rule.loc[train_rule['feature1']==6,'predict2'] = np.maximum(train_rule['feature2'],train_rule['feature3'])
train_rule.loc[train_rule['feature1']==7,'predict2'] = ((train_rule['feature2']*train_rule['feature3'])%9)*11
train_rule.loc[train_rule['feature1']==8,'predict2'] = (train_rule['feature2']**2+1)*train_rule['feature2']+train_rule['feature3']*(train_rule['feature3']+1)
train_rule.loc[train_rule['feature1']==9,'predict2'] = 50+train_rule['feature2']
train_rule['predict2'] = train_rule['predict2']%99
'''
# Predict using if condition
test_rule.loc[test_rule['feature1']==-1,'predict'] = test_rule['feature2']+test_rule['feature3']
test_rule.loc[test_rule['feature1']==0,'predict'] = test_rule['feature2']*test_rule['feature3']
test_rule.loc[test_rule['feature1']==1,'predict'] = abs(test_rule['feature2']-test_rule['feature3'])
test_rule.loc[test_rule['feature1']==2,'predict'] = (test_rule['feature2']+test_rule['feature3'])*abs(test_rule['feature2']-test_rule['feature3'])
test_rule.loc[test_rule['feature1']==3,'predict'] = abs((test_rule['feature3']*(test_rule['feature3']+1)-test_rule['feature2']*(test_rule['feature2']-1))/2)
test_rule.loc[test_rule['feature1']==4,'predict'] = 50+test_rule['feature2']-test_rule['feature3']
test_rule.loc[test_rule['feature1']==5,'predict'] = np.minimum(test_rule['feature2'],test_rule['feature3'])
test_rule.loc[test_rule['feature1']==6,'predict'] = np.maximum(test_rule['feature2'],test_rule['feature3'])
test_rule.loc[test_rule['feature1']==7,'predict'] = ((test_rule['feature2']*test_rule['feature3'])%9)*11
test_rule.loc[test_rule['feature1']==8,'predict'] = (test_rule['feature2']**2+1)*test_rule['feature2']+test_rule['feature3']*(test_rule['feature3']+1)
test_rule.loc[test_rule['feature1']==9,'predict'] = 50+test_rule['feature2']
test_rule['predict'] = test_rule['predict']%99
# Create a submission file
submission = test_rule.drop(['feature1','feature2','feature3'],axis=1)
submission.to_csv('submission.csv', index=False)