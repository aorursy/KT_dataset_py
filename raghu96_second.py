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
# from IPython.display import clear_output

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

training = pd.read_csv('../input/facial-keypoints-detection/training.zip')

test = pd.read_csv('../input/facial-keypoints-detection/test.zip')

lookid_data = pd.read_csv('../input/facial-keypoints-detection/IdLookupTable.csv')
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import cv2

import sys, requests, shutil, os

from urllib import request, error





def fetch_image(image_url,name):

    img_data = requests.get(image_url).content

    with open(f'../input/{name}', 'wb') as handler:

        handler.write(img_data)



def plot_image(image):

    plt.imshow(image,cmap='gray')

    plt.xticks([])

    plt.yticks([])



    

def pooling(image, kernel_shape=3):

    #showing max_pooling

    print ("shape before pooling",image.shape)

    y, x = image.shape

    new_image = []

    for i in range(0,y,kernel_shape):

        temp = []

        for j in range(0,x,kernel_shape):

            temp.append(np.max(image[i:i+kernel_shape, j:j+kernel_shape]))

        new_image.append(temp)

    new_image = np.array(new_image)

    print ("shape after pooling",new_image.shape)

    return (new_image)



def padding(image,top=1,bottom=1,left=1,right=1,values=0):

  # Create new rows/columns in the matrix and fill those with some values

  #return cv2.copyMakeBorder(image,top,bottom,left,right,cv2.BORDER_CONSTANT,value=values)

    

    x,y = image.shape

    #print (image.shape)

    arr = np.full((x+top+bottom,y+left+right),values,dtype=float)

    #print(image[0])

    #print (arr.shape)

    #print (top,x-bottom)

    #print (y,y-bottom)

    arr[top:x+top,left:y+left] = image

    #print(arr[top])

    return arr



def convolution2d(image, kernel, bias=0,strid=1,pad_val=()):

  #including padding,striding and convolution

    print ("shape before padding/striding",image.shape)

    if not pad_val:

        print (pad_val)

    image = padding(image,*pad_val)#(how many rows, columns to be padded, and of what type)

    m, n = kernel.shape

    y, x = image.shape

    y = y - m + 1

    x = x - m + 1

    new_image = []

    for i in range(0,y,strid):

        temp = []

        for j in range(0,x,strid):

            temp.append(np.sum(image[i:i+m, j:j+m]*kernel) + bias)

        new_image.append(temp)

    new_image = np.array(new_image)

    print ("shape after padding/striding",new_image.shape)

    return (new_image)
import numpy as np

import matplotlib.pyplot as plt

import sys

np.set_printoptions(threshold=sys.maxsize)

img_txt_input = training['Image'][0]

print("No of pixel values is :",len(img_txt_input.split(' ')),"Converting the pixel values into rows and column:",np.sqrt(len(img_txt_input.split(' '))),"*",np.sqrt(len(img_txt_input.split(' '))),"\n")

fn_reshape = lambda a: np.fromstring(a, dtype=int, sep=' ').reshape(96,96)

img = fn_reshape(img_txt_input)

print("Below is the pixel value conveted into an image")

plt.imshow(img,cmap='gray')

plt.show()
samp_imag = img.copy()

samp_imag = samp_imag/255.

h_kernal = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])

v_kernal = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])

h_image = cv2.filter2D(samp_imag,-1, h_kernal)

v_image = cv2.filter2D(samp_imag,-1, v_kernal)

#Laplacian filter

lap_filter = np.array([[0,1,0],[1,-4,1],[0,1,0]])

lap_image = cv2.filter2D(samp_imag,-1, lap_filter)



print ("Shapes before applying the filter:{} ".format(samp_imag.shape))

print ("Shapes after applying the filter:{} ".format(lap_image.shape))

plt.figure(figsize=(10,10))

plt.suptitle("Image and its transformations after applying filters")

plt.subplot(221)

plt.title("Actual Gray Scale Image")

plot_image(samp_imag)

plt.subplot(222)

plt.title("Horizontal Filter applied")

plot_image(h_image)

plt.subplot(223)

plt.title("Vertical Filter applied")

plot_image(v_image)

plt.subplot(224)

plt.title("Laplacian Filter applied")

plot_image(lap_image)

plt.show()
# print ("shape of actual image: {}".format(samp_imag.shape))

padded_image_5 = padding(samp_imag,*(5,5,5,5,1))

padded_image_10 = padding(samp_imag,*(10,10,10,10,0))

padded_bimage_10 = padding(samp_imag,*(10,10,10,10,1))

# # print ("shape of padded image: {}".format(padded_image.shape))



plt.figure(figsize=(10,10))

plt.suptitle("Padding")

plt.subplot(2,2,1)

plt.imshow(samp_imag,cmap='gray')

plt.title("Actual Image")

plt.subplot(2,2,2)

plt.imshow(padded_image_5,cmap='gray')

plt.title("Padding 5 border with white")

plt.subplot(2,2,3)

plt.imshow(padded_image_10,cmap='gray')

plt.title("Padding 10 border with black")

plt.subplot(2,2,4)

plt.imshow(padded_bimage_10,cmap='gray')

plt.title("Padding 10 border with white")

plt.show()
print ("Padding used is 1 for all the borders\nVertical filter is used")

fig, ax = plt.subplots(2, 2,figsize=(10,10))

plt.suptitle("Affect of Stride")

ax[0,0].set_title("Actual Image")

ax[0,0].imshow(samp_imag,cmap='gray')

rave = ax.ravel()

for i in range(1,4):

    print ("")

    print(f"striding value = {i}")

    custom_conv = convolution2d(samp_imag,v_kernal,strid=i,pad_val=(1,1,1,1,0))

    rave[i].set_title(f"striding value = {i}")

    rave[i].imshow(custom_conv,cmap='gray')

#print (custom_conv.shape)

#plt.imshow(custom_conv,cmap='gray')
print ("Pooling example")

fig, ax = plt.subplots(2, 2,figsize=(10,10))

plt.suptitle("Affect of Pooing")

ax[0,0].set_title("Actual Image")

ax[0,0].imshow(samp_imag,cmap='gray')

rave = ax.ravel()

for i in range(2,5):

    print ("")

    print(f"Pooling and striding value = {i}")

    custom_conv = pooling(samp_imag,i)

    rave[i-1].set_title(f"striding value = {i}")

    rave[i-1].imshow(custom_conv,cmap='gray')

#print (custom_conv.shape)

#plt.imshow(custom_conv,cmap='gray')
train_columns = training.columns[:-1].values

training.head().T
test.head()
training[training.columns[:-1]].describe(percentiles = [0.05,0.1,.25, .5, .75,0.9,0.95]).T
whisker_width = 1.5

total_rows = training.shape[0]

missing_col = 0

for col in training[training.columns[:-1]]:

    count = training[col].count()

    q1 = training[col].quantile(0.25)

    q3 = training[col].quantile(0.75)

    iqr = q3 - q1

    outliers = training[(training[col] < q1 - whisker_width*iqr)

                       | (training[col] > q3 + whisker_width*iqr)][col].count()

    print (f"dv:{col}, dv_rows:{count}, missing_pct:{round(100.*(1-count/total_rows),2)}%, outliers:{outliers}, outlier_pct:{round(100.*outliers/count,2)}%")

    if (100.*(1-count/total_rows)>65):

        missing_col+=1



print(f"DVs containing more than 65% of data missing : {missing_col} out of {len(training.columns[:-1])}")
def plot_loss(hist,name,plt,RMSE_TF=False):

    '''

    RMSE_TF: if True, then RMSE is plotted with original scale 

    '''

    loss = hist['loss']

    val_loss = hist['val_loss']

    if RMSE_TF:

        loss = np.sqrt(np.array(loss))*48 

        val_loss = np.sqrt(np.array(val_loss))*48 

        

    plt.plot(loss,"--",linewidth=3,label="train:"+name)

    plt.plot(val_loss,linewidth=3,label="val:"+name)



def plot_sample_val(X,y,axs,pred):

    '''

    kaggle picture is 96 by 96

    y is rescaled to range between -1 and 1

    '''

    

    axs.imshow(X.reshape(96,96),cmap="gray")

    axs.scatter(48*y[0::2]+ 48,48*y[1::2]+ 48, label='Actual')

    axs.scatter(48*pred[0::2]+ 48,48*pred[1::2]+ 48, label='Prediction')



def plot_sample(X,y,axs):

    '''

    kaggle picture is 96 by 96

    y is rescaled to range between -1 and 1

    '''

    

    axs.imshow(X.reshape(96,96),cmap="gray")

    axs.scatter(48*y[0::2]+ 48,48*y[1::2]+ 48)
main_features = ['left_eye_center_x', 'left_eye_center_y',

            'right_eye_center_x','right_eye_center_y',

            'nose_tip_x', 'nose_tip_y',

            'mouth_center_bottom_lip_x',

            'mouth_center_bottom_lip_y', 'Image']

#Create 2 different datasets.

train_8_csv = training[main_features].dropna().reset_index()

train_30_csv = training.dropna().reset_index()
train_30_csv.head()
from tqdm import tqdm
def str_to_array(pd_series):

    '''

    pd_series: a pandas series, contains img pixels as strings,

    each element is a long str (length = 96*96 = 9216),

    contains pixel values. eg:('29 34 122 244 12 ....').

    

    1- Convert str of pixel values to 2d array.

    2- Stack all arrays into one 3d array.

    

    returns 3d numpy array of shape of (num of images, 96, 96, 1).

    '''

    data_size = len(pd_series)

    #intialize output 3d array as numpy zeros array.

    X = np.zeros(shape=(data_size,96,96,1), dtype=np.float32)

    for i in tqdm(range(data_size)):

        img_str = pd_series[i]

        img_list = img_str.split(' ')

        img_array = np.array(img_list, dtype=np.float32)

        img_array = img_array.reshape(96,96,1)

        X[i] = img_array

    return X
#Wrap train data and labels into numpy arrays.

X_30 = str_to_array(train_30_csv['Image'])

X_30  = X_30/255.0

labels_30 =  train_30_csv.drop(['index','Image'], axis=1)

y_30 = labels_30.to_numpy(dtype=np.float32)

print('X_train with 30 feature shape: ', X_30.shape)

print('y_train with 30 feature shape: ', y_30.shape)
X_30[0]
#Wrap test data and labels into numpy arrays.

X_8 = str_to_array(train_8_csv['Image'])

X_8  = X_8/255.0

labels_8 =  train_8_csv.drop(['index','Image'], axis=1)

y_8 = labels_8.to_numpy(dtype=np.float32)

print('X_train with 8 feature shape: ', X_8.shape)

print('y_train with 8 feature shape: ', y_8.shape)
X_train_8, X_val_8, y_train_8, y_val_8 = train_test_split(X_8, y_8, test_size=0.3, random_state=0)

print("Train sample:",X_train_8.shape,"Val sample:",X_val_8.shape)



X_train_30, X_val_30, y_train_30, y_val_30 = train_test_split(X_30, y_30, test_size=0.2, random_state=0)

print("Train sample:",X_train_30.shape,"Val sample:",X_val_30.shape)
import keras

from keras.models import Sequential

from keras.layers import Dense, Conv2D, Flatten, AvgPool2D, BatchNormalization, Dropout, Activation, MaxPooling2D

from keras.optimizers import Adam

from keras import regularizers

from keras.layers.advanced_activations import LeakyReLU

from keras.models import Sequential, Model

from keras.layers import Activation, Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Conv2D,MaxPool2D, ZeroPadding2D
def get_model(n_out = 8):

    model = Sequential()



    model.add(Convolution2D(32, (3,3), padding='same', use_bias=False, input_shape=(96,96,1)))

    model.add(LeakyReLU(alpha = 0.1))

    model.add(BatchNormalization())



    model.add(Convolution2D(32, (3,3), padding='same', use_bias=False))

    model.add(LeakyReLU(alpha = 0.1))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2, 2)))



    model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))

    model.add(LeakyReLU(alpha = 0.1))

    model.add(BatchNormalization())



    model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))

    model.add(LeakyReLU(alpha = 0.1))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2, 2)))



    model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))

    model.add(LeakyReLU(alpha = 0.1))

    model.add(BatchNormalization())



    model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))

    model.add(LeakyReLU(alpha = 0.1))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2, 2)))



    model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))

    # model.add(BatchNormalization())

    model.add(LeakyReLU(alpha = 0.1))

    model.add(BatchNormalization())



    model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))

    model.add(LeakyReLU(alpha = 0.1))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2, 2)))



    model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))

    model.add(LeakyReLU(alpha = 0.1))

    model.add(BatchNormalization())



    model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))

    model.add(LeakyReLU(alpha = 0.1))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2, 2)))



    model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))

    model.add(LeakyReLU(alpha = 0.1))

    model.add(BatchNormalization())



    model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))

    model.add(LeakyReLU(alpha = 0.1))

    model.add(BatchNormalization())





    model.add(Flatten())

    model.add(Dense(512,activation='relu'))

    model.add(Dropout(0.1))

    model.add(Dense(n_out))

    return model
model_8 = get_model(8)

model_8.compile(optimizer='Adam', 

              loss='mse', 

              metrics=['mae'])

model_8.summary()
model_30 = get_model(30)

model_30.summary()
model_30.compile(optimizer='Adam', 

              loss='mse', 

              metrics=['mae'])
LR_callback_30 = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=4, verbose=10, factor=.4, min_lr=.00001)

EarlyStop_callback_30 = keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)



LR_callback_8 = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=4, verbose=10, factor=.4, min_lr=.00001)

EarlyStop_callback_8 = keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)
history_30 = model_30.fit(X_train_30, y_train_30, validation_data=(X_val_30, y_val_30), epochs=500,

                       callbacks=[LR_callback_30,EarlyStop_callback_30])
# Plot the loss and accuracy curves for training and validation

def plot_loss(history):

    fig, ax = plt.subplots(2,1)

    ax[0].plot(history.history['loss'], color='b', label="Training loss")

    ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

    legend = ax[0].legend(loc='best', shadow=True)



    ax[1].plot(history.history['mae'], color='b', label="Training mae")

    ax[1].plot(history.history['val_mae'], color='r',label="Validation mae")

    legend = ax[1].legend(loc='best', shadow=True)
plot_loss(history_30)
score = model_30.evaluate(X_val_30, y_val_30, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
model_30.save('keypoint_model30.h5')
test['Image'] = test['Image'].apply(lambda x: np.fromstring(x, dtype=int, sep=' ').reshape((96,96)))
X_test = np.asarray([test['Image']], dtype=np.uint8).reshape(test.shape[0],96,96,1)

X_test = X_test/255.0
y_hat_30 = model_30.predict(X_test)
history_8 = model_8.fit(X_train_8, y_train_8, validation_data=(X_val_8, y_val_8), epochs=500, 

                       callbacks=[LR_callback_8,EarlyStop_callback_8])
plot_loss(history_8)
# X_test = str_to_array(test['Image'])

# print('X_test shape: ', X_test.shape)
#Pridect points for each image using 2 different model.

y_hat_30 = model_30.predict(X_test) 

y_hat_8 = model_8.predict(X_test)

print('Predictions shape', y_hat_30.shape)

print('Predictions shape', y_hat_8.shape)
feature_8_ind = [0, 1, 2, 3, 20, 21, 28, 29]

#Merge 2 prediction from y_hat_30 and y_hat_8.

for i in range(8):

    print('Copy "{}" feature column from y_hat_8 --> y_hat_30'.format(main_features[i]))

    y_hat_30[:,feature_8_ind[i]] = y_hat_8[:,i]
def plot_face_pts(img, pts):

    plt.imshow(img[:,:,0], cmap='gray')

    for i in range(1,31,2):

        plt.plot(pts[i-1], pts[i], 'b.')
fig = plt.figure(figsize=(10, 10))

fig.subplots_adjust(

    left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i, f in enumerate(range(40,49)):

    ax = fig.add_subplot(3, 3, i + 1, xticks=[], yticks=[])

    plot_face_pts(X_test[i],y_hat_30[i])

plt.show()
#All required features in order.

required_features = list(lookid_data['FeatureName'])

#All images nmber in order.

imageID = list(lookid_data['ImageId']-1)

#Generate Directory to map feature name 'Str' into int from 0 to 29.

feature_to_num = dict(zip(required_features[0:30], range(30)))
feature_ind = []

for f in required_features:

    feature_ind.append(feature_to_num[f])
y_hat_30[0,1]
required_pred = []

for x,y in zip(imageID,feature_ind):

    required_pred.append(y_hat_30[x, y])
feature_names = list(lookid_data['FeatureName'])

image_ids = list(lookid_data['ImageId']-1)

row_ids = list(lookid_data['RowId'])



feature_list = []

for feature in feature_names:

    feature_list.append(feature_names.index(feature))

    

predictions = []

for x,y in zip(image_ids, feature_list):

    predictions.append(y_hat_30[x][y])

    

row_ids = pd.Series(row_ids, name = 'RowId')

locations = pd.Series(predictions, name = 'Location')

locations = locations.clip(0.0,96.0)

submission_result = pd.concat([row_ids,locations],axis = 1)

submission_result.to_csv('combo_model.csv',index = False)