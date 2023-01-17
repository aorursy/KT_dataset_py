import matplotlib.pyplot as plt
import cv2
img = cv2.imread('../input/image-files/pichai2.jpg',1)
plt.imshow(img) # default value is BGR

print("The Dimension of the matrix is ", img.shape)
# print("Matrix value of the blue color \n",img[:,:,0], "\n Y,X Pixels:",img[:,:,0].shape) ## 0 refer blue, 1 refer green and 2 refer red

fig=plt.figure(figsize=(20, 20))
fig.add_subplot(1, 4, 1)
#####################################################################
### Converting BGR to RGB
plt.imshow(img[:,:,[2,1,0]])
## im = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # using cvtcolor function also we can change the color
## plt.imshow(im)

fig.add_subplot(1, 4, 2)
## Select the pixels of the face
plt.imshow(img[40:275,500:750,[2,1,0]])


fig.add_subplot(1, 4, 3)
## Select the pixels of the eyes
plt.imshow(img[120:160,550:600,[2,1,0]])

fig.add_subplot(1, 4, 4)
## Select the pixels of the nose
plt.imshow(img[140:190,600:650,[2,1,0]])
plt.show()
color = ('r','g','b')

for i,col in enumerate(color):
    histogram2 = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histogram2,color = col)
    plt.xlim([0,256])
plt.show()    
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
def data_loader(data_frame):
    
    # Load dataset file
   
    data_frame['Image'] = data_frame['Image'].apply(lambda i: np.fromstring(i, sep=' '))
    data_frame = data_frame.dropna()  # Get only the data with 15 keypoints
   
    # Extract Images pixel values
    imgs_array = np.vstack(data_frame['Image'].values)/ 255.0
    imgs_array = imgs_array.astype(np.float32)    # Normalize, target values to (0, 1)
    imgs_array = imgs_array.reshape(-1, 96, 96, 1)
        
    # Extract labels (key point cords)
    labels_array = data_frame[data_frame.columns[:-1]].values
    labels_array = (labels_array - 48) / 48    # Normalize, traget cordinates to (-1, 1)
    labels_array = labels_array.astype(np.float32) 
    
    # shuffle the train data
#     imgs_array, labels_array = shuffle(imgs_array, labels_array, random_state=9)  
    
    return imgs_array, labels_array

def data_loader_test(data_frame):
    
    # Load dataset file
   
    data_frame['Image'] = data_frame['Image'].apply(lambda i: np.fromstring(i, sep=' '))
  
    # Extract Images pixel values
    imgs_array = np.vstack(data_frame['Image'].values)/ 255.0
    imgs_array = imgs_array.astype(np.float32)    # Normalize, target values to (0, 1)
    imgs_array = imgs_array.reshape(-1, 96, 96, 1)
    
    return imgs_array


X,Y = data_loader(training)
X_test = data_loader_test(test)
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.4, random_state=42)
print("Train sample:",X_train.shape,"Val sample:",X_val.shape)
class DataModifier(object):
    def fit(self,X_,y_):
        return(NotImplementedError)
    
class FlipPic(DataModifier):
    def __init__(self,flip_indices=None):
        if flip_indices is None:
            flip_indices = [
                (0, 2), (1, 3),
                (4, 8), (5, 9), (6, 10), (7, 11),
                (12, 16), (13, 17), (14, 18), (15, 19),
                (22, 24), (23, 25)
                ]
        
        self.flip_indices = flip_indices
        
    def fit(self,X_batch,y_batch):

        batch_size = X_batch.shape[0]
        indices = np.random.choice(batch_size, batch_size//2, replace=False)

        X_batch[indices] = X_batch[indices, :, ::-1,:]
        y_batch[indices, ::2] = y_batch[indices, ::2] * -1

        # flip left eye to right eye, left mouth to right mouth and so on .. 
        for a, b in self.flip_indices:
            y_batch[indices, a], y_batch[indices, b] = (
                    y_batch[indices, b], y_batch[indices, a]
                )
        return X_batch, y_batch
class ShiftFlipPic(FlipPic):
    def __init__(self,flip_indices=None,prop=0.1):
        super(ShiftFlipPic,self).__init__(flip_indices)
        self.prop = prop
        
    def fit(self,X,y):
        X, y = super(ShiftFlipPic,self).fit(X,y)
        X, y = self.shift_image(X,y,prop=self.prop)
        return(X,y)
    def random_shift(self,shift_range,n=96):
        '''
        :param shift_range: 
        The maximum number of columns/rows to shift
        :return: 
        keep(0):   minimum row/column index to keep
        keep(1):   maximum row/column index to keep
        assign(0): minimum row/column index to assign
        assign(1): maximum row/column index to assign
        shift:     amount to shift the landmark

        assign(1) - assign(0) == keep(1) - keep(0)
        '''
        shift = np.random.randint(-shift_range,
                                  shift_range)
        def shift_left(n,shift):
            shift = np.abs(shift)
            return(0,n - shift)
        def shift_right(n,shift):
            shift = np.abs(shift)
            return(shift,n)

        if shift < 0:
            keep = shift_left(n,shift) 
            assign = shift_right(n,shift)
        else:
            assign = shift_left(n,shift) ## less than 96
            keep = shift_right(n,shift)

        return((keep,  assign, shift))

    def shift_single_image(self,x_,y_,prop=0.1):
        '''
        :param x_: a single picture array (96, 96, 1)
        :param y_: 15 landmark locations 
                   [0::2] contains x axis values
                   [1::2] contains y axis values 
        :param prop: proportion of random horizontal and vertical shift
                     relative to the number of columns
                     e.g. prop = 0.1 then the picture is moved at least by 
                     0.1*96 = 8 columns/rows
        :return: 
        x_, y_
        '''
        w_shift_max = int(x_.shape[0] * prop)
        h_shift_max = int(x_.shape[1] * prop)

        w_keep,w_assign,w_shift = self.random_shift(w_shift_max)
        h_keep,h_assign,h_shift = self.random_shift(h_shift_max)

        x_[w_assign[0]:w_assign[1],
           h_assign[0]:h_assign[1],:] = x_[w_keep[0]:w_keep[1],
                                           h_keep[0]:h_keep[1],:]

        y_[0::2] = y_[0::2] - h_shift/float(x_.shape[0]/2.)
        y_[1::2] = y_[1::2] - w_shift/float(x_.shape[1]/2.)
        return(x_,y_)

    def shift_image(self,X,y,prop=0.1):
            ## This function may be modified to be more efficient e.g. get rid of loop?
            for irow in range(X.shape[0]):
                x_ = X[irow]
                y_ = y[irow]
                X[irow],y[irow] = self.shift_single_image(x_,y_,prop=prop)
            return(X,y)
from keras.preprocessing.image import ImageDataGenerator

generator = ImageDataGenerator()
modifier = FlipPic()
fig = plt.figure(figsize=(20,20))
count = 1
for batch in generator.flow(X[:4],Y[:4]):
    X_batch, y_batch = modifier.fit(*batch)
    ax = fig.add_subplot(5,4, count,xticks=[],yticks=[])  
    plot_sample(X_batch[0],y_batch[0],ax)
    count += 1
    if count == 10:
        break
plt.show()
from keras.preprocessing.image import ImageDataGenerator
generator = ImageDataGenerator()
shiftFlipPic = ShiftFlipPic(prop=0.1)

fig = plt.figure(figsize=(20,20))

count = 1
for batch in generator.flow(X[:4],Y[:4]):
    X_batch, y_batch = shiftFlipPic.fit(*batch)

    ax = fig.add_subplot(5,4, count,xticks=[],yticks=[])  
    plot_sample(X_batch[0],y_batch[0],ax)
    count += 1
    if count == 10:
        break
plt.show()
generator = ImageDataGenerator()
modifier = FlipPic()
shiftFlipPic_1 = ShiftFlipPic(prop=0.02)
shiftFlipPic_2 = ShiftFlipPic(prop=0.04)
shiftFlipPic_3 = ShiftFlipPic(prop=0.06)
shiftFlipPic_4 = ShiftFlipPic(prop=0.08)
shiftFlipPic_5 = ShiftFlipPic(prop=0.1)
shiftFlipPic_6 = ShiftFlipPic(prop=0.12)
shiftFlipPic_7 = ShiftFlipPic(prop=0.14)
batches = 0
for batch in generator.flow(X_train,y_train):
    X_batch, y_batch = modifier.fit(*batch)
    y_train = np.concatenate((y_train,y_batch))
    X_train = np.concatenate((X_train,X_batch))
    X_batch, y_batch = modifier.fit(*batch)
    y_train = np.concatenate((y_train,y_batch))
    X_train = np.concatenate((X_train,X_batch))
    X_batch, y_batch = modifier.fit(*batch)
    y_train = np.concatenate((y_train,y_batch))
    X_train = np.concatenate((X_train,X_batch))
    X_batch, y_batch = shiftFlipPic_1.fit(*batch)
    y_train = np.concatenate((y_train,y_batch))
    X_train = np.concatenate((X_train,X_batch))
    X_batch, y_batch = shiftFlipPic_2.fit(*batch)
    y_train = np.concatenate((y_train,y_batch))
    X_train = np.concatenate((X_train,X_batch))
    X_batch, y_batch = shiftFlipPic_3.fit(*batch)
    y_train = np.concatenate((y_train,y_batch))
    X_train = np.concatenate((X_train,X_batch))
    X_batch, y_batch = shiftFlipPic_4.fit(*batch)
    y_train = np.concatenate((y_train,y_batch))
    X_train = np.concatenate((X_train,X_batch))
    X_batch, y_batch = shiftFlipPic_5.fit(*batch)
    y_train = np.concatenate((y_train,y_batch))
    X_train = np.concatenate((X_train,X_batch))
    X_batch, y_batch = shiftFlipPic_6.fit(*batch)
    y_train = np.concatenate((y_train,y_batch))
    X_train = np.concatenate((X_train,X_batch))
    X_batch, y_batch = shiftFlipPic_7.fit(*batch)
    y_train = np.concatenate((y_train,y_batch))
    X_train = np.concatenate((X_train,X_batch))
    batches += 1
    if batches >= 40:
#         len(X_train) / 32
        # we need to break the loop by hand because
        # the generator loops indefinitely
        break  

print("Train sample:",X_train.shape,"Val sample:",X_val.shape)
from keras import backend
 
def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

from keras.models import Sequential, Model
from keras.layers import Activation, Convolution2D, MaxPooling2D, BatchNormalization, AvgPool2D, Flatten, Dense, Dropout, Conv2D,MaxPool2D, ZeroPadding2D
from keras.optimizers import Adam, SGD, RMSprop
from keras import regularizers
import keras.backend as K
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import numpy as np
from keras.layers.advanced_activations import LeakyReLU
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

def baseline_model(optimizer=1, init='glorot_uniform',momentum=0.8):
    model = Sequential()
    model.add(Convolution2D(32, (3,3), padding='same', use_bias=False, kernel_initializer=init, activation='relu', input_shape=(96,96,1)))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(64, (3,3), padding='same', use_bias=False, kernel_initializer=init, activation='relu'))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(96, (3,3), padding='same', use_bias=False, kernel_initializer=init, activation='relu'))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(128, (3,3),padding='same', use_bias=False, kernel_initializer=init, activation='relu'))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(256, (3,3),padding='same',use_bias=False, kernel_initializer=init, activation='relu'))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(512, (3,3), padding='same', use_bias=False, kernel_initializer=init, activation='relu'))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))


    model.add(Flatten())
    model.add(Dense(4608,kernel_initializer=init,activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(30))
    filepath="weights-improvement_1.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    if optimizer==1:
        optimizer_fn = SGD(lr=0.01, momentum=momentum, decay=0.0004, nesterov=False)
    elif optimizer==2:
        optimizer_fn = Adam(lr=0.01, momentum=momentum, decay=0.0004, nesterov=False)
    else:
        optimizer_fn = RMSprop(lr=0.01, momentum=momentum, decay=0.0004, nesterov=False)
    model.compile(optimizer= optimizer_fn,loss='mean_squared_error',metrics=[rmse])
    return model

momentum = [0.6,0.8, 0.9]
optimizer = [1,2]
init = ['glorot_uniform','normal']
#         ,'normal','uniform']

param_grid = dict(momentum=momentum, optimizer=optimizer,init=init)


model = KerasRegressor(build_fn=baseline_model, epochs=25)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid.fit(X_train,y_train)


# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
# model = Sequential()
# model.add(Convolution2D(32, (3,3), padding='same', use_bias=False, activation='relu', input_shape=(96,96,1)))
# model.add(LeakyReLU(alpha = 0.1))
# model.add(BatchNormalization())

# model.add(Convolution2D(64, (3,3), padding='same', use_bias=False,activation='relu'))
# model.add(LeakyReLU(alpha = 0.1))
# model.add(BatchNormalization())
# model.add(MaxPool2D(pool_size=(2, 2)))

# model.add(Convolution2D(64, (3,3), padding='same', use_bias=False,activation='relu'))
# model.add(LeakyReLU(alpha = 0.1))
# model.add(BatchNormalization())
# model.add(MaxPool2D(pool_size=(2, 2)))

# model.add(Convolution2D(96, (3,3), padding='same', use_bias=False,activation='relu'))
# model.add(LeakyReLU(alpha = 0.1))
# model.add(BatchNormalization())

# model.add(Convolution2D(96, (3,3), padding='same', use_bias=False,activation='relu'))
# model.add(LeakyReLU(alpha = 0.1))
# model.add(BatchNormalization())
# model.add(MaxPool2D(pool_size=(2, 2)))

# model.add(Convolution2D(128, (3,3),padding='same', use_bias=False,activation='relu'))
# # model.add(BatchNormalization())
# model.add(LeakyReLU(alpha = 0.1))
# model.add(BatchNormalization())

# model.add(Convolution2D(128, (3,3),padding='same', use_bias=False,activation='relu'))
# model.add(LeakyReLU(alpha = 0.1))
# model.add(BatchNormalization())
# model.add(MaxPool2D(pool_size=(2, 2)))

# model.add(Convolution2D(256, (3,3),padding='same',use_bias=False,activation='relu'))
# model.add(LeakyReLU(alpha = 0.1))
# model.add(BatchNormalization())

# model.add(Convolution2D(256, (3,3),padding='same',use_bias=False,activation='relu'))
# model.add(LeakyReLU(alpha = 0.1))
# model.add(BatchNormalization())
# model.add(MaxPool2D(pool_size=(2, 2)))

# model.add(Convolution2D(512, (3,3), padding='same', use_bias=False,activation='relu'))
# model.add(LeakyReLU(alpha = 0.1))
# model.add(BatchNormalization())

# model.add(Convolution2D(512, (3,3), padding='same', use_bias=False,activation='relu'))
# model.add(LeakyReLU(alpha = 0.1))
# model.add(BatchNormalization())


# model.add(Flatten())
# model.add(Dense(512,activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(30))
# model.summary()
# # model.compile(A(lr=0.0001),loss='mean_squared_error',  metrics=[rmse,mae,own_loss])


# model.compile(optimizer=SGD(lr=0.1,momentum = 0.9,decay=0.0004, nesterov=True), loss='mean_squared_error',metrics=[rmse,'mse', 'mae'])
# print(model.summary())
# filepath="weights-improvement_1.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='min')
# callbacks_list = [checkpoint]
# # Fit the model

hist = model.fit(X_train,y_train,validation_data=(X_val, y_val), epochs = 250,verbose=0)
scores = model.evaluate(X_train, y_train, verbose=0)
print(" Train %s: %.2f%% %s: %.2f%% %s: %.2f%%" % (model.metrics_names[1], scores[1]*100,model.metrics_names[2], scores[2]*100,model.metrics_names[3], scores[3]*100))
scores = model.evaluate(X_val, y_val, verbose=0)
print(" Val %s: %.2f%% %s: %.2f%% %s: %.2f%%" % (model.metrics_names[1], scores[1]*100,model.metrics_names[2], scores[2]*100,model.metrics_names[3], scores[3]*100))
def plot_loss(hist,name,plt,RMSE_TF=False):
    '''
    RMSE_TF: if True, then RMSE is plotted with original scale 
    '''
    loss = hist['rmse']
    val_loss = hist['rmse']
    if RMSE_TF:
        loss = np.sqrt(np.array(loss))*48 
        val_loss = np.sqrt(np.array(val_loss))*48 
        
    plt.plot(loss,"--",linewidth=3,label="train:"+name)
    plt.plot(val_loss,linewidth=3,label="val:"+name)

plot_loss(hist.history,"model 1",plt)
plt.legend()
plt.grid()
plt.xlabel("epoch")
plt.ylabel("RMSE")
plt.show()
  
pred = model.predict(X_val)

fig = plt.figure(figsize=(7, 7))
fig.subplots_adjust(hspace=0.13,wspace=0.0001,
                    left=0,right=1,bottom=0, top=1)
Npicture = 9
count = 1
for irow in range(Npicture):
    ipic = np.random.choice(X_val.shape[0])
    ax = fig.add_subplot(Npicture/3 , 3, count,xticks=[],yticks=[])        
    plot_sample_val(X_val[ipic],y_val[ipic], ax,pred[ipic])
    ax.legend( ncol = 1)
    ax.set_title("picture "+ str(ipic))
    count += 1
plt.show()
pred = model.predict(X_test)
label_points = (np.squeeze(pred)*48)+48

feature_names = list(lookid_data['FeatureName'])
image_ids = list(lookid_data['ImageId']-1)
row_ids = list(lookid_data['RowId'])

feature_list = []
for feature in feature_names:
    feature_list.append(feature_names.index(feature))
    
predictions = []
for x,y in zip(image_ids, feature_list):
    predictions.append(label_points[x][y])
    
row_ids = pd.Series(row_ids, name = 'RowId')
locations = pd.Series(predictions, name = 'Location')
locations = locations.clip(0.0,96.0)
submission_result = pd.concat([row_ids,locations],axis = 1)
submission_result.to_csv('face_key_detection_submission_10.csv',index = False)
from keras.models import model_from_json

def save_model(model,name):
    '''
    save model architecture and model weights
    '''
    json_string = model.to_json()
    open(name+'_architecture.json', 'w').write(json_string)
    model.save_weights(name+'_weights.h5')
    
def load_model(name):
    model = model_from_json(open(name+'_architecture.json').read())
    model.load_weights(name + '_weights.h5')
    return(model)

save_model(model,"model1")
model = load_model("model1")