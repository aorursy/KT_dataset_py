# import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
from skimage.io import imread, imshow
from skimage.transform import resize
import tensorflow as tf
## Create Numpy Dataset
os.listdir("../input/train/images")[:10]
filename = os.listdir("../input/train/images")[0]
img = imread( "../input/train/images/" + filename, as_gray=True)
img.shape
img[:,:]
imshow( img[:,:])
mask = imread( "../input/train/masks/" + filename, as_gray=True)/255/257
print(mask.shape)
imshow( mask )
print( mask[:10,:10] )
m = resize( np.array(mask), (128,128), mode='constant', preserve_range=True)
print( m[-10:,-10:] )
train_imgs = np.array([ resize( imread("../input/train/images/" + filename, as_gray=True), (128, 128),mode='constant',preserve_range=True )/255 for filename in os.listdir("../input/train/images") ])
train_masks = np.array([ resize(imread("../input/train/masks/" + filename, as_gray=True), (128,128),mode='constant',preserve_range=True)/255/257 for filename in os.listdir("../input/train/images") ])
train_ids = np.array([ filename for filename in os.listdir("../input/train/images")])
print( train_imgs.shape )
print( train_masks.shape )
print( train_ids.shape )
test_imgs = np.array([ resize(imread("../input/test/images/" + filename, as_gray=True),(128,128),mode='constant',preserve_range=True) for filename in os.listdir("../input/test/images") ])
test_ids = np.array([filename for filename in os.listdir("../input/test/images")])
print( test_imgs.shape )
print( test_ids.shape )
imshow( train_masks[0,:,:])
train_masks[0,-10:,-10:]
from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid, id_train, id_valid = train_test_split( 
    train_imgs.reshape( -1, 128, 128, 1),
    train_masks.reshape(-1, 128, 128, 1),
    train_ids,
    test_size = 0.2, random_state=14
)
x_test = test_imgs.reshape(-1, 128, 128, 1)
train_masks[0,:,:].max()
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


img_size_ori = 101
img_size_target = 128

def upsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)
    
def downsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)
from keras.models import Sequential
from keras.layers import Dropout, Conv2D, MaxPooling2D, Input, Conv2DTranspose, concatenate
def build_model( input_layer, start_neurons):
    conv1 = Conv2D( start_neurons*1, (3,3), activation="relu", padding="same")(input_layer)
    conv1 = Conv2D( start_neurons*1, (3,3), activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D( (2,2) )(conv1)
    pool1 = Dropout( 0.25 )(pool1)
    
    conv2 = Conv2D( start_neurons*2, (3,3), activation="relu", padding="same")(pool1)
    conv2 = Conv2D( start_neurons*2, (3,3), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D( (2,2) )(conv2)
    pool2 = Dropout( 0.5 )(pool2)
    
    conv3 = Conv2D( start_neurons*4, (3,3), activation="relu", padding="same")(pool2)
    conv3 = Conv2D( start_neurons*4, (3,3), activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D( (2,2) )(conv3)
    pool3 = Dropout(0.5)(pool3)
    
    conv4 = Conv2D( start_neurons*8, (3,3), activation="relu", padding="same")(pool3)
    conv4 = Conv2D( start_neurons*8, (3,3), activation="relu", padding="same")(conv4)
    pool4 = MaxPooling2D( (2,2) )(conv4)
    pool4 = Dropout(0.5)(pool4)
    
    convm = Conv2D( start_neurons*16, (3,3), activation="relu", padding="same")(pool4)
    convm = Conv2D( start_neurons*16, (3,3), activation="relu", padding="same")(convm)
    
    deconv4 = Conv2DTranspose(start_neurons*8, (3,3), strides=(2,2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(0.5)(uconv4)
    uconv4 = Conv2D( start_neurons*8, (3,3), activation="relu", padding="same")(uconv4)
    uconv4 = Conv2D( start_neurons*8, (3,3), activation="relu", padding="same")(uconv4)

    deconv3 = Conv2DTranspose(start_neurons*4, (3,3), strides=(2,2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(0.5)(uconv3)
    uconv3 = Conv2D( start_neurons*4, (3,3), activation="relu", padding="same")(uconv3)
    uconv3 = Conv2D( start_neurons*4, (3,3), activation="relu", padding="same")(uconv3)
    
    deconv2 = Conv2DTranspose(start_neurons*2, (3,3), strides=(2,2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(0.5)(uconv2)
    uconv2 = Conv2D( start_neurons*2, (3,3), activation="relu", padding="same")(uconv2)
    uconv2 = Conv2D( start_neurons*2, (3,3), activation="relu", padding="same")(uconv2)
    
    deconv1 = Conv2DTranspose(start_neurons*2, (3,3), strides=(2,2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(0.5)(uconv1)
    uconv1 = Conv2D( start_neurons*1, (3,3), activation="relu", padding="same")(uconv1)
    uconv1 = Conv2D( start_neurons*1, (3,3), activation="relu", padding="same")(uconv1)
    
    output_layer = Conv2D( 1, (1,1), padding="same", activation="sigmoid")(uconv1)
    
    return output_layer
y_train[0,:10,:10]
x_train[0,:10,:10]
from keras import Model
input_layer = Input( (128,128,1))
output_layer = build_model( input_layer, 8)
model = Model( input_layer, output_layer)
model.compile( loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
model.fit( x_train, y_train, validation_data=[x_valid, y_valid],
         epochs = 3, batch_size = 32 )
