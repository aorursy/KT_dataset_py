# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import imgaug
import glob
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score

import cv2
import imgaug as ia
import imgaug.augmenters as iaa


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
sam = pd.read_csv('../input/sample_submission.csv')
sam['label'].value_counts()

train_labels = pd.read_csv('../input/train_labels.csv')
train_labels.head()
train_label_mapping = {k:v for k,v in zip(train_labels['id'].values,train_labels['label'].values)}
train_label_mapping
train_imgs_path = glob.glob('../input/train/*.tif') #file paths of all train images
test_imgs_path = glob.glob('../input/test/*.tif') #filepaths of all test images
train_imgs_path[:5]
test_imgs_path[:5]
print("train_img size :",len(train_imgs_path))
print("test_imgs size :",len(test_imgs_path))
def get_id_from_img_path(img_path):
    return img_path.split(os.path.sep)[-1].replace('.tif','')
train_imgs_path = train_imgs_path[:80000]
train,test = train_test_split(train_imgs_path, test_size = 0.2)

def get_label(df):
    
    y = []
    for path in df:
        img_id = get_id_from_img_path(path)
        y.append(train_label_mapping[img_id])
    return y

y_train = get_label(train)
y_train = y_train[:50000]
y_test = get_label(test)







ia.seed(1)

# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second
# image.
sometimes = lambda aug: iaa.Sometimes(0.5, aug)
def get_seq():
    # Define our sequence of augmentation steps that will be applied to every image.
    seq = iaa.Sequential([
        iaa.Fliplr(0.5), #horizontally flip 50% images
        iaa.Flipud(0.2), #vertically flip 20% images
        
        # crop some of the images by 0-10% of their height/width
        sometimes(iaa.Crop(percent=(0,0.1))),
        
        # Apply affine transformations to some of the images
        # - scale to 80-120% of image height/width (each axis independently)
        # - translate by -20 to +20 relative to height/width (per axis)
        # - rotate by -45 to +45 degrees
        # - shear by -16 to +16 degrees
        # - order: use nearest neighbour or bilinear interpolation (fast)
        # - mode: use any available mode to fill newly created pixels
        #         see API or scikit-image for which modes are available
        # - cval: if the mode is constant, then use a random brightness
        #         for the newly created pixels (e.g. sometimes black,
        #         sometimes white)
        sometimes(iaa.Affine(
            scale = {"x":(0.8,1.2), "y":(0.8,1.2)},
            translate_percent = {"x":(-0.2,0.2), "y":(-0.2,0.2)},
            rotate = (-45,45),
            shear = (-16,16),
            order = [0,1],
            cval = (0,255),
            mode = ia.ALL
        )),
        
        #
        # Execute 0 to 5 of the following (less important) augmenters per
        # image. Don't execute all of them, as that would often be way too
        # strong.
        #
        iaa.SomeOf((0,5),
                   [
                       # Convert some images into their superpixel representation,
                       # sample between 20 and 200 superpixels per image, but do
                       # not replace all superpixels with their average, only
                       # some of them (p_replace).
                       sometimes(
                           iaa.Superpixels(
                           p_replace = (0, 1.0),
                           n_segments = (20, 200)
                           )
                       ),
                       
                       # Blur each image with varying strength using
                       # gaussian blur (sigma between 0 and 3.0),
                       # average/uniform blur (kernel size between 2x2 and 7x7)
                       # median blur (kernel size between 3x3 and 11x11).
                       iaa.OneOf([
                           iaa.GaussianBlur((0,3.0)),
                           iaa.AverageBlur(k=(2,6)),
                           iaa.MedianBlur(k=(3,7))
                       ]),
                       
                       # Sharpen each image, overlay the result with the original
                       # image using an alpha between 0 (no sharpening) and 1
                       # (full sharpening effect).
                       iaa.Sharpen(alpha=(0,1.0), lightness = (0.75, 1.5)),
                       
                       # Same as sharpen, but for an embossing effect.
                       iaa.Emboss(alpha=(0,1.0), strength=(0,2.0)),
                       
                       # Search in some images either for all edges or for
                       # directed edges. These edges are then marked in a black
                       # and white image and overlayed with the original image
                       # using an alpha of 0 to 0.7.
                       sometimes(iaa.OneOf([
                           iaa.EdgeDetect(alpha = (0,0.7)),
                           iaa.DirectedEdgeDetect(alpha=(0,0.7), direction=(0.0,1.0))
                       ])),
                       
                       # Add gaussian noise to some images.
                       # In 50% of these cases, the noise is randomly sampled per
                       # channel and pixel.
                       # In the other 50% of all cases it is sampled once per
                       # pixel (i.e. brightness change).
                       iaa.AdditiveGaussianNoise(loc=0,scale=(0.0,0.05*255),per_channel=0.5),

                       # Either drop randomly 1 to 10% of all pixels (i.e. set
                       # them to black) or drop them on an image with 2-5% percent
                       # of the original size, leading to large dropped
                       # rectangles.
                       iaa.OneOf([
                           iaa.Dropout((0.01, 0.1),per_channel=0.5),
                           iaa.CoarseDropout((0.03,0.15),size_percent=(0.02,0.05), per_channel=0.2)
                       ]),
                       
                       # Invert each image's chanell with 5% probability.
                       # This sets each pixel value v to 255-v.
                       iaa.Invert(0.05, per_channel=True), #Invert colour channels
                       
                       # Add a value of -10 to 10 to each pixel.
                       iaa.Add((-10,10), per_channel=0.5),
                       
                       # Change brightness of images (50-150% of original value).
                       iaa.Multiply((0.5,1.5), per_channel=0.5),
                       
                       # Improve or worsen the contrast of images.
                       iaa.ContrastNormalization((0.5,2.0),per_channel=0.5),
                       
                       # Convert each image to grayscale and then overlay the
                       # result with the original with random alpha. I.e. remove
                       # colors with varying strengths.
                       iaa.Grayscale(alpha=(0.0,1.0)),
                       
                       # In some images move pixels locally around (with random
                       # strengths).
                       sometimes(iaa.ElasticTransformation(alpha=(0.5,3.5),sigma=0.25)),
                       
                       # In some images distort local areas with varying strength.
                       sometimes(iaa.PiecewiseAffine(scale=(0.01,0.05)))
                   ],
                   #do all the above augmentations in random order
                   random_order = True
                  )
            ],
            random_order = True
        )
    return seq    
def get_batch(data, batch_size):
    return (data[i:i+batch_size] for i in range(0, len(data), batch_size))    
seq = get_seq() 
train_imgs = [cv2.imread(img_path) for img_path in train]
test_imgs = [cv2.imread(img_path) for img_path in test]
test_imgs = np.array(test_imgs)
np.array(train_imgs).shape
np.array(test_imgs).shape
train_imgs = train_imgs[:50000]
train_imgs = np.array(train_imgs)
test_imgs = np.array(test_imgs)
train_imgs = seq.augment_images(train_imgs)

# def data_aug(img_data,train_label_mapping,batch_size,augment = False):
#     seq = get_seq()
#     while True:
#         random.shuffle(img_data)
#         for batch in get_batch(img_data,batch_size):
#             X = [cv2.imread(img_path) for img_path in batch]
#             y = [train_label_mapping[get_id_from_img_path(img_path)] for img_path in batch]
            
#             if augment:
#                 X = seq.augment_images(X)
    
#             yield np.array(X), np.array(y)
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from tensorflow import set_random_seed
set_random_seed(42)

model = Sequential()
model.add(Conv2D(filters = 16, kernel_size = 3, padding = 'same', activation = 'relu', input_shape = (96, 96, 3)))
model.add(Conv2D(filters = 16, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 16, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size = 3)) 

model.add(Conv2D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu')) 
model.add(Conv2D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu')) 
model.add(Conv2D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size = 3)) 

model.add(Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size = 3))

model.add(Conv2D(filters = 128, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 128, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 256, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size = 3))
model.add(Flatten())
model.add(Dense(1, activation = 'sigmoid'))
model.compile(optimizer= 'adam', loss='binary_crossentropy', metrics=['accuracy'])
batch_size = 128
model.fit(train_imgs,np.array(y_train), batch_size = batch_size,epochs=16)
train_imgs, y_train = next(g)
pred = (model.predict(test_imgs).ravel()*model.predict(test_imgs[:,::-1,:,:]).ravel()*model.predict(test_imgs[:,:,::-1,:]).ravel()*model.predict(test_imgs[:,::-1,::-1,:]).ravel())**0.25
pred
roc_auc_score(np.array(y_test),pred)

for i in range(len(pred)):
    if pred[i] < 0.5:
        pred[i] = 0
    else:
        pred[i] = 1
pred

test_img = [cv2.imread(imgpath) for imgpath in test_imgs_path]
test_img = np.array(test_img)
test_img.shape
preds = (model.predict(test_img).ravel()*model.predict(test_img[:,::-1,:,:]).ravel()*model.predict(test_img[:,:,::-1,:]).ravel()*model.predict(test_img[:,::-1,::-1,:]).ravel())**0.25
preds
# for i in range(len(preds)):
#     if preds[i] < 0.5:
#         preds[i] = 0
#     else:
#         preds[i] = 1

test_imgs_path[:5]
id = []
for path in test_imgs_path:
    id.append(get_id_from_img_path(path))
df = pd.DataFrame({'id':id,'label':preds})
df.to_csv("sub1.csv",index=False)
