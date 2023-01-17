### Import libraries
import cv2
import pandas as pd
import numpy as np
import os
from tqdm import tqdm, tqdm_notebook
from keras.applications.densenet import preprocess_input, DenseNet201
## define params
train_df = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')
img_size = 256
batch_size = 64
ids = train_df['image_name'].values
# Takes the id of each image
n_batches = len(ids) // batch_size + 1
# Number of batches = length of ids divided by batch size + 1
## Resize to square
def resize_to_square(im):
    # Old size
    old_size = im.shape[:2] # old_size is in (height, width) format
    ratio = float(img_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    # We use this new size to resize images
    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]))
    delta_w = img_size - new_size[1]
    delta_h = img_size - new_size[0]
    # Delta width is the change in width
    ## Same for delta height
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    ## Define top and bottom dim
    ## Define left and right too
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)
    ## Square image
    return new_im

def load_image(path, ids):
    image = cv2.imread(f'{path}{ids}-1.jpg') # read new image
    new_image = resize_to_square(image) # resize to square
    new_image = preprocess_input(new_image) # now preprocess inputs for DenseNet
    return new_image
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling1D
import keras.backend as K

# input image
inp = Input((256,256,3))
# DenseNet model
backbone = DenseNet121(input_tensor = inp, include_top = False)
# To make sure we do not load the full thing
# we load the densenet output
x = backbone.output
# Make the output smaller (from 1024 output params)
x = GlobalAveragePooling2D()(x)
# Expands dimensions (very useful)
x = Lambda(lambda x: K.expand_dims(x,axis = -1))(x)
# Finally pools to 4
x = AveragePooling1D(4)(x)
# final output
out = Lambda(lambda x: x[:,:,0])(x)

m = Model(inp,out)
features = {}
for b in tqdm_notebook(range(n_batches)):
    start = b*batch_size
    end = (b+1)*batch_size
    batch_pets = ids[start:end]
    batch_images = np.zeros((len(batch_pets),img_size,img_size,3))
    for i,pet_id in enumerate(batch_pets):
        try:
            batch_images[i] = load_image("../input/siim-isic-melanoma-classification/train_images/", pet_id)
        except:
            pass
    batch_preds = m.predict(batch_images)
    for i,ids in enumerate(batch_pets):
        features[ids] = batch_preds[i]
train_feats = pd.DataFrame.from_dict(features, orient='index')
train_feats.to_csv('train_feats.csv') # convert to csv