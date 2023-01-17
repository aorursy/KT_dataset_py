# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Random

import random

from tqdm import tqdm_notebook

# Library for reading images 

from PIL import Image

# Plotting library

from matplotlib import pyplot as plt
# TRAIN DATA and manipulations:

###############################



train_df= pd.read_csv("../input/recognizing-faces-in-the-wild/train_relationships.csv")

train_df.shape
train_df.tail()
families_number = sorted(os.listdir("../input/recognizing-faces-in-the-wild/train/"))

print("There are {} families in the train set". format(len(families_number)))

print(families_number[:10])

members = {i:sorted(os.listdir(("../input/recognizing-faces-in-the-wild/train/")+i)) for i in families_number}

#print(members)
#train_im = "../input/recognizing-faces-in-the-wild/train"

#train_images = os.listdir(train_im)

#print(train_images[:10])
# TEST DATA and manipulations:

###############################

test_path= "../input/recognizing-faces-in-the-wild/test"

test_images_name = os.listdir(test_path)

print("Number of images in test set:", len(test_images_name))
def load_img(PATH): 

    return np.array(Image.open(PATH))



def plots(ims, figsize=(12,6), rows=1, titles=None):

    f = plt.figure(figsize=figsize)

    for i in range(len(ims)):

        sp = f.add_subplot(rows, len(ims)//rows, i+1)

        sp.axis('Off')

        if titles is not None: sp.set_title(titles[i], fontsize=16)

        plt.imshow(ims[i])
# To convert images into a matrix

test_images=np.array([load_img(os.path.join(test_path,image)) for image in test_images_name])
print(test_images.shape)
plots(test_images[:10], rows =2)
# LOAD MODEL FROM KERAS

########################



from keras.applications.densenet import DenseNet121 #Load trained Densenet121



def CNNdensenet (weights=None):

    model = DenseNet121(include_top=True, weights=None, input_tensor=None, input_shape=None, pooling=None, classes=1000)

    return model
#path_weights = "../input/imagenet_models/densenet121_weights_tf.h5" #load trained weights

path_weights = "../input/imagenet_models/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5" #load trained weights



CNN= CNNdensenet(weights=path_weights)

CNN.summary()
# To check results with the train set:



im = Image.open('../input/recognizing-faces-in-the-wild/train/F1000/MID4/P10582_face1.jpg')

im = np.array(im).astype(np.float32)

im2 = Image.open('../input/recognizing-faces-in-the-wild/train/F1000/MID4/P10582_face1.jpg')

im2 = np.array(im2).astype(np.float32)



im = np.expand_dims(im, axis=0)



im2 = np.expand_dims(im2, axis=0)

np.concatenate([im,im2]).shape
out = CNN.predict(np.concatenate([im,im2]))



print("Out dimension, one probe with 2 elements of train set:", out.shape)

def distance(x, y):

    return np.linalg.norm(x - y)

train_check= print("distance between 2 images of training set:", distance(out[0], out[1]))
# Read test set, CAREFUL¡¡¡¡¡¡ with the time and To calculate embedings of TEST set:



test_images = os.listdir(test_path)

test = np.array([load_img(os.path.join(test_path, i)) for i in test_images])

test_emb = CNN.predict(test)

print(test.shape, test_emb.shape) # TEst_emp esta normalizado
# To assign index to each image:

image_index = {imagen_numero:idx for idx, imagen_numero in enumerate(test_images)}
submission = pd.read_csv('../input/recognizing-faces-in-the-wild/sample_submission.csv')



splitting = [i.split('-') for i in submission.img_pair]  #por "-" , se divide un campo en dos

print("number of pairs for testing:", len(splitting))



distances  = []

for i in splitting:

    a=i[0]

    b=i[1]

    dist= distance(test_emb[image_index[a]], test_emb[image_index[b]])

    distances.append(dist)



distances = np.array(distances)/np.max(distances)

probability = 1- ((distances/np.max(distances)))# (0.2 is a security coeffcient)

print("distances:", distances)

print("Max distance:", np.max(distances))

print("Min distance:", np.min(distances))

print("Sum distances:", distances.sum()) 

print("Probabilities:", probability)

print("Max probability:", np.max(probability))





submission.head()
submission.is_related = probability

submission.to_csv('submission_SMILE3.csv', index=False)

submission.head()
print(os.listdir("../working"))