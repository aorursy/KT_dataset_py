from glob import glob

import os



path_to_train = '../input/train/train'

glob_train_imgs = os.path.join(path_to_train, '*_sat.jpg')

glob_train_masks = os.path.join(path_to_train, '*_msk.png')



train_img_paths = glob(glob_train_imgs)

train_mask_paths = glob(glob_train_masks)

print(train_img_paths[:10])

print(train_mask_paths[:10])





path_to_val = '../input/val/val'

glob_val_imgs = os.path.join(path_to_val, '*_sat.jpg')

val_img_paths = glob(glob_val_imgs)

print(val_img_paths[:5])
from skimage.io import imread

from skimage.transform import resize

from skimage.color import rgb2gray



# This will be useful so we can construct the corresponding mask

def get_img_id(img_path):

    img_basename = os.path.basename(img_path)

    img_id = os.path.splitext(img_basename)[0][:-len('_sat')]

    return img_id



# Write it like a normal function

def image_gen(img_paths, img_size=(128, 128)):

    # Iterate over all the image paths

    for img_path in img_paths:

        

        # Construct the corresponding mask path

        img_id = get_img_id(img_path)

        mask_path = os.path.join(path_to_train, img_id + '_msk.png')

        

        # Load the image and mask, and normalize it to 0-1 range

        img = imread(img_path) / 255.

        mask = rgb2gray(imread(mask_path))

        

        # Resize the images

        img = resize(img, img_size, preserve_range=True)

        mask = resize(mask, img_size, mode='constant', preserve_range=True)

        # Turn the mask back into a 0-1 mask

        mask = (mask >= 0.5).astype(float)

        

        # Yield the image mask pair

        yield img, mask
import matplotlib.pyplot as plt

import numpy as np



ig = image_gen(train_img_paths)



#first_img, first_mask = next(ig)



for i in range(1): 

    

    img, mask = next(ig)  

    plt.imshow(img)

    plt.show()

    plt.imshow(mask, cmap='gray')

    plt.show()
import numpy as np

import pandas as pd



# Create submission DataFrame

def create_submission(csv_name, predictions_gen):

    """

    csv_name -> string for csv ("XXXXXXX.csv")

    predictions -> generator that yields a pair of id, prediction

    """

    sub = pd.DataFrame()

    ids = []

    encodings = []

    num_images = len(val_img_paths)

    for i in range(num_images):

        if (i+1) % (num_images//10) == 0:

            print(i, num_images)

        img_id, pred = next(predictions_gen)

        #print(pred)

        ids.append(img_id)

        encodings.append(rle_encoding(pred))

        

    sub['EncodedPixels'] = encodings

    sub['ImageId'] = ids

    #sub['Height'] = [512]*num_images Nonger needed for DICE Scoring

    #sub['Width'] = [512]*num_images Nonger needed for DICE Scoring

    sub.to_csv(csv_name, index=False)



# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python

def rle_encoding(x):

    """

    x = numpyarray of size (height, width) representing the mask of an image

    if x[i,j] == 0:

        image[i,j] is not a road pixel

    if x[i,j] != 0:

        image[i,j] is a road pixel

    """

    dots = np.where(x.T.flatten() != 0)[0]

    run_lengths = []

    prev = -2

    for b in dots:

        if (b>prev+1): 

            run_lengths.extend((b+1, 0))

        run_lengths[-1] += 1

        prev = b

    return run_lengths
#the few images I selected to test the ANN with

images = ['../input/train/train/5398_sat.jpg',

          '../input/train/train/15254_sat.jpg',

          '../input/train/train/18859_sat.jpg',

          '../input/train/train/18_sat.jpg',

          '../input/train/train/19_sat.jpg',

          '../input/train/train/22_sat.jpg',

          '../input/train/train/48_sat.jpg',

          '../input/train/train/70_sat.jpg',

          '../input/train/train/2843_sat.jpg',

          '../input/train/train/6313_sat.jpg',

          '../input/train/train/4503_sat.jpg',

          '../input/train/train/30_sat.jpg',

          '../input/train/train/36_sat.jpg',

          '../input/train/train/841_sat.jpg',

          '../input/train/train/13677_sat.jpg',

          '../input/train/train/14078_sat.jpg',

          '../input/train/train/14091_sat.jpg',

          '../input/train/train/15062_sat.jpg',

          '../input/train/train/15043_sat.jpg',

          '../input/train/train/5911_sat.jpg',

          

         ]



ig = image_gen(images)

for i in range(len(images)):    

    img, mask = next(ig)

    #print (img)

    #print (mask)



    plt.imshow(img)

    plt.show()

    plt.imshow(mask, cmap='gray')

    plt.show()
from skimage.io import imread

from skimage.transform import resize

from skimage.color import rgb2gray

#selected the below images based on large quantity of roads and variety 

#get select pixels to run faster (around 500 total)

#road_images = ['../input/train/train/19_sat.jpg','../input/train/train/8_sat.jpg']

images = ['../input/train/train/5398_sat.jpg',

          '../input/train/train/15254_sat.jpg',

          '../input/train/train/18859_sat.jpg',

          '../input/train/train/18_sat.jpg',

          '../input/train/train/19_sat.jpg',

          '../input/train/train/22_sat.jpg',

          '../input/train/train/48_sat.jpg',

          '../input/train/train/70_sat.jpg',

          '../input/train/train/2843_sat.jpg',

          '../input/train/train/6313_sat.jpg',

          '../input/train/train/4503_sat.jpg',

          '../input/train/train/30_sat.jpg',

          '../input/train/train/36_sat.jpg',

          '../input/train/train/841_sat.jpg',

          '../input/train/train/13677_sat.jpg',

          '../input/train/train/14078_sat.jpg',

          '../input/train/train/14091_sat.jpg',

          '../input/train/train/15062_sat.jpg',

          '../input/train/train/15043_sat.jpg',

          '../input/train/train/45361_sat.jpg',

          '../input/train/train/45387_sat.jpg',

          '../input/train/train/45378_sat.jpg',

          '../input/train/train/45369_sat.jpg',

          '../input/train/train/183_sat.jpg',

          '../input/train/train/18373_sat.jpg',

          '../input/train/train/18313_sat.jpg',

          '../input/train/train/18304_sat.jpg',

          '../input/train/train/488_sat.jpg',

          '../input/train/train/537_sat.jpg',

          '../input/train/train/493_sat.jpg',

          '../input/train/train/480_sat.jpg',

          '../input/train/train/473_sat.jpg',    

          '../input/train/train/29702_sat.jpg', 

          '../input/train/train/49627_sat.jpg', 

          '../input/train/train/494_sat.jpg'

         ]

          

         



#ig = ANN_data_gen(train_img_paths)

def get_roads(img_paths, img_size = (128,128)):

    road_data = []

    road_labels = np.array([])

    for img_path in img_paths:

        img_id = get_img_id(img_path)

        mask_path = os.path.join(path_to_train, img_id + '_msk.png')      

        # Load the image and mask, and normalize it to 0-1 range

        img = imread(img_path) / 255.

        mask = rgb2gray(imread(mask_path))     

        # Resize the images

        img = resize(img, img_size, preserve_range=True)

        mask = resize(mask, img_size, mode='constant', preserve_range=True)

        # Turn the mask back into a 0-1 mask

        mask = (mask >= 0.5).astype(float)



        for j in range(2,126): #trim to 126 x 126

            for k in range(2,126):

                #print (img[j,k,:])

                #single_pixel = img[j,k,:]

                if mask[j,k] == 1: #and len(road_data)!=250:

                    single_example = img[j-2:j+3, k-2:k+3,:]

                    single_example = single_example.reshape((1,25,3)) #reshape into a row vector

                    #single_example = np.delete(single_example, 4, axis = 1) #delete the pixel itself from the example

                    #print (single_example[:,:,0])

                    #print (single_example.shape)

                    road_data.append(single_example)

                    #print (np.array(road_data).shape)

                    road_labels = np.append(road_labels, mask[j,k])

        

        

    road_data = np.array(road_data)   

    #road_data = road_data.reshape((250,1,8,3))

    return road_data, road_labels

    



def get_nonroads(img_paths, img_size = (128,128)):

    road_data = np.empty((0,1,25,3))

    road_labels = np.array([])

    for img_path in img_paths:

        img_id = get_img_id(img_path)

        mask_path = os.path.join(path_to_train, img_id + '_msk.png')      

        # Load the image and mask, and normalize it to 0-1 range

        img = imread(img_path) / 255.

        mask = rgb2gray(imread(mask_path))     

        # Resize the images

        img = resize(img, img_size, preserve_range=True)

        mask = resize(mask, img_size, mode='constant', preserve_range=True)

        # Turn the mask back into a 0-1 mask

        mask = (mask >= 0.5).astype(float)

        sub_data = []

        for j in range(2,126): #trim to 126 x 126

            for k in range(2,126):

                #print (img[j,k,:])

                #single_pixel = img[j,k,:]

                if mask[j,k] == 0: #and len(road_data)!=250:

                    single_example = img[j-2:j+3, k-2:k+3,:]

                    single_example = single_example.reshape((1,25,3)) #reshape into a row vector

                    #single_example = np.delete(single_example, 4, axis = 1) #delete the pixel itself from the example

                    #print (single_example[:,:,0])

                    #print (single_example.shape)

                    sub_data.append(single_example)

                    #print (np.array(road_data).shape)

                    road_labels = np.append(road_labels, mask[j,k])

        

        sub_data = np.array(sub_data)

        np.random.shuffle(sub_data)

        sub_data = sub_data[0:1500] #from each image take 1000 examples with no roads

        #print (sub_data.shape)

        road_data = np.vstack((road_data,sub_data))

        road_labels = road_labels[0:road_data.shape[0]]

    

    print (road_data.shape)

    return road_data, road_labels

    

    

r, rlabel = get_roads(images)

np.random.shuffle(r)

#r = r[0:1000]

#rlabel = rlabel[0:1000]

n, nlabel = get_nonroads(images)

"""

np.random.shuffle(n)

n = n[0:35000]

nlabel = nlabel[0:35000]

"""



#IMPORTANT- for some reason if you don't split 50/50 model will want to overfit and say 

#everything is not a road

X = np.vstack((r,n))



y = np.hstack((rlabel,nlabel))



print (X.shape)

print (y.shape)
#ANN model

import tensorflow as tf

from tensorflow import keras



model = keras.Sequential([

    keras.layers.Flatten(input_shape = (1,25,3)),

    #keras.layers.Dense(2048, activation = tf.nn.relu), #12 from paper

    keras.layers.Dense(15376, activation = tf.nn.relu), #12 from paper

    #keras.layers.Dense(12, activation = tf.nn.relu), #12 from paper

    keras.layers.Dense(2, activation = tf.nn.softmax) #two-classes (road or non-road)

])



model.compile(optimizer = 'adam',

              loss = 'sparse_categorical_crossentropy',

              metrics=['accuracy'])
#model.fit(ANN_data_gen(NN_images), steps_per_epoch = 15876, epochs = 5)



model.fit(X,y, epochs = 5)
#evaluate the accuracy on the training set

model.evaluate(X,y)
path_to_val = '../input/val/val'

glob_val_imgs = os.path.join(path_to_val, '*_sat.jpg')

val_img_paths = glob(glob_val_imgs)

val_img_paths = ['../input/val/val/68896_sat.jpg',

                 '../input/val/val/68784_sat.jpg',

                 '../input/val/val/68782_sat.jpg',

                 '../input/val/val/69056_sat.jpg',

                 '../input/val/val/68945_sat.jpg',

                 '../input/val/val/69050_sat.jpg',

                 '../input/val/val/69052_sat.jpg',

                 '../input/val/val/69519_sat.jpg',

                 '../input/val/val/69556_sat.jpg',

                 

                ]

#def generate_pixel_by_pixel_predictions_generator(val_paths):

    #while True:

i = 0

sub = pd.DataFrame()

ids = []

encodings = []

for img_path in val_img_paths:

    print (get_img_id(img_path))

    #print(i)

    #i+=1

    img = imread(img_path) / 255 #get the image from the image path and normalize

    img = resize(img, (128, 128), preserve_range=True) #resize to 128 x 128

    y = np.zeros((128,128)) #assume everything is not a road first



    for j in range(2,126):

        for k in range(2,126):

            #print (img[j,k,:])

            #single_pixel = img[j,k,:]

            single_example = img[j-2:j+3, k-2:k+3,:]

            #print (single_example.shape)

            #print (single_example[:,:,0])

            single_example = single_example.reshape((1,25,3)) #reshape into a row vector

            #single_example = np.delete(single_example, 4, axis = 1) #delete the pixel itself from the example

            single_example = single_example.reshape((1,1,25,3)) #reshape into a row vector

            classifications = model.predict(single_example)

            y[j,k] = np.argmax(classifications, axis = 1)[0]

            #print (np.argmax(classifications, axis = 1))

    ids.append(get_img_id(img_path))

    encodings.append(rle_encoding(y))

    plt.imshow(y, cmap='gray')

    plt.show()

"""

for img_path in val_img_paths[669:]:

    y = np.zeros((128,128))

    ids.append(get_img_id(img_path))

    encodings.append(rle_encoding(y))

"""

sub['EncodedPixels'] = encodings

sub['ImageId'] = ids

sub.to_csv('ANN.csv', index=False)

#print (y)

#plt.imshow(y, cmap='gray')

#plt.show()

#yield get_img_id(img_path), y

print ('Done!')
import time

tic = time.time()

create_submission("ANNsimple.csv", generate_pixel_by_pixel_predictions_generator(val_img_paths))

toc = time.time()

print(toc - tic)