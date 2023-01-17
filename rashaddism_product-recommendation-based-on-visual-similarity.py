# imports



from keras.applications import vgg16

from keras.preprocessing.image import load_img,img_to_array

from keras.models import Model

from keras.applications.imagenet_utils import preprocess_input



from PIL import Image

import os

import matplotlib.pyplot as plt

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
# parameters setup



imgs_path = "../input/style/"

imgs_model_width, imgs_model_height = 224, 224



nb_closest_images = 5 # number of most similar images to retrieve
# load the model

vgg_model = vgg16.VGG16(weights='imagenet')



# remove the last layers in order to get features instead of predictions

feat_extractor = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer("fc2").output)



# print the layers of the CNN

feat_extractor.summary()
files = [imgs_path + x for x in os.listdir(imgs_path) if "png" in x]



print("number of images:",len(files))
# load an image in PIL format

original = load_img(files[0], target_size=(imgs_model_width, imgs_model_height))

plt.imshow(original)

plt.show()

print("image loaded successfully!")
# convert the PIL image to a numpy array

# in PIL - image is in (width, height, channel)

# in Numpy - image is in (height, width, channel)

numpy_image = img_to_array(original)



# convert the image / images into batch format

# expand_dims will add an extra dimension to the data at a particular axis

# we want the input matrix to the network to be of the form (batchsize, height, width, channels)

# thus we add the extra dimension to the axis 0.

image_batch = np.expand_dims(numpy_image, axis=0)

print('image batch size', image_batch.shape)



# prepare the image for the VGG model

processed_image = preprocess_input(image_batch.copy())
# get the extracted features

img_features = feat_extractor.predict(processed_image)



print("features successfully extracted!")

print("number of image features:",img_features.size)

img_features
# load all the images and prepare them for feeding into the CNN



importedImages = []



for f in files:

    filename = f

    original = load_img(filename, target_size=(224, 224))

    numpy_image = img_to_array(original)

    image_batch = np.expand_dims(numpy_image, axis=0)

    

    importedImages.append(image_batch)

    

images = np.vstack(importedImages)



processed_imgs = preprocess_input(images.copy())

# extract the images features



imgs_features = feat_extractor.predict(processed_imgs)



print("features successfully extracted!")

imgs_features.shape
# compute cosine similarities between images



cosSimilarities = cosine_similarity(imgs_features)



# store the results into a pandas dataframe



cos_similarities_df = pd.DataFrame(cosSimilarities, columns=files, index=files)

cos_similarities_df.head()
# function to retrieve the most similar products for a given one



def retrieve_most_similar_products(given_img):



    print("-----------------------------------------------------------------------")

    print("original product:")



    original = load_img(given_img, target_size=(imgs_model_width, imgs_model_height))

    plt.imshow(original)

    plt.show()



    print("-----------------------------------------------------------------------")

    print("most similar products:")



    closest_imgs = cos_similarities_df[given_img].sort_values(ascending=False)[1:nb_closest_images+1].index

    closest_imgs_scores = cos_similarities_df[given_img].sort_values(ascending=False)[1:nb_closest_images+1]



    for i in range(0,len(closest_imgs)):

        original = load_img(closest_imgs[i], target_size=(imgs_model_width, imgs_model_height))

        plt.imshow(original)

        plt.show()

        print("similarity score : ",closest_imgs_scores[i])
retrieve_most_similar_products(files[1])
retrieve_most_similar_products(files[2])
retrieve_most_similar_products(files[3])
retrieve_most_similar_products(files[4])
retrieve_most_similar_products(files[5])