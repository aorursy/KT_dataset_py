from PIL import Image

import os

import matplotlib.pyplot as plt

import numpy as np



from keras.applications import vgg16

from keras.preprocessing.image import load_img,img_to_array

from keras.models import Model

from keras.applications.imagenet_utils import preprocess_input





from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
imgs_path = "/kaggle/input/fashion-product-images-dataset/fashion-dataset/fashion-dataset/images/"

imgs_model_width, imgs_model_height = 224, 224

nb_closest_images = 5
files = [imgs_path + x for x in os.listdir(imgs_path) if "jpg" in x]



print("Total number of images:",len(files))
#For reducing compilation time of the algorithm, we reduce the data to 5000 images or the system crashes!

files=files[0:5000]
original = load_img(files[9], target_size=(imgs_model_width, imgs_model_height))

plt.imshow(original)

plt.show()

print("Image loaded successfully!")
# load the model

vgg_model = vgg16.VGG16(weights='imagenet')



# remove the last layers in order to get features instead of predictions

feat_extractor = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer("fc2").output)



# print the layers of the CNN

feat_extractor.summary()
numpy_image = img_to_array(original)

# convert the image / images into batch format

# expand_dims will add an extra dimension to the data at a particular axis

# we want the input matrix to the network to be of the form (batchsize, height, width, channels)

# thus we add the extra dimension to the axis 0.

image_batch = np.expand_dims(numpy_image, axis=0)

print('Image Batch size', image_batch.shape)



# prepare the image for the VGG model

processed_image = preprocess_input(image_batch.copy())
img_features = feat_extractor.predict(processed_image)



print("Features successfully extracted for one image!")

print("Number of image features:",img_features.size)

img_features
importedImages = []



for f in files:

    filename = f

    original = load_img(filename, target_size=(224, 224))

    numpy_image = img_to_array(original)

    image_batch = np.expand_dims(numpy_image, axis=0)

    

    importedImages.append(image_batch)

    

images = np.vstack(importedImages)



processed_imgs = preprocess_input(images.copy())
imgs_features = feat_extractor.predict(processed_imgs)



print("features successfully extracted!")

imgs_features.shape
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
retrieve_most_similar_products(files[465])
retrieve_most_similar_products(files[50])
import pandas as pd



styles=pd.read_csv("../input/fashion-product-images-dataset/fashion-dataset/styles.csv", error_bad_lines=False)

import plotly.express as px

fig = px.pie(styles, styles['gender'],color_discrete_sequence=px.colors.sequential.RdBu)

fig.show()
catcounts=pd.value_counts(styles['masterCategory'])
import plotly.graph_objects as go



fig = go.Figure([go.Bar(x=catcounts.index, y=catcounts.values , text=catcounts.values,marker_color='darkblue')])

fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

fig.show()
seasons=pd.value_counts(styles['season'])
import plotly.graph_objects as go



fig = go.Figure(data=[go.Scatter(

    x=seasons.index, y=seasons.values,

    mode='markers',

    marker=dict(

        color=['rgb(93, 164, 214)', 'rgb(255, 144, 14)',

               'rgb(44, 160, 101)', 'rgb(255, 65, 54)'],

        opacity=[1, 0.8, 0.6, 0.4],

        size=[40, 60, 80, 100])

)]

               )



fig.show()
articles=pd.value_counts(styles['articleType'])

fig = go.Figure([go.Bar(x=articles.index, y=articles.values , text=articles.values,marker_color='indianred')])

fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

fig.show()