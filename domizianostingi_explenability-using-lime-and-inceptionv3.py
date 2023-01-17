from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from keras.preprocessing.image import load_img, img_to_array

import numpy as np 
import sklearn.cluster
model = InceptionV3()
from PIL import Image
import requests
url = 'https://i.guim.co.uk/img/media/c9b0aad22638133aa06cd68347bed2390b555e63/0_477_2945_1767/master/2945.jpg?width=1200&height=1200&quality=85&auto=format&fit=crop&s=97bf92d90f51da7067d00f8156512925'
image_cat = Image.open(requests.get(url, stream=True).raw)
image_cat = image_cat.resize((299,299))
image_cat
# Convert to numpy array, reshape and preprocess
image = img_to_array(image_cat)
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
image = preprocess_input(image)
image[0].shape
predictions = model.predict(image)
#function of keras, allows to see the prob of each prediction
decode_predictions(predictions)
# extract the index of the top 5 classes predicted by the model for the image selected 
model.predict(image).argsort()[0, -5:][::-1]

#keep the index of the first and the second class 
first_class = model.predict(image).argsort()[0, -5:][-1]
second_class = model.predict(image).argsort()[0, -5:][-2]
!pip install lime
from lime.lime_image import LimeImageExplainer
explainer = LimeImageExplainer()
explanation = explainer.explain_instance(image[0], #the image
                                         model.predict, 
                                         top_labels=2, #want just to see the 2 main classes predicted
                                         num_samples=500, # number of observation sampled from the original distribution in order to computer the linear regression
                                         random_seed=42)
from skimage.segmentation import mark_boundaries
from matplotlib import pyplot as plt
# maps for the first class predicted
temp, mask = explanation.get_image_and_mask(first_class, positive_only=True, num_features=5, hide_rest=True)
# plot image and mask together
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
print('mask for prediction of class: ',decode_predictions(predictions)[0][0][1])
# maps for the second class predicted 
temp, mask = explanation.get_image_and_mask(second_class, positive_only=True, num_features=5, hide_rest=True)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
print('mask for prediction of class: ', decode_predictions(predictions)[0][1][1])