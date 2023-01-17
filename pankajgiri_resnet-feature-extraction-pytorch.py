# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from PIL import Image

import torch

import torch.nn as nn

import torchvision.models as models

import torchvision.transforms as transforms

from torch.autograd import Variable



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# The idea here is to convert an image into a compressed 1D representation from a 2D image.

# The input image is converted to a 512 element vector from which we could hope to use a similarity metric

# to compare against other 2D images. Using Resnet to extract the compressed representation.

model = models.resnet18(pretrained='imagenet')
content_pd = pd.read_csv('../input/traincsv/train.csv')

content_pd.head()
#Resize the image to 224x224 px

scaler = transforms.Scale((224, 224))

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],

                                 std=[0.229, 0.224, 0.225])

to_tensor = transforms.ToTensor()



# Use the model object to select the desired layer

layer = model._modules.get('avgpool')



def extract_feature_vector(img):

    # 2. Create a PyTorch Variable with the transformed image

    #Unsqueeze actually converts to a tensor by adding the number of images as another dimension.

    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))



    # 3. Create a vector of zeros that will hold our feature vector

    #    The 'avgpool' layer has an output size of 512

    my_embedding = torch.zeros(1, 512, 1, 1)



    # 4. Define a function that will copy the output of a layer

    def copy_data(m, i, o):

        my_embedding.copy_(o.data)



    # 5. Attach that function to our selected layer

    h = layer.register_forward_hook(copy_data)



    # 6. Run the model on our transformed image

    model(t_img)



    # 7. Detach our copy function from the layer

    h.remove()



    # 8. Return the feature vector

    return my_embedding.squeeze().numpy()
from IPython.display import Image

from IPython.core.display import HTML 



def display_category(urls, category_name):

    img_style = "width: 180px; margin: 0px; float: left; border: 1px solid black;"

    images_list = ''.join([f"<img style='{img_style}' src='{u}' />" for _, u in urls.head(12).iteritems()])



    display(HTML(images_list))
# display a few images in the url

ids =  content_pd.landmark_id.value_counts().keys()[25]

urls = content_pd[content_pd.landmark_id == ids].url

display_category(urls, 'My Favourite')
from PIL import Image

import requests

from io import BytesIO



#Download the image into memory given the URL.

def get_image_from_url(url):

    response = requests.get(url)

    return Image.open(BytesIO(response.content))
# Now that you have the means for extracting the 1D representation from the given input image, you can use the cosine_similarity metric to compute the distance between

# the given image and all other images and sort them in the ascending order and return the results.

from sklearn.metrics.pairwise import cosine_similarity

import numpy as np



def get_cosine_distance(im_url1, im_url2):

    im1, im2 = get_image_from_url(im_url1), get_image_from_url(im_url2)

    image1 = extract_feature_vector(im1).reshape(1, -1)

    image2 = extract_feature_vector(im2).reshape(1, -1)

    return cosine_similarity(image1, image2)
# Print the distance between any 2 landmark images

ids =  content_pd.landmark_id.value_counts().keys()[1]

urls = content_pd[content_pd.landmark_id == ids].url

urls = [i for i in urls]

url1, url2 = urls[0], urls[1]

print('Distance between 2 images :', get_cosine_distance(url1, url2))