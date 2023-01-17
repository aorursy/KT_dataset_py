# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
## Importing Necessary Modules

import requests # to get image from the web

import shutil # to save it locally



## Set up the image URL and filename

image_url = "https://cdn.pixabay.com/photo/2020/02/06/09/39/summer-4823612_960_720.jpg"

filename = image_url.split("/")[-1]



# Open the url image, set stream to True, this will return the stream content.

r = requests.get(image_url, stream = True)



# Check if the image was retrieved successfully

if r.status_code == 200:

    # Set decode_content value to True, otherwise the downloaded image file's size will be zero.

    r.raw.decode_content = True

    

    # Open a local file with wb ( write binary ) permission.

    with open(filename,'wb') as f:

        shutil.copyfileobj(r.raw, f)

        

    print('Image sucessfully Downloaded: ',filename)

else:

    print('Image Couldn\'t be retreived')
!pip install wget
# First import wget python module.

import wget



# Set up the image URL

image_url = "https://cdn.pixabay.com/photo/2020/02/06/09/39/summer-4823612_960_720.jpg"



# Use wget download method to download specified image url.

image_filename = wget.download(image_url)



print('Image Successfully Downloaded: ', image_filename)
# importing required modules

import urllib.request



# setting filename and image URL

filename = 'sunshine_dog.jpg'

image_url = "https://cdn.pixabay.com/photo/2020/02/06/09/39/summer-4823612_960_720.jpg"



# calling urlretrieve function to get resource

urllib.request.urlretrieve(image_url, filename)
# Importing required libraries

import urllib.request



# Adding information about user agent

opener=urllib.request.build_opener()

opener.addheaders=[('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]

urllib.request.install_opener(opener)



# setting filename and image URL

filename = 'sunshine_dog.jpg'

image_url = "https://cdn.pixabay.com/photo/2020/02/06/09/39/summer-4823612_960_720.jpg"



# calling urlretrieve function to get resource

urllib.request.urlretrieve(image_url, filename)