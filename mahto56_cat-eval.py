# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#disable keras logs

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import load_model

from keras.preprocessing import image

import cv2

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from skimage import io, transform
mfile = '/kaggle/input/cnn-for-animals-cat-dog/cat_dog.h5'

#load model

try:

    model = load_model(mfile)

    print('model was loaded successfully')

except Exception as e:

    print('failed to load model, error: {0}'.format(e))
def download_image_and_predict(url):

    # Load & Preprocess test image

    filename = url.split('/')[-1] #get filename from url

    print('downloading file {0}'.format(filename))

    !wget $url -P /kaggle/input -q --show-progress #load image from internet on linux machine

    print('loading file: {0}'.format(filename))

    test_image=io.imread('/kaggle/input/'+filename)

    test_image=transform.resize(test_image, (100, 100))

    imgplot = plt.imshow(test_image)

    plt.show()

    pred = model.predict_classes(test_image.reshape(-1,100,100,3))

    if pred[0] == 1:

        print('It is a cat')

    elif pred[0] == 0:

        print('It is a dog')
download_image_and_predict('https://peopledotcom.files.wordpress.com/2018/09/bacon-the-dog-4.jpg')
download_image_and_predict('wget https://scx2.b-cdn.net/gfx/news/hires/2018/2-dog.jpg')
download_image_and_predict('https://www.guidedogs.org/wp-content/uploads/2015/05/Dog-Im-Not.jpg')
download_image_and_predict('https://amp.businessinsider.com/images/56c4bd4e6e97c627008b7bcb-750-633.jpg')
download_image_and_predict('https://media.wired.com/photos/5cdefc28b2569892c06b2ae4/master/w_2560%2Cc_limit/Culture-Grumpy-Cat-487386121-2.jpg')
download_image_and_predict('https://pbs.twimg.com/profile_images/378800000532546226/dbe5f0727b69487016ffd67a6689e75a_400x400.jpeg')
download_image_and_predict('https://static.independent.co.uk/s3fs-public/thumbnails/image/2018/11/24/16/cat.jpg')
download_image_and_predict('https://cdn.hswstatic.com/gif/whiskers-sam.jpg')
download_image_and_predict('https://coleandmarmalade.com/wp-content/uploads/2019/08/cat-coat-pattern-feature-e1565096580379.jpg')
download_image_and_predict('https://imgix.bustle.com/uploads/getty/2018/5/7/152fd35b-ac43-408c-8738-ae864ad7c3e7-getty-915832460.jpg')
download_image_and_predict('https://scx1.b-cdn.net/csz/news/800/2015/cat.jpg')
download_image_and_predict('https://images.squarespace-cdn.com/content/56f5fdc7c2ea5119892e22c2/1461340110045-W1N5ZWYZI4SS2ND61QW8/DOGFACE-Chase-024AFP.jpg')