!pip install face_recognition
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from PIL import Image

import glob

import pickle

import face_recognition

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from __future__ import print_function

list_im = glob.glob("../input/photo-dataset-generator/gphoto/gphoto/*.jpg") # 11 pages results from google "face photo" search



result = Image.new("RGB", (2479, 38544)) # stacked canvas dimension



for index, file in enumerate(list_im):

    path = os.path.expanduser(file)

    img = Image.open(path)

    #print(img.size)

    img.thumbnail((2479, 3504), Image.ANTIALIAS)

    x = index // 11 * 2479

    y = index % 11 * 3504

    w, h = img.size

    result.paste(img, (x, y, x + w, y + h))



result.save(os.path.expanduser('stackedPhotos.jpg')) #11 photos stacked in one photo
# Load the jpg file into a numpy array

image = face_recognition.load_image_file("stackedPhotos.jpg") # face recognition from one photo (stacked 707 photos)

face_locations = face_recognition.face_locations(image) #  face reco model
all_face_encodings ={}

row_dim = len(face_locations)

col_dim = 1000



pics_per_row = int(row_dim/10)

result2 = Image.new("RGB", (row_dim*10, col_dim))

thumbnail_size = int(col_dim/10)

for index , fl in enumerate(face_locations):

    try:

        top, right, bottom, left = fl

        face_image = image[top:bottom, left:right]

        pil_image = Image.fromarray(face_image)

        pil_image.thumbnail((thumbnail_size, thumbnail_size), Image.ANTIALIAS)

        x = index // 10 * thumbnail_size

        y = index % 10 * thumbnail_size

        w, h = pil_image.size

        result2.paste(pil_image, (x, y, x + w, y + h))

        all_face_encodings.update({index:face_recognition.face_encodings(face_image)[0]})

    except:

        pass

result2

result.save(os.path.expanduser('stackedfaces.jpg')) #face data save



#face data vector save

with open('stackedfaces.dat', 'wb') as f:

    pickle.dump(all_face_encodings, f) 