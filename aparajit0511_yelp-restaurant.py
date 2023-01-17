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
# Set your own project id here

PROJECT_ID = 'your-google-cloud-project'

from google.cloud import storage

storage_client = storage.Client(project=PROJECT_ID)
import pandas as pd

import cv2



# path = "../input/yelp-restaurant-photo-classification/train_photos.tgz"

import tarfile

tar = []

tar.append(tarfile.open('../input/yelp-restaurant-photo-classification/train_photos.tgz', "r:gz"))

tar.append(tarfile.open('../input/yelp-restaurant-photo-classification/train.csv.tgz', "r:gz"))

tar.append(tarfile.open('../input/yelp-restaurant-photo-classification/train_photo_to_biz_ids.csv.tgz', "r:gz"))



for items in tar:

    print(items.getmembers())

    print("break")
# train_photo_id = pd.read_csv('../input/yelp-restaurant-photo-classification/train_photo_to_biz_ids/train_photo_to_biz_ids.csv')





# train_photo_id = pd.read_csv('../input/yelp-restaurant-photo-classification/train_photo_to_biz_ids.csv', compression='gzip', header=0, sep=' ', quotechar='"', error_bad_lines=False)



with tarfile.open('../input/yelp-restaurant-photo-classification/train_photo_to_biz_ids.csv.tgz', "r:*") as tar:

    csv_path = tar.getnames()[1]

    train_photo_id = pd.read_csv(tar.extractfile(csv_path), header=0, sep=" ")

train_photo_id.head()
with tarfile.open('../input/yelp-restaurant-photo-classification/train.csv.tgz', "r:*") as tar:

    csv_path = tar.getnames()[0]

    train = pd.read_csv(tar.extractfile(csv_path), header=0, sep=" ",error_bad_lines=False)

train.head()