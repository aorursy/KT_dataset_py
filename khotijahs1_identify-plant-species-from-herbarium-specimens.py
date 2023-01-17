import matplotlib.pyplot as plt

from PIL import Image

import seaborn as sns

import pandas as pd

import numpy as np

import math

import sklearn

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error

import matplotlib.pyplot as plt

import seaborn as sb

from sklearn import preprocessing

from sklearn.feature_extraction.text import CountVectorizer

import time

import warnings 

warnings.filterwarnings('ignore')
import json, codecs

with codecs.open("../input/herbarium-2020-fgvc7/nybg2020/train/metadata.json", 'r',

                 encoding='utf-8', errors='ignore') as f:

    train= json.load(f)

    

with codecs.open("../input/herbarium-2020-fgvc7/nybg2020/test/metadata.json", 'r',

                 encoding='utf-8', errors='ignore') as f:

    test = json.load(f)
display(train.keys())

train_data = pd.DataFrame(train['annotations'])

display(train_data)
Cat = pd.DataFrame(train['categories'])

display(Cat)
train_img = pd.DataFrame(train['images'])

train_img.columns = ['file_name', 'height', 'image_id', 'license', 'width']

display(train_img)
licenses = pd.DataFrame(train['licenses'])

display(licenses)
regions = pd.DataFrame(train['regions'])

display(regions)
train_data = train_data.merge(Cat, on='id', how='outer')

train_data = train_data.merge(train_img, on='image_id', how='outer')

train_data = train_data.merge(regions, on='id', how='outer')
print(train_data.info())



display(train_data)
test_data = pd.DataFrame(test['images'])

test_data.columns = ['file_name', 'height', 'image_id', 'license', 'width']

print(test_data.info())

display(test_data)
print(len(train_data.id.unique()))

sub = pd.DataFrame()

sub['Id'] = test_data.image_id

sub['Predicted'] = list(map(int, np.random.randint(1, 32093, (test_data.shape[0]))))

display(sub)

sub.to_csv('submission.csv', index=False)