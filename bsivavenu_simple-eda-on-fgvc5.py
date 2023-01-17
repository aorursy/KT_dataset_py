# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

train = pd.read_json('../input/train.json')
test = pd.read_json('../input/test.json')
val = pd.read_json('../input/validation.json')
train.head()
val.head()
test.head()
train.columns
train['image_id'] = train.annotations.map(lambda x: x['image_id'])
train['label_id'] = train.annotations.map(lambda x: x['label_id'])
train['url'] = train.images.map(lambda x: x['url'][0])
train.drop(columns=['annotations', 'images'], inplace=True)
train.head()
train.isnull().sum()
val['image_id'] = val.annotations.map(lambda x: x['image_id'])
val['label_id'] = val.annotations.map(lambda x: x['label_id'])
val['url'] = val.images.map(lambda x: x['url'][0])
val.drop(columns=['annotations', 'images'], inplace=True)
val.head()
val.isnull().sum()
# test['image_id'] = test.annotations.map(lambda x: x['image_id'])
# test['label_id'] = test.annotations.map(lambda x: x['label_id'])
test['url'] = test.images.map(lambda x: x['url'][0])
test.drop(columns=[ 'images'], inplace=True)
test.head()
test.isnull().sum()
from IPython.display import Image
from IPython.core.display import HTML 
Image(url= train.url[50],width=200,height=200)
from IPython.display import Image
from IPython.core.display import HTML 

def display_category(urls, category_name):
    img_style = "width: 180px; margin: 0px; float: left; border: 1px solid black;"
    images_list = ''.join([f"<img style='{img_style}' src='{u}' />" for _, u in urls.head(12).iteritems()])

    display(HTML(images_list))

urls = train['url'][15:30]
display_category(urls, "")

urls = test['url'][15:30]
display_category(urls, "")

urls = val['url'][15:30]
display_category(urls, "")

train.label_id.value_counts().sort_values(ascending=False).head()
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(30,5))
sns.countplot(train.label_id)
plt.show()
(train.url[11])
train.columns
a = train.label_id.unique()
a
from IPython.core.display import HTML 
from ipywidgets import interact
from IPython.display import display

%matplotlib inline
from IPython.display import Image
from IPython.core.display import HTML 

def display_category1(urls, category_name):
    img_style = "width: 180px; margin: 0px; float: left; border: 1px solid black;"
    images_list = ''.join([f"<img style='{img_style}' src='{u}' />" for _, u in urls.head(12).iteritems()])

    display(HTML(images_list))

