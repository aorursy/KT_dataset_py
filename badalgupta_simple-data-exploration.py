# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train={}
test={}
validation={}
with open('../input/train.json') as json_data:
    train= json.load(json_data)
with open('../input/test.json') as json_data:
    test= json.load(json_data)
with open('../input/validation.json') as json_data:
    validation = json.load(json_data)
train_img_url=train['images']
train_data_1=pd.DataFrame(train_img_url)
train_ann=train['annotations']
train_data_2=pd.DataFrame(train_ann)
#Merging Them
train_data=pd.merge(train_data_1, train_data_2, on='imageId', how='inner')
train_data.head(5)
#test data 
test_data=pd.DataFrame(test['images'])
test_data.head()
#Validation Data
val_img_url=validation['images']
val_data_1=pd.DataFrame(val_img_url)
val_ann=train['annotations']
val_data_2=pd.DataFrame(val_ann)
#Merging Them
val_data=pd.merge(val_data_1, val_data_2, on='imageId', how='inner')
val_data.head()
total = train_data.isnull().sum().sort_values(ascending = False)
percent = (train_data.isnull().sum()/train_data.isnull().count()).sort_values(ascending = False)
missing_train_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_train_data.head()
# Missing In Test Data
total = test_data.isnull().sum().sort_values(ascending = False)
percent = (test_data.isnull().sum()/test_data.isnull().count()).sort_values(ascending = False)
missing_test_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_test_data.head()
# Missing In validation Data
total = val_data.isnull().sum().sort_values(ascending = False)
percent = (val_data.isnull().sum()/val_data.isnull().count()).sort_values(ascending = False)
missing_val_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_val_data.head()
from IPython.display import Image
from IPython.core.display import HTML 

def display_category(urls, category_name):
    img_style = "width: 180px; margin: 0px; float: left; border: 1px solid black;"
    images_list = ''.join([f"<img style='{img_style}' src='{u}' />" for _, u in urls.head(12).iteritems()])
    display(HTML(images_list))
#train_data Images
urls = train_data['url'][1:5]
display_category(urls, "")
#test data Images
urls = test_data['url'][1:5]
display_category(urls, "")
freq={}
for l in train_data['labelId']:
    for item in l:
        if item in freq:
            freq[str(item)]+=1
        else:
            freq[str(item)]=1
most_frequent_labels=sorted(freq.items(), key=lambda x: x[1], reverse=True)[:3]
most_frequent_labels
max_labels_count=0
_url=""
_id=""
for index, row in train_data.iterrows():
    c=len(row['labelId'])
    if c> max_labels_count:
        max_labels_count=c
        _url=row['url']
        _id=row['imageId']
print( _url ,max_labels_count, _id)
