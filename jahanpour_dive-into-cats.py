# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
from IPython.core.display import HTML  # for plotting images in a simpler format
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#load data
data = json.load(open('../input/train.json'))
#Saved On Data Frame
data_url = pd.DataFrame.from_dict(data['images'])
labels = pd.DataFrame.from_dict(data['annotations'])
train_data = data_url.merge(labels, how='inner', on=['image_id'])
train_data['url'] = train_data['url'].str.get(0)
del data, data_url, labels
train_data.head(5)
label_frequency = pd.DataFrame(train_data.label_id.value_counts())
label_frequency.reset_index(level=0, inplace=True)
label_frequency.plot('index', 'label_id', kind='bar', figsize=(20,10), title="distribution of the labels in training data")

#train_data.label_id.plot(kind='bar', alpha=0.5)
train_data.describe()
print('top three labels \n', str(train_data.label_id.value_counts().head(3)))  
print('least frequent three labels \n', str(train_data.label_id.value_counts().tail(3)))  
# Top 3 and bottom 3 labels in the images are below though all the images are not distributed Uniformly
def display_image(label_id, number_of_display, seed=123):
    labeled_data = train_data[train_data.label_id == label_id]
    labeled_data = labeled_data.sample(number_of_display, random_state=seed)
    img_style = "width: 180px; margin: 0px; float: left; border: 1px solid black;"
    images_list = ''.join([f"<img style='{img_style}' src='{u}' />" for u in labeled_data.url])
    display(HTML(images_list))
display_image(label_id=20, number_of_display=9)
display_image(label_id=42, number_of_display=9)
display_image(label_id=92, number_of_display=9)
display_image(label_id=124, number_of_display=9)
display_image(label_id=66, number_of_display=9)
display_image(label_id=83, number_of_display=9)

