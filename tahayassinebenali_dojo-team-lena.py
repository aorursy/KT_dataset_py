# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/train.csv')
train.head()
import cv2
import plotly.express as px
import matplotlib.pyplot as plt 
def read_image(image_id):
    image = cv2.imread(f'/kaggle/input/plant-pathology-2020-fgvc7/images/{image_id}.jpg')
    image = cv2.resize(image, (200,200))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def show_image(image_id):
    image = read_image(image_id)
    return px.imshow(image)
sample_healthy = train[train['healthy'] == 1].head()
sample_not_healthy = train[train['healthy'] == 0].head()
sample_healthy_list = list(sample_healthy['image_id'])
sample_not_healthy_list = list(sample_not_healthy['image_id'])
for image in sample_healthy_list:
    fig = show_image(image)
    fig.show()
for image in sample_not_healthy_list:
    fig = show_image(image)
    fig.show()
healthy = pd.DataFrame(train['healthy'].value_counts())
px.pie(healthy, values='healthy', names=healthy.index, title='Distribution des feuilles')
rust = pd.DataFrame(train['rust'].value_counts())
px.pie(rust, values='rust', names=rust.index, title='Distribution des feuilles atteintes du rust')
scab = pd.DataFrame(train['scab'].value_counts())
px.pie(scab, values='scab', names=scab.index, title='Distribution des feuilles atteintes du scab')
multi_disease = pd.DataFrame(train['multiple_diseases'].value_counts())
px.pie(multi_disease, values='multiple_diseases', names=multi_disease.index, title='Distribution des feuilles atteintes du scab')
