# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
import sys

import subprocess

import numpy as np 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import gc

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



pal = sns.color_palette()



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

from skimage import io

import scipy

from scipy import ndimage

from IPython.display import display

#import rasterio # reads and writes geospatial raster datasets
labels_df =  pd.read_csv('../input/train.csv')

labels_df.head()
labels_df.tail()
labels_df.describe()
labels = labels_df['tags'].apply(lambda x: x.split(' '))

from collections import Counter, defaultdict

counts = Counter()

for label in labels:

    counts += Counter(label)

# Build list with unique labels

labels_list = list(counts.keys())

len(labels_list)
# Add one-hot encoded features for every label

for label in labels_list:

    labels_df[label] = labels_df['tags'].apply(lambda x: 1 if label in x.split(' ') else 0)

    

# display head

labels_df.head()
data = [go.Bar(x = list(counts.keys()), y=list(counts.values()))]

layout = dict(height=600, width=600, title='Distribution of training labels')

fig = dict(data=data, layout=layout)

py.iplot(data, filename='training-label')
from sklearn.preprocessing import MinMaxScaler

def make_coocurence_matrix(labels):

    numeric_df = labels_df[labels];

    c_matrix = numeric_df.T.dot(numeric_df)

    scaler = MinMaxScaler()

    c_matrix.loc[:,:] = scaler.fit_transform(c_matrix) 

    data = [go.Heatmap(z=c_matrix.T.values.tolist(), x = list(c_matrix[:0][:]), y=list(c_matrix[:0][:]))]

    layout = go.Layout(height = 600, width =100, title='Co-occurence matrix of training labels')

    fig = dict(data=data, layout=layout)

    py.iplot(data)

    return c_matrix





# Compute the occurence matrix

make_coocurence_matrix(labels_list)
