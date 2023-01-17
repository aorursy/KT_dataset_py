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




import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc

import json

import math

import cv2

import PIL

from PIL import Image



import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression, Ridge

from sklearn.preprocessing import OneHotEncoder

from tqdm import tqdm

from sklearn.decomposition import PCA

import os

import imagesize



%matplotlib inline



data = pd.read_csv("../input/tv-shows-on-netflix-prime-video-hulu-and-disney/tv_shows.csv")



print("{} data".format(data.shape[0]))

data.head()

plt.figure(figsize=(12, 5))

plt.hist(data['IMDb'].values, bins=200)

plt.title('IMDb rating')

plt.xlabel('Value')

plt.ylabel('Count')

plt.show()
plt.figure(figsize=(12, 5))

plt.hist(data['Year'].values, bins=200)

plt.title('year vs count ')

plt.xlabel('year')

plt.ylabel('Count')

plt.show()
import plotly.graph_objects as go

fig = go.Figure(data=go.Heatmap(

                   z= data['Year'],

                   x=data['Title'],

                   y= data['Netflix'],

hoverongaps = False))

fig.update_layout(

    

    yaxis = dict(

        tickmode = 'array',

        tickvals = [2010, 2011, 2012, 2013, 2014, 

                    2015, 2016, 2017 , 2018, 2019, 2020],

        ticktext = ['2010', '2011', '2012', '2013', '2014', 

                    '2015', '2016', '2017' , '2018', '2019', '2020'],),

    paper_bgcolor='rgb(233,233,233)',

    

    )

fig.show()
data[data['Age'].apply(lambda Age: Age == '18+')].head()



import matplotlib.pyplot as plt

# pip install seaborn 

import seaborn as sns

# Graphics in retina format are more sharp and legible

%config InlineBackend.figure_format = 'retina'

sns.countplot(x='Age', hue='Netflix', data=data);