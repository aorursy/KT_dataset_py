# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns

%matplotlib inline

import plotly.express as px

import plotly.graph_objects as go

import plotly.offline as py

import plotly.express as px



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/17k-apple-app-store-strategy-games/appstore_games.csv', encoding='ISO-8859-2')

df.head()
df = df.rename(columns={'Icon URL':'icon'})
Description=df.sort_values('Description', ascending=False)

top10=Description.head(9)

f=['Name','icon']

displ=(top10[f])

displ.set_index('Name', inplace=True)
#Code from Niharika Pandit https://www.kaggle.com/niharika41298/netflix-vs-books-recommender-analysis-eda

from IPython.display import Image, HTML



def path_to_image_html(path):

    '''

     This function essentially convert the image url to 

     '<img src="'+ path + '"/>' format. And one can put any

     formatting adjustments to control the height, aspect ratio, size etc.

     within as in the below example. 

    '''



    return '<img src="'+ path + '""/>'



HTML(displ.to_html(escape=False ,formatters=dict(icon=path_to_image_html),justify='center'))