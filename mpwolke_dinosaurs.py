#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcS5TVHUgfULbG-mdHFJKjAggYWuLojWM37IfUIeARV7FewkM5aj',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px

from wordcloud import WordCloud



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/dinosaur-list/dinosaur.csv")

df.head()
def nonan(x):

    if type(x) == str:

        return x.replace("\n", "")

    else:

        return ""



text = ' '.join([nonan(abstract) for abstract in df["Name"]])

wordcloud = WordCloud(max_font_size=None,colormap='Set3', background_color='black', collocations=False,

                      width=1200, height=1000).generate(text)

fig = px.imshow(wordcloud)

fig.update_layout(title_text='Dinosaurs ')
def nonan(x):

    if type(x) == str:

        return x.replace("\n", "")

    else:

        return ""



text = ' '.join([nonan(abstract) for abstract in df["Period"]])

wordcloud = WordCloud(max_font_size=None,colormap='Set3', background_color='black', collocations=False,

                      width=1200, height=1000).generate(text)

fig = px.imshow(wordcloud)

fig.update_layout(title_text='Dinosaurs ')