#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/jpeg;base64,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',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px

import imageio

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from PIL import Image, ImageDraw

from ast import literal_eval

import cv2



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQ2QArJDnDSQDbeiASstxs8TIkSZGWoI5K3hh5OXidJ9IImPiL4&usqp=CAU',width=400,height=400)
PATH = '/kaggle/input/global-wheat-detection/'

fox_df = pd.read_csv('../input/global-wheat-detection/train.csv', encoding='ISO-8859-2')

fox_df.tail()
fox_images = pd.Series(os.listdir(PATH + '/train/')).sort_values(ascending=True).reset_index(drop=True)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTF10jqcjvSd8aRvhvqoAZe7Ouk4N4rdOOHegAUOBZ0pJXpyH07&usqp=CAU',width=400,height=400)
fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(10, 6))

k=0

for i, row in enumerate(ax):

    for j, col in enumerate(row):

        img = mpimg.imread(PATH + '/train/' + fox_images[k])

        col.imshow(img)

        col.set_title(fox_images[k])

        k=k+1

plt.suptitle('Samples from wheat Images', fontsize=14)

plt.show()
display_fox_image = fox_df.loc[fox_df['image_id'] == '5e0747034']
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcR2R51BcLDXTcDPsMSrlcxDTyiJ7YSQa7H7u9q6-dvoUIuItK_a&usqp=CAU',width=400,height=400)
display_fox_image.head()      
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRueMgBHT-80TLvCZVVndwGT-7eJoYwvp3OB3NwFGwpErIFS5qU&usqp=CAU',width=400,height=400)
import sqlite3

from tkinter import Tk, Button, Canvas

from PIL import Image, ImageFont



from ast import literal_eval

image_id = 'train/5e0747034'

image_path = os.path.join(PATH + image_id + ".jpg")

image = Image.open(image_path)



boxes = [literal_eval(box) for box in fox_df.loc[fox_df['image_id'] == '5e0747034']['bbox']]



print(boxes)
image = Image.open(image_path)

# visualize them

draw = ImageDraw.Draw(image)

for box in boxes:    

    draw.rectangle([box[0], box[1], box[0] + box[2], box[1] + box[3]], width=3)



plt.figure(figsize = (4,4))

plt.imshow(image)

plt.show()
sns.countplot(fox_df["source"])

plt.xticks(rotation=90)

plt.show()
fig = px.line(fox_df, x="source", y="height", 

              title="Wheat Fields and Foxes")

fig.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcR_UyVG8ThfElh45vJFWj41t0cAa7vHFRFehSqW1wv_YQBdY725&usqp=CAU',width=400,height=400)
fig = px.histogram(fox_df[fox_df.source.notna()],x="source",marginal="box",nbins=10)

fig.update_layout(

    title = "Wheat Fields and Foxes",

    xaxis_title="source",

    yaxis_title="height",

    barmode="group",

    bargap=0.1,

    xaxis = dict(

        tickmode = 'linear',

        tick0 = 0,

        dtick = 10),

    font=dict(

        family="Arial, monospace",

        size=15,

        color="#7f7f7f"

    )

    )

py.iplot(fig)
from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)



def show_wordcloud(data, title = None):

    wordcloud = WordCloud(

        background_color='black',

        stopwords=stopwords,

        max_words=200,

        colormap='Set3',

        max_font_size=40, 

        scale=3,

        random_state=1 # chosen at random by flipping a coin; it was heads

).generate(str(data))



    fig = plt.figure(1, figsize=(15, 15))

    plt.axis('off')

    if title: 

        fig.suptitle(title, fontsize=20)

        fig.subplots_adjust(top=2.3)



    plt.imshow(wordcloud)

    plt.show()



show_wordcloud(fox_df['source'])
fig = px.scatter(fox_df, x="width", y="source", color="height", marginal_y="rug", marginal_x="histogram")

fig
fig = px.density_contour(fox_df, x="width", y="source", color_discrete_sequence=['purple'])

fig.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRvVWWidXA8FgutFxsnJVcM-95dacIFfBdtZ5Ntfh3co5HEc_zw&usqp=CAU',width=400,height=400)