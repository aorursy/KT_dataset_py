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
df = pd.read_csv("../input/moodle.csv.txt")

df.describe()
#Take a look at the first rows of the dataset

df.head()
#List all different values for storypoints

story_points_series = df['storypoint']

story_points_series.unique()
import matplotlib.pyplot as plt

story_points_series.plot.hist(figsize=(20,10))
story_points_series_lower = story_points_series[lambda x: x <= 40]

story_points_series_lower.plot.hist(figsize=(20,10))
story_points_series.plot.box(figsize=(20,10))
from wordcloud import WordCloud



text = " ".join(t for t in df.title) 



def create_word_cloud(text_data):

  wordcloud = WordCloud(width=1600, height=800,max_font_size=200).generate(text_data)

  plt.figure(figsize=(12,10))

  plt.imshow(wordcloud, interpolation="bilinear")

  plt.axis("off")

  plt.show()



create_word_cloud(text)