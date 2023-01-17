# Importing libraries

import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# Set up matplotlib style 

plt.style.use('ggplot')



# Libraries for wordcloud making and image importing

from wordcloud import WordCloud, ImageColorGenerator

from PIL import Image



import tensorflow as tf

from tensorflow import keras

import random
input_dir = '../input/'

file_path = os.path.join(input_dir, 'Sarcasm_Headlines_Dataset.json')



data = pd.read_json(file_path, lines=True)
data.shape
data.head()
data['is_sarcastic'].value_counts()
data['website_domain'] = data['article_link'].apply(lambda x: x.split('.com')[0].split('.')[-1])
data.head()
data['website_domain'].value_counts()
data.groupby(['website_domain','is_sarcastic'])['headline'].aggregate('count').unstack().fillna(0)