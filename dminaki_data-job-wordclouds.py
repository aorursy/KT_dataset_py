import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import glob as glob

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
ds_posts = pd.read_csv('../input/alldata.csv').dropna()
ds_posts.head()
data_science_jobs = ds_posts[ds_posts.position.str.lower().str.contains('data scientist')]
data_science_jobs.shape
data_science_text = ''

for x in data_science_jobs['description'].str.lower():

    data_science_text = data_science_text + x
# lower max_font_size, change the maximum number of word and lighten the background:

plt.figure(figsize=(15,8))

wordcloud = WordCloud(max_font_size=50, max_words=30, background_color="white").generate(data_science_text)

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
data_analyst_jobs = ds_posts[ds_posts.position.str.lower().str.contains('data analyst')]
data_analyst_jobs.shape
data_analyst_text = ''

for x in data_analyst_jobs['description'].str.lower():

    data_analyst_text = data_analyst_text + x
# lower max_font_size, change the maximum number of word and lighten the background:

plt.figure(figsize=(15,8))

wordcloud = WordCloud(max_font_size=50, max_words=30, background_color="white").generate(data_analyst_text)

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
research_analyst_data = ds_posts[ds_posts.position.str.lower().str.contains('research analyst')]
research_analyst_data.shape
research_analyst_text = ''

for x in research_analyst_data['description'].str.lower():

    research_analyst_text = research_analyst_text + x
# lower max_font_size, change the maximum number of word and lighten the background:

plt.figure(figsize=(15,8))

wordcloud = WordCloud(max_font_size=50, max_words=30, background_color="white").generate(research_analyst_text)

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
machine_learning_data = ds_posts[ds_posts.position.str.lower().str.contains('machine learning')]
machine_learning_data.shape
machine_learning_text = ''

for x in machine_learning_data['description'].str.lower():

    machine_learning_text = machine_learning_text + x
# lower max_font_size, change the maximum number of word and lighten the background:

plt.figure(figsize=(15,8))

wordcloud = WordCloud(max_font_size=50, max_words=30, background_color="white").generate(machine_learning_text)

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()