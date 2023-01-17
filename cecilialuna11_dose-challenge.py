# for basic operations

import numpy as np

import pandas as pd



# for basic visualizations

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')



# for advanced visualizations

import plotly.offline as py

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

from plotly import tools

init_notebook_mode(connected = True)

import plotly.figure_factory as ff



# for providing the path

import os

print(os.listdir('../input/'))
data = pd.read_csv('../input/amazon-alexa-reviews/amazon_alexa.tsv', delimiter = '\t', quoting = 3)

# describing the data set



data.describe()
# Creating a Histogram of the ratings



plt.hist(data.rating);

plt.show();
color = plt.cm.copper(np.linspace(0, 1, 15))

data['variation'].value_counts().plot.bar(color = color, figsize = (15, 9))

plt.title('Distribution of Variations in Alexa', fontsize = 20)

plt.xlabel('variations')

plt.ylabel('count')

plt.show()
from sklearn.feature_extraction.text import CountVectorizer





cv = CountVectorizer(stop_words = 'english')

words_with_rating5 = data.verified_reviews[data.rating.eq(5)]

words = cv.fit_transform(words_with_rating5)

sum_words = words.sum(axis=0)





words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]

words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)

frequency = pd.DataFrame(words_freq, columns=['word', 'freq'])



plt.style.use('fivethirtyeight')

color = plt.cm.ocean(np.linspace(0, 1, 20))

frequency.head(20).plot(x='word', y='freq', kind='bar', figsize=(15, 6), color = color)

plt.title("Most Frequently Occuring Words")

plt.show()
