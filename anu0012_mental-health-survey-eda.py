# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import os

import pandas as pd

import sys

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set(color_codes=True)

from wordcloud import WordCloud



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/survey.csv")
df.head()
df.isnull().sum(axis=0)
df = df.fillna('Other')
df['Gender'].value_counts().head(10).plot.bar()
df['Country'].value_counts().head(10).plot.bar()
df['state'].value_counts().head(10).plot.bar()
df['self_employed'].value_counts().head().plot.bar()
# Generate a word cloud image

wordcloud = WordCloud(max_font_size=100).generate(' '.join(df['comments']))
# Display the generated image:



plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")