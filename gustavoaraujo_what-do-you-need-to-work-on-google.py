# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/job_skills.csv')
df.head()
categories = df.groupby('Category')
categories.size()
num_categories = []

for category in df.Category.unique():
    num_categories.append(categories.size()[category])

d = {'Category': df.Category.unique(), 'Vacancies': num_categories }
cat_df = pd.DataFrame(data=d)
bp = cat_df.plot(kind='bar', figsize=(15, 10), x='Category')
def makecloud(column, category, color):
    words = pd.Series(df.loc[df['Category'] == category][column]).str.cat(sep=' ')
    wc = WordCloud(stopwords=STOPWORDS, colormap=color, background_color='White', width=800, height=400).generate(words)
    plt.figure(figsize=(16,18))
    plt.imshow(wc)
    plt.axis('off')
    plt.title(category + " " + column);
    
def makeclouds(index, color):
    makecloud('Minimum Qualifications', df.Category.unique()[index], color)
    makecloud('Preferred Qualifications', df.Category.unique()[index], color)
makeclouds(0, 'Reds')
makeclouds(1, 'Greens')
makeclouds(2, 'Blues')
makeclouds(3, 'Reds')
makeclouds(4, 'Greens')
makeclouds(5, 'Blues')
makeclouds(6, 'Reds')
makeclouds(7, 'Greens')
makeclouds(8, 'Blues')
makeclouds(9, 'Reds')
makeclouds(10, 'Greens')
makeclouds(11, 'Blues')
makeclouds(12, 'Reds')
makeclouds(13, 'Greens')
makeclouds(14, 'Blues')
makeclouds(15, 'Reds')
makeclouds(16, 'Greens')
makeclouds(17, 'Blues')
makeclouds(18, 'Reds')
makeclouds(19, 'Greens')
makeclouds(20, 'Blues')
makeclouds(21, 'Reds')
makeclouds(22, 'Greens')