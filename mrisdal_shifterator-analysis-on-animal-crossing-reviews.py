!pip install shifterator
# Import packages

import pandas as pd
import numpy as np
import itertools
import collections
import nltk
from nltk.corpus import stopwords
import re

from shifterator import relative_shift as rs

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.5)
sns.set_style("whitegrid")
# Load the review CSV
reviews = pd.read_csv("../input/animal-crossing/user_reviews.csv", encoding='utf-8')
reviews.head()
reviews['date'] = pd.to_datetime(reviews['date'])
reviews.index = reviews['date'] 

fig, ax = plt.subplots(figsize=(12, 8))

mean_daily_grades = reviews.resample('D', on='date').mean().reset_index('date')

# Plot horizontal bar graph
monthly_plot = sns.lineplot(data = mean_daily_grades,
                      x = 'date',
                      y = 'grade',
                      color="purple"
                      )

ax.set_title("Average daily grade")
x_dates = mean_daily_grades['date'].dt.strftime('%m-%d').sort_values().unique()
ax.set_xticklabels(labels=x_dates, rotation=45, ha='right')

plt.show()
# Divide reviews into positive and negative based on the median grade for the dataset
median_grade = reviews.grade.median()

reviews.loc[reviews['grade'] <= median_grade, 'review_category'] = 'Negative' 
reviews.loc[reviews['grade'] > median_grade, 'review_category'] = 'Positive' 

reviews_neg = reviews[reviews['review_category'] == 'Negative']
reviews_pos = reviews[reviews['review_category'] == 'Positive']
texts = reviews['text'].tolist()
texts_neg = reviews_neg['text'].tolist()
texts_pos = reviews_pos['text'].tolist()
# We will want to remove stop words
stop_words = set(stopwords.words('english'))
def remove_punctuation(txt):
    """Replace URLs and other punctuation found in a text string with nothing 
    (i.e. it will remove the URL from the string).

    Parameters
    ----------
    txt : string
        A text string that you want to parse and remove urls.

    Returns
    -------
    The same txt string with URLs and punctuation removed.
    """

    return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", txt).split())
def clean_text(txt):
    """Removes punctuation, changes to lowercase, removes
        stopwords, removes "animal" and "crossing", and
        calculates word frequencies.

    Parameters
    ----------
    txt : string
        A text string that you want to clean.

    Returns
    -------
    Words and frequencies
    """
    
    tmp = [remove_punctuation(t) for t in txt]
    tmp = [t.lower().split() for t in tmp]
    
    tmp = [[w for w in t if not w in stop_words]
              for t in tmp]
    tmp = [[w for w in t if not w in ['animal', 'crossing']]
                     for t in tmp]
    
    tmp = list(itertools.chain(*tmp))
    tmp = collections.Counter(tmp)
        
    return tmp
# Clean up the review texts
clean_texts_neg = clean_text(texts_neg)
clean_texts_pos = clean_text(texts_pos)
# Dataframes for most frequent common words in positive and negative reviews
common_neg = pd.DataFrame(clean_texts_neg.most_common(15),
                             columns=['words', 'count'])
common_pos = pd.DataFrame(clean_texts_pos.most_common(15),
                             columns=['words', 'count'])
fig, ax = plt.subplots(figsize=(8, 8))

# Plot horizontal bar graph
common_neg.sort_values(by='count').plot.barh(x='words',
                      y='count',
                      ax=ax,
                      color="red")

ax.set_title("Common Words Found in Negative Reviews")

plt.show()
fig, ax = plt.subplots(figsize=(8, 8))

# Plot horizontal bar graph
common_pos.sort_values(by='count').plot.barh(x='words',
                      y='count',
                      ax=ax,
                      color="green")

ax.set_title("Common Words Found in Positive Reviews")

plt.show()
# From https://www.kaggle.com/prakashsadashivappa/word-cloud-of-abstracts-cord-19-dataset
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

wordcloud = WordCloud(
    width = 3000,
    height = 2000,
    background_color = 'black',
    ).generate(str(texts_neg))
fig = plt.figure(
    figsize = (10, 8),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()
# Get an entropy shift
entropy_shift = rs.EntropyShift(reference=clean_texts_neg,
                                comparison=clean_texts_pos,
                                base=2)
entropy_shift.get_shift_graph() 
# Get a Jensen-Shannon divergence shift
from shifterator import symmetric_shift as ss
jsd_shift = ss.JSDivergenceShift(system_1=clean_texts_neg,
                                 system_2=clean_texts_pos,
                                 base=2)
jsd_shift.get_shift_graph()