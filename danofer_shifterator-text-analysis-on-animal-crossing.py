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
def remove_punctuation(txt:str):

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



def clean_text(txt:str) -> {}:

    """Removes punctuation, changes to lowercase, removes

        stopwords, removes "animal" and "crossing", and

        calculates word frequencies (as counts).



    Parameters

    ----------

    txt : string

        A text string that you want to clean.



    Returns

    -------

    Words and frequency counts

    """

    

    tmp = [remove_punctuation(t) for t in txt]

    tmp = [t.lower().split() for t in tmp]

    

    tmp = [[w for w in t if not w in stop_words]

              for t in tmp]

#     tmp = [[w for w in t if not w in ['animal', 'crossing']]

#                      for t in tmp]

    

    tmp = list(itertools.chain(*tmp))

    tmp = collections.Counter(tmp)

        

    return tmp
# Load the review CSV

reviews = pd.read_csv("../input/animal-crossing/user_reviews.csv", encoding='utf-8')

print(reviews.shape)

reviews.head()
print(reviews.drop_duplicates("text").shape[0]) # 3 duplicate reviews. 
# do some text normalization to filter out more duplicate reviews , regardless of subsequent filtering

reviews["text"] = reviews["text"].str.lower().str.replace("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", case=False,regex=True)
reviews = reviews.drop_duplicates("text")

print(reviews.shape[0]) # 6 duplicate reviews. 
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
reviews.grade.hist();
mean_grade = reviews.grade.mean()

print(f"Average grade {mean_grade}")



median_grade = reviews.grade.median()

print(f"Median grade {median_grade}")





reviews.loc[reviews['grade'] <= mean_grade, 'review_category'] = 'Negative' 

reviews.loc[reviews['grade'] > mean_grade, 'review_category'] = 'Positive' 



reviews_neg = reviews[reviews['review_category'] == 'Negative']

reviews_pos = reviews[reviews['review_category'] == 'Positive']
texts = reviews['text'].tolist()

texts_neg = reviews_neg['text'].tolist()

texts_pos = reviews_pos['text'].tolist()
# Extend with custom stop words + animal + crossing

stop_words = set(stopwords.words('english'))

stop_words.update(['animal', 'crossing', "game"])

# stop_words
# Clean up the review texts

clean_texts_neg = clean_text(texts_neg)

clean_texts_pos = clean_text(texts_pos)
clean_texts_neg
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