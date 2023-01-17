import pandas as pd

import numpy as np



reviews = pd.read_csv("../input/tripadvisor_co_uk-travel_restaurant_reviews_sample.csv")

reviews = reviews.assign(

    rating = reviews.rating.map(lambda v: v.split(" ")[0] if pd.notnull(v) else np.nan),

    food = reviews.food.map(lambda v: v.split(" ")[0] if pd.notnull(v) else np.nan),

    value = reviews.value.map(lambda v: v.split(" ")[0] if pd.notnull(v) else np.nan),

    service = reviews.service.map(lambda v: v.split(" ")[0] if pd.notnull(v) else np.nan),

)

reviews.head(3)
import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("white")



f, axarr = plt.subplots(2, 2, figsize=(14, 7))

plt.suptitle('TripAdvisor UK Restaurant Review Metadata', fontsize=18)

f.subplots_adjust(hspace=0.5)



sns.countplot((reviews

                   .visited_on

                   .map(lambda v: float(v.split(" ")[-1]) if not pd.isnull(v) else v)

                   .dropna()

                   .astype(int)              

              ),

               ax=axarr[0][0])

axarr[0][0].set_title("Review Visit Timestamp Year")



(reviews

     .visited_on

     # Exclude 2017 because we don't have a full year of data.

     .map(lambda v: v.split(" ")[0] if not pd.isnull(v) and '2017' not in v else v)

     .value_counts()

     .reindex(['January', 'February', 'March', 'April', 'May', 'June',

               'July', 'August', 'September', 'October', 'November', 'December'])

).plot.bar(ax=axarr[0][1], color='darkseagreen')

axarr[0][1].set_title("Review Visit Timestamp Month")



(reviews

     .groupby('restaurant_id')

     .rating

     .count()

     .sort_values(ascending=False)

     .reset_index(drop=True)

).plot.line(ax=axarr[1][0])

axarr[1][0].set_title("Restaurants by Number of Reviews")



(reviews

     .groupby('author')

     .rating

     .count()

     .sort_values(ascending=False)

     .reset_index(drop=True)

).plot.line(ax=axarr[1][1])

axarr[1][1].set_title("Authors by Number of Reviews")



sns.despine()
reviews.rating = (reviews.rating

                      # Fix a format bug in the uploaded dataset.

                      .replace('April', np.nan).replace('September', np.nan)

                      .astype(float))
f, axarr = plt.subplots(2, 2, figsize=(14, 7))

plt.suptitle('TripAdvisor UK Restaurant Review Metadata', fontsize=18)

f.subplots_adjust(hspace=0.5)



sns.countplot(reviews.rating.dropna().astype(int), ax=axarr[0][0])

sns.countplot(reviews.food.sort_values(), ax=axarr[0][1])

sns.countplot(reviews.value.sort_values(), ax=axarr[1][0])

sns.countplot(reviews.service.sort_values(), ax=axarr[1][1])



sns.despine()
import itertools

restaurant_name_tokens = pd.Series(

    list(itertools.chain(*reviews.name.str.split(" ").values.tolist()))

)
from nltk.corpus import stopwords

from tqdm import tqdm

stopwords = set(stopwords.words('english'))
top_tokens = restaurant_name_tokens.value_counts()

top_tokens = top_tokens.iloc[

    np.argwhere(top_tokens.index.map(lambda t: str(t).lower() not in stopwords).values).flatten()

]

top_tokens = top_tokens.drop(['-', '&'])



top_tokens.head(20).plot.bar(figsize=(14, 5), fontsize=20)

sns.despine()
from nltk import word_tokenize



review_bag_of_words = list(

    itertools.chain(*(reviews

                           .title

                           .dropna()

                           .map(lambda t: word_tokenize(t[1:-1]))

                           .tolist()

                     ))

)

review_bag_of_words = pd.Series([w.lower() for w in review_bag_of_words])
wc = review_bag_of_words.value_counts()



wc.loc[

    ['awesome', 'excellent', 'great', 'good', 'ok', 'bad', 'awful', 'terrible', 'horrible']

].plot.bar(color='darkseagreen', figsize=(14, 5), fontsize=16)

plt.gca().set_title("Signal Words used in Review Titles", fontsize=20)

sns.despine()
from wordcloud import WordCloud

import matplotlib.pyplot as plt



plt.imshow(WordCloud().generate(" ".join(review_bag_of_words)), 

           interpolation='nearest', aspect='auto')
review_text_bag_of_words = list(itertools.chain(*(reviews

                               .review_text

                               .dropna()

                               .map(lambda t: word_tokenize(t[1:-1]))

                               .tolist()

                           )))

review_text_bag_of_words = pd.Series([w.lower() for w in review_bag_of_words])
review_text_bag_of_words.value_counts().loc[

    ['breakfast', 'brunch', 'lunch', 'dinner']

].plot.bar(color='darkseagreen', figsize=(14, 5), fontsize=16)

plt.gca().set_title("Time to Eat - Mealtime Mentions in Restaurant Reviews", fontsize=20)

sns.despine()