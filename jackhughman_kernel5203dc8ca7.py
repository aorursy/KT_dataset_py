# init notebook

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from wordcloud import WordCloud



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/amazon-alexa-reviews/amazon_alexa.tsv', delimiter = '\t')

print(df.info())
df['rating'].value_counts().sort_index(ascending = False).plot.bar(figsize = (12, 7))

plt.xticks(rotation = "horizontal", fontsize = 13)

plt.yticks(fontsize = 13)

plt.title('Number of Ratings for Rating Values', fontsize = 25)

plt.xlabel('Rating Value', fontsize = 25)

plt.ylabel('Rating Count', fontsize = 25)

plt.show()
df.groupby('variation').mean()['rating'].plot.bar(figsize=(12, 7))

plt.yticks(fontsize = 13)

plt.xlabel('Variation', fontsize = 13)

plt.ylabel('Average Rating', fontsize = 13)

plt.title("Average Rating per Variation", fontsize = 25);

plt.show()
textFiveStar = " ".join(df[df['rating'] == 5]['verified_reviews'])

wordcloud = WordCloud(background_color="white").generate(textFiveStar)

plt.figure(figsize=(21, 13))

plt.axis("off")

plt.imshow(wordcloud, interpolation='bilinear')

plt.title("Wordcloud of Five-Star Reviews", fontsize=20)

plt.show()

textOneStar = " ".join(df[df['rating'] == 1]['verified_reviews'])

wordcloud = WordCloud(background_color="white").generate(textOneStar)

plt.figure(figsize=(21, 13))

plt.axis("off")

plt.imshow(wordcloud, interpolation='bilinear')

plt.title("Wordcloud of One-Star Reviews", fontsize=20)

plt.show()