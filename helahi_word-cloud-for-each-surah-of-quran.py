import pandas as pd

from csv import QUOTE_NONE
def read_and_reformat(csv_path):

    df = pd.read_csv(csv_path,

                     dtype=object)

    return df
df = read_and_reformat('../input/en.yusufali.csv')

df.head()
import re

surah_verse_dict = {}

surah_text = {}

for i in range(1,115):

    surah_verse_dict[str(i)] = {}

    surah_text[str(i)] = ""

for i, row in df.iterrows():

    try:

        surah_verse_dict[row['Surah']][row['Ayah']] = row['Text']

        surah_text[row['Surah']] += row['Text'] + " "

    except:

        pass
from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt



custom_stopwords = ['ye', 'verily', 'will', 'said', 'say', 'us', 'thy', 'thee']



for sw in custom_stopwords:

    STOPWORDS.add(sw);



for key in surah_text.keys():

    print("Surah #" + key)

    wordcloud = WordCloud(stopwords=STOPWORDS, background_color='black', width=800, height=400).generate(surah_text[key])

    plt.figure( figsize=(20,10) )

    plt.imshow(wordcloud)

    plt.axis('off')

    plt.show()