# For TF-IDF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords 
# For Word Count Visualization
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input/train"))
# loading CSV data to check the Description content
train_csv = pd.read_csv("../input/train/train.csv")
train_csv.head()
train_csv.sample(20).Description
train_csv.loc[[6440,10779]].Description
print('There are ' + str(len(train_csv[train_csv.Description.isna()])) + ' records without description. They were removed!')
train_csv.dropna(subset=['Description'], inplace=True)
def get_top_n_words(corpus, n=None):
    """
    List the top n words in a vocabulary according to occurrence in a text corpus.
    """
    vec = CountVectorizer(
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{3,}', # vectorize 3-character words or more
            stop_words='english',
            ngram_range=(1, 2),
            max_features=30000
        ).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

words_count_by_adoption_speed = []
for adoption_speed in range(5):
    descriptions_by_adoption_speed = train_csv[train_csv.AdoptionSpeed == adoption_speed].Description
    top_words = get_top_n_words(descriptions_by_adoption_speed, 25)
    words_count_by_adoption_speed.append(pd.DataFrame(top_words, columns = ['Word', 'Count'])) 
    words_count_by_adoption_speed[adoption_speed].plot.bar(x='Word',y='Count',title="Top 25 words X Adoption Speed " + str(adoption_speed))