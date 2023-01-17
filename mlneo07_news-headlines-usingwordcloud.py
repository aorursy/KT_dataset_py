#importing required library

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS

data = pd.read_csv('../input/abcnews-date-text.csv')
#since date is not arraged and dealing with missing values
#lets managed it.

data.publish_date = pd.to_datetime(data.publish_date,format="%Y%m%d")
cleaned = data.dropna()
cleaned.head()
#Using wordcloud for most word used
#worldcloud using gray scale color
import random
def grey_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(60, 100)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(
                          background_color='black',
                          stopwords=stopwords,
                          random_state=42
                         ).generate(str(cleaned['headline_text']))

print(wordcloud)
#plot
plt.imshow(wordcloud.recolor(color_func=grey_color_func, random_state=3),
           interpolation="bilinear")
plt.axis("off")
plt.figure()
plt.show()

reindexed_data = cleaned['headline_text']
reindexed_data.index = cleaned['publish_date']
def get_top_n_words(n_top_words, count_vectorizer, text_data): 
    vectorized_headlines = count_vectorizer.fit_transform(text_data.as_matrix())
    
    vectorized_total = np.sum(vectorized_headlines, axis=0)
    word_indices = np.flip(np.argsort(vectorized_total)[0,:], 1)
    word_values = np.flip(np.sort(vectorized_total)[0,:],1)
    
    word_vectors = np.zeros((n_top_words, vectorized_headlines.shape[1]))
    for i in range(n_top_words):
        word_vectors[i,word_indices[0,i]] = 1

    words = [word[0].encode('ascii').decode('utf-8') for word in count_vectorizer.inverse_transform(word_vectors)]

    return (words, word_values[0,:n_top_words].tolist()[0])
from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer(stop_words='english')
words, word_values = get_top_n_words(n_top_words=20, count_vectorizer=count_vectorizer, text_data=reindexed_data)
wordcloud = WordCloud(
                          background_color='blue',
                          stopwords=stopwords,
                          random_state=42
                         ).generate(str(words))

print(wordcloud)
#plot
plt.imshow(wordcloud.recolor(color_func=grey_color_func, random_state=3),
           interpolation="bilinear")
plt.axis("off")
plt.figure()
plt.show()