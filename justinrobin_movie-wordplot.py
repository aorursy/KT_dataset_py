# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
metadata=pd.read_csv("/kaggle/input/the-movies-dataset/movies_metadata.csv")
links = pd.read_csv("/kaggle/input/the-movies-dataset/links.csv")
links_small=pd.read_csv("/kaggle/input/the-movies-dataset/links_small.csv")
ratings_small=pd.read_csv("/kaggle/input/the-movies-dataset/ratings.csv")
ratings=pd.read_csv("/kaggle/input/the-movies-dataset/ratings.csv")
keywods=pd.read_csv("/kaggle/input/the-movies-dataset/keywords.csv")
metadata.info()
del metadata['homepage']
links.info()
ratings.head()
metadata.head()
import nltk as nlp
import re
description_list_out = []
for description in metadata['genres']:
    description = re.sub("[^a-zA-Z]",' ',str(description))
    description = description.lower() 
    description = nlp.word_tokenize(description)
    #description = [ word for word in description if not word in set(stopwords.words("english"))]
    lemma = nlp.WordNetLemmatizer()
    description = [ lemma.lemmatize(word) for word in description]
    description = " ".join(description)
    description_list_out.append(description) #we hide all word one section
#We make bag of word it is including number of all word's info
from sklearn.feature_extraction.text import CountVectorizer 
max_features = 3000 #We use the most common word
count_vectorizer = CountVectorizer(max_features = max_features, stop_words = "english")
sparce_matrix = count_vectorizer.fit_transform(description_list_out).toarray()
print("the most using {} words: {}".format(max_features,count_vectorizer.get_feature_names()))
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 

wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = STOPWORDS, 
                min_font_size = 10).generate(str(count_vectorizer.get_feature_names()).replace('id','')) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off")
plt.title("Most watched")
plt.tight_layout(pad = 0) 
  
plt.show()
spoken_languages_list_out = []
for description in metadata['spoken_languages']:
    description = re.sub("[^a-zA-Z]",' ',str(description))
    description = description.lower() 
    description = nlp.word_tokenize(description)
    #description = [ word for word in description if not word in set(stopwords.words("english"))]
    lemma = nlp.WordNetLemmatizer()
    description = [ lemma.lemmatize(word) for word in description]
    description = " ".join(description)
    spoken_languages_list_out.append(description) #we hide all word one section
#We make bag of word it is including number of all word's info
from sklearn.feature_extraction.text import CountVectorizer 
max_features = 3000 #We use the most common word
count_vectorizer = CountVectorizer(max_features = max_features, stop_words = "english")
spoken_languages_list_out_matrix = count_vectorizer.fit_transform(spoken_languages_list_out).toarray()
print("the most using {} words: {}".format(max_features,count_vectorizer.get_feature_names()))
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 

wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = STOPWORDS, 
                min_font_size = 10).generate(str(count_vectorizer.get_feature_names())) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off")
plt.title("top languages")
plt.tight_layout(pad = 0) 
  
plt.show()