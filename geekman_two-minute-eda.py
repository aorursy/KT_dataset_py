# data exploration (for fun)
import os
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer


tokenizer = RegexpTokenizer(r'\w+')
wordnet_lemmatizer = WordNetLemmatizer()
# NLTK Stop words
stop_words = stopwords.words('english')


train_df = pd.read_csv('../input/train.csv')
# tokenize, remove stopwords, and lemmatize
def prepare_text(txt):
    tokens = tokenizer.tokenize(txt)
    tokens = [token for token in tokens if token.lower() not in stop_words]
    tokens = [wordnet_lemmatizer.lemmatize(token) for token in tokens]
    return tokens


# process toxic questions
processed_list = [prepare_text(x) for x in train_df['question_text']]
toxic_processed_list = [processed_list[x] for x in train_df[train_df['target'] == 1].index]
flat_toxic_processed_list = [item for sublist in toxic_processed_list for item in sublist]
flat_toxic_string = " ".join(x for x in flat_toxic_processed_list)
from wordcloud import WordCloud
# generate wordcloud
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='black', 
                min_font_size = 10).generate(flat_toxic_string) 
  

import matplotlib.pyplot as plt
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show() 