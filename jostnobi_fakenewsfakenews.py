# Setup all imports
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
import gensim
import nltk

from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from collections import defaultdict

# Load data
file_list = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        file_list += [os.path.join(dirname, filename)]
        
fake_news =  pd.read_csv('/kaggle/input/fake-and-real-news-dataset/Fake.csv',)
true_news =  pd.read_csv('/kaggle/input/fake-and-real-news-dataset/True.csv')

# Add classifiaction
fake_news["classification"] = "false"
true_news["classification"] = "true"

# Concarnate data and make all text lowercase
news_data = pd.concat([true_news, fake_news], axis =0)
news_data["title"] = news_data["title"].apply(lambda x: x.lower())
news_data["text"] = news_data["text"].apply(lambda x: x.lower())

print("Setup finished")
# Preprocess titles
title_corpus         = news_data.title.tolist()
true_title_corpus    = true_news.title.tolist()
fake_title_corpus    = fake_news.title.tolist()

# Split corpus by white space and filter out stopwords
stoplist          = set('for a of the and to in u.s. as on [video] (video) '.split(' '))
title_corpus      = [[word for word in title.split() if word not in stoplist]  for title in title_corpus]
true_title_corpus = [[word for word in title.split() if word not in stoplist]  for title in true_title_corpus]
fake_title_corpus = [[word for word in title.split() if word not in stoplist]  for title in fake_title_corpus]

#### CONTINUE FROM HERE


# Count word frequencies
frequency = defaultdict(int)
for title in title_corpus:
    for word in title:
        frequency[word] += 1
        
# Make dataframe of word frequencies
words, count = list(frequency.keys()), list(frequency.values())
frequency_df = pd.DataFrame({"words": words, "count": count})

# Remove low frequency words
title_corpus = [[token for token in title if frequency[token] > 1] for title in title_corpus]
# Run to add function for plotting wordcloud
def plot_wordcloud(text, mask = None, max_words = 400, max_font_size = 120, more_stopwords = None, figure_size = (16.0,8.0), 
                   title = None, title_size = 40, image_color = False):
    stopwords = set(STOPWORDS)
    if(more_stopwords != None):
        stopwords = stopwords.union(more_stopwords)

    wordcloud = WordCloud(background_color='white',
                    stopwords = stopwords,
                    max_words = max_words,
                    max_font_size = max_font_size, 
                    random_state = 42,
                    mask = mask)
    wordcloud.generate(text)
    
    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageColorGenerator(mask);
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");
        plt.title(title, fontdict={'size': title_size,  
                                  'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud);
        plt.title(title, fontdict={'size': title_size, 'color': 'green', 
                                  'verticalalignment': 'bottom'})
    plt.axis('off');
    plt.tight_layout()  
title_corpus_bag_of_words = ""
for title in title_corpus:
    for word in title:
        title_corpus_bag_of_words += word + " "
        
word_cloud_mask = np.array(Image.open('/kaggle/input/trump-pic/trump2.jpg'))

plot_wordcloud (title_corpus_bag_of_words, 
                mask         = word_cloud_mask, 
                max_words    =300, 
                max_font_size=120, 
                title        = 'Fake news', 
                title_size   =50)    
