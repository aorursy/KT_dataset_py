#importing bunch of libraries
from plotly.offline import init_notebook_mode, iplot
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from textblob import TextBlob 

import plotly.plotly as py
from plotly import tools
import pandas as pd
import string, os, random

init_notebook_mode(connected=True)
punc = string.punctuation
review_csv = pd.read_csv('../input/Reviews.csv',encoding = 'latin-1')
review_csv.head()
review_csv.describe()
#helper function for plotting wordcloud
def plot_wordcloud(text, mask=None, max_words=400, max_font_size=120, figure_size=(24.0,16.0), 
                   title = None, title_size=40, image_color=False):
    stopwords = set(STOPWORDS)
    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'book', 'read', 'reading'}
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
#Statistical analysis
review_csv['word_count'] = review_csv['ReviewContent'].apply(lambda x : len(x.split()))
review_csv['char_count'] = review_csv['ReviewContent'].apply(lambda x : len(x.replace(" ","")))
review_csv['word_density'] = review_csv['word_count'] / (review_csv['char_count'] + 1)
review_csv['punc_count'] = review_csv['ReviewContent'].apply(lambda x : len([a for a in x if a in punc]))
review_csv.head()
#Statistical analysis
review_csv.describe()
#helper function for sentiment analysis
def get_polarity(text):
    try:
        blob = TextBlob(text)
        pol = blob.sentiment.polarity
    except:
        pol = 0.0
    return pol
def get_subjectivity(text):
    try:
        blob = TextBlob(text)
        subj = blob.sentiment.subjectivity
    except:
        subj = 0.0
    return subj
df1 = pd.DataFrame(columns = ['polarity'])
df2 = pd.DataFrame(columns = ['subjectivity'])
total = review_csv['ReviewContent'].count()
for i in range (0,total):
    polarity = get_polarity(review_csv.at[i,'ReviewContent'])
    subjectivity = get_subjectivity(review_csv.at[i,'ReviewContent'])
    df1 = df1.append({'polarity': polarity}, ignore_index=True)
    df2 = df2.append({'subjectivity': subjectivity}, ignore_index=True)
print(df1.head())
print(df2.head())
review_csv = review_csv.join(df1)
review_csv = review_csv.join(df2)
review_csv.head()
review_csv.describe()
reviews_text = str(review_csv.ReviewContent)
plot_wordcloud(reviews_text, max_words=400, max_font_size=120, 
               title = 'Most frequently used words', title_size=50)
positive_review = review_csv.where(review_csv['polarity'] > 0)
positive_review = positive_review.dropna(axis=0, how='any')
reviews_text = str(positive_review.ReviewContent)
plot_wordcloud(reviews_text, max_words=929, max_font_size=120, 
               title = 'Most frequently used words from positive reviews', title_size=50)
negative_review = review_csv.where(review_csv['polarity'] < 0)
negative_review = negative_review.dropna(axis=0, how='any')
reviews_text = str(negative_review.ReviewContent)
plot_wordcloud(reviews_text, max_words=929, max_font_size=120, 
               title = 'Most frequently used words from negative reviews', title_size=50)
#let us see if there's any interesting correlation
review_csv.corr()
#and there's no interesting correlation (i kinda hope to see if there's corellation between polarity and subjectivity)
#but it turns out that positive or negative polarity is pretty random, so like 50% objective and 50% subjective
#let's visualize polarity vs subjectivity
plt.scatter(review_csv.polarity, review_csv.subjectivity, color='b')
plt.show()
#so...many reviews are actually in (kinda) neutral region