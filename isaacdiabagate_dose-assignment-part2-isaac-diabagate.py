# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
from wordcloud import WordCloud,STOPWORDS
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/amazon-alexa-reviews/amazon_alexa.tsv', sep='\t')
data
ratings_frequency = data['rating']
plt.figure()
bins =[1,2,3,4,5]
ratings_frequency.plot.hist(bins=bins, edgecolor='black', color='orange', figsize=(7,6))

plt.title('Rating Frequency')
plt.xlabel('Ratings')
plt.ylabel('Frequency')
plt.show()
data['count']=1
variation_rating = data.groupby(['variation', 'rating']).count()['count'].unstack()

plt.figure()
variation_rating.plot.barh(figsize=(12,7), stacked=True)

plt.title('Ratings and Variation',fontsize=25)
plt.xlabel ('Total Count', fontsize=20)
xt = range(0,600, 50)
plt.xticks(xt)
plt.ylabel('Variations',fontsize=20)
plt.legend(fontsize= 25)

plt.show()
rating_review = data[['rating', 'verified_reviews']]

word_freq2 = rating_review.verified_reviews.loc[rating_review['rating']==1].str.replace(r'.!?,',"").str.lower().str.split(expand=True).unstack().value_counts()
word_freq1 = rating_review.verified_reviews.loc[rating_review['rating']==5].str.replace(r'.!?,',"").str.lower().str.split(expand=True).unstack().value_counts()


# get top words that only appear in rating 5 or rating 1
set5=set(word_freq1[0:100].keys())
set1= set(word_freq2[0:100].keys())
unique5 = list(set5 - set1)
unique1 = list(set1 - set5)


# Top words from each side by side
d = {'count in ratings: 5': word_freq1[0:25], "Count in Ratings : 1": word_freq2[0:25]}
b=pd.DataFrame(d)
b.index.name = 'Most Common Words for ratings 1 and 5'
b


z = {'unique words: rating 5': unique5, "Unique Words: rating 1": unique1}
y= pd.DataFrame(z)
y.index.name = 'Most Common Unique Words for ratings 1 and 5'
y

stopwords = set(STOPWORDS) 
wordcloud = WordCloud(width = 1000, height = 600, 
                background_color ='pink', 
                stopwords = stopwords, 
                min_font_size = 3, max_words=20).generate(' '.join(unique5) )

plt.figure(figsize = (10, 5), facecolor = None) 
plt.imshow(wordcloud) 
plt.title('Top Words seen for Ratings of 5', fontsize=25)
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 
  
stopwords = set(STOPWORDS) 
wordcloud = WordCloud(width = 1000, height = 600, 
                background_color ='grey', 
                stopwords = stopwords, 
                min_font_size = 4, max_words=20).generate(' '.join(unique1))

plt.figure(figsize = (10, 5), facecolor = None) 
plt.imshow(wordcloud) 
plt.title('Top Words seen for Ratings of 1', fontsize=20)
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 