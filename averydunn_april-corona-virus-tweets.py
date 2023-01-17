# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load



import pandas as pd 
import numpy as np 
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#uncomment in the tweets for the days in April, and concatenate a few days if necessary
pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)

tweets_april1 = pd.read_csv('/kaggle/input/coronavirus-covid19-tweets-early-april/2020-04-01 Coronavirus Tweets.CSV')
# tweets_april2 = pd.read_csv('/kaggle/input/coronavirus-covid19-tweets-early-april/2020-04-02 Coronavirus Tweets.CSV')
# tweets_april3 = pd.read_csv('/kaggle/input/coronavirus-covid19-tweets-early-april/2020-04-03 Coronavirus Tweets.CSV')
# tweets_april4 = pd.read_csv('/kaggle/input/coronavirus-covid19-tweets-early-april/2020-04-04 Coronavirus Tweets.CSV')
# tweets_april5 = pd.read_csv('/kaggle/input/coronavirus-covid19-tweets-early-april/2020-04-05 Coronavirus Tweets.CSV')

# tweets = pd.concat([tweets_april1, tweets_april2, tweets_april3, tweets_april4, tweets_april5])
print("Size is: ", tweets_april1.shape)
print("Columns are: ", tweets_april1.columns)

tweets_april1.head()
april1_df = tweets_april1.loc[:,['text', 'retweet_count']]
april1_df
!pip install vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

def detectSentiment(data_set):
    for text in data_set.text:
        analysis = analyzer.polarity_scores(text)
        print('{} ----- {}'.format(str(text), str(analysis)))
detectSentiment(april1_df.head(20))

pos_count = 0
neg_count = 0
neu_count = 0 

for text in april1_df.text.head(75): 
    if analyzer.polarity_scores(text)['compound'] >= 0.05: 
        pos_count += 1
    elif analyzer.polarity_scores(text)['compound'] <= -0.05: 
        neg_count += 1 
    else: 
        neu_count += 1
total = pos_count+neg_count+neu_count
c = [(pos_count/total)*100, (neg_count/total)*100, (neu_count/total)*100]
labels=['Positive', 'Negative', 'Neutral']
plt.pie(c, labels=labels, autopct='%1.1f%%')
plt.show()
analyzer = SentimentIntensityAnalyzer()

pos_list = []
neg_list = []
neu_list = []

for text in april1_df.text.head(50): 
    if analyzer.polarity_scores(text)['compound'] >= 0.05:
        pos_list.append(text)
    elif analyzer.polarity_scores(text)['compound'] <= - 0.05:
        neg_list.append(text)
    else:
        neu_list.append(text)
pos_df = pd.DataFrame(pos_list, columns = ['positive_tweets'])
neg_df = pd.DataFrame(neg_list, columns=['negative_tweets'])
neu_df = pd.DataFrame(neu_list, columns=['neutral_tweets'])




    
comment_words = ''
stopwords = set(STOPWORDS)

for text in pos_df.positive_tweets:
    a = text.replace('#', '')
    b = a.replace("'", '')
    c = b.replace('#', '')
    d = c.replace('https', '')
    e = d.replace(':', '')
    f = e.replace('/', '')
    final = f.replace('RT', '')
    final = str(final)
    tokens = final.split()
    
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
    comment_words += ' '.join(tokens)+' '

wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words) 
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 
        
    
    