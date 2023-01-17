# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
data=pd.read_csv("../input/Tweets.csv")

# Check the ratio of positive and negative tweets for each airline
data['countval']=1
groupby_object=data[['airline','airline_sentiment','countval']].groupby(['airline','airline_sentiment']).aggregate(sum)
groupby_object.unstack(level=1).plot(kind='bar')
plt.show()

#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# Imports
import re
from sklearn.feature_extraction.text import TfidfVectorizer

#####
data['text']=data['text'].map(lambda x: re.sub("^@[^\s]+\s","",x))
def getHashtag(x):
	g=re.match("^[^#]+#([^\s]+).*",x)
	if g:
		return g.group(1)
	else:
		return ""
	
data['hashtags']=data['text'].map(getHashtag)
# Convert to lower case
data['hashtags']=data['hashtags'].str.lower() 

from nltk.corpus import stopwords
def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = raw_review
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))   

Reviews=[]
dataPos=data[data['airline_sentiment']=='positive']
for i in range(0, len(dataPos)):
    Reviews.append(review_to_words(dataPos['text'].tolist()[i]))

vect = TfidfVectorizer(sublinear_tf=True, max_df=0.5, analyzer='word',stop_words='english')
vect.fit(Reviews)
idf = vect._tfidf.idf_
wordDict=dict(zip(vect.get_feature_names(), idf))

word1={k: v for k, v in wordDict.items() if v < 4}
from wordcloud import WordCloud
wordcloud = WordCloud().generate(' '.join(word1.keys()))
plt.title('POSITIVE : TFID FREQ < 4')
plt.imshow(wordcloud)
word1={k: v for k, v in wordDict.items() if v > 4 and v <=5}
from wordcloud import WordCloud
wordcloud = WordCloud().generate(' '.join(word1.keys()))
plt.title('POSITIVE : TFID FREQ between 4 and 5')
plt.imshow(wordcloud)
word1={k: v for k, v in wordDict.items() if v > 5 and v <= 6}
from wordcloud import WordCloud
wordcloud = WordCloud().generate(' '.join(word1.keys()))
plt.title('POSITIVE: TFID FREQ between 5 and 6')
plt.imshow(wordcloud)