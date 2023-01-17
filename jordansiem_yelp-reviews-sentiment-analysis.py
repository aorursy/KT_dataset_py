import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import nltk.data



from nltk import sentiment

from nltk import word_tokenize

import collections



#graded words already with vader

from nltk.sentiment.vader import SentimentIntensityAnalyzer







from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize



%matplotlib inline





#important link: https://programminghistorian.org/en/lessons/sentiment-analysis
yelp = pd.read_csv('../input/yelp-reviews-data-science-bootcamp/yelp.csv')



allwords = pd.read_csv('../input/words-file/Words File.csv')



sid = SentimentIntensityAnalyzer()

yelp.head()

print('\n')

print('\n')

yelp.info()

print('\n')

print('\n')

yelp.describe()
scores_list = []

adjusted_score_list = []



for words in yelp['text']:

    scores = sid.polarity_scores(words)

    scores = (scores['compound'])

    scores = float(scores)

    scores_list.append(scores)

    

yelp['sentiment score'] = scores_list
yelp.head()
#After viewing some of the views and words used, I'm curious how filler words are effecting the sentiment score as well as
#if it'd be possible to get more specific adjectives and descriptors of what reviewers thought about the business. What were the top words
#used etc. This would definitely be more effective for one business, however, still interesting to see.
yelp.head()
#Adjusting reviews removing stop words (sentence fillers) and recalculating sentiment score



stop_words = set(stopwords.words('english'))

adjusted_review = []

word_count_list = []
print(stop_words)
for words in yelp['text']:

    filter = word_tokenize(words)

    #Remove stop words

    filtered_sentence = [w for w in filter if not w in stop_words]

    #Rebuild sentence without stop words

    filtered_sentence = ' '.join(w for w in filtered_sentence)

    adjusted_review.append(filtered_sentence)
yelp['adjusted review'] = adjusted_review

for words in yelp['adjusted review']:

    scores = sid.polarity_scores(words)

    scores = (scores['compound'])

    scores = float(scores)

    adjusted_score_list.append(scores)
#Lets get crazy and combine all the reviews together to find more trends in word choice



combinedtext = ''

for i in adjusted_review:

    combinedtext +=i



#Dictionary listing the word and how many times it was used

def word_count(str):

    counts = dict()

    words = str.split()



    for word in words:

        if word in counts:

            counts[word] += 1

        else:

            counts[word] = 1



    return counts

yelp.head()
y = word_count(combinedtext)

#Trying to disregard some of the random words not really describing the business or values that aren't words at all.

#Originally I tried a list of adjectives, however, I didn't have any positive matches which is why I extended my words list.



word_count_list = []



allwords_list = allwords['Words'].tolist()



key_list = []

values_list = []

scores_list = []

word_rating_list = []





for key, value in sorted(y.items(), key = lambda kv:( kv[1], kv[0]),reverse=True):

    key = str(key)

    key = ''.join(a for a in key if a not in ' /\\\'\":-.,')

    key = key.lower()

    if key in allwords_list:

        scores = sid.polarity_scores(key)

        scores = (scores['compound'])

        scores = float(scores)

        if scores != 0:

            if scores > 0 and scores <= .3:

                word_rating = 'low positive'

            if scores > .3  and scores <= .6:

                word_rating = 'moderately positive'

            if scores > .6:

                word_rating = 'high positive'

            if scores < 0 and scores >= -.3:

                word_rating = 'low negative'

            if scores < -.3 and scores >= -.6:

                word_rating = 'moderately negative'

            if scores < -.6:

                word_rating = 'high negative'

            x = ("%s: %d" % (key, value))

            x = str(x) +':' + str(scores) + ':' + word_rating

            word_rating_list.append(word_rating)

            scores_list.append(scores)

            key_list.append(key)

            values_list.append(value)

            

#Combining lists

zipped_List =  list(zip(key_list,values_list,scores_list,word_rating_list))



words_table = pd.DataFrame(zipped_List, columns = ['Word' , 'Number of Occurrences', 'Sentiment Score', 'Rating'])



print(words_table.head(30))



count =words_table.groupby('Rating').count()
print(count)
yelp['adjusted sentiment score'] = adjusted_score_list
yelp.head()

print('\n')

print('\n')

yelp.info()
yelp['review comparision'] = yelp['text'].equals(yelp['adjusted review'])



check = yelp['review comparision'].value_counts()

print(check)



yelp['score difference'] = yelp['sentiment score'] - yelp['adjusted sentiment score']



#Cleaning reviews of stop words made some difference



yelp.head(1)
sns.set_style('white')
#Review length based on number of stars given
yelp['text length'] = yelp['text'].apply(len)





g = sns.FacetGrid(yelp,col='stars')

g.map(plt.hist,'text length',bins=50)
g = sns.FacetGrid(yelp,col='stars')

g.map(plt.hist,'sentiment score',bins=50)
sns.boxplot(x='stars',y='text length',data=yelp,palette='rainbow')
sns.boxplot(x='stars',y='sentiment score',data=yelp,palette='rainbow')
sns.countplot(x='stars',data=yelp,palette='rainbow')
stars = yelp.groupby('stars').mean()
stars.corr()
sns.heatmap(stars.corr(),cmap='YlGnBu',annot=True)