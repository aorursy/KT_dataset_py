# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas  as pd
train=pd.read_csv("../input/nlp-getting-started/train.csv")
test=pd.read_csv("../input/nlp-getting-started/test.csv")
train.shape,test.shape
train.isnull().sum()/train.shape[0]
from matplotlib import pyplot as plt

#creates figure of height 5inch and width 5 inch
fig=plt.figure(figsize=(5,5))

#labels for each class in target variable 
labels='Real','Not Real'
#Getting the count of each type of class from train data 
target_counts=train.target.value_counts()
sizes=[target_counts[1],target_counts[0]]

plt.pie(sizes,labels=labels,autopct='%1.1f%%')
# Equal aspect ratio ensures that pie is drawn as a circle.
plt.axis('equal') 

#display the chart
plt.show()

import re                                  # library for regular expression operations
import string                              # for string operations

from nltk.corpus import stopwords          # module for stop words that come with NLTK
from nltk.stem import PorterStemmer        # module for stemming
from nltk.tokenize import TweetTokenizer   # module for tokenizing strings
def remove_hashtag_url(tweet):
    # remove old style retweet text "RT"
    tweet2 = re.sub(r'^RT[\s]+', '', tweet)
    ## remove hyperlinks
    tweet2 = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet2)
    #removing hashtag but preservering hasthtag name
    tweet2 = re.sub(r'#', '', tweet2)
    #removing all mentions in tweet using @
    tweet2=  re.sub("@[A-Za-z0-9]+"," ",tweet2)
    return tweet2
    
#verifying what we have done is right
train['text_v2']=train.text.apply(remove_hashtag_url)
def display_all(df):
    with pd.option_context("display.max_rows", 1000): 
        with pd.option_context("display.max_columns", 1000): 
            display(df)
display_all(train[['text','text_v2']].transpose())
tokenizer=TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
train['text_v3']=train.text_v2.apply(tokenizer.tokenize)
display_all(train[['text','text_v2','text_v3']].transpose())
stopwords_english = stopwords.words('english') 
def clean_tweets(tweet_tokens):
    clean_tweet=[]
    for word in tweet_tokens:
        if( word not in stopwords_english and word not in string.punctuation ):
            clean_tweet.append(word)
    return clean_tweet
train['text_v4']=train.text_v3.apply(clean_tweets)

display_all(train[['text','text_v2','text_v3','text_v4']].transpose())
# Instantiate stemming class
stemmer = PorterStemmer() 
def stem(tweet):
    tweets_stem=[]
    for word in tweet:
        stem_word=stemmer.stem(word)
        tweets_stem.append(stem_word)
    return tweets_stem

train['text_v5']=train['text_v4'].apply(stem)
display_all(train[['text','text_v2','text_v3','text_v4','text_v5']].transpose())
#objective : to create a dictionary with key as (word,target_type) and value being #times it has been observed

#defining empty dictionary
freqs={}
#for every row in training set 
for row in train.itertuples(): 
    #we go over every word of cleaned tweet
    for word in row.text_v5:
        #key would be word and target class type
        pair=(word,row.target)
        #if key already exists , add one to it
        if pair in freqs:
            freqs[pair]+=1
        #else create the key
        else:
            freqs[pair]=1
"""
objective : to create a dictionary with keys as word and values as absolute of difference in number of times it
has appeared in all real tweets and number of times it has appeared in not real ones
"""
diff_freqs={}
#for every pair in freq 
for pair in freqs:
    # get word and target class type 
    word,target=pair[0],pair[1]
    #create key with word and other target class type ,let's call this compliment key
    comp_pair=(word,abs(target-1))
    #if the compiment key is present , take only half of the difference  as we would be taking another half when we iterate over compliment pair
    if comp_pair in freqs:
        if word in diff_freqs:
            diff_freqs[word]+=float( abs(freqs[pair]-freqs[comp_pair]) /2 )
        else:
            diff_freqs[word]=float( abs(freqs[pair]-freqs[comp_pair]) /2 )
    #else take full difference
    else:
        diff_freqs[word]=abs( freqs[pair] )        
#sorting the dictionary based on value 
diff_freqs2={k: v for k, v in sorted(diff_freqs.items(), key=lambda item: item[1],reverse=True)}
#extracting top 30 words 
top_words=[]
for item in diff_freqs2.items():
    top_words.append(item[0])
    if len(top_words)>=30:
        break;
import numpy as np

# list representing our table of word counts.
# each element consist of a sublist with this pattern: [<word>, <positive_count>, <negative_count>]
data = []

# loop through our selected words
for word in top_words:
    
    # initialize positive and negative counts
    pos = 0
    neg = 0
    
    # retrieve number of positive counts
    if (word, 1) in freqs:
        pos = freqs[(word, 1)]
        
    # retrieve number of negative counts
    if (word, 0) in freqs:
        neg = freqs[(word, 0)]
        
    # append the word counts to the table
    data.append([word, pos, neg])
    
fig, ax = plt.subplots(figsize = (15, 15))

# convert positive raw counts to logarithmic scale. we add 1 to avoid log(0)
x = np.log([x[1] + 1 for x in data])  

# do the same for the negative counts
y = np.log([x[2] + 1 for x in data]) 

# Plot a dot for each pair of words
ax.scatter(x, y)  

# assign axis labels
plt.xlabel("Log Real count")
plt.ylabel("Log Not Real count")

# Add the word as the label at the same position as you added the points just before
for i in range(0, len(data)):
    ax.annotate(data[i][0], (x[i], y[i]), fontsize=12)

ax.plot([2, 6], [2, 6], color = 'red') # Plot the red line that divides the 2 areas.
plt.show()
def pos_sum(tweet):
    pos_sum=0
    for word in set(tweet):
        pair=(word,1)
        if pair in freqs:
            pos_sum+=freqs[pair]
        else:
            pos_sum+=0
    return pos_sum
def neg_sum(tweet):
    neg_sum=0
    for word in set(tweet):
        pair=(word,0)
        if pair in freqs:
            neg_sum+=freqs[pair]
        else:
            neg_sum+=0
    return neg_sum

train['pos_sum']=train.text_v5.apply(pos_sum)
train['neg_sum']=train.text_v5.apply(neg_sum)         
# let's plot before running a logistic regression model to check whether it would be good model
import seaborn as sns
g =sns.scatterplot(x="pos_sum", y="neg_sum",
              hue="target",
              data=train);
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
param_grid={'C':[0.001,0.01,0.1,1,10,100,1000]}
lr=LogisticRegression()
grid=GridSearchCV(lr,param_grid=param_grid,cv=5,scoring='f1')
grid.fit(train[['pos_sum','neg_sum']],train.target)
grid.best_score_
#removing hashtag ,urls, and handles
test['text_v2']=test.text.apply(remove_hashtag_url)
#tokenizing the data and lower casing 
test['text_v3']=test.text_v2.apply(tokenizer.tokenize)
#removing stopwords and punctuations
test['text_v4']=test.text_v3.apply(clean_tweets)
#stemming 
test['text_v5']=test.text_v4.apply(stem)
# calculating #times each word in tweet appearing in positive corpurs and later summing it up
test['pos_sum']=train.text_v5.apply(pos_sum)
# calculating #times each word in tweet appearing in negative corpurs and later summing it up
test['neg_sum']=train.text_v5.apply(neg_sum)
#predicting based on best model
test['target']=grid.best_estimator_.predict(test[['pos_sum','neg_sum']])

#saving the model for submission
test[['id','target']].to_csv('submission_lr.csv',index=False)
#for each word calculate log of ratio of appearing in positive corpus and negative corpus 

def get_required_stats(freqs):
    V_n= len(set(freqs.keys()))
    #initialize total frequency  of positive corpus and negative corpus as zero
    pos_sum_freq=0
    neg_sum_freq=0
    
    # for every key word in freqs dictionary 
    for key in freqs:
        word,cat=key
        # if the key belongs to positive class , increment the total frequency of positive class by frequency of that word
        if cat==1:
            pos_sum_freq+=freqs.get(key,0)
        # if the key belongs to negative class , increment the total frequency of negative class by frequency of that word
        else:
            neg_sum_freq+=freqs.get(key,0)
    return V_n,pos_sum_freq,neg_sum_freq



def get_log_ratio(word,freqs,V_n,pos_sum_freq,neg_sum_freq):
    #get how many times word has appeared in postive corpus 
    pos_freq=freqs.get((word,1),0)
    #get how many times word has appeared in negative corpus 
    neg_freq=freqs.get((word,0),0)
    # get length of overall corpus
    
    #calculate the conditional probability that word appears given the tweet is positive 
    pos_ratio=(pos_freq+1)/(pos_sum_freq+V_n)
    #calculate the conditional probability that word appears given the tweet is negative 
    neg_ratio=(neg_freq+1)/(neg_sum_freq+V_n)
    
    #return ratio of them 
    return np.log( pos_ratio/ neg_ratio)

def get_log_prior(train):
    #calculating prior probability 
    # number of positive observations in the train 
    num_pos_class=train.target.value_counts()[1]
    #number of negative observations in the train 
    num_neg_class=train.target.value_counts()[0]
    return np.log( num_pos_class/num_neg_class)

log_prior=get_log_prior(train)

V_n,pos_sum_freq,neg_sum_freq=get_required_stats(freqs)
def get_log_tweet_prob(tweet,freqs,log_prior):
    #initialize log of posterior probability( probabilty that tweet is positive given set of words) with log(prior )
    prob=log_prior
    #add every words log conditional prob
    for word in tweet:
        prob+=get_log_ratio(word,freqs,V_n,pos_sum_freq,neg_sum_freq)
    
    if prob>=0:
        return 1
    else:
        return 0
    
#calculate f prediction on train 

train['prediction']=train.apply(lambda row: get_log_tweet_prob(row['text_v5'],freqs,log_prior),axis=1)
from sklearn.metrics import f1_score
f1_score(train.target,train.prediction)
test['target']=test.apply(lambda row: get_log_tweet_prob(row['text_v5'],freqs,log_prior),axis=1)
test[['id','target']].to_csv('submission_nb.csv',index=False)