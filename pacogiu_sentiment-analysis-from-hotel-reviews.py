# Standard batch import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier
from nltk.tokenize import sent_tokenize, word_tokenize
import time
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
df=pd.read_csv('../input/Hotel_Reviews.csv')
# Lets look at some stats shall we?
display(df.shape)
display(df.describe())
display(df.describe(include=['O']))
# Keys handy for getting columns later on
df.keys()
# Just for fun, create a database with only hotels info
hotels=df[['Hotel_Name','Average_Score','Total_Number_of_Reviews',\
           'Hotel_Address', 'Additional_Number_of_Scoring','lat', 'lng']].drop_duplicates().reset_index()
print(hotels.shape)
hotels.head(3)
# Now lets get some new columns running
# Hotel address ---> Country ('Netherlands', 'UK', 'France', 'Spain', 'Italy', 'Austria')
hotels['Country']=hotels['Hotel_Address'].apply(lambda x: x.split()[-1]).replace('Kingdom','UK')
hotels.Country.unique()
# I would be curious to see which country gives more reviews (UK most of the time)
# So lets get the most common nationality of reviews for each hotel
most_national=df[['Hotel_Name','Hotel_Address','Reviewer_Nationality']].groupby(['Hotel_Name','Hotel_Address']).agg(lambda x:x.value_counts().index[0])
# In case we want to use the Tags, here I clean up the format 
# and create list from Tags ranked them by occurence
tags_rank=pd.Series(re.findall(r'[\']\s([\w\s]+)\s[\']',''.join(df.Tags))).value_counts()
# For some reason days where not numbers but string so..
# Correct day of the review into integer
df['days_since_review']=pd.to_numeric(df['days_since_review'].str.replace(r'[a-z]+', ""))
# Lets isolate the reviews that have some information into negative/positive
# For simplicity I start getting the badly scored and highly scored reviews 
# to feed later on to the classifier.
# This was he/she/it will understand how a bad or good review looks like
# DataFrames with Negative reviews and positive reviews given
neg_rev=df[df.Negative_Review!='No Negative'].reset_index().drop('index',1)
pos_rev=df[df.Positive_Review!='No Positive'].reset_index().drop('index',1)
neg_rev = neg_rev[neg_rev['Reviewer_Score']<5].reset_index().drop('index',1)
pos_rev = pos_rev[pos_rev['Reviewer_Score']>8].reset_index().drop('index',1)
# Takes review and gives back the clean list of words

#TokTok faster than word_tokenize
from nltk.tokenize import ToktokTokenizer
toktok = ToktokTokenizer()

# Stopwords, numbers and punctuation to remove
remove_punct_and_digits = dict([(ord(punct), ' ') for punct in string.punctuation + string.digits])
stopWords = set(stopwords.words('english'))


def word_cleaner(data):
    cleaned_word = data.lower().translate(remove_punct_and_digits)
    words = word_tokenize(cleaned_word)
    words = [toktok.tokenize(sent) for sent in sent_tokenize(cleaned_word)]
    wordsFiltered = []
    if not words:
        pass
    else:
        for w in words[0]:
            if w not in stopWords:
                wordsFiltered.append(w)
                end=time.time()
    return wordsFiltered


# Example
wordsFiltered = word_cleaner(neg_rev.Negative_Review[1])
print(neg_rev.Negative_Review[1])
print(wordsFiltered)
# We take a small sample within our database to speed up Learning
# with a decent machine and some time to spare we can easily skip this step
neg_red=neg_rev[:50000].copy()
pos_red=pos_rev[:50000].copy()
# Create set related to positive and negative review
def word_feats(words):
    return dict([(word, True) for word in words])
neg_set=[(word_feats(word_feats(word_cleaner(neg_red.loc[i,'Negative_Review']))), 0) for i in range(len(neg_red))]
pos_set=[(word_feats(word_feats(word_cleaner(pos_red.loc[i,'Positive_Review']))), 1) for i in range(len(pos_red))]
# Finally some Machine is Learning!
# Train the model and use CV to test accuracy

negcutoff = int(len(neg_set)*3/4)
poscutoff = int(len(pos_set)*3/4)
 
trainfeats = neg_set[:negcutoff] + pos_set[:poscutoff]
testfeats = neg_set[negcutoff:] + pos_set[poscutoff:]
print(len(trainfeats), len(testfeats))
 
classifier = NaiveBayesClassifier.train(trainfeats)
print( 'accuracy:', nltk.classify.util.accuracy(classifier, testfeats))
# Now we want to have a clear overview of the most hated/loved words so...
# Builds the dataframe of words with respective sentiment and score

cpdist = classifier._feature_probdist
word=[]
score=[]
sentiment=[]
for (fname, fval) in classifier.most_informative_features(100):
            def labelprob(l):
                return cpdist[l, fname].prob(fval)

            labels = sorted([l for l in classifier._labels
                             if fval in cpdist[l, fname].samples()],
                            key=labelprob)
            if len(labels) == 1:
                continue
            l0 = labels[0]
            l1 = labels[-1]
            if cpdist[l0, fname].prob(fval) == 0:
                ratio = 'INF'
            else:
                ratio = '%8.1f' % (cpdist[l1, fname].prob(fval) /
                                   cpdist[l0, fname].prob(fval))
            sentiment.append(int(l1))
            word.append(fname)
            score.append(float(ratio))
# Divides scores into negative and positive
word_scores=pd.DataFrame({'word':word,'sentiment':sentiment,'score':score})
neg_word_scores=word_scores[word_scores.sentiment==0]
pos_word_scores=word_scores[word_scores.sentiment==1]
display(word_scores[word_scores['sentiment']==1].head())
display(word_scores[word_scores['sentiment']==0].head())
neg_given=df[df.Negative_Review!='No Negative'].reset_index().drop('index',1)
pos_given=df[df.Positive_Review!='No Positive'].reset_index().drop('index',1)
# I want to create two new columns, one that will give a positive and one a negative score
# Sums positive and negative scores for a given review
def pos_sentiment_sum(review):
    pos=0
    asd=word_cleaner(review)
    set_w=set(pos_word_scores.word)-set(['no','negative','positive'])
    
    for word in asd:
        if word in set_w:
            pos+=pos_word_scores[pos_word_scores['word']==word].score.iloc[0]
    
    return pos

def neg_sentiment_sum(review):
    neg=0
    asd=word_cleaner(review)
    set_w=set(neg_word_scores.word)-set(['no','negative','positive'])
    
    for word in asd:
        if word in set_w:
            neg+=neg_word_scores[neg_word_scores['word']==word].score.iloc[0]
    
    return neg

# TEST
print('negative score:',neg_sentiment_sum(df.Negative_Review[6584]),\
       'positive score:', pos_sentiment_sum(df.Positive_Review[6584]))
# VERY SLOW! AROUND 45min for full database
# I comment it out so it does not need to run when I submit

# This is the final step where we get the additional columns 
# Produce the pos and neg colums in database

#=================
#pos_col=[]
#for i in range(len(df)):
#    if df.Positive_Review[i]=='No Positive':
#        pos_col.append(int(0))
#    else:
#        pos_col.append(pos_sentiment_sum(df.Positive_Review[i]))
#df['pos_score']=pos_col
#
#neg_col=[]
#for i in range(len(df)):
#    if df.Negative_Review[i]=='No Negative':
#        neg_col.append(int(0))
#    else:
#        neg_col.append(neg_sentiment_sum(df.Negative_Review[i]))
#df['neg_score']=neg_col
#=================
# The analysis can go on, but this is the last step for now
# Reviews grouped by rate band
score_9=df[df.Reviewer_Score>9].copy()
score_4=df[df.Reviewer_Score<4].copy()
score_6=df[df.Reviewer_Score<7].copy()
score_7=df[(df.Reviewer_Score>7)&(df.Reviewer_Score<8)].copy()
score_8=df[(df.Reviewer_Score>8)&(df.Reviewer_Score<9)].copy()
print(score_6.shape)
print(score_7.shape)
print(score_8.shape)
print(score_9.shape)
# Some plots

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
plt.style.use('fivethirtyeight')

plt.figure(figsize=(7,7))
plt.hist(df['Reviewer_Score'],bins=20)
plt.ylabel('Number_Reviewers',fontsize=16)
plt.xlabel('Rating',fontsize=16)
plt.title('Ratings accross Users',fontsize=16)
plt.axvline(df['Reviewer_Score'].mean(), color='k', linestyle='dashed', linewidth=1)
plt.savefig('Ratings_user.png')

plt.figure(figsize=(7,7))
plt.hist(df['Average_Score'],bins=20)
plt.ylabel('Number Hotels',fontsize=16)
plt.xlabel('Rating',fontsize=16)
plt.title('Ratings accross Hotels',fontsize=16)
plt.axvline(df['Average_Score'].mean(), color='k', linestyle='dashed', linewidth=1)
plt.savefig('Ratings_hotel.png')

plt.figure(figsize=(7,4))
week_bins=int(np.floor((max(df['days_since_review'])-min(df['days_since_review']))/7))
vals = plt.hist(df['days_since_review'],bins=week_bins);
plt.ylabel('Number ratings',fontsize=16)
plt.xlabel('Days passed',fontsize=16)
plt.title('Ratings per week',fontsize=16)
plt.axhline(vals[0].mean(), color='k', linestyle='dashed', linewidth=1)
plt.savefig('Ratings_week.png')

plt.figure(figsize=(7,4))
plt.plot(vals[0]);
plt.ylabel('Number ratings',fontsize=16)
plt.xlabel('Weeks passed',fontsize=16)
plt.title('Ratings per week',fontsize=16)
plt.axhline(vals[0].mean(), color='k', linestyle='dashed', linewidth=1)
plt.savefig('Ratings_week_plot.png')