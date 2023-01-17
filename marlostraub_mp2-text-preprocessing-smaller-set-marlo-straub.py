# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
!pip install --upgrade pip
!pip install spacy
import spacy
sp = spacy.load('en_core_web_sm')
!pip install git+https://github.com/crazyfrogspb/RedditScore.git

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

from sklearn.feature_extraction import text #text processing
from sklearn.feature_extraction.text import CountVectorizer #spliting and numeric representation (Bag-of-words/n-grams)
from sklearn.feature_extraction.text import TfidfTransformer #calculating word importance score (TF/IDF)
#check out the data 
df=pd.read_csv("/kaggle/input/hcde530/comments20200605.csv")
df.head() #load the first 5 rows of the D&S dataframe using this method
print(df.shape) #show me the shape of this dataframe 
print(df.columns) #question for instructors: do I need to leave this in to show you I am looking at this, as proof or process?
                  #or is stuff like this typicaly left out?
#convert epoch time to python timestamp format
#I ended up not using this data for this assignment, but the content owner was interested in pre and post-COVID comments so I left the code in
import datetime #import this module 
df['published'] = df['comment_publish_date'].apply(lambda x: datetime.datetime.fromtimestamp(x)) #update the argument from pd.datetime to datetime.datetime per datetime documentation
#c.head() #check the output
#remove unnecessary columns

c=df.drop(['video_id','commenter_channel_url','comment_publish_date', 'commenter_rating', 'comment_id','commenter_channel_id','commenter_channel_display_name','comment_parent_id','collection_date',], axis=1)
print(c.shape)
c.head()
#clean up newlines from text column (general question: is it better to do this in pandas or in sklearn/spacy, etc?)
cleaned = c.replace(r'\\n',' ', regex=True) 
cleaned = c.replace('\n','', regex=True)
#expand how much of the text column I can read to make it easier to work with
pd.set_option('display.max_colwidth', None)
print(cleaned.text[50:60])
#sort the comments by like count
cl_likes=cleaned.sort_values(by=['comment_like_count'], ascending=False)
cl_likes.head()
#create a smaller dataframe to clean for the sake of scoping down this project
clbiglikes=cl_likes.loc[cl_likes['comment_like_count']>500]
print(type(clbiglikes))
print(clbiglikes.shape)
clbiglikes.to_csv("comments501.csv", index=False) #save to csv to share with content creator
#create a medium sized dataframe to mess around with
cl1like=cl_likes.loc[cl_likes['comment_like_count']>1]
print(type(cl1like))
print(cl1like.shape)
cl1like.to_csv("comments1like.csv", index=False) #save to csv to share with content creator
#code is from Rafal per video 2
import nltk
from nltk.stem.porter import *
from nltk import word_tokenize

#Create own tokenization (method of splitting text into individual word tokens)
#Problem 3 - Fix typos by replacing words, e.g represent 'the best' and 'good' the same way
class LemmaTokenizer:
    def __init__(self):
        self.sp = spacy.load('en_core_web_sm') #load english language data
    def __call__(self, doc):
        replacements = {'Kuomintang (KMT)': 'Kuomintang', '(KMT)': 'Kuomintang','oofed':'offed'} #replace specific tokens; I chose these from a manual visual skim of the output csv
        tokens = self.sp(doc) #tokenize the document (split into words) #should I be splitting this into sentences too?
        return [replacements.get(t.lemma_,t.lemma_) for t in tokens] #replace some tokens #I actually do not know what this does, so I left it in

class PorterTokenizer: 
    def __init__(self):
        self.stemmer = PorterStemmer() #Create porter tokenizer
    def __call__(self, doc):
        replacements = {'Kuomintang (KMT)': 'Kuomintang', '(KMT)': 'Kuomintang','oofed':'offed'} #replace specific tokens; I chose these from a manual visual skim of the output csv
        return [replacements.get(self.stemmer.stem(t), self.stemmer.stem(t)) for t in word_tokenize(doc)]  
#Converting words to word ids using Scikit-learn CountVectorizer
#Let's try LemmaTokenizer
count_vect = CountVectorizer( 
    tokenizer=LemmaTokenizer(),
    ngram_range=(1,2)
)#create the CountVectorizer object
count_vect.fit(clbiglikes['text'].values) #fit into our dataset

#Get a list of unique words found in the document (vocabulary)
word_list = count_vect.get_feature_names()
print("This is the length of the LemmaTokenizer feature list: "+str(len(word_list)))

#Check all the words that were extracted and their ids:
print("These are the outputs using LemmaTokenizer:")
for word_id, word in enumerate(word_list):
    print(str(word_id)+" -> "+str(word_list[word_id])) #Show ID and word

#Let's try PorterTokenizer
count_vect = CountVectorizer( 
    tokenizer=PorterTokenizer(),
    ngram_range=(1,2) #let's try ngram range up to 2 to try and capture some of the entities
)#create the CountVectorizer object
count_vect.fit(clbiglikes['text'].values) #fit into our dataset

#Get a list of unique words found in the document (vocabulary)
word_list = count_vect.get_feature_names()
print("This is the length of the PorterTokenizer feature list: "+str(len(word_list))) #cast integer to string

#Check all the words that were extracted and their ids:
print("These are the outputs using Porter Tokenizer:")
for word_id, word in enumerate(word_list):
    print(str(word_id)+" -> "+str(word_list[word_id])) #Show ID and word
from redditscore import tokenizer
from redditscore.tokenizer import CrazyTokenizer #discovered while trying to figure out how to clean some of the stranger punctuation formats at https://redditscore.readthedocs.io/en/master/tokenizing.html
#create the CrazyTokenizer object
tokenizer = CrazyTokenizer( #documentation at https://redditscore.readthedocs.io/en/master/apis/tokenizer.html
    lowercase=True, 
    normalize=4, #get rid of the strings like, "craaaazy" and "craaaaaaaazy"
    ignore_quotes=True,
    ignore_stopwords=True, #ignore stopwords taken from english nltk package
    remove_breaks=True,
    remove_nonunicode=True,#this matters less for this set, but the original comment set had lots of nonunicode characters
    pos_emojis=True, neg_emojis=True, neutral_emojis=True,#separate consecutive emojis using built in code
    ngrams=2,
    whitespaces_to_underscores=False
)
X=clbiglikes['text'].apply(tokenizer.tokenize) #apply the tokenizer method defined above on the 'text' column of df clbiglikes; name the output X
print(len(X))
print(type(X))
print(X)

#I am struggling to find the equivalent command to "word_list = count_vect.get_feature_names()"but for CrazyTokenizer
#word_list = tokenizer.get_feature_names()
#but maybe I don't need to see the list of features to proceed?

from redditscore.models import bow_mod #import the bag of words models

# estimator = ??? The documentation indicates that it will take an sklearn classifier or regressor.. https://redditscore.readthedocs.io/en/master/redditscore.models.html#redditscore.models.redditmodel.RedditModel
#...but I don't think I want either of these types of estimators since I don't know the names or number of categories of comment types or topics

#this code was taken from the documentation at https://redditscore.readthedocs.io/en/master/modelling.html
#commenting it out since it produces an error unless I have an estimator defined
# bow_model = bow_mod.BoWModel(estimator, ngrams=2, tfidf=True) #per the documentation, this is the objet I would create, I just don't know what to fill in for estimator so this will return an error
# bow_model.fit_transform(X) #this is how I would apply the fit_transform method to my text data 
#I'm not sure if this would actually work with the definition of X provided in the previous code block or if I would have to convert the series to a string first
#create a CountVectorizer object for splitting the text into words and replacing them with numbers. I got stuck here so commenting it out.
#count_vect = CountVectorizer(
    #tokenizer = CrazyTokenizer(), #crap, I realized I don't know how to define a CrazyTokenizer object
    #stop_words = ['about', 'this', 'quite', 'that', 'the', '!', '.', 'i', '-PRON-', 'with'],
   # ngram_range=(1,2)#Learn the vocabulary and transform our dataset from words to ids
#I am basically stuck between the steps above and below; I can't find a way to get the list of unique words and the counts of those words in the text
word_counts = count_vect.fit_transform(df['text'].values) #Transform the text into numbers (bag of words features)

dir(tokenizer) #I looked at this to try and find more documentaiton on what I can do to figure out. 
#It may be that I am tired, but I can't find documentation on what these other methods are