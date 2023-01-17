# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
tweet=pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test=pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
tweet['Data']='tweet'   #for identification when concat the dfs

tweet.head(40)
test['Data']='test'

test.head()
tweet.info()
#tweet['target'].unique()      #0= no disaster, 1= disaster

x=tweet['target'].value_counts()

sns.barplot(x.index,x)                 #no disaster tweets are more than disaster ones
test.info()
# count numer of words

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))

tweet_len=tweet[tweet['target']==1]['text'].str.split().map(lambda x: len(x))

ax1.hist(tweet_len,color='blue')

ax1.set_title('Disaster Tweets')

#tweet_len

tweet_len=tweet[tweet['target']==0]['text'].str.split().map(lambda x: len(x))

ax2.hist(tweet_len,color='red')

ax2.set_title('Non-Disaster Tweets')
#combining DFs for data cleaning

df_both=pd.concat([tweet,test])

df_both.reset_index(inplace=True)
#dropping unwanted column 'location' as large no. of null values

df_both = df_both.drop(['location'],axis=1)

df_both.shape
#REMOVING URLs

df_both['text_nourl']  = df_both['text'].replace('http\S+','', regex=True)   

#looks for http till the whitespace character comes up and replaces it with space
df_both['text_nourl'].tail(5)   #URL links removed
##REMOVAL OF EMOJIS & EMOTICONS

!pip install emoji

import emoji

emo=emoji.UNICODE_EMOJI

def remove_emojis(text):

    noemoji=[word for word in text if word not in emo]

    return ''.join(noemoji)

df_both['text_nourl_noemoji']=df_both['text_nourl'].apply(lambda x: remove_emojis(x))
#### Emojis like this ;) and :) has not been removed, they can be removed by Emoticon Cleaning

#not working

#!pip install emot #This may be required for the Colab notebook

#from emot.emo_unicode import UNICODE_EMO, EMOTICONS

# Function for removing emoticons

#def remove_emoticons(text):

    #emoticon_pattern = re.compile(u'(' + u'|'.join(k for k in   EMOTICONS) + u')')

    #return emoticon_pattern.sub(r'', text)

#df_both['text_nourl_noemoji']=df_both['text_nourl_noemoji'].apply(lambda x: remove_emoticons(x))
df_both['text_nourl_noemoji'].head(30)    
#REMOVAL OF HTML tags

import re

def remove_html_tags(text):

    """Remove html tags from a string"""

    clean = re.compile('<.*?>')

    return re.sub(clean, ' ',text)

df_both['text_nourl_noemoji_nohtmltag']=df_both['text_nourl_noemoji'].apply(lambda x: remove_html_tags(x))
df_both['text_nourl_noemoji_nohtmltag'].head()
#KEEPING only ASCII letters(removing all the weird ones)

df_both['text_nourl_noemoji_nohtmltag_ascii'] = df_both['text_nourl_noemoji_nohtmltag'].str.encode('ascii', 'ignore').str.decode('ascii')
#pd.set_option('display.max_colwidth', -1)

df_both['text_nourl_noemoji_nohtmltag_ascii'].tail(25)   #only ascii letters
# PUNCTUATION REMOVAL

import string

string.punctuation
def remove_punc(text):

    no_punct=[words for words in text if words not in string.punctuation]

    return ''.join(no_punct)

df_both['text_nourl_noemoji_nohtmltag_ascii_nopunct']=df_both['text_nourl_noemoji_nohtmltag_ascii'].apply(lambda x: remove_punc(x))
df_both.tail(10)
#DO THE TOKENIZATION(SPLITING INTO A LIST OF WORDS)(USING REGEX)

import re

def tokens(text):

    split=re.split("\W+",text)    #“\W+” splits on one or more non-word character

    return split

df_both['text_nourl_noemoji_nohtmltag_ascii_nopunct_split']=df_both['text_nourl_noemoji_nohtmltag_ascii_nopunct'].apply(lambda x: tokens(x.lower()))
df_both.head()
#REMOVAL OF STOP WORDS

###will use “nltk” library for stop-words 

!pip install nltk
import nltk

stopword = nltk.corpus.stopwords.words('english')      #consists 179 stop words

print(stopword[:10])
def stop_wrds(text):

    nsw=[words for words in text if words not in stopword]

    #nsw_join=' '.join(nsw)

    return nsw

df_both['text_nourl_noemoji_nohtmltag_ascii_nopunct_split_nostopwords']=df_both['text_nourl_noemoji_nohtmltag_ascii_nopunct_split'].apply(lambda x: stop_wrds(x))
df_both.head()
freq = pd.Series((' '.join(df_both['text_nourl_noemoji_nohtmltag_ascii_nopunct_split_nostopwords'].apply(lambda x: ' '.join(x)))).split()).value_counts()[:30]

freq     #30 most frequently occuring words for further stopwords removal
# Adding common words from our document to stop_words as mentioned below as of no significance

from nltk.corpus import stopwords

add_words = ["im", "rt", "2","us","would"]

stop_words = set(stopwords.words("english"))

stop_added = stop_words.union(add_words)

def stop_w(text):

    sw=[words for words in text if words not in stop_added]

    return sw

df_both['text_nourl_noemoji_nohtmltag_ascii_nopunct_split_nostopwords']=df_both['text_nourl_noemoji_nohtmltag_ascii_nopunct_split_nostopwords'].apply(lambda x: stop_w(x))



df_both['text_nourl_noemoji_nohtmltag_ascii_nopunct_split_nostopwords'].head(10)
#STEMMING

from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()

df_both['text_nourl_noemoji_nohtmltag_ascii_nopunct_split_nostopwords_stem'] =df_both['text_nourl_noemoji_nohtmltag_ascii_nopunct_split_nostopwords'].apply(lambda x: [porter.stem(i) for i in x])

#########Stemming works on elements not on lists so if list is there then have to put a loop that can run through all the elements of the lists and do the stemming.########
df_both.head()

#stemming seems quite inefficient as it reduces the word to a word that doesnot make much sense
#LEMMATIZATION

from nltk.stem import WordNetLemmatizer 

lemmatizer = WordNetLemmatizer() 

df_both['text_nourl_noemoji_nohtmltag_ascii_nopunct_split_nostopwords_lemma']=df_both['text_nourl_noemoji_nohtmltag_ascii_nopunct_split_nostopwords'].apply(lambda x: [lemmatizer.lemmatize(i) for i in x])

#######Lemmatization works on elements not on lists so if list have to put a loop that can run through all the elements of the lists and do the lemmatization.########
df_both.head(10)

#Lemmatization seems more efficient as it reduces the word to the root form which makes more sense so will choose Lemmatization
df_dataclean_final = df_both[['index','id','keyword','text','target','Data','text_nourl_noemoji_nohtmltag_ascii_nopunct_split_nostopwords_stem','text_nourl_noemoji_nohtmltag_ascii_nopunct_split_nostopwords_lemma']]

df_dataclean_final.head()
#df_dataclean_final.rename(columns={'text_nourl_noemoji_nohtmltag_ascii_nopunct_split_nostopwords_lemma':'text_lemma'},inplace=True)

#df_dataclean_final.rename(columns={'text_nourl_noemoji_nohtmltag_ascii_nopunct_split_nostopwords_stem':'text_stem'},inplace=True)

df_dataclean_final['text_lemma']= [' '.join(map(str,l)) for l in df_dataclean_final['text_nourl_noemoji_nohtmltag_ascii_nopunct_split_nostopwords_lemma']]  

df_dataclean_final['text_stem']= [' '.join(map(str,t)) for t in df_dataclean_final['text_nourl_noemoji_nohtmltag_ascii_nopunct_split_nostopwords_stem']]                                                                  
##WORDCLOUD PLOT

##Visualizing all the words in column "text_lemma" in our data using the wordcloud plot.



all_words_lemma = ' '.join([word for word in df_dataclean_final['text_lemma']])

from wordcloud import WordCloud

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words_lemma)

plt.figure(figsize=(10, 7))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis('off')

plt.title("Most Common words in column Text Lemma")

plt.show()





##Visualizing all the words in column "text_stem" in our data using the wordcloud plot.



all_words_stem = ' '.join([word for word in df_dataclean_final['text_stem']])

from wordcloud import WordCloud

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words_stem)

plt.figure(figsize=(10, 7))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis('off')

plt.title("Most Common words in column Text stem")

plt.show()



###### Wordcloud is different for stem and lemma words
# Importing library

from sklearn.feature_extraction.text import CountVectorizer

bow_vectorizer = CountVectorizer(max_df=0.90, min_df=3, max_features=1000, stop_words='english')

bow_vectorizer

####### max_df is used for removing terms that appear too frequently(here 90%)

####### min_df is used for removing terms that appear too infrequently(here ignore terms that appear in less than 3 documents)
# 1.1 Bag-Of-Words feature matrix - For columns "df_dataclean_final['text_lemma']"

bow_lemma = bow_vectorizer.fit_transform(df_dataclean_final['text_lemma'])

bow_lemma
# 1.2 Bag-Of-Words feature matrix - For columns "df_dataclean_final['text_stem']"

bow_stem = bow_vectorizer.fit_transform(df_dataclean_final['text_stem'])

bow_stem
# Importing library

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')

tfidf_vectorizer
# 2.1 TF-IDF feature matrix - For columns "df_dataclean_final['text_lemma']"

tfidf_lemma = tfidf_vectorizer.fit_transform(df_dataclean_final['text_lemma'])

tfidf_lemma
# 2.2 TF-IDF feature matrix - For columns "df_dataclean_final['text_stem']"

tfidf_stem = tfidf_vectorizer.fit_transform(df_dataclean_final['text_stem'])

tfidf_stem
# Importing Libraries

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score,accuracy_score,precision_score
#  A.1) For columns "df_dataclean_final['text_stem']"



bow_stem =  bow_stem.tocsr()          #converting coo_object(coordinate format) to csr format

train_bow_stem = bow_stem[:7613,:]    #for same number of elements



## splitting data into training and validation set

X_train_bow, X_test_bow, y_train, y_test = train_test_split(train_bow_stem, tweet['target'], test_size=0.3, random_state=0)



logreg = LogisticRegression()

logreg.fit(X_train_bow, y_train)       # training the model



##Predicting the test set results and calculating the accuracy

y_pred_bow = logreg.predict(X_test_bow)



print ('Accuracy:', accuracy_score(y_test, y_pred_bow))    # calc. accuracy score

print ('Precision%:', precision_score(y_test, y_pred_bow)*100)

A1 = f1_score(y_test, y_pred_bow,average='weighted')    # calculating f1 score

print(A1)
type(X_test_bow[0])
y_pred_bow


print(' The prediction: {}, {}'.format(X_test_bow,y_pred_bow))
#  A.2) For columns "df_dataclean_final['text_lemma']"



bow_lemma =  bow_lemma.tocsr()          #converting coo_object(coordinate format) to csr format

train_bow_lemma = bow_lemma[:7613,:]    #for same number of elements



## splitting data into training and validation set

X_train_bow2 , X_test_bow2 , y_train2 , y_test2 = train_test_split(train_bow_lemma , tweet['target'], test_size=0.3, random_state=0)



logreg = LogisticRegression()

logreg.fit(X_train_bow2 , y_train2)       # training the model



##Predicting the test set results and calculating the accuracy

y_pred_bow2 = logreg.predict(X_test_bow2)



print ('Accuracy:', accuracy_score(y_test2 , y_pred_bow2))    # calc. accuracy score

print ('Precision%:', precision_score(y_test2, y_pred_bow2)*100)

A2 = f1_score(y_test2 , y_pred_bow2 ,average='weighted')    # calculating f1 score

print(A2)

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test2, y_pred_bow2))
#  B.1) For columns "df_dataclean_final['text_stem']"



#bow_stem =  bow_stem.tocsr()          #converting coo_object(coordinate format) to csr format

train_tfidf_stem = tfidf_stem[:7613,:]    #for same number of elements as in tweet df



## splitting data into training and validation set

X_train_tfidf, X_test_tfidf, y_train3 , y_test3 = train_test_split(train_tfidf_stem, tweet['target'], test_size=0.3, random_state=0)



logreg = LogisticRegression()

logreg.fit(X_train_tfidf, y_train3)       # training the model



##Predicting the test set results and calculating the accuracy

y_pred_tfidf = logreg.predict(X_test_tfidf)



print ('Accuracy:', accuracy_score(y_test3, y_pred_tfidf))    # calc. accuracy score

print ('Precision%:', precision_score(y_test3 , y_pred_tfidf)*100)

B1 = f1_score(y_test3 , y_pred_tfidf,average='weighted')    # calculating f1 score

print(B1)
tfidf_lemma
#  B.1) For columns "df_dataclean_final['text_lemma']"



#bow_stem =  bow_stem.tocsr()          #converting coo_object(coordinate format) to csr format

train_tfidf_lemma = tfidf_lemma[:7613,:]    #for same number of elements as in tweet df





## splitting data into training and validation set

X_train_tfidf4, X_test_tfidf4 , y_train4 , y_test4 = train_test_split(train_tfidf_lemma , tweet['target'], test_size=0.3, random_state=0)



logreg = LogisticRegression()

logreg.fit(X_train_tfidf4, y_train4)       # training the model



##Predicting the test set results and calculating the accuracy

y_pred_tfidf4 = logreg.predict(X_test_tfidf4)



print ('Accuracy:', accuracy_score(y_test4, y_pred_tfidf4))    # calc. accuracy score

print ('Precision%:', precision_score(y_test4 , y_pred_tfidf4)*100)

B2 = f1_score(y_test4 , y_pred_tfidf4 ,average='weighted')    # calculating f1 score

print(B2)

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test4, y_pred_tfidf4))
print("F1 - Score Chart For Logistic Regression")

print("** F1-Score - Model using Bag-of-Words features")

print("   F1-Score = ",A1," - For column tweets are stemmed")

print("   F1-Score = ",A2," - For column tweets are Lemmatized")

print("** F1-Score - Model using TF-IDF features")

print("   F1-Score = ",B1," - For column tweets are stemmed")

print("   F1-Score = ",B2," - For column tweets are Lemmatized")
#import library

from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import export_graphviz

from pydot import graph_from_dot_data
#  B.1) For columns "df_dataclean_final['text_stem']"



train_tfidf_stem = tfidf_stem[:7613,:]    #for same number of elements as in tweet df



## splitting data into training and validation set

X_train_tfidf_stem_dt, X_test_tfidf_stem_dt , y_train_dt1 , y_test_dt1 = train_test_split(train_tfidf_stem , tweet['target'], test_size=0.3, random_state= 100)
## Fitting Decision Tree Models to the Training set



clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100, splitter='best',

                               max_depth=3, min_samples_leaf=5)            #min_samples_leaf: The minimum number of samples required to be at a leaf node

clf_gini = clf_gini.fit(X_train_tfidf_stem_dt, y_train_dt1)



#prediction using gini index

y_pred_stem_dt = clf_gini.predict(X_test_tfidf_stem_dt)

#y_pred_stem_dt



print ('Accuracy:', accuracy_score(y_test_dt1, y_pred_stem_dt))    # calc. accuracy score

print ('Precision%:', precision_score(y_test_dt1 , y_pred_stem_dt)*100)

DT_f1score = f1_score(y_test_dt1 , y_pred_stem_dt ,average='weighted')    # calculating f1 score

print(DT_f1score)
#  B.1) For columns "df_dataclean_final['text_stem']"



train_tfidf_lemma = tfidf_lemma[:7613,:]    #for same number of elements as in tweet df



## splitting data into training and validation set

X_train_tfidf_lemma_dt, X_test_tfidf_lemma_dt , y_train_dt2 , y_test_dt2 = train_test_split(train_tfidf_lemma , tweet['target'], test_size=0.3, random_state= 100)



## Fitting Decision Tree Models to the Training set



clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100, splitter='best',

                               max_depth=3, min_samples_leaf=5)            #min_samples_leaf: The minimum number of samples required to be at a leaf node

clf_gini = clf_gini.fit(X_train_tfidf_lemma_dt, y_train_dt2)



#prediction using gini index

y_pred_lemma_dt = clf_gini.predict(X_test_tfidf_lemma_dt)

#y_pred_stem_dt



print ('Accuracy:', accuracy_score(y_test_dt2, y_pred_lemma_dt))    # calc. accuracy score

print ('Precision%:', precision_score(y_test_dt2 , y_pred_lemma_dt)*100)

DT__f1score = f1_score(y_test_dt2 , y_pred_lemma_dt ,average='weighted')    # calculating f1 score

print(DT__f1score)
#importing libraries

!pip install xgboost

import xgboost as xgb
#  B.1) For columns "df_dataclean_final['text_stem']"



train_tfidf_stem = tfidf_stem[:7613,:]    #for same number of elements as in tweet df



## splitting data into training and validation set

X_train_tfidf_stem_xg, X_test_tfidf_stem_xg , y_train_xg1 , y_test_xg1 = train_test_split(train_tfidf_stem , tweet['target'], test_size=0.3, random_state= 100)



#trainig of the model

xg_reg = xgb.XGBClassifier(objective ='binary:logistic', colsample_bytree = 0.3, learning_rate = 0.3,

                max_depth = 5, n_estimators = 10)     #n_estimators: number of trees you want to build

                                                                  #reg:logistic for classification problems with only decision

xg_reg.fit(X_train_tfidf_stem_xg, y_train_xg1)

#predicting

pred_stem_xg = xg_reg.predict(X_test_tfidf_stem_xg)



print ('Accuracy:', accuracy_score(y_test_xg1, pred_stem_xg))    # calc. accuracy score

print ('Precision%:', precision_score(y_test_xg1 , pred_stem_xg)*100)

xg_stem_f1score = f1_score(y_test_xg1 , pred_stem_xg ,average='weighted')    # calculating f1 score

print(xg_stem_f1score)

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test_xg1 ,pred_stem_xg))

##precision (also called positive predictive value) is the fraction of relevant instances among the retrieved instances, 

## recall (also known as sensitivity) is the fraction of the total amount of relevant instances that were actually retrieved. 
#  B.1) For columns "df_dataclean_final['text_stem']"



train_tfidf_lemma = tfidf_lemma[:7613,:]    #for same number of elements as in tweet df



## splitting data into training and validation set

X_train_tfidf_lemma_xg, X_test_tfidf_lemma_xg , y_train_xg2 , y_test_xg2 = train_test_split(train_tfidf_lemma , tweet['target'], test_size=0.3, random_state= 100)



#trainig of the model

xg_reg = xgb.XGBClassifier(objective ='binary:logistic', colsample_bytree = 0.3, learning_rate = 0.3,

                max_depth = 5, n_estimators = 10)     #n_estimators: number of trees you want to build

                                                                  #reg:logistic for classification problems with only decision

xg_reg.fit(X_train_tfidf_lemma_xg, y_train_xg2)

#predicting

pred_lemma_xg = xg_reg.predict(X_test_tfidf_lemma_xg)



print ('Accuracy:', accuracy_score(y_test_xg2, pred_lemma_xg))    # calc. accuracy score

print ('Precision%:', precision_score(y_test_xg2, pred_lemma_xg)*100)

xg_lemma_f1score = f1_score(y_test_xg2 , pred_lemma_xg ,average='weighted')    # calculating f1 score

print(xg_lemma_f1score)

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test_xg2 ,pred_lemma_xg))
##Train/Test split is done to see which model is performing best and then we apply the best performing model to our test data to generate the final results*****

# PREDICTING on test data



test_tfidf_lemma = tfidf_lemma[7613:,:]

logreg.fit(train_tfidf_lemma,tweet['target'] )

y_pred = logreg.predict(test_tfidf_lemma)

y_pred
#Fetching Id to differnt frame

y_test_id = test['id']
#Creating Submission dataframe

submission_df_lr = pd.DataFrame({"id":y_test_id,"target":y_pred})

submission_df_lr
#Converting into CSV file for submission

submission_df_lr.to_csv("submission_lr.csv")