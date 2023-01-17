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
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#creating a dataframe object from out train csv file
tweets_df=pd.read_csv('/kaggle/input/twitter-sentiment-analysis-hatred-speech/train.csv')

tweets_df.info()
#label=0 tells the tweet is positive and 1 tells that it was a negative tweet
tweets_df.describe()
tweets_df.head()
#here, we can see , we do not need the coloumn id , since it is not required for analysis, so we drop it
tweets_df=tweets_df.drop(['id'],axis=1) #axis=1 because we want to drop the entire column
tweets_df.isnull().sum()
#there are no null entries or NaN
sns.heatmap(tweets_df.isnull(),yticklabels=False,cbar=False,cmap='Blues')
#Everything is blue , there are no null values
tweets_df['label'].hist(bins=30,figsize=(13,5),color='g')
#positive tweets are around 29000 and negative tweets are around 2400
sns.countplot(tweets_df['label'])
#this gives a clearer and prioritized picture.
#let's find the length of tweets and see their popularity
# creating a column for length of tweets in our data set
tweets_df['length']=tweets_df['tweet'].apply(len) # apply len will return the length of each tweet and assign it to our new column
#let's plot the length description data
tweets_df['length'].plot(bins=100,kind='hist',color='g')
tweets_df.describe()
#minimum length of tweets is 11 and max is 274 while avg is 85
#let's see the what is the tweet with min and max lengths
print(tweets_df[tweets_df['length']==11].iloc[0:])
print('-------------------------------------------------------------------------------------------------')
print(tweets_df[tweets_df['length']==274].iloc[0:])
#both are positive/harmless tweets
#most common length tweets are...
print(tweets_df[tweets_df['length']==85].iloc[0:2])
positive=tweets_df[tweets_df['label']==0]
negative=tweets_df[tweets_df['label']==1]
#gets the object for pos and neg tweets
#grab the tweet column and convert into one massive string
sentences=tweets_df['tweet'].tolist()
sentences=''.join(sentences)
from wordcloud import WordCloud
plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(sentences))
#let's see the positive tweets' words
plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(''.join(positive['tweet'].tolist())))
#LOVE HAPPY USER THANKFUL POSITIVE TIME GIRL etc are the words used in positive tweets
#similarly, for negative tweets
plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(''.join(negative['tweet'].tolist())))
import string #for punctuation
import nltk #natural language tool kit
from nltk.corpus import stopwords

print(string.punctuation)
print('------------------------------------------------------------------------------------------------------------------------')
print(stopwords.words('english'))
def text_cleaning(sentence):
    sentence_punc_removed=[letter for letter in sentence if letter not in string.punctuation]
    sentence_punc_removed=''.join(sentence_punc_removed)
    sentence_clean=[word for word in sentence_punc_removed.split() if word.lower() not in stopwords.words('english')]
    return sentence_clean
#NO NEED TO RUN THIS>>>>> JUST TO GET THE IDEA
# tweets_df_clean=tweets_df['tweet'].apply(text_cleaning)
# #we get list of clean messages
# print(tweets_df_clean.head)
from sklearn.feature_extraction.text import CountVectorizer

#here, we performed data cleaning and count vectorization sequentially altogether !

tweets_vectorizer=CountVectorizer(analyzer=text_cleaning,dtype='uint8').fit_transform(tweets_df['tweet']) #transforms text into numeric vectorized format
X=tweets_vectorizer.toarray()
print(X.shape)
y=tweets_df['label']
#Let's first predict our accuracy by splitting our train data into test(say, 20 % of train) and train 
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2) #setting up train and test datasets

from sklearn.naive_bayes import MultinomialNB

NB_classifier=MultinomialNB()
NB_classifier.fit(X_train,y_train) #training our model using test datasets
from sklearn.metrics import confusion_matrix , classification_report

y_test_predictions=NB_classifier.predict(X_test)
cm=confusion_matrix(y_test,y_test_predictions)
sns.heatmap(cm,annot=True)
# this means 5700+250 are correctly predicted whereas 220+180 are falsely predicted
print(classification_report(y_test,y_test_predictions))
# accuracy - 0.94 (okay !)
print(X_train)
tweets_test_df=pd.read_csv('/kaggle/input/twitter-sentiment-analysis-hatred-speech/test.csv')

tweets_test_df.head()
tweets_test_df=tweets_test_df.drop(['id'],axis=1)
print(tweets_test_df.head())
from sklearn.feature_extraction.text import CountVectorizer

tweets_test_classifier=CountVectorizer(analyzer=text_cleaning,dtype='uint8').fit_transform(tweets_test_df['tweet'])

X1=tweets_test_classifier.toarray()
from sklearn.naive_bayes import MultinomialNB

NB_test_classifier=MultinomialNB()
NB_test_classifier.fit(X[:,0:31242],y)
y_final_predictions=NB_test_classifier.predict(X1)
new_column=y_final_predictions.T
tweets_test_df['label']=new_column

print(y.shape)
print(X[:,0:31242])
tweets_test_df.head