import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_tweets = pd.read_csv('/kaggle/input/twitter-coursera/twitter.csv')
df_tweets.describe()
df_tweets = df_tweets.drop(['id'], axis=1)
#Check if there is null value

sns.heatmap(df_tweets.isnull(), yticklabels = False, cbar = False, cmap="Blues")



#The heatmmat shows us that there is no null value
df_tweets.hist(bins = 30, figsize = (13,5), color = 'r')
sns.countplot(df_tweets['label'], label='Count')
df_tweets['length'] = df_tweets['tweet'].apply(len)
df_tweets['length'].plot.hist(bins=100)
df_tweets.describe()
df_tweets[df_tweets['length']==11]['tweet'].iloc[0]
positive = df_tweets[df_tweets['label']==0]

negative = df_tweets[df_tweets['label']==1]
!pip install WordCloud
sentences = df_tweets['tweet'].tolist()
sentences_as_one_string = " ".join(sentences)
from wordcloud import WordCloud, STOPWORDS



plt.figure(figsize=(20,20))

plt.imshow(WordCloud().generate(sentences_as_one_string))
negative_list = negative['tweet'].tolist()

negative_sentences_as_one_string = " ".join(negative_list)



plt.imshow(WordCloud().generate(negative_sentences_as_one_string))
import string

string.punctuation
Test = 'Good morning beautiful people :)... I am having fun learning Machine learning and AI!!'
test_removed = [ char for char in Test if char not in string.punctuation]
test_removed

test_join = ''.join(test_removed)

test_join
import nltk



nltk.download('stopwords')
from nltk.corpus import stopwords



stopwords.words('english')
mini_challenge = 'Here is a mini challenge, that will teach you how to remove stopwords and punctuations!'

challege = [ char     for char in mini_challenge  if char not in string.punctuation ]

challenge = ''.join(challege)

challenge = [  word for word in challenge.split() if word.lower() not in stopwords.words('english')  ] 

challenge
from sklearn.feature_extraction.text import CountVectorizer

sample_data = ['This is the first paper.','This document is the second paper.','And this is the third one.','Is this the first paper?']



vectorizer = CountVectorizer()

X = vectorizer.fit_transform(sample_data)
X.toarray()
# Let's define a pipeline to clean up all the messages 

# The pipeline performs the following: (1) remove punctuations, (2) remove stopwords



def message_cleaning(message):

    Test_punc_removed = [char for char in message if char not in string.punctuation]

    Test_punc_removed_join = ''.join(Test_punc_removed)

    Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]

    return Test_punc_removed_join_clean
# Let's test the newly added function

tweets_df_clean = df_tweets['tweet'].apply(message_cleaning)
print(tweets_df_clean[5]) # show the cleaned up version
print(df_tweets['tweet'][5]) # show the original version
from sklearn.feature_extraction.text import CountVectorizer

# Define the cleaning pipeline we defined earlier

vectorizer = CountVectorizer(analyzer = message_cleaning)

tweets_countvectorizer = vectorizer.fit_transform(df_tweets['tweet'])
tweets = pd.DataFrame(tweets_countvectorizer.toarray())

X = tweets

y = df_tweets['label']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.naive_bayes import MultinomialNB



NB_classifier = MultinomialNB()

NB_classifier.fit(X_train, y_train)
from sklearn.metrics import classification_report, confusion_matrix
# Predicting the Test set results

y_predict_test = NB_classifier.predict(X_test)

cm = confusion_matrix(y_test, y_predict_test)

sns.heatmap(cm, annot=True)
print(classification_report(y_test, y_predict_test))