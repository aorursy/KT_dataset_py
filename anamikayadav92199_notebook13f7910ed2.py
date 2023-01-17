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
import seaborn as sns 

import matplotlib.pyplot as plt
twitter_df = pd.read_csv('/kaggle/input/twitter-sentiment-analysis-hatred-speech/train.csv')

twitter_test = pd.read_csv('/kaggle/input/twitter-sentiment-analysis-hatred-speech/test.csv')
twitter_df.head()
twitter_df.describe
twitter_df.info()
twitter_df = twitter_df.drop(['id'],axis = 1)

twitter_test = twitter_test.drop(['id'],axis = 1)
twitter_df
twitter_test
sns.heatmap(twitter_df.isnull(),yticklabels = False,cbar = False , cmap = "Blues")
twitter_df.hist(bins = 40,figsize = (14,5),color = 'r')
sns.countplot(twitter_df['label'],label = 'count')
twitter_df['length'] = twitter_df['tweet'].apply(len)
twitter_df
twitter_df['length'].plot(bins = 100,kind = 'hist')
twitter_df.describe
twitter_df.describe()
twitter_df[twitter_df['length']==11]['tweet'].iloc[0]
twitter_df[twitter_df['length']==84]['tweet'].iloc[0]
positive = twitter_df[twitter_df['label']==0]
negative = twitter_df[twitter_df['label']==1]
positive
negative
sentences = twitter_df['tweet'].tolist()
sentences
len(sentences)
sentences_as_one_string = " ".join(sentences)
!pip install Wordcloud
!pip install WordCloud
from wordcloud import WordCloud
import numpy as np # linear algebra

import pandas as pd 

%pylab

import seaborn as sns 

import matplotlib.pyplot as plt

plt.figure(figsize(20,20))

plt.imshow(WordCloud().generate(sentences_as_one_string))
import string

string.punctuation
Test = 'Good morning beautiful people :)... I am having fun learning Machine learning and AI!!'
Test_punc_rem = [char for char in Test if char not in string.punctuation]
Test_punc_rem
Test_punc_rem_join = ''.join(Test_punc_rem)

Test_punc_rem_join
import nltk # Natural Language tool kit 



nltk.download('stopwords')
import nltk # Natural Language tool kit 



nltk.download('stopwords')
import re                                  # library for regular expression operations

import string                              # for string operations



from nltk.corpus import stopwords          # module for stop words that come with NLTK

from nltk.stem import PorterStemmer        # module for stemming

from nltk.tokenize import TweetTokenizer 
stopwords.words('english')
Test_punc_rem_join_clean = [word for word in Test_punc_rem_join.split() if word.lower() not in stopwords.words('english')]
Test_punc_rem_join_clean 
import sklearn

from sklearn.feature_extraction.text import CountVectorizer
sample_data = ['This is the first paper.','This document is the second paper.','And this is the third one.','Is this the first paper?']
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sample_data)
print(X.toarray())
def message_cleaning(message):

    Test_punc_rem = [char for char in Test if char not in string.punctuation]

    Test_punc_rem_join = ''.join(Test_punc_rem)

    Test_punc_rem_join_clean = [word for word in Test_punc_rem_join.split() if word.lower() not in stopwords.words('english')]

    return Test_punc_rem_join_clean
# Let's test the newly added function

twitter_df_clean = twitter_df['tweet'].apply(message_cleaning)
print(twitter_df_clean[5]) 
print(twitter_df['tweet'][5])
from sklearn.feature_extraction.text import CountVectorizer

# Define the cleaning pipeline we defined earlier

vectorizer = CountVectorizer(analyzer = message_cleaning)

twitter_countvectorizer = CountVectorizer(analyzer = message_cleaning,dtype = 'uint8').fit_transform(twitter_df['tweet']).toarray()
twitter_countvectorizer.shape
x = twitter_countvectorizer

y = twitter_df['label']
x.shape
y.shape
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
from sklearn.naive_bayes import MultinomialNB



NB_classifier = MultinomialNB()

NB_classifier.fit(x_train, y_train)
from sklearn.metrics import classification_report, confusion_matrix


y_predict_test = NB_classifier.predict(x_test)

cm = confusion_matrix(y_test,y_predict_test)



sns.heatmap(cm,annot = True)
print(classification_report(y_test, y_predict_test))
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35)



from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
# Fitting Logistic Regression

log_reg_model = LogisticRegression()

log_reg_model.fit(x_train, y_train)
# Scoring

train_prediction = log_reg_model.predict(x_train)

test_prediction = log_reg_model.predict(x_test)

accuracy_train = accuracy_score(train_prediction, y_train)

accuracy_test = accuracy_score(test_prediction, y_test)



print(f"Score on training set: {accuracy_train}")

print(f"Score on test set: {accuracy_test}")