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
tweets_test_df = pd.read_csv('/kaggle/input/twitter-sentiment-analysis-hatred-speech/test.csv')

tweets_train_df = pd.read_csv('/kaggle/input/twitter-sentiment-analysis-hatred-speech/train.csv')
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
tweets_train_df.head()
tweets_train_df.info()
tweets_train_df.describe()
tweets_train_df['tweet']
tweets_train_df.drop(['id'], axis=1, inplace=True)
tweets_train_df.head()
sns.heatmap(tweets_train_df.isnull(), yticklabels = False, cbar = False, cmap = 'Blues')
tweets_train_df.hist(bins= 30, figsize = (12,5), color = 'b')
# These plots clearly shows that its a complete unbalanced data.

sns.countplot(x=tweets_train_df['label'] ,data=tweets_train_df)
tweets_test_df.head()
tweets_train_df['lengths'] = tweets_train_df['tweet'].apply(len)
tweets_train_df.head()
#distribution of tweets

tweets_train_df['lengths'].plot(bins=100, kind = 'hist')
tweets_train_df.describe()
#Min length is 11, so let's see it

tweets_train_df[tweets_train_df['lengths']==11]['tweet'].iloc[0]
# lets viwe the meesage with average length

tweets_train_df[tweets_train_df['lengths']==85]
# Now separting positive and negative tweets

positive = tweets_train_df[tweets_train_df['label']==0]

positive
negative = tweets_train_df[tweets_train_df['label']==1]

negative
# Plot the word cloud

from wordcloud import WordCloud
sentences = tweets_train_df['tweet'].tolist()
#All tweets has been converted to a list

#sentences
len(sentences)
#Joining sentences (combining all the sentences that we have)

sentences_as_single_string = " ".join(sentences)
plt.figure(figsize=(20,20))

plt.imshow(WordCloud().generate(sentences_as_single_string))
# Lets plot wordcloud of negative words.

negative_sentences = negative['tweet'].tolist()

negative_string = " ".join(negative_sentences)
plt.figure(figsize=(20,20))

plt.imshow(WordCloud().generate(negative_string))
import string

string.punctuation
sample = 'Hi! everyone :) ; enjoy learning real world example of NLP !.....'
sample_punc_removed = [char   for char in sample if char not in string.punctuation]
sample_punc_removed
#Now join again

test_punc_removed_string = ''.join(sample_punc_removed)

test_punc_removed_string
# Second and efficient method

out = sample.translate(str.maketrans('', '', string.punctuation))

out
# Third and basic method

punc_removed = []

for char in sample:

    if char not in string.punctuation:

        punc_removed.append(char)

        

punc_removed_join = ''.join(punc_removed)

punc_removed_join
# The Question is what are stopwords, so lets download and plot them using Natural languae toolkit

import nltk #Natural language toolkit

nltk.download('stopwords')
#Lets import stopword and see the common words stored there. These are words that don't convey any specific information

from nltk.corpus import stopwords

stopwords.words('english')
# Lets remove common words and retain only unique words

test_punc_removed_string_clean = [word for word in test_punc_removed_string.split() if word.lower() not in stopwords.words('english')]
test_punc_removed_string_clean
# Lets try Pipeline approach to accomplish removal of punctuation and stopwords

test_sample = 'A sample to learn,; that how can we remove punctuations and stopwords in a pipeline fashion!!!'

pipe_punc_removed_cleaned = [char for char in test_sample if char not in string.punctuation]

pipe_punc_removed_cleaned = ''.join(pipe_punc_removed_cleaned)

pipe_punc_removed_cleaned = [word for word in pipe_punc_removed_cleaned.split() if word.lower() not in stopwords.words('english')]

pipe_punc_removed_cleaned
# This will take unique words utilized in text as features, and then count that how many time each word is utilized in that sentence. 

from sklearn.feature_extraction.text import CountVectorizer

sample_new = ['This is first method.', 'This method is the second method.', 'This new one is the third one.' ]
vectorizer = CountVectorizer()

X = vectorizer.fit_transform(sample_new)
#Lets see the extracted feature names (unique words)

print (vectorizer.get_feature_names())
X
#We can see that in first sentence, only four features (unique words) are present there (first three and last feature).

#In second sentence of sample_new, word method is repeated two times, so we can see 2 at corresponding feature position

print(X.toarray())
# We can see with following example that Countvectroizer always convert each character to lower case before transforming.

second_sample = ['Hello World.', 'Hello Hello World', 'Hello World world world']

XX = vectorizer.fit_transform(second_sample)

print(XX.toarray())
def text_cleaning(text):

    remv_punc = [char for char in text.lower() if char not in string.punctuation]

    remv_punc_join = ''.join(remv_punc)

    remv_punc_clean = [word for word in remv_punc_join.split() if word.lower() not in stopwords.words('english')]

    return remv_punc_clean
#Lets visualize the newly created function

tweets_df_clean = tweets_train_df['tweet'].apply(text_cleaning)

print(tweets_df_clean[5])
#Actual version of selected tweet, we can see that we have removed all punctuations and stopwords using a single user defined function

tweets_train_df['tweet'][5]
# Now we will use "analyser" to apply countvectorization. 

#In other words, analyzer is an preprocess step before applying countVectorization step.

vectorizer_analyzer = CountVectorizer(analyzer = text_cleaning)

countvectorizer_tweets = CountVectorizer(analyzer= text_cleaning, dtype= 'uint8').fit_transform(tweets_train_df['tweet']).toarray()
countvectorizer_tweets.shape
X_features = countvectorizer_tweets

y_label = tweets_train_df['label']
X_features.shape
y_label.shape
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_features, y_label, test_size = 0.2, random_state = 1)
from sklearn.naive_bayes import MultinomialNB

NaiveBclassifier = MultinomialNB()

NaiveBclassifier.fit(X_train,y_train)
from sklearn.metrics import classification_report, confusion_matrix
# Predicting test cases

y_pred_test = NaiveBclassifier.predict(X_test)
# Confusion matrix

cm = confusion_matrix(y_test, y_pred_test)

sns.heatmap(cm, annot= True)