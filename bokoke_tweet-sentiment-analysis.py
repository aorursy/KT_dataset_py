import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#pd.options.display.max_rows = 9999
import seaborn as sns
import matplotlib.pyplot as plt


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
plt.style.use('ggplot')
%matplotlib inline
tweet_df = pd.read_csv('/kaggle/input/tweet-sentiment/twitter.csv')
tweet_df.info()
tweet_df.head(10)
tweet_df = tweet_df.drop('id', 1) #drop the 'id' column
sns.heatmap(tweet_df.isnull(), yticklabels=False, cbar=False, cmap='Blues')
tweet_df.hist(bins=30, figsize=(13,5), color='r')
sns.countplot(tweet_df['label'], label='Count')
tweet_df['length'] = tweet_df['tweet'].apply(len)
tweet_df.head(10)
tweet_df.length.plot(bins=100, kind='hist')
plt.xlabel('Length')
plt.show()
tweet_df.describe()
tweet_df[tweet_df['length']==11]['tweet'].iloc[0]
tweet_df[tweet_df['length']==84]['tweet'].iloc[0]
positive = tweet_df[tweet_df.label==0]
positive.shape
negative = tweet_df[tweet_df.label==1]
negative.shape
sentences = tweet_df.tweet.to_list()
sentences_as_one_string = " ".join(sentences)
from wordcloud import WordCloud

wordCloud = WordCloud(max_words=100, background_color="white").generate(sentences_as_one_string)
plt.figure(figsize=(15,15))
plt.imshow(wordCloud, interpolation='bilinear')
plt.axis('off')

#to save the image
wordCloud.to_file("first_review.png")
negtive_sens_as_one_str = " ".join(negative.tweet.to_list())

wordCloud = WordCloud(max_words=100, background_color="white").generate(negtive_sens_as_one_str)
plt.figure(figsize=(15,15))
plt.imshow(wordCloud, interpolation='bilinear')
plt.axis('off')
import string

string.punctuation
test = 'Good morning!!!, guys :Vv. You are gonna have  a breakfast'
test_punc_removed = ''.join(ch for ch in [char for char in test if char not in string.punctuation])
test_punc_removed
from nltk.corpus import stopwords

stopwords.words('english')[:10]
test_punc_removed_clean = [word for word in test_punc_removed.split()
                            if word.lower() not in stopwords.words('english')]
test_punc_removed_clean
from sklearn.feature_extraction.text import CountVectorizer

sample = ['This is the first paper', 'This paper is the second paper', 'And this is the third one',
          'Is this the first paper?']
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sample)
vectorizer.get_feature_names()
X.toarray()
import matplotlib.image as mpimg

img = mpimg.imread('../input/tweet-sentiment/Capture.PNG')
plt.figure(figsize=(15,15))
plt.axis('off')
plt.imshow(img)
def tweet_cleaning(tweet):
    tweet_punc_removed = ''.join(ch for ch in [char for char in tweet if char not in string.punctuation])
    tweet_punc_removed_clean = [word for word in tweet_punc_removed.split() 
                                if word.lower() not in stopwords.words('english')]
    return tweet_punc_removed_clean
tweet_df_clean = tweet_df.tweet.apply(tweet_cleaning)
tweet_df_clean[5]
tweet_df.tweet[5]
vectorizer = CountVectorizer(analyzer=tweet_cleaning)
tweets_countvectorizer = CountVectorizer(analyzer=tweet_cleaning, dtype='uint8').fit_transform(np.array(tweet_df.tweet))
tweets_countvectorizer.shape
X = tweets_countvectorizer
y = tweet_df['label']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()
classifier.fit(X_train, y_train)
from sklearn.metrics import classification_report, confusion_matrix

y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, square=True, fmt='d', linewidths=.2, cbar=0, cmap = "Paired")
plt.ylabel('True labels')
plt.xlabel('Predicted labels')
print(classification_report(y_test, y_pred))