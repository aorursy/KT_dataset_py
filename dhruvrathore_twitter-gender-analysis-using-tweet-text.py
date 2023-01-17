# Importing necessary packages 

import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
twitter = pd.read_csv('../input/twitter-user-gender-classification/gender-classifier-DFE-791531.csv',encoding='latin-1')
twitter.head()
twitter.shape
twitter.describe()
twitter.info()
twitter.columns
twitter.isnull().sum()
twitter['tweet_count'].value_counts()
twitter['retweet_count'].value_counts()
sns.barplot (x = 'gender', y = 'tweet_count',data = twitter)
sns.barplot (x = 'gender', y = 'retweet_count',data = twitter)
# Visualizing null values to get a better idea of the dataset & it's trends

plt.subplots(figsize=(15,15))
sns.heatmap(twitter.isnull(), cbar=False)
#Dropping irrelevant columns from dataset

twitter = twitter.drop(['_unit_id', '_golden', '_unit_state', '_last_judgment_at', 'gender:confidence', 'profile_yn', 'profile_yn:confidence', 
                        'created', 'fav_number', 'gender_gold', 'name', 'profile_yn_gold', 'profileimage', 'retweet_count', 
                        'tweet_coord', 'tweet_count', 'tweet_created', 'tweet_id', 'tweet_location', 'user_timezone', 
                        '_trusted_judgments'], axis = 1)
twitter.head()
twitter['gender'].count()
twitter['gender'].value_counts(dropna=False) 
sns.countplot(twitter['gender'],label="Gender")
# dropping all the null values from 'gender'

twitter = twitter.dropna(subset=['gender'],how ='any')  
twitter.head()
# Merging the 'text' & 'description' to combine all sorts of text and then find out common words

twitter['text_description'] = twitter['text'].str.cat(twitter['description'], sep=' ')
twitter = twitter.drop(['description','text'],axis=1)
twitter.head()
# Junk words & letters other than the English vocab words are filtered out
import re
def cleaning(s):
    s = str(s)
    s = s.lower()
    s = s.replace(",","")
    s = re.sub('[!@#$_]', '', s)
    s = re.sub('\W,\s',' ',s)
    s = re.sub(r'[^\w]', ' ', s)
    s = re.sub('\s\W',' ',s)
    s = re.sub("\d+", "", s)
    s = re.sub('\s+',' ',s)
    s = s.replace("co","")
    s = s.replace("https","")
    s = s.replace("[\w*"," ")
    return s

twitter['text_description'] = [cleaning(s) for s in twitter['text_description']]
twitter.head()
from collections import Counter
words = Counter()
for twit in twitter['text_description']:
    for x in twit.split(' '):
        words[x] += 1

words.most_common(20)
# Filtering out 'text_description' and printing most commonly used words by elimination stopwords

from nltk.corpus import stopwords
stopwords = stopwords.words('english')
words_filtered = Counter()
for x, y in words.items():
    if not x in stopwords:
        words_filtered[x]=y

words_filtered.most_common(20)
# This will clear out the rest of the remaining junk

import re
def preprocessor(text_description):
    text_description = re.sub("[^a-zA-z]", " ",text_description)
    text_description = re.sub('<[^>]*>', '', text_description)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text_description)
    text_description = (re.sub('[\W]+', ' ', text_description.lower()) + ' ' + ' '.join(emoticons).replace('-', ''))
    return text_description
from nltk.stem import PorterStemmer

porter = PorterStemmer()

def tokenizer(text_description): #tokenizer to break down our twits in individual words
    return text_description.split()

def tokenizer_porter(text_description):
    return [porter.stem(word) for word in text_description.split()]
twitter.text_description

male_sidebar_color = twitter[twitter['gender'] == 'male']['sidebar_color'].value_counts().head(7)
male_sidebar_color_idx = male_sidebar_color.index
male_top_color = male_sidebar_color_idx.values

male_top_color[2] = '000000'
print (male_top_color)

l = lambda x: '#'+x

sns.set_style("darkgrid")
sns.barplot (x = male_sidebar_color, y = male_top_color) 
female_sidebar_color = twitter[twitter['gender'] == 'female']['sidebar_color'].value_counts().head(7)
female_sidebar_color_idx = female_sidebar_color.index
female_top_color = female_sidebar_color_idx.values

female_top_color[2] = '000000'
print (female_top_color)

l = lambda x: '#'+x

sns.set_style("darkgrid")
sns.barplot (x = female_sidebar_color, y = female_top_color)
male_link_color = twitter[twitter['gender'] == 'male']['link_color'].value_counts().head(7)
male_link_color_idx = male_link_color.index
male_top_color = male_link_color_idx.values
male_top_color[1] = '009999'
male_top_color[5] = '000000'
print(male_top_color)

l = lambda x: '#'+x

sns.set_style("whitegrid", {"axes.facecolor": "white"})
sns.barplot (x = male_link_color, y = male_link_color_idx)
female_link_color = twitter[twitter['gender'] == 'female']['link_color'].value_counts().head(7)
female_link_color_idx = female_link_color.index
female_top_color = female_link_color_idx.values

l = lambda x: '#'+x

sns.set_style("whitegrid", {"axes.facecolor": "white"})
sns.barplot (x = female_link_color, y = female_link_color_idx, palette=list(map(l, female_top_color)))
# The frequency of the words will be helpful in classifying the gender of the users.

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Setting up training and testing data 
encoder = LabelEncoder()
y = encoder.fit_transform(twitter['gender'])
X = twitter['text_description']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

tfidf = TfidfVectorizer(lowercase=False,
                        tokenizer=tokenizer_porter)
clf = Pipeline([('vect', tfidf),
                ('clf', LogisticRegression(multi_class='ovr', random_state=0))])

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
print('Accuracy:',accuracy_score(y_test,predictions))
from sklearn.ensemble import RandomForestClassifier

n = range (1,100,10)

tfidf = TfidfVectorizer(lowercase=False,
                        tokenizer=tokenizer_porter)
clf = Pipeline([('vect', tfidf),
                ('clf', RandomForestClassifier(n_estimators = 40, random_state=0))])

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
print('Accuracy:',accuracy_score(y_test,predictions))
from sklearn.svm import SVC

tfidf = TfidfVectorizer(lowercase=False,
                        tokenizer=tokenizer_porter)
clf = Pipeline([('vect', tfidf),
                ('clf', SVC(kernel = 'linear'))])

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
print('Accuracy:',accuracy_score(y_test,predictions))
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y = encoder.fit_transform(twitter['gender'])
X = twitter['sidebar_color']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
from sklearn.linear_model import LogisticRegression

tfidf = TfidfVectorizer(lowercase=False,
                        tokenizer=tokenizer_porter)
clf = Pipeline([('vect', tfidf),
                ('clf', LogisticRegression(multi_class='ovr', random_state=0))])

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
print('Accuracy:',accuracy_score(y_test,predictions))
from sklearn.ensemble import RandomForestClassifier

n = range (1,100,10)

tfidf = TfidfVectorizer(lowercase=False,
                        tokenizer=tokenizer_porter)
clf = Pipeline([('vect', tfidf),
                ('clf', RandomForestClassifier(n_estimators = 40, random_state=0))])

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
print('Accuracy:',accuracy_score(y_test,predictions))
from sklearn.svm import SVC

tfidf = TfidfVectorizer(lowercase=False,
                        tokenizer=tokenizer_porter)
clf = Pipeline([('vect', tfidf),
                ('clf', SVC(kernel = 'linear'))])
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
print('Accuracy:',accuracy_score(y_test,predictions))
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y = encoder.fit_transform(twitter['gender'])
X = twitter['link_color']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
from sklearn.linear_model import LogisticRegression

tfidf = TfidfVectorizer(lowercase=False,
                        tokenizer=tokenizer_porter)
clf = Pipeline([('vect', tfidf),
                ('clf', LogisticRegression(multi_class='ovr', random_state=0))])

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
print('Accuracy:',accuracy_score(y_test,predictions))
from sklearn.ensemble import RandomForestClassifier

n = range (1,100,10)

tfidf = TfidfVectorizer(lowercase=False,
                        tokenizer=tokenizer_porter)
clf = Pipeline([('vect', tfidf),
                ('clf', RandomForestClassifier(n_estimators = 40, random_state=0))])

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
print('Accuracy:',accuracy_score(y_test,predictions))
from sklearn.svm import SVC

tfidf = TfidfVectorizer(lowercase=False,
                        tokenizer=tokenizer_porter)
clf = Pipeline([('vect', tfidf),
                ('clf', SVC(kernel = 'linear'))])
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
print('Accuracy:',accuracy_score(y_test,predictions))