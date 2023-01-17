# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Bigram = Very Good; here, 2 words "together" make it more meaningful



# annotated data: someone has done the analysis

# airline_sentiment,

# airline_sentiment_confidence, etc.
import numpy as np

import pandas as pd

import re

import nltk

nltk.download('stopwords')

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# other packages: spacey, textplot
tweets = pd.read_csv('/kaggle/input/tweets/Tweets.csv')
tweets.head().T
tweets.info()
pd.value_counts(tweets['airline_sentiment'])   # datasets is un-balanced
tweets.isnull().sum() / len(tweets) * 100
cols = ['airline_sentiment_gold', 'negativereason_gold', 'tweet_coord']

for col in cols:

    del tweets[col]



# del is in-place operation only; drop we have an option

# tweets.drop['airline_sentiment_gold', inplace=True, axis=1]

    

tweets.info()
tweets.airline.nunique()
pd.value_counts(tweets.airline)
pd.value_counts(tweets.airline).sort_values().plot(kind='bar');
sns.countplot(x="airline_sentiment", hue="airline", data=tweets, palette = 'pastel');



# United, US Airways, American: significantly high negative sentiment
tweets_united = tweets[tweets['airline'] == "United"]

sns.countplot(x="airline_sentiment", data=tweets_united, palette = 'pastel');
# tweets_united = tweets.query("airline == United")    <<< runs only on numeric
tweets['word_count'] = tweets['text'].apply(lambda x: len(str(x).split()))

tweets[['text', 'word_count']].head()
from nltk.corpus import stopwords

stop = stopwords.words('english')

print(stop)

print(len(stop))
# count of stopwords

tweets['stopwords'] = tweets['text'].apply(lambda x: len([x for x in str(x).split() if x in stop]))

tweets[['text', 'stopwords']].head()
# count of hashtags

tweets['hashtags'] = tweets['text'].apply(lambda x: len([x for x in str(x).split() if x.startswith('@')]))

tweets[['text', 'hashtags']].head()
# count of digits

tweets['digits'] = tweets['text'].apply(lambda x: len([x for x in str(x).split() if x.isdigit()]))

tweets[['text', 'digits']].head()
# count of uppercase words

tweets['uppercase'] = tweets['text'].apply(lambda x: len([x for x in str(x).split() if x.isupper()]))

tweets[['text', 'uppercase']].head()
##!pip install wordcloud
from wordcloud import WordCloud, STOPWORDS

new_df = tweets[tweets['airline_sentiment'] == 'negative']

words = " ".join(new_df['text'])

cleaned_word = " ".join([word for word in words.split()

                         if 'http' not in word

                         and not word.startswith('@')

                         and word != 'RT'

                         and word not in stop

                        ])

wordcloud = WordCloud(stopwords = STOPWORDS,

                     background_color='black',

                     width=3000,

                     height=2500).generate(cleaned_word)

plt.figure(1, figsize=(12,12))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
pd.value_counts(tweets['negativereason']).plot(kind='bar');



# Customer Service Issue is the main reason for negative sentiment; followed by Late Flight
plt.figure(1, figsize=(20,5))

sns.countplot(x='negativereason', hue='airline', data=tweets)

plt.show();



# US Airways, American, United have the max number of Customer Service issues
tweets['lower'] = tweets['text'].apply(lambda x: str(x).lower())

tweets['lower'].head()
tweets['lower'] = tweets['lower'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

tweets['lower'].head()
tweets['lower'] = tweets['lower'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

tweets['lower'].head()
pd.Series(" ".join(tweets['lower']).split()).value_counts()[:30]
#frequent

remove_words = ['flight', 'get', 'im', 'us', '2']
tweets['lower'] = tweets['lower'].apply(lambda x: " ".join(x for x in x.split() if x not in remove_words))

tweets['lower'].head()
pd.Series(" ".join(tweets['lower']).split()).value_counts()[-10:]
#rare

remove_words = ['saleim', 'virus', '1151', 'lou']
tweets['lower'] = tweets['lower'].apply(lambda x: " ".join(x for x in x.split() if x not in remove_words))

tweets['lower'].head()
from nltk.stem import PorterStemmer

st = PorterStemmer()

tweets['lower'] = tweets['lower'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))

tweets['lower'].head()
tweets.info()
from sklearn.feature_extraction.text import TfidfVectorizer



vectorizer = TfidfVectorizer(max_features=2500,    # pick top 2500 frequent words

                             min_df=7,             # if a word occurs in <7 documents, drop it

                             max_df=0.8)           # if a word occurs in >80% of documents, drop it

processed_features = vectorizer.fit_transform(tweets['lower'])
labels = tweets['airline_sentiment']
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=123)
x_train
from time import time

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn import metrics

from sklearn import model_selection
classifiers = [tree.DecisionTreeClassifier(random_state=1), 

               RandomForestClassifier(n_estimators=200),

               LogisticRegression(max_iter=1000)]

#               LinearDiscriminantAnalysis()]
df_method_score = pd.DataFrame(columns = ['Classifier', 'Train score', 'Test score', 'Training time'])

for classifier in classifiers:

    string = 'currently training: '

    string+= classifier.__class__.__name__

    print(string)

    ts=time()

    classifier.fit(x_train, y_train)

    te = time() - ts

    trn_score = classifier.score(x_train, y_train)

    tst_score = classifier.score(x_test, y_test)

    

    df_method_score = df_method_score.append({'Classifier': classifier.__class__.__name__, 

                                              'Train score': trn_score, 

                                              'Test score': tst_score, 

                                              'Training time': te}, ignore_index = True)
df_method_score
lr = LogisticRegression(max_iter=1000)

lr.fit(x_train, y_train)
y_train_pred = lr.predict(x_train)

model_score = lr.score(x_train, y_train)

print(model_score)

print(metrics.confusion_matrix(y_train, y_train_pred))

print(metrics.classification_report(y_train, y_train_pred))
y_test_pred = lr.predict(x_test)

model_score = lr.score(x_test, y_test)

print(model_score)

print(metrics.confusion_matrix(y_test, y_test_pred))

print(metrics.classification_report(y_test, y_test_pred))



# recall that "positive" was the smallest category
param_grid = {'n_estimators': [100, 200, 300]}

rfcl = RandomForestClassifier()

grid_search = model_selection.GridSearchCV(estimator = rfcl, param_grid = param_grid, cv=5)
grid_search.fit(x_train, y_train)
grid_search.best_params_
best_grid = grid_search.best_estimator_
y_train_pred = best_grid.predict(x_train)

model_score = best_grid.score(x_train, y_train)

print(model_score)

print(metrics.confusion_matrix(y_train, y_train_pred))

print(metrics.classification_report(y_train, y_train_pred))
y_test_pred = best_grid.predict(x_test)

model_score = best_grid.score(x_test, y_test)

print(model_score)

print(metrics.confusion_matrix(y_test, y_test_pred))

print(metrics.classification_report(y_test, y_test_pred))
DT_model = tree.DecisionTreeClassifier(criterion='gini', max_depth=8, min_samples_leaf=10, min_samples_split=30)

DT_model.fit(x_train, y_train)
y_train_pred = DT_model.predict(x_train)

model_score = DT_model.score(x_train, y_train)

print(model_score)

print(metrics.confusion_matrix(y_train, y_train_pred))

print(metrics.classification_report(y_train, y_train_pred))
y_test_pred = DT_model.predict(x_test)

model_score = DT_model.score(x_test, y_test)

print(model_score)

print(metrics.confusion_matrix(y_test, y_test_pred))

print(metrics.classification_report(y_test, y_test_pred))