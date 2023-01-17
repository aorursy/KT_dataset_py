# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



Tweets = pd.read_csv("../input/Tweets.csv")

Tweets.head()

#having a look at the data set
Tweets.info()
Tweets.isna().sum()/len(Tweets)

#cheking for missing values
#droping columns with missing values

Tweets.drop(['airline_sentiment_gold','negativereason_gold','tweet_coord'],axis=1,inplace = True)
SentimentCount = Tweets['airline_sentiment'].value_counts()

SentimentCount
sns.countplot(x='airline_sentiment',data=Tweets,order=['negative','neutral','positive'])

plt.show()
sns.factorplot(x = 'airline_sentiment',data=Tweets,order = ['negative','neutral','positive'],kind = 'count',col_wrap=3,col='airline',size=4,aspect=1,sharex=False,sharey=False)

plt.show()
Tweets['negativereason'].value_counts()
sns.factorplot(x = 'airline',data = Tweets,kind = 'count',hue='negativereason',size=6,aspect=2)

plt.show()
import re

#remove words which are starts with @ symbols

Tweets['text'] = Tweets['text'].map(lambda x:re.sub('@\w*','',str(x)))
#remove special characters except [a-zA-Z]

Tweets['text'] = Tweets['text'].map(lambda x:re.sub('[^a-zA-Z]',' ',str(x)))
#remove link starts with https

Tweets['text'] = Tweets['text'].map(lambda x:re.sub('http.*','',str(x)))
Tweets['text'] = Tweets['text'].map(lambda x:str(x).lower())
from nltk.corpus import stopwords

stop = stopwords.words('english')
Tweets['text'] = Tweets['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
Tweets['text'].head(10)
from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()

Tweets['text'] = Tweets['text'].apply(lambda x: " ".join(lem.lemmatize(x,pos='a') for x in x.split()))

Tweets['text'] = Tweets['text'].apply(lambda x: " ".join(lem.lemmatize(x,pos='r') for x in x.split()))

Tweets['text'] = Tweets['text'].apply(lambda x: " ".join(lem.lemmatize(x,pos='n') for x in x.split()))

Tweets['text'] = Tweets['text'].apply(lambda x: " ".join(lem.lemmatize(x,pos='v') for x in x.split()))
Tweets['text']
#stemming made the words lose their sense
X = Tweets[['text']]
X
y = Tweets['airline_sentiment'].map({'neutral':1,'negative':0,'positive':2})
y
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
from sklearn.feature_extraction.text import TfidfVectorizer



#TFIDF



vector = TfidfVectorizer(stop_words='english',sublinear_tf=True,strip_accents='unicode',analyzer='word',token_pattern=r'\w{2,}',ngram_range=(1,1),max_features=30000)

#token_patten #2 for word length greater than 2>=

X_train_word_feature = vector.fit_transform(X_train['text']).toarray()

X_test_word_feature = vector.transform(X_test['text']).toarray()

print(X_train_word_feature.shape,X_test_word_feature.shape)
#training

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

classifier = LogisticRegression()

classifier.fit(X_train_word_feature,y_train)
y_pred = classifier.predict(X_test_word_feature)

cm = confusion_matrix(y_test,y_pred)

acc_score = accuracy_score(y_test,y_pred)

print(classification_report(y_test,y_pred),'\n',cm,'\n',acc_score)