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
df_train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

df_test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
df_train
import re

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
cleaned_tweet=[]

for i in range(0,len(df_train)):

    tweet=re.sub('[^a-zA-Z]',' ',df_train['text'][i])

    tweet=tweet.lower()

    tweet=tweet.split()

    ps=PorterStemmer()

    tweet=[ps.stem(word) for word in tweet if not word in set(stopwords.words('english'))]

    tweet=' '.join(tweet)

    cleaned_tweet.append(tweet)
cleaned_tweet_test=[]

for i in range(0,len(df_test)):

    tweet=re.sub('[^a-zA-Z]',' ',df_test['text'][i])

    tweet=tweet.lower()

    tweet=tweet.split()

    ps=PorterStemmer()

    tweet=[ps.stem(word) for word in tweet if not word in set(stopwords.words('english'))]

    tweet=' '.join(tweet)

    cleaned_tweet_test.append(tweet)
print(cleaned_tweet_test)
from sklearn.feature_extraction.text import CountVectorizer

cv=CountVectorizer()

X = cv.fit_transform(cleaned_tweet)

test_vectors = cv.transform(cleaned_tweet_test)

y=df_train['target']
from sklearn import feature_extraction, linear_model, model_selection, preprocessing

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
classifier = linear_model.RidgeClassifier()

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

y_pred
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

print(cm)

accuracy_score(y_test, y_pred)
y_pred_test = classifier.predict(test_vectors)

y_pred_test
testpred=pd.DataFrame(y_pred_test)
sub_df=pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

datasets=pd.concat([sub_df['id'],testpred],axis=1)

datasets.columns=['id','target']

datasets.to_csv('NewSubmission.csv',index=False)